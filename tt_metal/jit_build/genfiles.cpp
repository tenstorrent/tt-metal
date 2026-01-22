// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build/genfiles.hpp"

#include <circular_buffer_constants.h>
#include "data_format.hpp"
#include <cstdint>
#include <tt_backend_api_types.hpp>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <ostream>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/unreachable.hpp>
#include "build.hpp"
#include "hlk_desc.hpp"
#include "jit_build_options.hpp"
#include "jit_build_settings.hpp"
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"
#include "impl/kernels/kernel.hpp"

enum class UnpackToDestMode : uint8_t;

namespace fs = std::filesystem;

using namespace std;

namespace tt::tt_metal {

namespace {

string get_kernel_source_to_include(const KernelSource& kernel_src) {
    switch (kernel_src.source_type_) {
        case KernelSource::FILE_PATH: {
            return "#include \"" + kernel_src.path_.string() + "\"\n";
        }
        case KernelSource::SOURCE_CODE: return kernel_src.source_;
    }
    ttsl::unreachable();
}

// Simple kernel syntax refers to declaring kernel entry point as just "void kernel_main()"
// This is in contrast to legacy syntax: "namespace NAMESPACE { void MAIN() { ... } }"
// Eventually we may want to deprecate legacy syntax, but for now we support both.
// This namespace isolates the logic related to simple kernel syntax.
namespace simple_kernel_syntax {

const std::regex kernel_main_pattern(R"(\bvoid\s+kernel_main\s*\(\s*\)\s*\{)");

size_t find_kernel_main_definition(const string& source) {
    std::smatch match;
    if (std::regex_search(source, match, kernel_main_pattern)) {
        return static_cast<size_t>(match.position());
    }
    return string::npos;
}

bool has_legacy_syntax_markers(const string& source) {
    // Check for legacy syntax markers: "namespace NAMESPACE" or "void MAIN"
    // If found, the file uses legacy syntax (possibly mixed with kernel_main for data movement)
    return source.find("namespace NAMESPACE") != string::npos || source.find("void MAIN") != string::npos;
}

size_t count_kernel_main_definitions(const string& source) {
    auto begin = std::sregex_iterator(source.begin(), source.end(), kernel_main_pattern);
    auto end = std::sregex_iterator();
    return std::distance(begin, end);
}

bool is_used_in_source(const string& source) {
    // Use simplified syntax only if kernel_main is found AND no legacy markers present.
    // This handles kernels with multiple entrypoints that have kernel_main() for data movement
    // but legacy syntax for compute - we must not transform those.
    if (find_kernel_main_definition(source) == string::npos) {
        return false;
    }
    if (has_legacy_syntax_markers(source)) {
        return false;
    }
    // Multiple kernel_main() with simplified syntax for compute is not supported.
    // We cannot determine which kernel_main belongs to compute. Use legacy syntax for compute.
    if (count_kernel_main_definitions(source) > 1) {
        throw std::runtime_error(
            "Multiple kernel_main() definitions found. Kernels with multiple entrypoints must use "
            "legacy syntax (namespace NAMESPACE { void MAIN { } }) for the compute path.");
    }
    return true;
}

// Transforms simplified kernel to legacy format:
//   - Splits at "void kernel_main()"
//   - Preamble (#includes) stays outside namespace
//   - Function body wrapped in namespace, renamed to func_name
string transform_to_legacy_syntax(const string& source, const char* ns_name, const char* func_name) {
    size_t func_pos = find_kernel_main_definition(source);
    if (func_pos == string::npos) {
        throw std::runtime_error("Could not find 'void kernel_main() {' in source");
    }

    string preamble = source.substr(0, func_pos);
    string function_part = source.substr(func_pos);

    // Rename kernel_main -> func_name
    size_t name_pos = function_part.find("kernel_main");
    if (name_pos != string::npos) {
        function_part.replace(name_pos, strlen("kernel_main"), func_name);
    }

    ostringstream result;
    result << preamble;
    result << "namespace " << ns_name << " {\n";
    result << function_part;
    result << "\n}  // namespace " << ns_name << "\n";
    return result.str();
}
}  // namespace simple_kernel_syntax

// Generates TRISC prolog: #define + #include for defines_generated.h
string build_trisc_prolog(const char* trisc_define) {
    ostringstream prolog;
    prolog << "#define " << trisc_define << "\n";
    prolog << "#include \"defines_generated.h\"\n";
    return prolog.str();
}

// Writes content to a file, throwing on failure
void write_file(const string& path, const string& content) {
    std::ofstream f(path);
    if (!f) {
        throw std::runtime_error("Cannot create file: " + path);
    }
    f << content;
    if (!f) {
        throw std::runtime_error("Failed to write file: " + path);
    }
}

}  // namespace

void jit_build_genfiles_kernel_include(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src) {
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for BRISC/NCRISC/ERISC user kernel");

    string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";
    string kernel_header = out_dir + "kernel_includes.hpp";

    const string& kernel_src_to_include = get_kernel_source_to_include(kernel_src);
    write_file(kernel_header, kernel_src_to_include);
}

void jit_build_genfiles_triscs_src(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src) {
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for TRISCs");

    const string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";
    const string unpack_cpp = out_dir + "chlkc_unpack.cpp";
    const string math_cpp = out_dir + "chlkc_math.cpp";
    const string pack_cpp = out_dir + "chlkc_pack.cpp";

    // Read content for syntax detection (needed for both paths)
    const string kernel_content = kernel_src.get_content();
    const bool simplified = simple_kernel_syntax::is_used_in_source(kernel_content);

    if (simplified) {
        log_trace(tt::LogBuildKernels, "Detected simplified compute kernel syntax (kernel_main)");
    } else {
        log_warning(
            tt::LogBuildKernels,
            "Compute kernel '{}' uses deprecated 'namespace NAMESPACE {{ void MAIN {{ }} }}' syntax. "
            "Please migrate to simplified 'void kernel_main() {{ }}' syntax.",
            settings.get_full_kernel_name());
    }

    // Build prologs (same for both syntaxes)
    const string unpack_prolog = build_trisc_prolog("TRISC_UNPACK");
    const string math_prolog = build_trisc_prolog("TRISC_MATH");
    const string pack_prolog = build_trisc_prolog("TRISC_PACK");

    // Determine kernel source for each TRISC.
    //
    // Why the if-else structure is necessary:
    // - Simplified syntax: MUST transform source, so we inline the transformed content
    // - Legacy syntax: use existing get_kernel_source_to_include() which returns:
    //   - FILE_PATH: #include directive (preserves file refs in compiler errors)
    //   - SOURCE_CODE: the source directly
    string unpack_src, math_src, pack_src;
    if (simplified) {
        // For FILE_PATH sources, add #line directive to preserve original file's line numbers
        // in compiler diagnostics and __LINE__ macro. This ensures error messages reference
        // the original kernel file, not the generated file.
        string line_directive;
        if (kernel_src.source_type_ == KernelSource::FILE_PATH) {
            line_directive = "#line 1 \"" + kernel_src.path_.string() + "\"\n";
        }
        unpack_src = line_directive + simple_kernel_syntax::transform_to_legacy_syntax(kernel_content, "chlkc_unpack", "unpack_main");
        math_src = line_directive + simple_kernel_syntax::transform_to_legacy_syntax(kernel_content, "chlkc_math", "math_main");
        pack_src = line_directive + simple_kernel_syntax::transform_to_legacy_syntax(kernel_content, "chlkc_pack", "pack_main");
    } else {
        // Legacy: use existing helper that handles FILE_PATH vs SOURCE_CODE appropriately
        const string src = get_kernel_source_to_include(kernel_src);
        unpack_src = math_src = pack_src = src;
    }

    // Generate the three TRISC source files
    write_file(unpack_cpp, unpack_prolog + unpack_src);
    write_file(math_cpp, math_prolog + math_src);
    write_file(pack_cpp, pack_prolog + pack_src);

    // Here we generate an auxiliary header with defines added via add_define() call
    // this header is then included from the kernel
    // We also append the include path to generated dir to hlkc cmldline.
    const string generated_defines_fname = out_dir + "defines_generated.h";
    std::ofstream gen_defines_file(generated_defines_fname);
    if (!gen_defines_file) {
        throw std::runtime_error("Cannot create file: " + generated_defines_fname);
    }
    settings.process_defines([&gen_defines_file](const string& define, const string& value) {
        gen_defines_file << "#define " << define << " " << value << endl;
    });
    if (!gen_defines_file) {
        throw std::runtime_error("Failed to write file: " + generated_defines_fname);
    }
}

namespace {

std::string data_format_vec_to_string(const vector<DataFormat>& formats) {
    std::string formats_string;
    for (const auto& format : formats) {
        formats_string += to_string((int)format) + ",";
    }
    return formats_string;
}

std::string create_formats_array_string(
    const std::string& array_type, const std::string& array_name, int array_size, const std::string& array_data) {
    stringstream str_stream;

    str_stream << array_type << " " << array_name << "[" << array_size << "] = {" << endl;
    str_stream << "    " << array_data << endl;
    str_stream << "};" << endl;

    return str_stream.str();
}

std::pair<std::vector<DataFormat>, std::vector<DataFormat>> generate_unpack_data_formats(
    tt_hlk_desc& desc,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    std::vector<UnpackToDestMode> unpack_to_dest_mode) {
    vector<DataFormat> src_formats = tt::get_unpack_src_formats(desc.buf_dataformat_arr);

    vector<DataFormat> dst_formats = tt::get_unpack_dst_formats(
        desc.buf_dataformat_arr, unpack_conditional_dst_format, fp32_dest_acc_en, std::move(unpack_to_dest_mode));

    TT_ASSERT(src_formats.size() == NUM_CIRCULAR_BUFFERS);
    TT_ASSERT(dst_formats.size() == NUM_CIRCULAR_BUFFERS);

    return std::make_pair(src_formats, dst_formats);
}

void emit_unpack_data_formats(
    const std::string& unpack_data_format_descs,
    const std::vector<DataFormat>& src_formats_all_cbs,
    const std::vector<DataFormat>& dst_formats_all_cbs) {
    // TODO: we should be emitting "unsigned char", no reason to use up 4B per data format
    ofstream file_stream;
    file_stream.open(unpack_data_format_descs);
    file_stream << "#pragma once\n\n";
    file_stream << create_formats_array_string(
        "constexpr std::int32_t",
        "unpack_src_format",
        NUM_CIRCULAR_BUFFERS,
        data_format_vec_to_string(src_formats_all_cbs));
    file_stream << create_formats_array_string(
        "constexpr std::int32_t",
        "unpack_dst_format",
        NUM_CIRCULAR_BUFFERS,
        data_format_vec_to_string(dst_formats_all_cbs));
    file_stream.close();
}

std::pair<std::vector<DataFormat>, std::vector<DataFormat>> generate_pack_data_formats(
    tt_hlk_desc& desc,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    bool bfp8_pack_precise,
    const tt::ARCH arch) {
    vector<DataFormat> src_formats = tt::get_pack_src_formats(
        desc.buf_dataformat_arr,
        unpack_conditional_dst_format,
        fp32_dest_acc_en,
        bfp8_pack_precise,
        false,
        arch);

    vector<DataFormat> dst_formats = tt::get_pack_dst_formats(
        desc.buf_dataformat_arr);

    TT_ASSERT(src_formats.size() == NUM_CIRCULAR_BUFFERS);
    TT_ASSERT(dst_formats.size() == NUM_CIRCULAR_BUFFERS);

    return std::make_pair(src_formats, dst_formats);
}

void emit_pack_data_formats(
    const std::string& pack_data_format_descs,
    const std::vector<DataFormat>& src_formats_all_cbs,
    const std::vector<DataFormat>& dst_formats_all_cbs) {
    ofstream file_stream;
    file_stream.open(pack_data_format_descs);
    file_stream << "#pragma once\n\n";
    file_stream << create_formats_array_string(
        "constexpr unsigned char",
        "pack_src_format",
        NUM_CIRCULAR_BUFFERS,
        data_format_vec_to_string(src_formats_all_cbs));
    file_stream << create_formats_array_string(
        "constexpr unsigned char",
        "pack_dst_format",
        NUM_CIRCULAR_BUFFERS,
        data_format_vec_to_string(dst_formats_all_cbs));

    // budabackend-style format array
    // file_stream << create_formats_array_string("const std::int32_t", "pack_src_format", 16,
    // data_format_vec_to_string(src_formats)); file_stream << create_formats_array_string("const std::int32_t",
    // "pack_dst_format", 16, data_format_vec_to_string(dst_formats));

    file_stream.close();
}

void equalize_data_format_vectors(std::vector<DataFormat>& v1, std::vector<DataFormat>& v2) {
    // Check that the vectors have the same size
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("vectors have different sizes");
    }

    for (size_t i = 0; i < v1.size(); ++i) {
        // Check if both values are the same
        if (v1[i] == v2[i]) {
            // Do nothing
            continue;
        }

        // values are not same:

        // 1) check if both values are non-DataFormat::Invalid -- not allowed
        if (v1[i] != DataFormat::Invalid && v2[i] != DataFormat::Invalid) {
            throw std::invalid_argument("vectors have different non-DataFormat::Invalid values");
        }

        // 2) if only one of the values is non-DataFormat::Invalid, assign the same value to the other vector
        if (v1[i] != DataFormat::Invalid) {
            v2[i] = v1[i];
        } else {
            v1[i] = v2[i];
        }
    }
}

void generate_data_format_descriptors(JitBuildOptions& options, const tt::ARCH arch) {
    string out_file_name_base = "chlkc_";
    string out_file_name_suffix = "_data_format.h";
    string unpack_data_format_descs = options.path + out_file_name_base + "unpack" + out_file_name_suffix;
    string pack_data_format_descs = options.path + out_file_name_base + "pack" + out_file_name_suffix;

    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = options.hlk_desc;

    // Determine dst format under ambiguous conditions (either or both l1 input & output formats are Float32)
    ExpPrecision exp_prec = tt::get_data_exp_precision(desc.buf_dataformat_arr);
    DataFormat unpack_conditional_dst_format = (exp_prec == ExpPrecision::A) ? DataFormat::Float16 : DataFormat::Float16_b;

    if (options.fp32_dest_acc_en && (tt::is_all_fp32_formats(desc.buf_dataformat_arr) || (exp_prec == ExpPrecision::B))) {
        unpack_conditional_dst_format = DataFormat::Tf32;
    }

    tt::check_valid_formats_in_out_data_formats(
        desc.buf_dataformat_arr);

    vector<DataFormat> unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs;
    tie(unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs) = generate_unpack_data_formats(desc, unpack_conditional_dst_format, options.fp32_dest_acc_en, options.unpack_to_dest_mode);

    vector<DataFormat> pack_src_formats_all_cbs, pack_dst_formats_all_cbs;
    tie(pack_src_formats_all_cbs, pack_dst_formats_all_cbs) =
        generate_pack_data_formats(desc, unpack_conditional_dst_format, options.fp32_dest_acc_en, options.bfp8_pack_precise, arch);

    // equalize "upack src" and "pack dst" data format vectors
    // both "unpack src" and "pack dst" refer to data in L1, "unpack src" == L1, and "pack dst" == L1
    // in order to allow any CB to be read and written to/from L1, these formats should be the same (one cannot be
    // DataFromat::Invalid if the other is set) if both formats are DataFormat::Invalid then this CB is not used this
    // allows any CB to be used as both in & out in non-compute kernels (readers/writers)
    // TODO: for any CB to be used as both in & out of compute kernels (ie intermediate), additional work is required to
    // propagate formats to "unpack dst (SRCA/B REG)" / "pack src (DST REG)"
    equalize_data_format_vectors(unpack_src_formats_all_cbs, pack_dst_formats_all_cbs);

    emit_unpack_data_formats(unpack_data_format_descs, unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs);
    emit_pack_data_formats(pack_data_format_descs, pack_src_formats_all_cbs, pack_dst_formats_all_cbs);
}

std::string array_to_string(const uint32_t arr[]) {
    std::string formats_string;
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        formats_string += to_string((int)arr[i]) + ",";
    }
    return formats_string;
}

void emit_unpack_tile_dims(const std::string& unpack_tile_dims_descs, tt_hlk_desc& desc) {
    ofstream file_stream;
    file_stream.open(unpack_tile_dims_descs);
    file_stream << "#pragma once\n\n";
    file_stream << create_formats_array_string("constexpr uint8_t", "unpack_tile_num_faces", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_num_faces_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "unpack_partial_face", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_partial_face_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "unpack_tile_face_r_dim", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_face_r_dim_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "unpack_narrow_tile", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_narrow_tile_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "unpack_tile_r_dim", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_tile_r_dim_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "unpack_tile_c_dim", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_tile_c_dim_arr));
    file_stream << create_formats_array_string("constexpr uint16_t", "unpack_tile_size", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_tile_size_arr));
    file_stream.close();
}

void emit_pack_tile_dims(const std::string& pack_tile_dims_descs, tt_hlk_desc& desc) {
    ofstream file_stream;
    file_stream.open(pack_tile_dims_descs);
    file_stream << "#pragma once\n\n";
    file_stream << create_formats_array_string("constexpr uint8_t", "pack_tile_num_faces", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_num_faces_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "pack_partial_face", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_partial_face_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "pack_tile_face_r_dim", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_face_r_dim_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "pack_narrow_tile", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_narrow_tile_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "pack_tile_r_dim", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_tile_r_dim_arr));
    file_stream << create_formats_array_string("constexpr uint8_t", "pack_tile_c_dim", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_tile_c_dim_arr));
    file_stream << create_formats_array_string("constexpr uint16_t", "pack_tile_size", NUM_CIRCULAR_BUFFERS, array_to_string(desc.buf_tile_size_arr));
    file_stream.close();
}

void generate_tile_dims_descriptors(JitBuildOptions& options, const tt::ARCH /*arch*/) {
    string out_file_name_base = "chlkc_";
    string out_file_name_suffix = "_tile_dims.h";
    string unpack_tile_dims_descs = options.path + out_file_name_base + "unpack" + out_file_name_suffix;
    string pack_tile_dims_descs = options.path + out_file_name_base + "pack" + out_file_name_suffix;

    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = options.hlk_desc;

    emit_unpack_tile_dims(unpack_tile_dims_descs, desc);
    emit_pack_tile_dims(pack_tile_dims_descs, desc);
}

void generate_dst_accum_mode_descriptor(JitBuildOptions& options) {
    string dst_accum_format_descriptor = options.path + "chlkc_dst_accum_mode.h";

    ofstream file_stream;

    file_stream.open(dst_accum_format_descriptor);

    if (options.fp32_dest_acc_en == 0) {
        file_stream << "constexpr bool DST_ACCUM_MODE = false;" << endl;
    } else {
        file_stream << "constexpr bool DST_ACCUM_MODE = true;" << endl;
    }

    file_stream.close();
}

void generate_dst_sync_mode_descriptor(JitBuildOptions& options) {
    string dst_sync_mode_descriptor = options.path + "chlkc_dst_sync_mode.h";

    ofstream file_stream;

    file_stream.open(dst_sync_mode_descriptor);

    if (options.dst_full_sync_en) {
        file_stream << "#define DST_SYNC_MODE DstSync::SyncFull" << endl;
    } else {
        file_stream << "#define DST_SYNC_MODE DstSync::SyncHalf" << endl;
    }

    file_stream.close();
}

void generate_math_fidelity_descriptor(JitBuildOptions& options) {
    string math_fidelity_descriptor = options.path + "chlkc_math_fidelity.h";
    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = options.hlk_desc;

    ofstream file_stream;

    file_stream.open(math_fidelity_descriptor);
    file_stream << "constexpr std::int32_t MATH_FIDELITY = " << (int)desc.get_hlk_math_fidelity() << ";" << endl;
    file_stream.close();
}

void generate_math_approx_mode_descriptor(JitBuildOptions& options) {
    string approx_descriptor = options.path + "chlkc_math_approx_mode.h";

    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = options.hlk_desc;

    ofstream file_stream;

    file_stream.open(approx_descriptor);
    file_stream << "constexpr bool APPROX = " << std::boolalpha << desc.get_hlk_math_approx_mode() << ";" << endl;
    file_stream.close();
}

}  // namespace

// clang-format off
void jit_build_genfiles_descriptors(const JitBuildEnv& env, JitBuildOptions& options) {
    //ZoneScoped;
    //const std::string tracyPrefix = "generate_descriptors_";
    //ZoneName((tracyPrefix + options.name).c_str(), options.name.length() + tracyPrefix.length());
    fs::create_directories(options.path);
    try {
        std::thread td( [&]() { generate_data_format_descriptors(options, env.get_arch()); } );
        std::thread tt( [&]() { generate_tile_dims_descriptors(options, env.get_arch()); } );
        std::thread tm( [&]() { generate_math_fidelity_descriptor(options); } );
        std::thread ta( [&]() { generate_math_approx_mode_descriptor(options); } );
        std::thread tf( [&]() { generate_dst_accum_mode_descriptor(options); } );
        std::thread ts( [&]() { generate_dst_sync_mode_descriptor(options); } );
        td.join();
        tt.join();
        tm.join();
        ta.join();
        tf.join();
        ts.join();
    } catch (std::runtime_error& ex) {
        std::cerr << "EXCEPTION FROM THREADING IN GENERATE_DESCRIPTORS: " << ex.what() << std::endl;
    }
}
// clang-format on

}  // namespace tt::tt_metal
