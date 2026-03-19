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
#include <iterator>
#include <iostream>
#include <ostream>
#include <fstream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <tt_stl/assert.hpp>
#include <tt_stl/unreachable.hpp>
#include "build.hpp"
#include "hlk_desc.hpp"
#include "jit_build/jit_build_utils.hpp"
#include "jit_build_options.hpp"
#include "jit_build_settings.hpp"
#include <tt-logger/tt-logger.hpp>
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

// Generates TRISC prolog: #define + #include for defines_generated.h
string build_trisc_prolog(const char* trisc_define) {
    ostringstream prolog;
    prolog << "#define " << trisc_define << "\n";
    prolog << "#include \"defines_generated.h\"\n";
    return prolog.str();
}

// Writes content to a file, throwing on failure
void write_file(const string& path, const string& content) {
    jit_build::utils::FileRenamer tmp(path);
    std::ofstream f(tmp.path());
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
    const string isolate_sfpu_cpp = out_dir + "chlkc_isolate_sfpu.cpp";

    // Build prologs for each TRISC
    const string unpack_prolog = build_trisc_prolog("TRISC_UNPACK");
    const string math_prolog = build_trisc_prolog("TRISC_MATH");
    const string pack_prolog = build_trisc_prolog("TRISC_PACK");
    const string isolate_sfpu_prolog = build_trisc_prolog("TRISC_ISOLATE_SFPU");

    // All TRISCs get the same kernel source (differentiated by TRISC_* defines)
    const string kernel_src_to_include = get_kernel_source_to_include(kernel_src);

    // Generate the four TRISC source files (fourth only used on Quasar)
    write_file(unpack_cpp, unpack_prolog + kernel_src_to_include);
    write_file(math_cpp, math_prolog + kernel_src_to_include);
    write_file(pack_cpp, pack_prolog + kernel_src_to_include);
    write_file(isolate_sfpu_cpp, isolate_sfpu_prolog + kernel_src_to_include);
    // Here we generate an auxiliary header with defines added via add_define() call
    // this header is then included from the kernel
    // We also append the include path to generated dir to hlkc cmldline.
    const string generated_defines_fname = out_dir + "defines_generated.h";
    jit_build::utils::FileRenamer tmp(generated_defines_fname);
    std::ofstream gen_defines_file(tmp.path());
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

template <std::ranges::range Range>
void emit_formats_array(
    std::ostream& out, std::string_view array_type, std::string_view array_name, int array_size, const Range& arr) {
    fmt::format_to(
        std::ostreambuf_iterator<char>(out),
        "{} {}[{}] = {{\n    {}\n}};\n",
        array_type,
        array_name,
        array_size,
        fmt::join(arr, ","));
}

void emit_formats_array(
    std::ostream& out,
    std::string_view array_type,
    std::string_view array_name,
    int array_size,
    const std::vector<DataFormat>& formats) {
    auto as_int = [](DataFormat f) { return static_cast<std::underlying_type_t<DataFormat>>(f); };
    emit_formats_array(out, array_type, array_name, array_size, formats | std::views::transform(as_int));
}

std::pair<std::vector<DataFormat>, std::vector<DataFormat>> generate_unpack_data_formats(
    const tt_hlk_desc& desc,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    std::vector<UnpackToDestMode> unpack_to_dest_mode,
    uint32_t max_cbs) {
    vector<DataFormat> src_formats = tt::get_unpack_src_formats(desc.buf_dataformat_arr);

    vector<DataFormat> dst_formats = tt::get_unpack_dst_formats(
        desc.buf_dataformat_arr, unpack_conditional_dst_format, fp32_dest_acc_en, std::move(unpack_to_dest_mode));

    TT_ASSERT(src_formats.size() == max_cbs);
    TT_ASSERT(dst_formats.size() == max_cbs);

    return std::make_pair(src_formats, dst_formats);
}

void emit_unpack_data_formats(
    std::ostream& out,
    const std::vector<DataFormat>& src_formats_all_cbs,
    const std::vector<DataFormat>& dst_formats_all_cbs,
    uint32_t max_cbs) {
    // TODO: we should be emitting "unsigned char", no reason to use up 4B per data format
    emit_formats_array(out, "constexpr std::int32_t", "unpack_src_format", max_cbs, src_formats_all_cbs);
    emit_formats_array(out, "constexpr std::int32_t", "unpack_dst_format", max_cbs, dst_formats_all_cbs);
}

std::pair<std::vector<DataFormat>, std::vector<DataFormat>> generate_pack_data_formats(
    const tt_hlk_desc& desc,
    DataFormat unpack_conditional_dst_format,
    bool fp32_dest_acc_en,
    bool bfp8_pack_precise,
    const tt::ARCH arch,
    uint32_t max_cbs) {
    vector<DataFormat> src_formats = tt::get_pack_src_formats(
        desc.buf_dataformat_arr,
        unpack_conditional_dst_format,
        fp32_dest_acc_en,
        bfp8_pack_precise,
        false,
        arch);

    vector<DataFormat> dst_formats = tt::get_pack_dst_formats(
        desc.buf_dataformat_arr);

    TT_ASSERT(src_formats.size() == max_cbs);
    TT_ASSERT(dst_formats.size() == max_cbs);

    return std::make_pair(src_formats, dst_formats);
}

void emit_pack_data_formats(
    std::ostream& out,
    const std::vector<DataFormat>& src_formats_all_cbs,
    const std::vector<DataFormat>& dst_formats_all_cbs,
    uint32_t max_cbs) {
    emit_formats_array(out, "constexpr unsigned char", "pack_src_format", max_cbs, src_formats_all_cbs);
    emit_formats_array(out, "constexpr unsigned char", "pack_dst_format", max_cbs, dst_formats_all_cbs);
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

struct ComputedDataFormats {
    std::vector<DataFormat> unpack_src, unpack_dst, pack_src, pack_dst;
};

ComputedDataFormats compute_data_formats(const JitBuildOptions& options, tt::ARCH arch, uint32_t max_cbs) {
    // assuming all cores within a op have the same desc
    const tt_hlk_desc& desc = options.hlk_desc;

    // Determine dst format under ambiguous conditions (either or both l1 input & output formats are Float32)
    ExpPrecision exp_prec = tt::get_data_exp_precision(desc.buf_dataformat_arr);
    DataFormat unpack_conditional_dst_format =
        (exp_prec == ExpPrecision::A) ? DataFormat::Float16 : DataFormat::Float16_b;
    if (options.fp32_dest_acc_en &&
        (tt::is_all_fp32_formats(desc.buf_dataformat_arr) || (exp_prec == ExpPrecision::B))) {
        unpack_conditional_dst_format = DataFormat::Tf32;
    }

    tt::check_valid_formats_in_out_data_formats(desc.buf_dataformat_arr);
    auto [unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs] = generate_unpack_data_formats(
        desc, unpack_conditional_dst_format, options.fp32_dest_acc_en, options.unpack_to_dest_mode, max_cbs);

    auto [pack_src_formats_all_cbs, pack_dst_formats_all_cbs] = generate_pack_data_formats(
        desc, unpack_conditional_dst_format, options.fp32_dest_acc_en, options.bfp8_pack_precise, arch, max_cbs);

    // equalize "unpack src" and "pack dst" data format vectors
    // both "unpack src" and "pack dst" refer to data in L1, "unpack src" == L1, and "pack dst" == L1
    // in order to allow any CB to be read and written to/from L1, these formats should be the same (one cannot be
    // DataFromat::Invalid if the other is set) if both formats are DataFormat::Invalid then this CB is not used this
    // allows any CB to be used as both in & out in non-compute kernels (readers/writers)
    // TODO: for any CB to be used as both in & out of compute kernels (ie intermediate), additional work is required to
    // propagate formats to "unpack dst (SRCA/B REG)" / "pack src (DST REG)"
    equalize_data_format_vectors(unpack_src_formats_all_cbs, pack_dst_formats_all_cbs);

    return {
        std::move(unpack_src_formats_all_cbs),
        std::move(unpack_dst_formats_all_cbs),
        std::move(pack_src_formats_all_cbs),
        std::move(pack_dst_formats_all_cbs)};
}

// Decomposes tile dimensions into (num_faces_r_dim, num_faces_c_dim) per CB.
// Derived directly from tile_r_dim / face_r_dim and tile_c_dim / face_c_dim.
// Runs on host at JIT time so division is fine.
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> compute_num_faces_rc_dims(
    const std::vector<uint32_t>& tile_r_dim_arr,
    const std::vector<uint32_t>& tile_c_dim_arr,
    const std::vector<uint32_t>& face_r_dim_arr) {
    TT_FATAL(
        tile_r_dim_arr.size() == tile_c_dim_arr.size(),
        "tile_r_dim_arr size ({}) must match tile_c_dim_arr size ({})",
        tile_r_dim_arr.size(),
        tile_c_dim_arr.size());
    TT_FATAL(
        tile_r_dim_arr.size() == face_r_dim_arr.size(),
        "tile_r_dim_arr size ({}) must match face_r_dim_arr size ({})",
        tile_r_dim_arr.size(),
        face_r_dim_arr.size());
    const size_t n = tile_r_dim_arr.size();
    std::vector<uint32_t> r_dims(n);
    std::vector<uint32_t> c_dims(n);
    for (size_t i = 0; i < n; ++i) {
        TT_FATAL(face_r_dim_arr[i] > 0, "face_r_dim must be > 0 at index {}", i);
        TT_FATAL(
            tile_r_dim_arr[i] % face_r_dim_arr[i] == 0,
            "tile_r_dim ({}) must be a multiple of face_r_dim ({})",
            tile_r_dim_arr[i],
            face_r_dim_arr[i]);
        TT_FATAL(
            tile_c_dim_arr[i] % constants::FACE_WIDTH == 0,
            "tile_c_dim ({}) must be a multiple of FACE_WIDTH ({})",
            tile_c_dim_arr[i],
            constants::FACE_WIDTH);
        r_dims[i] = tile_r_dim_arr[i] / face_r_dim_arr[i];
        c_dims[i] = tile_c_dim_arr[i] / constants::FACE_WIDTH;
    }
    return {r_dims, c_dims};
}

void emit_unpack_tile_dims(std::ostream& out, const tt_hlk_desc& desc, uint32_t max_cbs) {
    emit_formats_array(out, "constexpr uint8_t", "unpack_tile_num_faces", max_cbs, desc.buf_num_faces_arr);
    emit_formats_array(out, "constexpr uint8_t", "unpack_partial_face", max_cbs, desc.buf_partial_face_arr);
    emit_formats_array(out, "constexpr uint8_t", "unpack_tile_face_r_dim", max_cbs, desc.buf_face_r_dim_arr);
    emit_formats_array(out, "constexpr uint8_t", "unpack_narrow_tile", max_cbs, desc.buf_narrow_tile_arr);
    emit_formats_array(out, "constexpr uint8_t", "unpack_tile_r_dim", max_cbs, desc.buf_tile_r_dim_arr);
    emit_formats_array(out, "constexpr uint8_t", "unpack_tile_c_dim", max_cbs, desc.buf_tile_c_dim_arr);
    emit_formats_array(out, "constexpr uint16_t", "unpack_tile_size", max_cbs, desc.buf_tile_size_arr);

    auto [r_dims, c_dims] = compute_num_faces_rc_dims(
        desc.buf_tile_r_dim_arr, desc.buf_tile_c_dim_arr, desc.buf_face_r_dim_arr);
    emit_formats_array(out, "constexpr uint8_t", "unpack_num_faces_r_dim", max_cbs, r_dims);
    emit_formats_array(out, "constexpr uint8_t", "unpack_num_faces_c_dim", max_cbs, c_dims);
}

void emit_pack_tile_dims(std::ostream& out, const tt_hlk_desc& desc, uint32_t max_cbs) {
    emit_formats_array(out, "constexpr uint8_t", "pack_tile_num_faces", max_cbs, desc.buf_num_faces_arr);
    emit_formats_array(out, "constexpr uint8_t", "pack_partial_face", max_cbs, desc.buf_partial_face_arr);
    emit_formats_array(out, "constexpr uint8_t", "pack_tile_face_r_dim", max_cbs, desc.buf_face_r_dim_arr);
    emit_formats_array(out, "constexpr uint8_t", "pack_narrow_tile", max_cbs, desc.buf_narrow_tile_arr);
    emit_formats_array(out, "constexpr uint8_t", "pack_tile_r_dim", max_cbs, desc.buf_tile_r_dim_arr);
    emit_formats_array(out, "constexpr uint8_t", "pack_tile_c_dim", max_cbs, desc.buf_tile_c_dim_arr);
    emit_formats_array(out, "constexpr uint16_t", "pack_tile_size", max_cbs, desc.buf_tile_size_arr);

    auto [r_dims, c_dims] = compute_num_faces_rc_dims(
        desc.buf_tile_r_dim_arr, desc.buf_tile_c_dim_arr, desc.buf_face_r_dim_arr);
    emit_formats_array(out, "constexpr uint8_t", "pack_num_faces_r_dim", max_cbs, r_dims);
    emit_formats_array(out, "constexpr uint8_t", "pack_num_faces_c_dim", max_cbs, c_dims);
}

void emit_compute_scalar_descriptors(std::ostream& out, const JitBuildOptions& options) {
    fmt::format_to(
        std::ostreambuf_iterator<char>(out),
        "constexpr bool DST_ACCUM_MODE = {};\n"
        "#define DST_SYNC_MODE DstSync::Sync{}\n",
        options.fp32_dest_acc_en,
        options.dst_full_sync_en ? "Full" : "Half");
}

void emit_math_scalar_descriptors(std::ostream& out, const tt_hlk_desc& desc) {
    fmt::format_to(
        std::ostreambuf_iterator<char>(out),
        "constexpr ckernel::MathFidelity MATH_FIDELITY = static_cast<ckernel::MathFidelity>({});\n"
        "constexpr bool APPROX = {};\n",
        static_cast<std::uint32_t>(desc.get_hlk_math_fidelity()),
        desc.get_hlk_math_approx_mode());
}

void generate_all_descriptors(const JitBuildEnv& env, const JitBuildOptions& options) {
    const uint32_t max_cbs = env.get_max_cbs();
    const tt_hlk_desc& desc = options.hlk_desc;

    const string descriptors_path = options.path + "chlkc_descriptors.h";
    jit_build::utils::FileRenamer tmp(descriptors_path);
    ofstream out(tmp.path());
    if (!out) {
        throw std::runtime_error("Cannot create file: " + descriptors_path);
    }

    auto fmts = compute_data_formats(options, env.get_arch(), max_cbs);

    out << "#pragma once\n\n"
           "#if defined(UCK_CHLKC_MATH)\n"
           "#include \"llk_defs.h\"\n";
    emit_math_scalar_descriptors(out, desc);
    out << "#endif\n\n";

    out << "#if !defined(UCK_CHLKC_PACK)\n";
    emit_unpack_data_formats(out, fmts.unpack_src, fmts.unpack_dst, max_cbs);
    emit_unpack_tile_dims(out, desc, max_cbs);
    out << "#endif\n\n";

    out << "#if !defined(UCK_CHLKC_MATH) && !defined(UCK_CHLKC_UNPACK)\n";
    emit_pack_data_formats(out, fmts.pack_src, fmts.pack_dst, max_cbs);
    emit_pack_tile_dims(out, desc, max_cbs);
    out << "#endif\n\n";

    out << "#if defined(UCK_CHLKC_MATH) || defined(UCK_CHLKC_PACK) || defined(UCK_CHLKC_UNPACK) || "
           "defined(UCK_CHLKC_ISOLATE_SFPU)\n";
    emit_compute_scalar_descriptors(out, options);
    out << "#endif\n";

    if (!out) {
        throw std::runtime_error("Failed to write file: " + descriptors_path);
    }
}

}  // namespace

// clang-format off
void jit_build_genfiles_descriptors(const JitBuildEnv& env, const JitBuildOptions& options) {
    //ZoneScoped;
    //const std::string tracyPrefix = "generate_descriptors_";
    //ZoneName((tracyPrefix + options.name).c_str(), options.name.length() + tracyPrefix.length());
    fs::create_directories(options.path);
    generate_all_descriptors(env, options);
}
// clang-format on

}  // namespace tt::tt_metal
