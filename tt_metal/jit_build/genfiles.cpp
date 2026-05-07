// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build/genfiles.hpp"

#include <circular_buffer_constants.h>
#include "data_format.hpp"
#include <algorithm>
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
#include <sstream>
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
#include "impl/kernels/kernel_source.hpp"

namespace tt::tt_metal {
enum class UnpackToDestMode : uint8_t;
}  // namespace tt::tt_metal

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

// Generates TRISC prolog: #define + includes for JIT-generated headers and defines_generated.h
// Kernels using Metal 2.0 get additional JIT-generated headers (not included for legacy kernels)
string build_trisc_prolog(const char* trisc_define, bool is_metal2_kernel) {
    ostringstream prolog;
    prolog << "#define " << trisc_define << "\n";
    if (is_metal2_kernel) {
        prolog << "#include \"kernel_bindings_generated.h\"\n";
        prolog << "#include \"kernel_args_generated.h\"\n";
    }
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

// METAL 2.0 only:
// This is only invoked for Metal 2.0 kernels created via the new ProgramSpec host APIs.
// Legacy kernels (created via CreateKernel) do not get kernel_bindings_generated.h.
void write_kernel_bindings_generated_header(const string& out_dir, const JitBuildSettings& settings) {
    const string path = out_dir + "kernel_bindings_generated.h";

    // Get the DFB bindings from the settings callback
    // Sort them to ensure the file output is deterministic for the JIT build cache
    // (aka the on-disk per-object dephash cache)
    vector<pair<string, uint16_t>> dfb_entries;
    settings.process_dataflow_buffer_local_accessor_handles(
        [&dfb_entries](const string& name, uint16_t id) { dfb_entries.emplace_back(name, id); });
    sort(dfb_entries.begin(), dfb_entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Get the semaphore bindings from the settings callback
    // Sort them to ensure the file output is deterministic for the JIT build cache
    // (aka the on-disk per-object dephash cache)
    vector<pair<string, uint16_t>> sem_entries;
    settings.process_semaphore_local_accessor_handles(
        [&sem_entries](const string& name, uint16_t id) { sem_entries.emplace_back(name, id); });
    sort(sem_entries.begin(), sem_entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    // Emit the header content:
    //  - DFB accessors are emitted into the dfb namespace
    //  - Semaphore accessors are emitted into the sem namespace
    //
    // NOTE: Both accessor types are emitted as constexpr variables, i.e. as implicit CTAs.
    //       This is a design decision; we could alternatively emit them as implicit CRTAs.
    //       (Or, we could give the user the choice via the Metal 2.0 host API, on a per-kernel or per-accessor basis.)
    //       Implicit CTA is simpler and cheaper, but could theoretically cause unnecessary kernel cache hit misses.
    //       We are starting simple and can adjust later if problems arise.
    //       Legacy kernels passed semaphores both ways, kernel folks think this was more random than intentional.
    ostringstream content;
    content << "// AUTO-GENERATED — do not edit.\n\n"
               "#pragma once\n\n";
    if (dfb_entries.empty() && sem_entries.empty()) {
        content << "// No bindings for this kernel.\n";
    } else {
        if (!dfb_entries.empty()) {
            content << "#include \"experimental/dataflow_buffer.h\"\n";
        }
        if (!sem_entries.empty()) {
            content << "#include <cstdint>\n";
        }
        content << "\n";

        if (!dfb_entries.empty()) {
            content << "namespace dfb {\n";
            for (const auto& [name, id] : dfb_entries) {
                content << "constexpr experimental::DFBAccessor " << name << "{" << id << "};\n";
            }
            content << "}  // namespace dfb\n";
        }

        if (!sem_entries.empty()) {
            content << "namespace sem {\n";
            for (const auto& [name, id] : sem_entries) {
                content << "constexpr std::uint32_t " << name << " = " << id << "u;\n";
            }
            content << "}  // namespace sem\n";
        }
    }
    write_file(path, content.str());
}

// METAL 2.0 only:
// Emits per-kernel accessors for named RTAs, CRTAs, and CTAs inside the `args` namespace.
// Also emits get_vararg() / get_common_vararg() helpers with the named-args offset baked
// in, so that vararg indices in kernel code are stable across schema changes.
//
// NOTE: This is only invoked for Metal 2.0 kernels created via the new host API.
//       Legacy kernels do not get kernel_args_generated.h.
void write_kernel_args_generated_header(const std::filesystem::path& out_dir, const JitBuildSettings& settings) {
    const fs::path path = out_dir / "kernel_args_generated.h";

    // Named RTAs/CRTAs come straight from the settings as ordered vectors.
    const vector<string>& rta_names = settings.get_named_runtime_args();
    const vector<string>& crta_names = settings.get_named_common_runtime_args();

    // Named CTAs come through the legacy unordered_map path (Kernel internal storage).
    // The order in which we emit them DOES matter!
    // We sort them to ensure the file output is deterministic for the JIT build cache
    // (aka the on-disk per-object dephash cache)
    vector<pair<string, uint32_t>> cta_entries;
    settings.process_named_compile_time_args(
        [&cta_entries](const std::unordered_map<std::string, uint32_t>& named_args) {
            for (const auto& [name, value] : named_args) {
                cta_entries.emplace_back(name, value);
            }
        });
    sort(cta_entries.begin(), cta_entries.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

    ostringstream content;
    content << "// AUTO-GENERATED — do not edit.\n\n"
               "#pragma once\n\n"
               "#include \"experimental/kernel_args.h\"\n\n";

    // Named args namespace: emit only when the kernel has at least one named arg or CTA.
    // A kernel with only varargs (and no named anything) still needs the vararg helpers below,
    // so we keep emitting those unconditionally.
    const bool has_named_args = !rta_names.empty() || !crta_names.empty() || !cta_entries.empty();
    if (has_named_args) {
        content << "namespace args {\n";

        // Named RTAs
        // Here, rta_offset tracks the byte_offset of the RTA in the dispatch buffer.
        // (Only uint32_t arg types are currently supported, but we later want to extend this.)
        uint32_t rta_offset = 0;
        for (const auto& name : rta_names) {
            content << "constexpr experimental::RtaArg<uint32_t> " << name << "{" << rta_offset << "};\n";
            rta_offset += sizeof(uint32_t);
        }
        // Named CRTAs
        uint32_t crta_offset = 0;
        for (const auto& name : crta_names) {
            content << "constexpr experimental::CrtaArg<uint32_t> " << name << "{" << crta_offset << "};\n";
            crta_offset += sizeof(uint32_t);
        }
        // Named CTAs
        // No offsets to deal with here; CTA values are emitted directly into the generated header.
        for (const auto& [name, value] : cta_entries) {
            content << "constexpr experimental::CtaVal<uint32_t> " << name << "{" << value << "u};\n";
        }

        content << "}  // namespace args\n\n";
    }

    // Vararg helpers — always emitted.
    // The starting offset (named_arg_count) is baked in so kernel code uses 0-based
    // indexing: get_vararg(0) is the first vararg, regardless of named-arg count. When
    // there are no named args, the offset is zero and these helpers are just thin wrappers
    // around get_arg_val / get_common_arg_val.
    const uint32_t named_rta_words = static_cast<uint32_t>(rta_names.size());
    const uint32_t named_crta_words = static_cast<uint32_t>(crta_names.size());
    content << "FORCE_INLINE uint32_t get_vararg(uint32_t idx) { return get_arg_val<uint32_t>(" << named_rta_words
            << " + idx); }\n"
            << "FORCE_INLINE uint32_t get_common_vararg(uint32_t idx) { return get_common_arg_val<uint32_t>("
            << named_crta_words << " + idx); }\n";

    write_file(path, content.str());
}

}  // namespace

void jit_build_genfiles_kernel_include(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src) {
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for BRISC/NCRISC/ERISC user kernel");

    string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";

    // Metal 2.0 generated headers and their includes are emitted only for Metal 2.0 kernels.
    // Legacy kernels created via the old host API are fenced out of this code path.
    const bool is_metal2 = settings.is_metal2_kernel();
    string kernel_header_content;
    if (is_metal2) {
        write_kernel_bindings_generated_header(out_dir, settings);
        write_kernel_args_generated_header(out_dir, settings);
        kernel_header_content =
            string("#include \"kernel_bindings_generated.h\"\n#include \"kernel_args_generated.h\"\n");
    }
    kernel_header_content += get_kernel_source_to_include(kernel_src);

    string kernel_header = out_dir + "kernel_includes.hpp";
    write_file(kernel_header, kernel_header_content);
}

void jit_build_genfiles_triscs_src(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src) {
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for TRISCs");

    const string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";

    // Metal 2.0 generated headers are emitted and referenced only for Metal 2.0 kernels.
    const bool is_metal2 = settings.is_metal2_kernel();
    if (is_metal2) {
        write_kernel_bindings_generated_header(out_dir, settings);
        write_kernel_args_generated_header(out_dir, settings);
    }

    const string unpack_cpp = out_dir + "chlkc_unpack.cpp";
    const string math_cpp = out_dir + "chlkc_math.cpp";
    const string pack_cpp = out_dir + "chlkc_pack.cpp";
    const string isolate_sfpu_cpp = out_dir + "chlkc_isolate_sfpu.cpp";

    // Build prologs for each TRISC
    const string unpack_prolog = build_trisc_prolog("TRISC_UNPACK", is_metal2);
    const string math_prolog = build_trisc_prolog("TRISC_MATH", is_metal2);
    const string pack_prolog = build_trisc_prolog("TRISC_PACK", is_metal2);
    const string isolate_sfpu_prolog = build_trisc_prolog("TRISC_ISOLATE_SFPU", is_metal2);

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
    // Remap host-only enum values to HW values for device compilation.
    // Int16 has a unique host value (13) to avoid colliding with UInt16 (9),
    // but the Quasar HW expects Int16 = 9 in tensix_types.h.
    auto as_int = [](DataFormat f) -> std::underlying_type_t<DataFormat> {
        if (f == DataFormat::Int16) {
            return 9;  // HW value from tensix_types.h
        }
        return static_cast<std::underlying_type_t<DataFormat>>(f);
    };
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

    // Fp8_e4m3 is always unpacked to Float16 (A-family) in source/dest registers.
    // Without fp32_dest_acc, the dest register holds Float16 (A-family) data when
    // the input is Fp8, so non-Fp8 output CBs need A-family pack_src to match.
    // With fp32_dest_acc, dest holds Float32 and pack_src semantics differ, so skip.
    // CBs that are themselves Fp8_e4m3 are already handled by get_single_pack_src_format.
    if (!fp32_dest_acc_en &&
        std::any_of(desc.buf_dataformat_arr.begin(), desc.buf_dataformat_arr.end(), [](DataFormat f) {
            return f == DataFormat::Fp8_e4m3;
        })) {
        for (size_t i = 0; i < src_formats.size(); i++) {
            if (desc.buf_dataformat_arr[i] == DataFormat::Fp8_e4m3) {
                continue;
            }
            switch (src_formats[i]) {
                case DataFormat::Float16_b: src_formats[i] = DataFormat::Float16; break;
                case DataFormat::Bfp8_b: src_formats[i] = DataFormat::Bfp8; break;
                case DataFormat::Bfp4_b: src_formats[i] = DataFormat::Bfp4; break;
                case DataFormat::Bfp2_b: src_formats[i] = DataFormat::Bfp2; break;
                default: break;
            }
        }
    }

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

    if (std::any_of(desc.buf_dataformat_arr.begin(), desc.buf_dataformat_arr.end(), [](DataFormat f) {
            return f == DataFormat::MxFp4;
        })) {
        TT_FATAL(arch == tt::ARCH::QUASAR, "MxFp4 format is only supported on Quasar");
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

    out << "#if defined(UCK_CHLKC_PACK)\n"
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
    // For Blackhole tilize workaround, PACK needs access to unpack_src_format to determine
    // if the original input format is 8-bit (Int8, UInt8, Fp8_e4m3, Lf8) since those formats
    // do not require the tilize workaround. This is needed to determine whether to skip the workaround in llk_pack_init.
    out << "#if defined(UCK_CHLKC_PACK)\n";
    emit_formats_array(out, "constexpr std::int32_t", "unpack_src_format", max_cbs, fmts.unpack_src);
    out << "#endif\n";   // if pack
    out << "#endif\n\n"; // if not math and not unpack

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
