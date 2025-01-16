// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build/genfiles.hpp"

#include <bit>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <utility>

#include <tt_backend_api_types.hpp>
#include <utils.hpp>
#include "hostdevcommon/common_values.hpp"
#include <build.hpp>
#include <data_format.hpp>
#include <settings.hpp>

#include <circular_buffer_constants.h>

namespace fs = std::filesystem;

using namespace std;

namespace tt::tt_metal {

static void gen_kernel_cpp(const string& src, const string& dst_name, const vector<string>& prolog) {
    std::ofstream out(dst_name);
    for (const string& s : prolog) out << s;
    out << src;
}

static void gen_kernel_cpp(const string& src, const string& dst_name) {
    vector<string> empty_prolog;
    gen_kernel_cpp(src, dst_name, empty_prolog);
}

static fs::path get_file_path_relative_to_dir(const string& dir, const fs::path& file_path) {
    const string& path_relative_to_dir = dir + file_path.string();
    fs::path file_path_relative_to_dir(path_relative_to_dir);

    if (!fs::exists(file_path_relative_to_dir)) {
        file_path_relative_to_dir.clear();
    }

    return file_path_relative_to_dir;
}

static fs::path get_relative_file_path_from_config(const fs::path& file_path) {
    fs::path file_path_relative_to_dir;

    if (llrt::RunTimeOptions::get_instance().is_root_dir_specified()) {
        file_path_relative_to_dir = get_file_path_relative_to_dir(llrt::RunTimeOptions::get_instance().get_root_dir(), file_path);
    }

    if (!fs::exists(file_path_relative_to_dir) && llrt::RunTimeOptions::get_instance().is_kernel_dir_specified()) {
        file_path_relative_to_dir = get_file_path_relative_to_dir(llrt::RunTimeOptions::get_instance().get_kernel_dir(), file_path);
    }

    return file_path_relative_to_dir;
}

static fs::path get_file_path_relative_to_src(const fs::path& file_path) {
    fs::path file_path_relative_to_src;
    if (fs::exists(file_path)) {
        file_path_relative_to_src = file_path;
    } else {
        // If the path doesn't exist as a absolute/relative path, then it must be relative to
        // TT_METAL_HOME/TT_METAL_KERNEL_PATH.
        file_path_relative_to_src = get_relative_file_path_from_config(file_path);
    }
    return file_path_relative_to_src;
}

static string get_absolute_path(const string& file_path_string) {
    const fs::path& file_path = get_file_path_relative_to_src(file_path_string);

    const bool does_file_exist = fs::exists(file_path);
    TT_FATAL(does_file_exist, "Kernel file {} doesn't exist!", file_path_string);

    const fs::path& absolute_file_path = fs::absolute(file_path);
    return absolute_file_path.string();
}

static string get_kernel_source_to_include(const KernelSource& kernel_src) {
    switch (kernel_src.source_type_) {
        case KernelSource::FILE_PATH: return "#include \"" + get_absolute_path(kernel_src.source_) + "\"\n";
        case KernelSource::SOURCE_CODE: return kernel_src.source_;
        default: {
            TT_THROW("Unsupported kernel source type!");
        }
    }
}

void jit_build_genfiles_kernel_include(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src) {
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for BRISC/NCRISC/ERISC user kernel");

    string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";
    string kernel_header = out_dir + "kernel_includes.hpp";

    const string& kernel_src_to_include = get_kernel_source_to_include(kernel_src);

    gen_kernel_cpp(kernel_src_to_include, kernel_header);
}

void jit_build_genfiles_triscs_src(
    const JitBuildEnv& env, const JitBuildSettings& settings, const KernelSource& kernel_src) {
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for TRISCs");

    string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";
    string unpack_base = out_dir + "chlkc_unpack";
    string math_base = out_dir + "chlkc_math";
    string pack_base = out_dir + "chlkc_pack";
    string unpack_cpp = unpack_base + ".cpp";
    string unpack_llk_args_h = unpack_base + "_llk_args.h";
    string math_cpp = math_base + ".cpp";
    string math_llk_args_h = math_base + "_llk_args.h";
    string pack_cpp = pack_base + ".cpp";
    string pack_llk_args_h = pack_base + "_llk_args.h";

    const string& kernel_src_to_include = get_kernel_source_to_include(kernel_src);

    vector<string> unpack_prolog;
    unpack_prolog.push_back("#define TRISC_UNPACK\n");
    unpack_prolog.push_back("#include \"defines_generated.h\"\n");
    vector<string> math_prolog;
    math_prolog.push_back("#define TRISC_MATH\n");
    math_prolog.push_back("#include \"defines_generated.h\"\n");
    vector<string> pack_prolog;
    pack_prolog.push_back("#define TRISC_PACK\n");
    pack_prolog.push_back("#include \"defines_generated.h\"\n");

    // TODO(pgk) - is this really worth it?
    std::thread t0([&]() { gen_kernel_cpp(kernel_src_to_include, unpack_cpp, unpack_prolog); });
    std::thread t1([&]() { gen_kernel_cpp(kernel_src_to_include, math_cpp, math_prolog); });
    std::thread t2([&]() { gen_kernel_cpp(kernel_src_to_include, pack_cpp, pack_prolog); });
    t0.join();
    t1.join();
    t2.join();

    // Here we generate an auxiliary header with defines added via add_define() call
    // this header is then included from the kernel
    // We also append the include path to generated dir to hlkc cmldline.
    std::ofstream gen_defines_file;
    string generated_defines_fname = out_dir + "/defines_generated.h";
    gen_defines_file.open(generated_defines_fname, std::ios_base::out);
    settings.process_defines([&gen_defines_file](const string& define, const string& value) {
        gen_defines_file << "#define " << define << " " << value << endl;
    });
}


static std::string data_format_vec_to_string(const vector<DataFormat>& formats) {
    std::string formats_string = "";
    for (int i = 0; i < formats.size(); i++) {
        formats_string += to_string((int)formats[i]) + ",";
    }
    return formats_string;
}

static std::string create_formats_array_string(
    const std::string& array_type, const std::string& array_name, int array_size, const std::string& array_data) {
    stringstream str_stream;

    str_stream << array_type << " " << array_name << "[" << array_size << "] = {" << endl;
    str_stream << "    " << array_data << endl;
    str_stream << "};" << endl;

    return str_stream.str();
}

static std::pair<std::vector<DataFormat>, std::vector<DataFormat>>
generate_unpack_data_formats(tt_hlk_desc& desc, DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en, std::vector<UnpackToDestMode> unpack_to_dest_mode) {

    vector<DataFormat> src_formats = tt::get_unpack_src_formats(desc.buf_dataformat_arr);

    vector<DataFormat> dst_formats = tt::get_unpack_dst_formats(
        desc.buf_dataformat_arr, unpack_conditional_dst_format, fp32_dest_acc_en, std::move(unpack_to_dest_mode));

    TT_ASSERT(src_formats.size() == NUM_CIRCULAR_BUFFERS);
    TT_ASSERT(dst_formats.size() == NUM_CIRCULAR_BUFFERS);

    return std::make_pair(src_formats, dst_formats);
}

static void emit_unpack_data_formats(
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
        data_format_vec_to_string(std::move(src_formats_all_cbs)));
    file_stream << create_formats_array_string(
        "constexpr std::int32_t",
        "unpack_dst_format",
        NUM_CIRCULAR_BUFFERS,
        data_format_vec_to_string(std::move(dst_formats_all_cbs)));
    file_stream.close();
}

static std::pair<std::vector<DataFormat>, std::vector<DataFormat>> generate_pack_data_formats(
    tt_hlk_desc& desc, DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en, bool bfp8_pack_precise, const tt::ARCH arch) {
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

static void emit_pack_data_formats(
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
        data_format_vec_to_string(std::move(src_formats_all_cbs)));
    file_stream << create_formats_array_string(
        "constexpr unsigned char",
        "pack_dst_format",
        NUM_CIRCULAR_BUFFERS,
        data_format_vec_to_string(std::move(dst_formats_all_cbs)));

    // budabackend-style format array
    // file_stream << create_formats_array_string("const std::int32_t", "pack_src_format", 16,
    // data_format_vec_to_string(src_formats)); file_stream << create_formats_array_string("const std::int32_t",
    // "pack_dst_format", 16, data_format_vec_to_string(dst_formats));

    file_stream.close();
}

static void equalize_data_format_vectors(std::vector<DataFormat>& v1, std::vector<DataFormat>& v2) {
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

static void generate_data_format_descriptors(JitBuildOptions& options, const tt::ARCH arch) {
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

static std::string array_to_string(const uint32_t arr[]) {
    std::string formats_string = "";
    for (int i = 0; i < NUM_CIRCULAR_BUFFERS; i++) {
        formats_string += to_string((int)arr[i]) + ",";
    }
    return formats_string;
}

static void emit_unpack_tile_dims(const std::string& unpack_tile_dims_descs, tt_hlk_desc& desc) {
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

static void emit_pack_tile_dims(const std::string& pack_tile_dims_descs, tt_hlk_desc& desc) {
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

static void generate_tile_dims_descriptors(JitBuildOptions& options, const tt::ARCH arch) {
    string out_file_name_base = "chlkc_";
    string out_file_name_suffix = "_tile_dims.h";
    string unpack_tile_dims_descs = options.path + out_file_name_base + "unpack" + out_file_name_suffix;
    string pack_tile_dims_descs = options.path + out_file_name_base + "pack" + out_file_name_suffix;

    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = options.hlk_desc;

    emit_unpack_tile_dims(unpack_tile_dims_descs, desc);
    emit_pack_tile_dims(pack_tile_dims_descs, desc);
}

static void generate_dst_accum_mode_descriptor(JitBuildOptions& options) {
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

static void generate_dst_sync_mode_descriptor(JitBuildOptions& options) {
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

static void generate_math_fidelity_descriptor(JitBuildOptions& options) {
    string math_fidelity_descriptor = options.path + "chlkc_math_fidelity.h";
    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = options.hlk_desc;

    ofstream file_stream;

    file_stream.open(math_fidelity_descriptor);
    file_stream << "constexpr std::int32_t MATH_FIDELITY = " << (int)desc.get_hlk_math_fidelity() << ";" << endl;
    file_stream.close();
}

static void generate_math_approx_mode_descriptor(JitBuildOptions& options) {
    string approx_descriptor = options.path + "chlkc_math_approx_mode.h";

    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = options.hlk_desc;

    ofstream file_stream;

    file_stream.open(approx_descriptor);
    file_stream << "constexpr bool APPROX = " << std::boolalpha << desc.get_hlk_math_approx_mode() << ";" << endl;
    file_stream.close();
}

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

}  // namespace tt::tt_metal
