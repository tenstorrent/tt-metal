// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <fstream>
#include <filesystem>
#include <iostream>

#include "common/utils.hpp"
#include "common/tt_backend_api_types.hpp"
#include "hostdevcommon/common_values.hpp"
#include "hostdevcommon/common_runtime_address_map.h"
#include "jit_build/build.hpp"
#include "jit_build/genfiles.hpp"
#include "jit_build/settings.hpp"

#include "noc/noc_parameters.h"

namespace fs = std::filesystem;

using namespace std;

namespace tt::tt_metal {

static void gen_kernel_cpp(const string& src_name, const string& dst_name, vector<string>& prolog)
{
    std::ofstream out(dst_name);
    for (auto s: prolog)
        out << s;
    out << "#include \"" << src_name << "\"\n";
}

static string get_absolute_path(const string& file_path_string) {
    fs::path file_path(file_path_string);

    // If the path doesn't exist as a absolute/relative path, then it must be relative to TT_METAL_HOME.
    if (!fs::exists(file_path)) {
        file_path = fs::path(llrt::OptionsG.get_root_dir() + file_path_string);
        TT_FATAL(fs::exists(file_path), "Kernel file {} doesn't exist!", file_path_string);
    }

    // Convert to absolute path and return
    return fs::absolute(file_path).string();
}

void jit_build_genfiles_kernel_include(const JitBuildEnv& env,
                                   const JitBuildSettings& settings,
                                   const string& input_hlk_file_path) {
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for BRISC/NCRISC/ERISC user kernel");

    string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";
    string kernel_header = out_dir + "kernel_includes.hpp";

    // Get absolute path of kernel file to include
    string abs_file_path = get_absolute_path(input_hlk_file_path);

    vector<string> prolog;
    gen_kernel_cpp(abs_file_path, kernel_header, prolog);
}
void jit_build_genfiles_triscs_src(const JitBuildEnv& env,
                                   const JitBuildSettings& settings,
                                   const string& input_hlk_file_path)
{
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for TRISCs");

    string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";
    string unpack_base        = out_dir + "chlkc_unpack";
    string math_base          = out_dir + "chlkc_math";
    string pack_base          = out_dir + "chlkc_pack";
    string unpack_cpp         = unpack_base + ".cpp";
    string unpack_llk_args_h  = unpack_base + "_llk_args.h";
    string math_cpp           = math_base + ".cpp";
    string math_llk_args_h    = math_base + "_llk_args.h";
    string pack_cpp           = pack_base + ".cpp";
    string pack_llk_args_h    = pack_base + "_llk_args.h";

    // Get absolute path of kernel file to include
    string abs_file_path = get_absolute_path(input_hlk_file_path);

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
    std::thread t0( [&]() { gen_kernel_cpp(abs_file_path, unpack_cpp, unpack_prolog); } );
    std::thread t1( [&]() { gen_kernel_cpp(abs_file_path, math_cpp, math_prolog); } );
    std::thread t2( [&]() { gen_kernel_cpp(abs_file_path, pack_cpp, pack_prolog); } );
    t0.join(); t1.join(); t2.join();

    // Here we generate an auxiliary header with defines added via add_define() call
    // this header is then included from the kernel
    // We also append the include path to generated dir to hlkc cmldline.
    std::ofstream gen_defines_file;
    string generated_defines_fname = out_dir + "/defines_generated.h";
    gen_defines_file.open(generated_defines_fname, std::ios_base::out);
    settings.process_defines([&gen_defines_file] (const string& define, const string& value) {
        gen_defines_file << "#define " << define << " " << value << endl;
    });
}

static std::pair<vector<DataFormat>,vector<DataFormat>> extend_unpack_data_format_vectors_to_all_cbs(const vector<DataFormat> &src_formats, const vector<DataFormat> &dst_formats) {
    // for the purposes of consistency and brevity of the LLK code that uses these arrays,
    // extend unpack data formats to all 32 CBs
    // [out0...out7] is missing from the vector, insert invalid (not used by the unpacker)

    vector<DataFormat> src_formats_all_cbs;
    vector<DataFormat> dst_formats_all_cbs;

    // copy inputs and params
    for (int i=0 ; i<16 ; i++) {
        src_formats_all_cbs.push_back(src_formats[i]);
        dst_formats_all_cbs.push_back(dst_formats[i]);
    }

    // insert invalid data format for output [out0...out7]
    for (int i=0 ; i<8 ; i++) {
        src_formats_all_cbs.push_back(DataFormat::Invalid);
        dst_formats_all_cbs.push_back(DataFormat::Invalid);
    }

    // copy intermediates
    for (int i=0 ; i<8 ; i++) {
        src_formats_all_cbs.push_back(src_formats[16+i]);
        dst_formats_all_cbs.push_back(dst_formats[16+i]);
    }

    return std::make_pair(src_formats_all_cbs, dst_formats_all_cbs);
}

static std::pair<vector<DataFormat>,vector<DataFormat>> extend_pack_data_format_vectors_to_all_cbs(const vector<DataFormat> &src_formats, const vector<DataFormat> &dst_formats) {
    // for the purposes of consistency and brevity of the LLK code that uses these arrays,
    // extend pack data formats to all 32 CBs
    // [in0...in7, param0...param7] are missing from the vector, insert invalid (not used by the unpacker)

    vector<DataFormat> src_formats_all_cbs;
    vector<DataFormat> dst_formats_all_cbs;

    // insert invalid for inputs and params
    for (int i=0 ; i<16 ; i++) {
        src_formats_all_cbs.push_back(DataFormat::Invalid);
        dst_formats_all_cbs.push_back(DataFormat::Invalid);
    }

    // copy outputs and intermediates
    for (int i=0 ; i<16 ; i++) {
        src_formats_all_cbs.push_back(src_formats[i]);
        dst_formats_all_cbs.push_back(dst_formats[i]);
    }

    return std::make_pair(src_formats_all_cbs, dst_formats_all_cbs);
}

static std::string data_format_vec_to_string(const vector<DataFormat> formats) {
    std::string formats_string = "";
    for (int i = 0; i < formats.size(); i++) {
        formats_string += to_string((int)formats[i]) + ",";
    }
    return formats_string;
}

static std::string create_formats_array_string(std::string array_type, std::string array_name, int array_size, std::string array_data) {
    stringstream str_stream;

    str_stream << array_type << " " << array_name << "[" << array_size << "] = {" << endl;
    str_stream << "    " << array_data << endl;
    str_stream << "};" << endl;

    return str_stream.str();
}

static std::pair<std::vector<DataFormat>, std::vector<DataFormat>>
generate_unpack_data_formats(tt_hlk_desc& desc, DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en, bool preserve_fp32_precision) {

    vector<DataFormat> src_formats = tt::get_unpack_src_formats(
        desc.input_buf_dataformat_arr, desc.param_buf_dataformat_arr, desc.intermediate_buf_dataformat_arr);

    vector<DataFormat> dst_formats = tt::get_unpack_dst_formats(
        desc.input_buf_dataformat_arr, desc.param_buf_dataformat_arr, desc.intermediate_buf_dataformat_arr,
        desc.output_buf_dataformat_arr, unpack_conditional_dst_format, fp32_dest_acc_en, preserve_fp32_precision);

    TT_ASSERT(src_formats.size() == 24 && dst_formats.size() == 24,
        "There must be 8 unpack src/dst formats for each input, param, and intermediate operands.");

    vector<DataFormat> src_formats_all_cbs;
    vector<DataFormat> dst_formats_all_cbs;
    tie(src_formats_all_cbs, dst_formats_all_cbs) = extend_unpack_data_format_vectors_to_all_cbs(src_formats, dst_formats);

    TT_ASSERT(src_formats_all_cbs.size() == NUM_CIRCULAR_BUFFERS);
    TT_ASSERT(dst_formats_all_cbs.size() == NUM_CIRCULAR_BUFFERS);

    return std::make_pair(src_formats_all_cbs, dst_formats_all_cbs);
}

static void emit_unpack_data_formats(std::string unpack_data_format_descs, std::vector<DataFormat> src_formats_all_cbs, std::vector<DataFormat> dst_formats_all_cbs) {
    // TODO: we should be emitting "unsigned char", no reason to use up 4B per data format
    ofstream file_stream;
    file_stream.open(unpack_data_format_descs);
    file_stream << create_formats_array_string("constexpr std::int32_t", "unpack_src_format", NUM_CIRCULAR_BUFFERS, data_format_vec_to_string(src_formats_all_cbs));
    file_stream << create_formats_array_string("constexpr std::int32_t", "unpack_dst_format", NUM_CIRCULAR_BUFFERS, data_format_vec_to_string(dst_formats_all_cbs));
    file_stream.close();
}

static std::pair<std::vector<DataFormat>, std::vector<DataFormat>>
generate_pack_data_formats(tt_hlk_desc& desc, DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en, const tt::ARCH arch) {
    vector<DataFormat> src_formats = tt::get_pack_src_formats(
        desc.input_buf_dataformat_arr, desc.param_buf_dataformat_arr, desc.intermediate_buf_dataformat_arr,
        desc.output_buf_dataformat_arr, unpack_conditional_dst_format, fp32_dest_acc_en, false, arch);

    vector<DataFormat> dst_formats = tt::get_pack_dst_formats(
        desc.input_buf_dataformat_arr, desc.param_buf_dataformat_arr, desc.intermediate_buf_dataformat_arr, desc.output_buf_dataformat_arr);

    TT_ASSERT(src_formats.size() == 16 && dst_formats.size() == 16,
        "There must be 8 pack src/dst formats for each output, and intermediate operands.");

    vector<DataFormat> src_formats_all_cbs;
    vector<DataFormat> dst_formats_all_cbs;
    tie(src_formats_all_cbs, dst_formats_all_cbs) = extend_pack_data_format_vectors_to_all_cbs(src_formats, dst_formats);

    TT_ASSERT(src_formats_all_cbs.size() == NUM_CIRCULAR_BUFFERS);
    TT_ASSERT(dst_formats_all_cbs.size() == NUM_CIRCULAR_BUFFERS);

    return std::make_pair(src_formats_all_cbs, dst_formats_all_cbs);
}

static void emit_pack_data_formats(std::string pack_data_format_descs, std::vector<DataFormat> src_formats_all_cbs, std::vector<DataFormat> dst_formats_all_cbs) {
    ofstream file_stream;
    file_stream.open(pack_data_format_descs);
    file_stream << create_formats_array_string("constexpr unsigned char", "pack_src_format", NUM_CIRCULAR_BUFFERS, data_format_vec_to_string(src_formats_all_cbs));
    file_stream << create_formats_array_string("constexpr unsigned char", "pack_dst_format", NUM_CIRCULAR_BUFFERS, data_format_vec_to_string(dst_formats_all_cbs));

    // budabackend-style format array
    // file_stream << create_formats_array_string("const std::int32_t", "pack_src_format", 16, data_format_vec_to_string(src_formats));
    // file_stream << create_formats_array_string("const std::int32_t", "pack_dst_format", 16, data_format_vec_to_string(dst_formats));

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

    // Determine what the packformat should be
    DataFormat pack_format =
        tt::get_pack_data_format(desc.output_buf_dataformat_arr, desc.intermediate_buf_dataformat_arr);

    // Determine dst format under ambiguous conditions (either or both l1 input & output formats are Float32)
    DataFormat unpack_conditional_dst_format = DataFormat::Invalid;
    if (pack_format == DataFormat::Float32) {
        ExpPrecision unpack_exp_prec = tt::get_data_exp_precision(desc.input_buf_dataformat_arr);
        unpack_conditional_dst_format =
            (unpack_exp_prec == ExpPrecision::A) ? DataFormat::Float16 : DataFormat::Float16_b;
    } else {
        ExpPrecision pack_exp_prec = tt::get_data_exp_precision(desc.output_buf_dataformat_arr);
        unpack_conditional_dst_format =
            (pack_exp_prec == ExpPrecision::A) ? DataFormat::Float16 : DataFormat::Float16_b;
    }

    if ((tt::is_all_fp32_formats(desc.input_buf_dataformat_arr) || options.preserve_fp32_precision) && options.fp32_dest_acc_en){
        if (options.preserve_fp32_precision) {
            unpack_conditional_dst_format = DataFormat::Float32;
        } else {
            unpack_conditional_dst_format = DataFormat::Tf32;
        }
    }

    tt::check_valid_in_out_data_formats(
        desc.input_buf_dataformat_arr,
        desc.output_buf_dataformat_arr,
        desc.param_buf_dataformat_arr,
        desc.intermediate_buf_dataformat_arr);

    vector<DataFormat> unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs;
    tie(unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs) = generate_unpack_data_formats(desc, unpack_conditional_dst_format, options.fp32_dest_acc_en, options.preserve_fp32_precision);

    vector<DataFormat> pack_src_formats_all_cbs, pack_dst_formats_all_cbs;
    tie(pack_src_formats_all_cbs, pack_dst_formats_all_cbs) = generate_pack_data_formats(desc, unpack_conditional_dst_format, options.fp32_dest_acc_en, arch);

    // equalize "upack src" and "pack dst" data format vectors
    // both "unpack src" and "pack dst" refer to data in L1, "unpack src" == L1, and "pack dst" == L1
    // in order to allow any CB to be read and written to/from L1, these formats should be the same (one cannot be DataFromat::Invalid if the other is set)
    // if both formats are DataFormat::Invalid then this CB is not used
    // this allows any CB to be used as both in & out in non-compute kernels (readers/writers)
    // TODO: for any CB to be used as both in & out of compute kernels (ie intermediate), additional work is required to propagate formats to "unpack dst (SRCA/B REG)" / "pack src (DST REG)"
    equalize_data_format_vectors(unpack_src_formats_all_cbs, pack_dst_formats_all_cbs);

    emit_unpack_data_formats(unpack_data_format_descs, unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs);
    emit_pack_data_formats(pack_data_format_descs, pack_src_formats_all_cbs, pack_dst_formats_all_cbs);
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

void jit_build_genfiles_descriptors(const JitBuildEnv& env,
                                    JitBuildOptions& options)
{
    ZoneScoped;
    const std::string tracyPrefix = "generate_descriptors_";
    ZoneName( (tracyPrefix + options.name).c_str(), options.name.length() + tracyPrefix.length());
    fs::create_directories(options.path);
    try {
        std::thread td( [&]() { generate_data_format_descriptors(options, env.get_arch()); } );
        std::thread tm( [&]() { generate_math_fidelity_descriptor(options); } );
        std::thread ta( [&]() { generate_math_approx_mode_descriptor(options); } );
        std::thread tf( [&]() { generate_dst_accum_mode_descriptor(options); } );
        td.join();
        tm.join();
        ta.join();
        tf.join();
    } catch (std::runtime_error &ex) {
        std::cerr << "EXCEPTION FROM THREADING IN GENERATE_DESCRIPTORS: " << ex.what() << std::endl;
    }
}

std::string generate_bank_to_noc_coord_descriptor_string(
    tt_xy_pair grid_size,
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<int32_t>& dram_bank_offset_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<int32_t>& l1_bank_offset_map,
    int core_count_per_dram,
    const std::map<CoreCoord, int32_t>& profiler_flat_id_map
) {
    stringstream ss;
    bool is_dram_pow2 = ceil(log2(dram_bank_map.size())) == log2(dram_bank_map.size());
    bool is_l1_pow2 = ceil(log2(l1_bank_map.size())) == log2(l1_bank_map.size());

    ss << "// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc." << endl;
    ss << "//" << endl;
    ss << "// SPDX-License-Identifier: Apache-2.0" << endl;
    ss << endl;
    ss << "/*" << endl;
    ss << " * This file is autogenerated by tt-metal runtime" << endl;
    ss << " * DO NOT EDIT" << endl;
    ss << " * This file contains values that are visible to the device compiled code." << endl;
    ss << " * CAREFUL: when included in the FW_BUILD, it defines global variables." << endl;
    ss << " * When included in KERNEL_BUILD, it declares global variables." << endl;
    ss << " */" << endl;
    ss << endl;
    ss << "#pragma once" << endl;
    ss << endl;
    ss << "#include <noc/noc_parameters.h>" << endl;
    ss << endl;

    ss << "#define LOG_BASE_2_OF_ALLOCATOR_ALIGNMENT " << std::bit_width(ALLOCATOR_ALIGNMENT) - 1 << endl;
    ss << "#define NUM_DRAM_BANKS " << dram_bank_map.size() << endl;
    ss << "#define NUM_L1_BANKS " << l1_bank_map.size() << endl;

    if (is_dram_pow2) {
        ss << "#define LOG_BASE_2_OF_NUM_DRAM_BANKS " << log2(dram_bank_map.size()) << endl;
    } else {
        ss << "#define IS_NOT_POW2_NUM_DRAM_BANKS 1" << endl;
    }
    if (is_l1_pow2) {
        ss << "#define LOG_BASE_2_OF_NUM_L1_BANKS " << log2(l1_bank_map.size()) << endl;
    } else {
        ss << "#define IS_NOT_POW2_NUM_L1_BANKS 1" << endl;
    }
    ss << endl;

    ss << "constexpr uint8_t noc_size_x = " << grid_size.x << ";" << endl;
    ss << "constexpr uint8_t noc_size_y = " << grid_size.y << ";" << endl;
    ss << endl;

    ss << "static_assert(NUM_NOCS == 2);" << endl;
    ss << endl;

    ss << "#ifdef KERNEL_BUILD" << endl;
    ss << endl;
    ss << "extern uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS];" << endl;
    ss << "extern int32_t bank_to_dram_offset[NUM_DRAM_BANKS];" << endl;
    ss << "extern uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS];" << endl;
    ss << "extern int32_t bank_to_l1_offset[NUM_L1_BANKS];" << endl;
#if defined(TRACY_ENABLE)
    ss << "#if defined(PROFILE_KERNEL) && (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC))" << endl;
    ss << "extern uint8_t noc_xy_to_profiler_flat_id[noc_size_x][noc_size_y];" << endl;
    ss << "extern uint16_t profiler_core_count_per_dram;" << endl;
    ss << "#endif" << endl;
#endif

    ss << endl;
    ss << "#else // !KERNEL_BUILD (FW_BUILD)" << endl;
    ss << endl;

    ss << "uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used)) = {" << endl;
    for (unsigned int noc = 0; noc < 2; noc++) {
        ss << "    {" << "\t// noc=" << noc << endl;
        for (unsigned int bank_id = 0; bank_id < dram_bank_map.size(); bank_id++) {
            uint16_t noc_x = NOC_0_X(noc, grid_size.x, dram_bank_map[bank_id].x);
            uint16_t noc_y = NOC_0_Y(noc, grid_size.y, dram_bank_map[bank_id].y);
            uint16_t xy = ((noc_y << NOC_ADDR_NODE_ID_BITS) | noc_x) << NOC_COORD_REG_OFFSET;
            ss << "        " << xy << "," << "\t// NOC_X=" << noc_x << " NOC_Y=" << noc_y << endl;
        }
        ss << "    }," << endl;
    }
    ss << "};" << endl;
    ss << endl;
    ss << "int32_t bank_to_dram_offset[NUM_DRAM_BANKS] __attribute__((used)) = {" << endl;
    for (unsigned int bank_id = 0; bank_id < dram_bank_map.size(); bank_id++) {
        ss << "    " << dram_bank_offset_map[bank_id] << "," << endl;
    }
    ss << "};" << endl;
    ss << endl;

#if defined(TRACY_ENABLE)
    /*
     * This part is adding the 2D array for sharing the flat IDs soc descriptor has assigned to every NOC coordinate,
     * and the ceiled number of cores per DRAM banks.
     *
     * The logic of flat ID assignment can be optimized to lower NOC traffic. With this design the heuristic can be implemented
     * in host and device just does look up to the table.
     *
     * For DRAM banks in particular, integer division of flat_id/core_count_per_dram gives the dram bank id and the modulo
     * is the offset.
     * */
    ss << "#if defined(PROFILE_KERNEL) && (defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_ERISC))" << endl;
    ss << "uint16_t profiler_core_count_per_dram __attribute__((used)) = ";
    ss << core_count_per_dram <<  ";" << endl;
    ss << endl;

    ss << "uint8_t noc_xy_to_profiler_flat_id[noc_size_x][noc_size_y] __attribute__((used)) = {" << endl;
    for (unsigned int x = 0; x < grid_size.x; x++) {
        ss << "    {" << endl;
        for (unsigned int y = 0; y < grid_size.y; y++) {
            CoreCoord core = {x,y};
            if (profiler_flat_id_map.find(core) == profiler_flat_id_map.end()){
                ss << "        " << 255 << "," << endl;
            }
            else{
                ss << "        " << profiler_flat_id_map.at(core) << "," << endl;
            }
        }
        ss << "    }," << endl;
    }
    ss << "};" << endl;
    ss << endl;
    ss << "#endif" << endl;

#endif


    ss << "uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used)) = {" << endl;
    for (unsigned int noc = 0; noc < 2; noc++) {
        ss << "    {" << "\t// noc=" << noc << endl;
        for (unsigned int bank_id = 0; bank_id < l1_bank_map.size(); bank_id++) {
            uint16_t noc_x = NOC_0_X(noc, grid_size.x, l1_bank_map[bank_id].x);
            uint16_t noc_y = NOC_0_Y(noc, grid_size.y, l1_bank_map[bank_id].y);
            uint16_t xy = ((noc_y << NOC_ADDR_NODE_ID_BITS) | noc_x) << NOC_COORD_REG_OFFSET;
            ss << "        " << xy << "," << "\t// NOC_X=" << noc_x << " NOC_Y=" << noc_y << endl;
        }
        ss << "    }," << endl;
    }
    ss << "};" << endl;
    ss << endl;
    ss << "int32_t bank_to_l1_offset[NUM_L1_BANKS]  __attribute__((used)) = {" << endl;
    for (unsigned int bank_id = 0; bank_id < l1_bank_map.size(); bank_id++) {
        ss << "    " << l1_bank_offset_map[bank_id] << "," << endl;
    }
    ss << "};" << endl;
    ss << endl;

    ss << "#endif // FW_BUILD" << endl;

    return ss.str();
}
void jit_build_genfiles_bank_to_noc_coord_descriptor(
    const string& path,
    tt_xy_pair grid_size,
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<int32_t>& dram_bank_offset_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<int32_t>& l1_bank_offset_map,
    int core_count_per_dram,
    const std::map<CoreCoord, int32_t>& profiler_flat_id_map
) {
    string output_string = generate_bank_to_noc_coord_descriptor_string(grid_size, dram_bank_map, dram_bank_offset_map, l1_bank_map, l1_bank_offset_map, core_count_per_dram, profiler_flat_id_map);

    fs::create_directories(path + "/brisc");
    ofstream file_stream_br(path + "/brisc/generated_bank_to_noc_coord_mapping.h");
    file_stream_br << output_string;
    file_stream_br.close();
    fs::create_directories(path + "/ncrisc");
    ofstream file_stream_nc(path + "/ncrisc/generated_bank_to_noc_coord_mapping.h");
    file_stream_nc << output_string;
    file_stream_nc.close();
    fs::create_directories(path + "/erisc");
    ofstream file_stream_ec(path + "/erisc/generated_bank_to_noc_coord_mapping.h");
    file_stream_ec << output_string;
    file_stream_ec.close();
    fs::create_directories(path + "/idle_erisc");
    ofstream file_stream_iec(path + "/idle_erisc/generated_bank_to_noc_coord_mapping.h");
    file_stream_iec << output_string;
    file_stream_iec.close();
}

static string generate_noc_core_xy_range_define(const std::vector<CoreCoord>& cores) {
    stringstream ss;

    string end_of_line = " \\\n    ( \\";
    for (const auto& core : cores) {
        ss << end_of_line << endl;
        ss << "    ((x) == NOC_0_X(noc_idx, noc_size_x, (uint32_t)" << core.x
           << ") && (y) == NOC_0_Y(noc_idx, noc_size_y, (uint32_t)" << core.y << "))";
        end_of_line = " || \\";
    }
    ss << ")" << endl;

    return ss.str();
}

} // namespace tt::tt_metal
