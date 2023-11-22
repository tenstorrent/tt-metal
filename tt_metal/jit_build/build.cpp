// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <string>

#include "jit_build/build.hpp"
#include "dev_mem_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "tools/profiler/profiler_state.hpp"

#include "noc/noc_parameters.h"

using namespace std;
using namespace tt;

namespace tt::tt_metal {

static std::string get_string_aliased_arch_lowercase(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "grayskull"; break;
        case tt::ARCH::WORMHOLE: return "wormhole"; break;
        case tt::ARCH::WORMHOLE_B0: return "wormhole"; break;
        default: return "invalid"; break;
    }
}

JitBuildEnv::JitBuildEnv()
{
}

void JitBuildEnv::init(uint32_t device_id, tt::ARCH arch)
{
    // Paths
    this->root_ = llrt::OptionsG.get_root_dir();
    this->out_root_ = this->root_ + "built/";
    this->arch_ = arch;
    this->arch_name_ = get_string_lowercase(arch);
    this->aliased_arch_name_ = get_string_aliased_arch_lowercase(arch);

    this->out_firmware_root_ = this->out_root_ + to_string(device_id) + "/firmware/";
    this->out_kernel_root_ = this->out_root_ + to_string(device_id) + "/kernels/";

    // Tools
    this->gpp_ = this->root_ + "tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-g++ ";
    this->objcopy_ = this->root_ + "tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-objcopy ";
    this->hex8tohex32_ = string("python3 ") + this->root_ + "tt_metal/hw/toolchain/hex8tohex32.py ";

    // Flags
    string common_flags;
    switch (arch) {
    case ARCH::GRAYSKULL:
        common_flags = "-mgrayskull -march=rv32iy -mtune=rvtt-b1 -mabi=ilp32 ";
        break;
    case ARCH::WORMHOLE_B0:
        common_flags = "-mwormhole -march=rv32imw -mtune=rvtt-b1 -mabi=ilp32 ";
        break;
    default:
        TT_ASSERT(false, "Invalid arch");
        break;
    }
    common_flags += "-std=c++17 -g -flto -ffast-math ";
    this->cflags_ = common_flags;
    this->cflags_ +=
        "-fno-use-cxa-atexit -fno-exceptions "
        "-Wall -Werror -Wno-unknown-pragmas "
        "-Wno-error=multistatement-macros -Wno-error=parentheses "
        "-Wno-error=unused-but-set-variable -Wno-unused-variable ";

    // Defines
    switch (arch) {
    case ARCH::GRAYSKULL:
        this->defines_ = "-DARCH_GRAYSKULL ";
        break;
    case ARCH::WORMHOLE_B0:
        this->defines_ = "-DARCH_WORMHOLE ";
        break;
    default:
        break;
    }
    this->defines_ += "-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 ";

    if (tt::tt_metal::getDeviceProfilerState()) {
        this->defines_ += "-DPROFILE_KERNEL=1 ";
    }

    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        this->defines_ += "-DWATCHER_ENABLED ";
    }

    if (tt::llrt::OptionsG.get_dprint_enabled()) {
        this->defines_ += "-DDEBUG_PRINT_ENABLED ";
    }

    // Includes
    // TODO(pgk) this list is insane
    this->includes_ = string("") +
        "-I. " +
        "-I.. " +
        "-I" + this->root_ + " " +
        "-I" + this->root_ + "tt_metal " +
        "-I" + this->root_ + "tt_metal/include " +
        "-I" + this->root_ + "tt_metal/hw/inc " +
        "-I" + this->root_ + "tt_metal/hw/inc/" + this->aliased_arch_name_ + " " +
        "-I" + this->root_ + "tt_metal/hw/inc/" + this->aliased_arch_name_ + "/" + this->arch_name_ + "_defines " +
        "-I" + this->root_ + "tt_metal/hw/inc/" + this->aliased_arch_name_ + "/noc " +
        "-I" + this->root_ + "tt_metal/third_party/umd/device/" + this->arch_name_ + " " + // TODO(fixme)
        "-I" + this->root_ + "tt_metal/hw/ckernels/" + this->arch_name_ + "/common/inc "; // TODO(fixme) datamovement fw shouldn't read this

    this->lflags_ = common_flags;
    this->lflags_ += "-fno-exceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";
}

JitBuildState::JitBuildState(const JitBuildEnv& env, int which, bool is_fw) : env_(env), core_id_(which), is_fw_(is_fw)
{
}

// Fill in common state derived from the default state set up in the constructors
void JitBuildState::finish_init()
{
    if (this->is_fw_) {
        this->defines_ += "-DFW_BUILD ";
    } else {
        this->defines_ += "-DKERNEL_BUILD ";
    }

    // Create the objs from the srcs
    for (string src : srcs_) {
        // Lop off the right side from the last "."
        string stub = src.substr(0, src.find_last_of("."));
        // Lop off the leading path
        stub = stub.substr(stub.find_last_of("/") + 1, stub.length());
        this->objs_.push_back(stub + ".o");
    }

    // Prepend root path to srcs, but not to outputs (objs) due to device dependency
    for (string& src : this->srcs_) {
        src = env_.root_ + src;
    }

    // Create list of object files for link
    for (const string& obj : this->objs_) {
        this->link_objs_ += obj + " ";
    }

    // Note the preceding slash which defies convention as this gets appended to
    // the kernel name used as a path which doesn't have a slash
    this->target_full_path_ = "/" + this->target_name_ + "/" + this->target_name_ + ".hex";
}

JitBuildDataMovement::JitBuildDataMovement(const JitBuildEnv& env, int which, bool is_fw) : JitBuildState(env, which, is_fw)
{
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 2, "Invalid data movement processor");

    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->cflags_ = env_.cflags_ +
        "-Os " +
        "-fno-tree-loop-distribute-patterns "; // don't use memcpy for cpy loops
    this->includes_ = env_.includes_ +
        "-I " + env_.root_ + "tt_metal/hw/firmware/src ";

    this->defines_ = env_.defines_;

    // TODO(pgk): build these once at init into built/libs!
    this->srcs_.push_back("tt_metal/hw/firmware/src/risc_common.cc");
    this->srcs_.push_back("tt_metal/hw/toolchain/substitutes.cpp");

    this->lflags_ = env_.lflags_ + "-Os ";

    switch (this->core_id_) {
    case 0:
        this->target_name_ = "brisc";

        this->defines_ += "-DCOMPILE_FOR_BRISC ";

        this->srcs_.push_back("tt_metal/hw/firmware/src/tdma_xmov.c");
        this->srcs_.push_back("tt_metal/hw/firmware/src/" + env_.aliased_arch_name_ + "/noc.c");
        if (this->is_fw_) {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/brisc.cc");
        } else {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0k.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/brisck.cc");
        }

        this->lflags_ +=
            "-T" + env_.root_ + "build/hw/toolchain/brisc.ld ";

        break;

    case 1:
        this->target_name_ = "ncrisc";

        this->defines_ += "-DCOMPILE_FOR_NCRISC ";

        if (this->is_fw_) {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/ncrisc.cc");
            this->srcs_.push_back("tt_metal/hw/toolchain/ncrisc-halt.S");
        } else {
            this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0k.S");
            this->srcs_.push_back("tt_metal/hw/firmware/src/ncrisck.cc");
        }

        this->lflags_ +=
            "-T" + env_.root_ + "build/hw/toolchain/ncrisc.ld ";

        break;
    }

    this->process_defines_at_compile = true;

    finish_init();
}

JitBuildCompute::JitBuildCompute(const JitBuildEnv& env, int which, bool is_fw) : JitBuildState(env, which, is_fw)
{
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 3, "Invalid compute processor");

    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->cflags_ = env_.cflags_ +
        "-O3 ";

    this->defines_ = env_.defines_;

    this->includes_ = env_.includes_ +
        "-I" + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/inc " +
        "-I" + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/llk_lib " +
        "-I" + env_.root_ + "tt_metal/third_party/sfpi/include " +
        "-I" + env_.root_ + "tt_metal/hw/firmware/src ";

    this->srcs_.push_back("tt_metal/hw/toolchain/substitutes.cpp");
    if (this->is_fw_) {
        this->srcs_.push_back("tt_metal/hw/firmware/src/trisc.cc");
        this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0.S");
    } else {
        this->srcs_.push_back("tt_metal/hw/ckernels/" + env_.arch_name_ + "/common/src/ckernel_template.cc");
        this->srcs_.push_back("tt_metal/hw/firmware/src/trisck.cc");
        this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0k.S");
    }

    this->lflags_ = env_.lflags_ + "-O3 ";

    switch (this->core_id_) {
    case 0:
        this->target_name_ = "trisc0";

        this->defines_ += "-DUCK_CHLKC_UNPACK ";
        this->defines_ += "-DNAMESPACE=chlkc_unpack ";
        this->defines_ += "-DCOMPILE_FOR_TRISC=0 ";

        this->lflags_ +=
            "-T" + env_.root_ + "build/hw/toolchain/trisc0.ld ";

        break;

    case 1:
        this->target_name_ = "trisc1";

        this->defines_ += "-DUCK_CHLKC_MATH ";
        this->defines_ += "-DNAMESPACE=chlkc_math ";
        this->defines_ += "-DCOMPILE_FOR_TRISC=1 ";

        this->lflags_ +=
            "-T" + env_.root_ + "build/hw/toolchain/trisc1.ld ";

        break;

    case 2:
        this->target_name_ = "trisc2";

        this->defines_ += "-DUCK_CHLKC_PACK ";
        this->defines_ += "-DNAMESPACE=chlkc_pack ";
        this->defines_ += "-DCOMPILE_FOR_TRISC=2 ";

        this->lflags_ +=
            "-T" + env_.root_ + "build/hw/toolchain/trisc2.ld ";

        break;
    }

    this->process_defines_at_compile = false;

    finish_init();
}

JitBuildEthernet::JitBuildEthernet(const JitBuildEnv& env, int which, bool is_fw) : JitBuildState(env, which, is_fw)
{
    this->target_name_ = "erisc";

    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->cflags_ = env_.cflags_ + "-Os -fno-delete-null-pointer-checks ";

    this->defines_ = env_.defines_ +
        "-DCOMPILE_FOR_ERISC "
        "-DERISC "
        "-DRISC_B0_HW ";
    if (this->is_fw_) {
        this->defines_ += "-DLOADING_NOC=0 ";
    }

    this->includes_ = env_.includes_ +
        "-I " + env_.root_ + "tt_metal/hw/inc/ethernet ";

    this->srcs_.push_back("tt_metal/hw/toolchain/substitutes.cpp");
    if (this->is_fw_) {
        this->srcs_.push_back("tt_metal/hw/firmware/src/erisc.cc");
    } else {
        this->srcs_.push_back("tt_metal/hw/firmware/src/erisck.cc");
        this->srcs_.push_back("tt_metal/hw/toolchain/tmu-crt0k.S");
    }

    this->lflags_ = env_.lflags_ +
        "-Os "
        "-L" + env_.root_ + "/tt_metal/hw/toolchain "
        "-T" + env_.root_ + "tt_metal/hw/toolchain/erisc-b0-app.ld ";

    this->process_defines_at_compile = true;

    finish_init();
}

//////////////////
// TRISCs       //
//////////////////
static void gen_trisc_cpp(const string& src_name, const string& dst_name, vector<string>& prolog)
{
    std::ofstream out(dst_name);
    for (auto s: prolog)
        out << s;
    out << "#include \"" << src_name << "\"\n";
}

void jit_build_genfiles_triscs_src(const JitBuildEnv& env,
                                   const JitBuildSettings& settings,
                                   const string& input_hlk_file_path)
{
    // Note: assumes dirs (and descriptors) already created
    log_trace(tt::LogBuildKernels, "Generating defines for TRISCs");

    string out_dir = env.get_out_kernel_root_path() + settings.get_full_kernel_name() + "/";;
    string unpack_base        = out_dir + "chlkc_unpack";
    string math_base          = out_dir + "chlkc_math";
    string pack_base          = out_dir + "chlkc_pack";
    string unpack_cpp         = unpack_base + ".cpp";
    string unpack_llk_args_h  = unpack_base + "_llk_args.h";
    string math_cpp           = math_base + ".cpp";
    string math_llk_args_h    = math_base + "_llk_args.h";
    string pack_cpp           = pack_base + ".cpp";
    string pack_llk_args_h    = pack_base + "_llk_args.h";

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
    std::thread t0( [&]() { gen_trisc_cpp(input_hlk_file_path, unpack_cpp, unpack_prolog); } );
    std::thread t1( [&]() { gen_trisc_cpp(input_hlk_file_path, math_cpp, math_prolog); } );
    std::thread t2( [&]() { gen_trisc_cpp(input_hlk_file_path, pack_cpp, pack_prolog); } );
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

std::pair<vector<DataFormat>,vector<DataFormat>> extend_unpack_data_format_vectors_to_all_cbs(const vector<DataFormat> &src_formats, const vector<DataFormat> &dst_formats) {
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

std::pair<vector<DataFormat>,vector<DataFormat>> extend_pack_data_format_vectors_to_all_cbs(const vector<DataFormat> &src_formats, const vector<DataFormat> &dst_formats) {
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

std::string data_format_vec_to_string(const vector<DataFormat> formats) {
    std::string formats_string = "";
    for (int i = 0; i < formats.size(); i++) {
        formats_string += to_string((int)formats[i]) + ",";
    }
    return formats_string;
}

std::string create_formats_array_string(std::string array_type, std::string array_name, int array_size, std::string array_data) {
    stringstream str_stream;

    str_stream << array_type << " " << array_name << "[" << array_size << "] = {" << endl;
    str_stream << "    " << array_data << endl;
    str_stream << "};" << endl;

    return str_stream.str();
}

std::pair<std::vector<DataFormat>, std::vector<DataFormat>>
generate_unpack_data_formats(tt_hlk_desc& desc, DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en) {

    vector<DataFormat> src_formats = tt::get_unpack_src_formats(
        desc.input_buf_dataformat_arr, desc.param_buf_dataformat_arr, desc.intermediate_buf_dataformat_arr);

    vector<DataFormat> dst_formats = tt::get_unpack_dst_formats(
        desc.input_buf_dataformat_arr, desc.param_buf_dataformat_arr, desc.intermediate_buf_dataformat_arr,
        desc.output_buf_dataformat_arr, unpack_conditional_dst_format, fp32_dest_acc_en);

    TT_ASSERT(src_formats.size() == 24 && dst_formats.size() == 24,
        "There must be 8 unpack src/dst formats for each input, param, and intermediate operands.");

    vector<DataFormat> src_formats_all_cbs;
    vector<DataFormat> dst_formats_all_cbs;
    tie(src_formats_all_cbs, dst_formats_all_cbs) = extend_unpack_data_format_vectors_to_all_cbs(src_formats, dst_formats);

    TT_ASSERT(src_formats_all_cbs.size() == NUM_CIRCULAR_BUFFERS);
    TT_ASSERT(dst_formats_all_cbs.size() == NUM_CIRCULAR_BUFFERS);

    return std::make_pair(src_formats_all_cbs, dst_formats_all_cbs);
}

void emit_unpack_data_formats(std::string unpack_data_format_descs, std::vector<DataFormat> src_formats_all_cbs, std::vector<DataFormat> dst_formats_all_cbs) {
    // TODO: we should be emitting "unsigned char", no reason to use up 4B per data format
    ofstream file_stream;
    file_stream.open(unpack_data_format_descs);
    file_stream << create_formats_array_string("const std::int32_t", "unpack_src_format", NUM_CIRCULAR_BUFFERS, data_format_vec_to_string(src_formats_all_cbs));
    file_stream << create_formats_array_string("const std::int32_t", "unpack_dst_format", NUM_CIRCULAR_BUFFERS, data_format_vec_to_string(dst_formats_all_cbs));
    file_stream.close();
}

std::pair<std::vector<DataFormat>, std::vector<DataFormat>>
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

void emit_pack_data_formats(std::string pack_data_format_descs, std::vector<DataFormat> src_formats_all_cbs, std::vector<DataFormat> dst_formats_all_cbs) {
    ofstream file_stream;
    file_stream.open(pack_data_format_descs);
    // TODO: we should be emitting "unsigned char", no reason to use 4B per data format
    file_stream << create_formats_array_string("const std::int32_t", "pack_src_format", NUM_CIRCULAR_BUFFERS, data_format_vec_to_string(src_formats_all_cbs));
    file_stream << create_formats_array_string("const std::int32_t", "pack_dst_format", NUM_CIRCULAR_BUFFERS, data_format_vec_to_string(dst_formats_all_cbs));

    // budabackend-style format array
    // file_stream << create_formats_array_string("const std::int32_t", "pack_src_format", 16, data_format_vec_to_string(src_formats));
    // file_stream << create_formats_array_string("const std::int32_t", "pack_dst_format", 16, data_format_vec_to_string(dst_formats));

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

    if (tt::is_all_fp32_formats(desc.input_buf_dataformat_arr) && options.fp32_dest_acc_en){
        unpack_conditional_dst_format = DataFormat::Tf32;
    }

    tt::check_valid_in_out_data_formats(
        desc.input_buf_dataformat_arr,
        desc.output_buf_dataformat_arr,
        desc.param_buf_dataformat_arr,
        desc.intermediate_buf_dataformat_arr);

    vector<DataFormat> unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs;
    tie(unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs) = generate_unpack_data_formats(desc, unpack_conditional_dst_format, options.fp32_dest_acc_en);

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

#define NOC_X(noc_index, noc_size_x, x) (noc_index == 0 ? (x) : (noc_size_x-1-(x)))
#define NOC_Y(noc_index, noc_size_y, y) (noc_index == 0 ? (y) : (noc_size_y-1-(y)))

std::string generate_bank_to_noc_coord_descriptor_string(
    tt_xy_pair grid_size,
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<int32_t>& dram_bank_offset_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<int32_t>& l1_bank_offset_map
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

    ss << endl;
    ss << "#else // !KERNEL_BUILD (FW_BUILD)" << endl;
    ss << endl;

    ss << "uint16_t dram_bank_to_noc_xy[NUM_NOCS][NUM_DRAM_BANKS] __attribute__((used)) = {" << endl;
    for (unsigned int noc = 0; noc < 2; noc++) {
        ss << "    {" << endl;
        for (unsigned int bank_id = 0; bank_id < dram_bank_map.size(); bank_id++) {
            uint16_t xy =
                ((NOC_Y(noc, grid_size.y, dram_bank_map[bank_id].y) << NOC_ADDR_NODE_ID_BITS) |
                 NOC_X(noc, grid_size.x, dram_bank_map[bank_id].x)) <<
                (NOC_ADDR_LOCAL_BITS - 32);
            ss << "        " << xy << "," << endl;
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

    ss << "uint16_t l1_bank_to_noc_xy[NUM_NOCS][NUM_L1_BANKS] __attribute__((used)) = {" << endl;
    for (unsigned int noc = 0; noc < 2; noc++) {
        ss << "    {" << endl;
        for (unsigned int bank_id = 0; bank_id < l1_bank_map.size(); bank_id++) {
            uint16_t xy =
                ((NOC_Y(noc, grid_size.y, l1_bank_map[bank_id].y) << NOC_ADDR_NODE_ID_BITS) |
                 NOC_X(noc, grid_size.x, l1_bank_map[bank_id].x)) <<
                (NOC_ADDR_LOCAL_BITS - 32);
            ss << "        " << xy << "," << endl;
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
void generate_bank_to_noc_coord_descriptor(
    const string& path,
    tt_xy_pair grid_size,
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<int32_t>& dram_bank_offset_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<int32_t>& l1_bank_offset_map
) {
    string output_string = generate_bank_to_noc_coord_descriptor_string(grid_size, dram_bank_map, dram_bank_offset_map, l1_bank_map, l1_bank_offset_map);

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
}

static string generate_noc_core_xy_range_define(const std::vector<CoreCoord>& cores) {
    stringstream ss;

    string end_of_line = " \\\n    ( \\";
    for (const auto& core : cores) {
        ss << end_of_line << endl;
        ss << "    ((x) == NOC_X((uint32_t)" << core.x << ") && (y) == NOC_Y((uint32_t)" << core.y << "))";
        end_of_line = " || \\";
    }
    ss << ")" << endl;

    return ss.str();
}

static string generate_noc_addr_ranges_string(
    uint64_t pcie_addr_base,
    uint64_t pcie_addr_size,
    uint64_t dram_addr_base,
    uint64_t dram_addr_size,
    const std::vector<CoreCoord>& pcie_cores,
    const std::vector<CoreCoord>& dram_cores,
    const std::vector<CoreCoord>& ethernet_cores,
    CoreCoord grid_size,
    const std::vector<uint32_t>& harvested_rows,
    const vector<CoreCoord>& dispatch_cores) {

    TT_ASSERT(dispatch_cores.size() == 1, "Only 1 dispatch core supported so far");

    stringstream ss;

    ss << "// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc." << endl;
    ss << "//" << endl;
    ss << "// SPDX-License-Identifier: Apache-2.0" << endl;
    ss << endl;

    ss << "/*" << endl;
    ss << " * This file is autogenerated by tt-metal API" << endl;
    ss << " * DO NOT EDIT" << endl;
    ss << " * This file contains values for use in device compiled code." << endl;
    ss << " * The macros here can be used to, eg, validate the sanity of noc addresses." << endl;
    ss << " */" << endl;
    ss << endl;
    ss << "#pragma once" << endl;
    ss << endl;

    ss << "#define NOC_PCIE_ADDR_BASE (uint64_t) 0x" << std::hex << pcie_addr_base << std::dec << endl;
    ss << "#define NOC_PCIE_ADDR_SIZE (uint64_t) 0x" << std::hex << pcie_addr_size << std::dec << endl;
    ss << "#define NOC_PCIE_ADDR_END (NOC_PCIE_ADDR_BASE + NOC_PCIE_ADDR_SIZE)" << endl;
    ss << endl;
    ss << "#define NOC_DRAM_ADDR_BASE 0x" << std::hex << dram_addr_base << std::dec << endl;
    ss << "#define NOC_DRAM_ADDR_SIZE 0x" << std::hex << dram_addr_size << std::dec << endl;
    ss << "#define NOC_DRAM_ADDR_END (NOC_DRAM_ADDR_BASE + NOC_DRAM_ADDR_SIZE)" << endl;
    ss << endl;

    if (pcie_addr_base == pcie_addr_size) {
        // If the address range is 0, then there are no PCIe cores (non-mmio device)
        ss << "#define NOC_PCIE_XY_P(x, y) false" << endl;
    } else {
        ss << "#define NOC_PCIE_XY_P(x, y)";
        ss << generate_noc_core_xy_range_define(pcie_cores);
    }
    ss << endl;

    ss << "#define NOC_DRAM_XY_P(x, y)";
    ss << generate_noc_core_xy_range_define(dram_cores);
    ss << endl;

    ss << "#define NOC_ETH_XY_P(x, y)";
    if (ethernet_cores.size() == 0) {
        ss << " false" << endl;
    } else {
        ss << generate_noc_core_xy_range_define(ethernet_cores);
    }
    ss << endl;

    ss << "#define NOC_HARVESTED_Y_P(y)";
    if (harvested_rows.size() == 0) {
        ss << " false" << endl;
    } else {
        string join = " \\\n    ( \\\n";
        for (const auto& y : harvested_rows) {
            ss << join << "     (NOC_Y((y)) == " << y << ")";
            join = " || \\\n";
        }
        ss << ")" << endl;
    }
    ss << endl;

    ss << "#define NOC_WORKER_XY_P(x, y) \\" << endl;
    ss << "    (!NOC_PCIE_XY_P(x, y) && \\" << endl;
    ss << "     !NOC_DRAM_XY_P(x, y) && \\" << endl;
    ss << "     !NOC_ETH_XY_P(x, y) && \\" << endl;
    ss << "     !NOC_HARVESTED_Y_P(y) && \\" << endl;
    ss << "     ((noc_index == 0) ? \\" << endl;
    ss << "      ((x) >= NOC_X((uint32_t)" << 1 << ") && \\" << endl;
    ss << "       (x) <= NOC_X((uint32_t)" << grid_size.x - 1 << ") && \\" << endl;
    ss << "       (y) >= NOC_Y((uint32_t)" << 1 << ") && \\" << endl;
    ss << "       (y) <= NOC_Y((uint32_t)" << grid_size.y - 1 << ")) : \\" << endl;
    ss << "      ((x) <= NOC_X((uint32_t)" << 1 << ") && \\" << endl;
    ss << "       (x) >= NOC_X((uint32_t)" << grid_size.x - 1 << ") && \\" << endl;
    ss << "       (y) <= NOC_Y((uint32_t)" << 1 << ") && \\" << endl;
    ss << "       (y) >= NOC_Y((uint32_t)" << grid_size.y - 1<< "))))";
    ss << endl;

    ss << endl;
    ss << "#define DISPATCH_CORE_X " << dispatch_cores[0].x << endl;
    ss << "#define DISPATCH_CORE_Y " << dispatch_cores[0].y << endl;

    return ss.str();
}

void generate_noc_addr_ranges_header(
    const std::string& path,
    uint64_t pcie_addr_base,
    uint64_t pcie_addr_size,
    uint64_t dram_addr_base,
    uint64_t dram_addr_size,
    const std::vector<CoreCoord>& pcie_cores,
    const std::vector<CoreCoord>& dram_cores,
    const std::vector<CoreCoord>& ethernet_cores,
    CoreCoord grid_size,
    const std::vector<uint32_t>& harvested_rows,
    const vector<CoreCoord>& dispatch_cores) {

    string output_string = generate_noc_addr_ranges_string(pcie_addr_base, pcie_addr_size, dram_addr_base, dram_addr_size,
                                                           pcie_cores, dram_cores, ethernet_cores, grid_size, harvested_rows, dispatch_cores);

    ofstream file_stream_br(path + "/brisc/noc_addr_ranges_gen.h");
    file_stream_br << output_string;
    file_stream_br.close();

    fs::create_directories(path + "/ncrisc");
    ofstream file_stream_nc(path + "/ncrisc/noc_addr_ranges_gen.h");
    file_stream_nc << output_string;
    file_stream_nc.close();

    fs::create_directories(path + "/erisc");
    ofstream file_stream_er(path + "/erisc/noc_addr_ranges_gen.h");
    file_stream_er << output_string;
    file_stream_er.close();
}

static void build_failure(const string& target_name,
                          const string& op,
                          const string& cmd,
                          const string& log_file)
{
    log_info(tt::LogBuildKernels, "{} {} failure -- cmd: {}", target_name, op, cmd);
    string cat = "cat " + log_file;
    if (fs::exists(log_file)) {
        // XXXX PGK(TODO) not portable
        if (system(cat.c_str())) {
            TT_THROW("Failed system comand {}", cat);
        }
    }
    TT_THROW("{} build failed", target_name);
}

void JitBuildState::pre_compile(const string& kernel_in_path, const string& op_out_path) const
{
}

void JitBuildState::copy_kernel(const string& kernel_in_path, const string& op_out_path) const
{
    // TODO(pgk): get rid of this copy, compile kernel file in place as its own .o
    const string out_dir = this->out_path_ + op_out_path + this->target_name_;
    const string src = env_.get_root_path() + kernel_in_path;
    const string dst = out_dir + "/kernel.cpp";
    fs::copy(src, dst, fs::copy_options::overwrite_existing);
}

void JitBuildDataMovement::pre_compile(const string& kernel_in_path, const string& op_out_path) const
{
    copy_kernel(kernel_in_path, op_out_path);
}

void JitBuildEthernet::pre_compile(const string& kernel_in_path, const string& op_out_path) const
{
    copy_kernel(kernel_in_path, op_out_path);
}


void JitBuildState::compile_one(const string& log_file,
                                const string& out_dir,
                                const JitBuildSettings *settings,
                                const string& src,
                                const string& obj) const
{
    fs::create_directories(out_dir);

    // Add kernel specific defines
    string defines = this->defines_;
    if (settings != nullptr) {
        if (process_defines_at_compile) {
            settings->process_defines([&defines] (const string& define, const string& value) {
                defines += "-D" + define + "=" + value + " ";
            });
        }

        settings->process_compile_time_args([&defines] (int i, uint32_t value) {
            defines += "-DKERNEL_COMPILE_TIME_ARG_" + to_string(i) + "=" + to_string(value) + " ";
        });
    }

    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.gpp_;
    cmd += this->cflags_;
    cmd += defines;
    cmd += this->includes_;
    cmd += "-c -o " + obj + " " + src;

    log_debug(tt::LogBuildKernels, "    g++ compile cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "compiile", cmd, log_file);
    }
}

void JitBuildState::compile(const string& log_file, const string& out_dir, const JitBuildSettings *settings) const
{
    // Compile each of the srcs to an obj in parallel
    std::vector<std::thread> threads;
    threads.resize(this->srcs_.size());;
    for (int i = 0; i < this->srcs_.size(); i++) {
        threads[i] = thread(&JitBuildState::compile_one, &*this,
                            ref(log_file), ref(out_dir), settings, ref(this->srcs_[i]), ref(this->objs_[i]));
    }

    for (auto& th: threads) {
        th.join();
    }
}

void JitBuildState::link(const string& log_file, const string& out_dir) const
{
    string lflags = this->lflags_;
    if (tt::llrt::OptionsG.get_build_map_enabled()) {
        lflags += " -Wl,-Map=" + out_dir + "linker.map";
    }

    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.gpp_;
    cmd += this->lflags_;
    cmd += this->link_objs_;

    if (!this->is_fw_) {
        string weakened_elf_name = env_.out_firmware_root_ + this->target_name_ + "/" + this->target_name_  + "_weakened.elf";
        cmd += " -Xlinker \"--just-symbols=" + weakened_elf_name + "\" ";
    }

    cmd += "-o " + out_dir + this->target_name_ + ".elf";
    log_debug(tt::LogBuildKernels, "    g++ link cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "link", cmd, log_file);
    }
}

void JitBuildState::elf_to_hex8(const string& log_file, const string& out_dir) const
{
    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.objcopy_;
    cmd += " -O verilog " + this->target_name_ + ".elf" + " " + this->target_name_ + ".hex.tmp";

    log_debug(tt::LogBuildKernels, "    objcopy cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "objcopy", cmd, log_file);
    }
}

void JitBuildState::hex8_to_hex32(const string& log_file, const string& out_dir) const
{
    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.hex8tohex32_ + this->target_name_ + ".hex.tmp " + this->target_name_ + ".hex";

    log_debug(tt::LogBuildKernels, "    hex8tohex32 cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "hex8tohex32.py", cmd, log_file);
    }
}

// Given this elf (A) and a later elf (B):
// weakens symbols in A so that it can be used as a "library" for B. B imports A's weakened symbols, B's symbols of the
// same name don't result in duplicate symbols but B can reference A's symbols. Force the fw_export symbols to remain
// strong so to propogate link addresses
void JitBuildState::weaken(const string& log_file, const string& out_dir) const
{
    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.objcopy_;
    cmd += " --wildcard --weaken-symbol \"*\" --weaken-symbol \"!__fw_export_*\" " +
        this->target_name_ + ".elf " + this->target_name_ + "_weakened.elf";

    log_debug(tt::LogBuildKernels, "    objcopy cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "objcopy weaken", cmd, log_file);
    }
}

void JitBuildState::build(const JitBuildSettings *settings) const
{
    string out_dir = (settings == nullptr) ?
        this->out_path_ + this->target_name_ + "/" :
        this->out_path_ + settings->get_full_kernel_name() + this->target_name_ + "/";

    string log_file = out_dir + "build.log";
    if (fs::exists(log_file)) {
        std::remove(log_file.c_str());
    }

    compile(log_file, out_dir, settings);
    link(log_file, out_dir);
    elf_to_hex8(log_file, out_dir);
    hex8_to_hex32(log_file, out_dir);
    if (this->is_fw_) {
        weaken(log_file, out_dir);
    }
}

void jit_build(const JitBuildState& build,
               const JitBuildSettings *settings,
               const string& kernel_in_path)
{
    ZoneScoped;
    const std::string tracyPrefix = "jit_build";

    if (settings != nullptr) {
        build.pre_compile(kernel_in_path, settings->get_full_kernel_name());
    }

    build.build(settings);
}

void jit_build_set(const JitBuildStateSet& build_set,
                   const JitBuildSettings *settings,
                   const string& kernel_in_path)
{
    ZoneScoped;
    const std::string tracyPrefix = "jit_build_set";
    ZoneName( tracyPrefix, tracyPrefix.length());

    std::vector<std::thread> threads;
    threads.resize(build_set.size());
    for (int i = 0; i < build_set.size(); i++) {
        const JitBuildState& build = *(build_set[i]);
        std::function<void()> lambda = [&build, &kernel_in_path, &settings] () {
            if (settings != nullptr) {
                build.pre_compile(kernel_in_path, settings->get_full_kernel_name());
            }
            build.build(settings);
        };
        threads[i] = thread(lambda);
    }

    for (auto& th: threads) {
        th.join();
    }
}

void jit_build_subset(const JitBuildStateSubset& build_subset,
                      const JitBuildSettings *settings,
                      const string& kernel_in_path)
{
    ZoneScoped;
    const std::string tracyPrefix = "jit_build_subset";
    ZoneName( tracyPrefix, tracyPrefix.length());

    std::vector<std::thread> threads;
    threads.resize(build_subset.size);
    for (int i = 0; i < build_subset.size; i++) {
        const JitBuildState& build = *(build_subset.build_ptr[i]);
        std::function<void()> lambda = [&build, &kernel_in_path, &settings] () {
            if (settings != nullptr) {
                build.pre_compile(kernel_in_path, settings->get_full_kernel_name());
            }
            build.build(settings);
        };
        threads[i] = thread(lambda);
    }

    for (auto& th: threads) {
        th.join();
    }
}

// TODO(AP): can move joins for these threads to happen later (to compiler launch)
void generate_descriptors(const JitBuildEnv& env,
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
        td.join();
        tm.join();
        ta.join();
    } catch (std::runtime_error &ex) {
        std::cerr << "EXCEPTION FROM THREADING IN GENERATE_DESCRIPTORS: " << ex.what() << std::endl;
    }
}

} // namespace tt_metal
