
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <string>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "hlkc/hlkc_api.h"
#include "l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"

using namespace std;
using namespace tt;

void add_profile_kernel_to_make_cmd(std::stringstream &make_cmd) {
    make_cmd << " PROFILE_KERNEL=1";
}

void add_noc_index_to_make_cmd(std::stringstream &make_cmd, const std::uint8_t noc_index) {
    make_cmd << " NOC_INDEX=" << std::to_string(noc_index);
}

void add_kernel_compile_time_args_to_make_cmd(std::stringstream &make_cmd, const std::vector<std::uint32_t>& kernel_compile_time_args) {
    for (int i = 0; i<kernel_compile_time_args.size(); i++) {
        make_cmd << " KERNEL_COMPILE_TIME_ARG_" << i << "=" << std::to_string(kernel_compile_time_args[i]);
    }
}

void add_kernel_compile_time_args_to_make_cmd(std::string &make_cmd_string, const std::vector<std::uint32_t>& kernel_compile_time_args) {
    stringstream make_cmd;
    for (int i = 0; i<kernel_compile_time_args.size(); i++) {
        make_cmd << " KERNEL_COMPILE_TIME_ARG_" << i << "=" << std::to_string(kernel_compile_time_args[i]);
    }
    make_cmd_string += make_cmd.str() + " ";
}



//////////////////
// BRISC       //
//////////////////

void generate_binary_for_brisc(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, const std::string &out_dir_path, const std::string& arch_name, const std::uint8_t noc_index, const std::vector<std::uint32_t>& kernel_compile_time_args, bool profile_kernel) {
    log_info(tt::LogBuildKernels, "Compiling BRISC");

    string root_dir = std::getenv("TT_METAL_HOME");
    auto old_cwd = fs::current_path(); //getting path
    fs::current_path(root_dir); //setting path

    stringstream make_clean_cmd;
    stringstream make_cmd;
    std::string make_flags = " OUTPUT_DIR=" + fs::absolute(out_dir_path).string() + "/brisc ";

    // this is in the top-level outdir because "/brisc" is deleted by make clean, but we log the output of make clean as well
    string log_file = fs::absolute(out_dir_path).string() + "/brisc_build.log";
    utils::create_file(log_file);

    // Build clean
    make_clean_cmd << " make -C src/firmware/riscv/targets/brisc clean " << make_flags;

    // default ARCH_NAME is grayskull in Makefile
    TT_ASSERT(
        (arch_name.compare("grayskull") == 0) || (arch_name.compare("wormhole") == 0) ||
        (arch_name.compare("wormhole_b0") == 0));

    // Build
    if (arch_name.compare("grayskull") != 0) {
        make_cmd << " ARCH_NAME=" << arch_name;
    }

    make_cmd << " make -C src/firmware/riscv/targets/brisc";
    make_cmd << make_flags;

    // Make a directory "brisc_kernel" in the out_dir_path
    std::string kernel_dir = out_dir_path + "/brisc_kernel";
    fs::create_directories(kernel_dir);
    // Pass this directoy as include path the the BRISC makefile
    make_cmd << " KERNEL_INC=" << fs::absolute(kernel_dir).string();

    // If flag is set, print kernel profiling timer marks
    if (profile_kernel)
    {
        add_profile_kernel_to_make_cmd(make_cmd);
    }
    // Add NOC_INDEX for brisc
    add_noc_index_to_make_cmd(make_cmd, noc_index);
    // Add a list of KERNEL_COMPILE_TIME_ARGs for brisc
    add_kernel_compile_time_args_to_make_cmd(make_cmd, kernel_compile_time_args);

    // copy the BRISC kernel to that directory, w/ a generic filename kernel.cpp (this is what brisc.cc includes)
    string kernel_file_name = build_kernel_for_riscv_options->brisc_kernel_file_name;
    fs::copy(kernel_file_name, kernel_dir + "/kernel.cpp", fs::copy_options::overwrite_existing);

    // copy unpack/pack data formats to the kernel dir
    if (fs::exists(out_dir_path + "/chlkc_unpack_data_format.h")) {
        fs::copy(out_dir_path + "/chlkc_unpack_data_format.h", kernel_dir, fs::copy_options::overwrite_existing);
        fs::copy(out_dir_path + "/chlkc_pack_data_format.h", kernel_dir, fs::copy_options::overwrite_existing);
    }

    log_debug(tt::LogBuildKernels, "    Make clean cmd: {}", make_clean_cmd.str());
    if (!tt::utils::run_command(make_clean_cmd.str(), log_file, false)) {
        log_fatal(tt::LogBuildKernels, " BRISC clean failed -- cmd: {}", make_clean_cmd.str());
        exit(1);
    }
    log_debug(tt::LogBuildKernels, "    Make compile cmd: {}", make_cmd.str());
    if (!tt::utils::run_command(make_cmd.str(), log_file, false)) {
        log_fatal(tt::LogBuildKernels, " BRISC Build failed -- cmd: {}", make_cmd.str());
        exit(1);
    }

    fs::current_path(old_cwd); // restore cwd
}


//////////////////
// NCRISC       //
//////////////////

void generate_binary_for_ncrisc(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, const std::string &out_dir_path, const std::string& arch_name, const std::uint8_t noc_index, const std::vector<std::uint32_t>& kernel_compile_time_args, bool profile_kernel) {
    log_info(tt::LogBuildKernels, "Compiling NCRISC");

    string root_dir = std::getenv("TT_METAL_HOME");
    auto old_cwd = fs::current_path();
    fs::current_path(root_dir); // make build work from any cwd

    stringstream make_clean_cmd;
    stringstream make_cmd;
    std::string make_flags = " OUTPUT_DIR=" + fs::absolute(out_dir_path).string() + "/ncrisc ";

    // this is in the top-level outdir because "/ncrisc" is deleted by make clean, but we log the output of make clean as well
    string log_file = fs::absolute(out_dir_path).string() + "/ncrisc_build.log";
    utils::create_file(log_file);

    bool is_perf_dump_en = false;
    bool is_perf_spill_dram = false;
    uint32_t perf_dump_level = 0;

    // Build clean
    make_clean_cmd << " make -C src/firmware/riscv/targets/ncrisc clean " << make_flags;

    // default ARCH_NAME is grayskull in Makefile
    TT_ASSERT(
        (arch_name.compare("grayskull") == 0) || (arch_name.compare("wormhole") == 0) ||
        (arch_name.compare("wormhole_b0") == 0));

    // Build
    if (arch_name.compare("grayskull") != 0) {
        make_cmd << " ARCH_NAME=" << arch_name;
    }

    make_cmd << " make -C src/firmware/riscv/targets/ncrisc";
    make_cmd << make_flags;
    make_cmd << " PERF_DUMP=" << to_string(is_perf_dump_en);
    make_cmd << " INTERMED_DUMP=" << to_string(is_perf_spill_dram);
    make_cmd << " PERF_DUMP_LEVEL=" << to_string(perf_dump_level);

    // Make a directory "ncrisc_kernel" in the out_dir_path
    std::string kernel_dir = out_dir_path + "/ncrisc_kernel";
    fs::create_directories(kernel_dir);
    // Pass this directoy as include path the the NCRISC makefile
    make_cmd << " KERNEL_INC=" << fs::absolute(kernel_dir).string();

    // If flag is set, print kernel profiling timer marks
    if (profile_kernel)
    {
        add_profile_kernel_to_make_cmd(make_cmd);
    }
    // Add NOC_INDEX for ncrisc
    add_noc_index_to_make_cmd(make_cmd, noc_index);
    // Add a list of KERNEL_COMPILE_TIME_ARGs for ncrisc
    add_kernel_compile_time_args_to_make_cmd(make_cmd, kernel_compile_time_args);

    // copy the NCRISC kernel to that directory, w/ a generic filename kernel.cpp (this is what ncrisc.cc includes)
    string kernel_file_name = build_kernel_for_riscv_options->ncrisc_kernel_file_name;
    fs::copy(kernel_file_name, kernel_dir + "/kernel.cpp", fs::copy_options::overwrite_existing);

    // copy unpack/pack data formats to the kernel dir
    if (fs::exists(out_dir_path + "/chlkc_unpack_data_format.h")) {
        fs::copy(out_dir_path + "/chlkc_unpack_data_format.h", kernel_dir, fs::copy_options::overwrite_existing);
        fs::copy(out_dir_path + "/chlkc_pack_data_format.h", kernel_dir, fs::copy_options::overwrite_existing);
    }

    log_debug(tt::LogBuildKernels, "    Make clean cmd: {}", make_clean_cmd.str());
    if (!tt::utils::run_command(make_clean_cmd.str(), log_file, false)) {
        log_fatal(tt::LogBuildKernels, " NCRISC clean failed -- cmd: {}", make_clean_cmd.str());
        exit(1);
    }
    log_debug(tt::LogBuildKernels, "    Make compile cmd: {}", make_cmd.str());
    if (!tt::utils::run_command(make_cmd.str(), log_file, false)) {
        log_fatal(tt::LogBuildKernels, " NCRISC Build failed -- cmd: {}", make_cmd.str());
        exit(1);
    }
    fs::current_path(old_cwd); // restore cwd
}


//////////////////
// TRISCs       //
//////////////////

// fwd declarations (these funcs are used only in this file)
void compile_ckernels_for_all_triscs(string, string root, string chlkc_src_dir, vector<uint32_t> kernel_compile_time_args);
void compile_ckernels_for_trisc(string chlkc_src_dir, string output_dir, string make_src_args, string make_args, uint32_t trisc_mailbox_addr, int thread_id, string kernel_file_name);
void generate_data_format_descriptors(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path);
void generate_loop_count_file(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, int epoch, string out_dir_path);
void generate_math_approx_mode_descriptor(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path);
void generate_math_fidelity_descriptor(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path);

void generate_binaries_for_triscs(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, const string &out_dir_path, const string& arch_name, bool skip_hlkc, bool parallel, std::vector<uint32_t> kernel_compile_time_args) {

    string root_dir = std::getenv("TT_METAL_HOME");

    log_info(tt::LogBuildKernels, "Compiling HLK to TRISCs");

    // currently assuming HLK is the same across all cores
    string hlk_file_name = build_kernel_for_riscv_options->hlk_desc.get_hlk_file_name();
    auto hlk_defines = build_kernel_for_riscv_options->hlk_defines;

    // we have a directory per op in the output directory
    string op_path = out_dir_path;
    fs::create_directories(op_path);

    log_debug(tt::LogBuildKernels, "build_kernel_for_riscv_options->fp32_dest_acc_en = {}", build_kernel_for_riscv_options->fp32_dest_acc_en);

    if (not skip_hlkc) {
        // hlkc_api
        hlk_file_name = compile_hlk(
            hlk_file_name,
            op_path,
            arch_name,
            hlk_defines,
            false, // is_perf_dump_en,
            false, // untilize
            false, // enable_cache,
            false,  // pack_microblocks -- not supported
            build_kernel_for_riscv_options->fp32_dest_acc_en,
            parallel);

        generate_data_format_descriptors(build_kernel_for_riscv_options, op_path);
        generate_math_fidelity_descriptor(build_kernel_for_riscv_options, op_path);
        generate_math_approx_mode_descriptor(build_kernel_for_riscv_options, op_path);
        generate_loop_count_file(build_kernel_for_riscv_options, 1 , op_path); // loop_count = 1
    }

    compile_ckernels_for_all_triscs(arch_name, root_dir, op_path, kernel_compile_time_args);
}

void generate_binaries_for_triscs_new(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, const string &out_dir_path, const string& arch_name, bool parallel, std::vector<uint32_t> kernel_compile_time_args) {

    string root_dir = std::getenv("TT_METAL_HOME");

    // Right now, to support compiling multiple threads, I am overloading
    // the concept of hlk filename to point to a directory with llks
    // Really need to refactor soon, however fastest change to check in
    // critical work
    string llk_directory_name = build_kernel_for_riscv_options->hlk_desc.get_hlk_file_name();

    auto hlk_defines = build_kernel_for_riscv_options->hlk_defines;

    // we have a directory per op in the output directory
    string op_path = out_dir_path;
    fs::create_directories(op_path);

    log_debug(tt::LogBuildKernels, "build_kernel_for_riscv_options->fp32_dest_acc_en = {}", build_kernel_for_riscv_options->fp32_dest_acc_en);

    // Copy llks into new folder
    fs::copy(llk_directory_name + "/chlkc_unpack.cpp", op_path + "/chlkc_unpack.cpp", fs::copy_options::overwrite_existing);
    fs::copy(llk_directory_name + "/chlkc_math.cpp", op_path + "/chlkc_math.cpp", fs::copy_options::overwrite_existing);
    fs::copy(llk_directory_name + "/chlkc_pack.cpp", op_path + "/chlkc_pack.cpp", fs::copy_options::overwrite_existing);

    generate_data_format_descriptors(build_kernel_for_riscv_options, op_path);
    generate_math_fidelity_descriptor(build_kernel_for_riscv_options, op_path);
    generate_math_approx_mode_descriptor(build_kernel_for_riscv_options, op_path);

    // For time being
    const string make_ckernels_compile_dir = root_dir + "/src/ckernels/" + arch_name + "/common";
    const string make_ckernels_link_dir = root_dir + "/src/ckernels";

    string make_src_args = "-C ";
    make_src_args += make_ckernels_compile_dir;
    make_src_args += " HLKC_KERNELS=1 ";
    make_src_args += " -j8 ";

    add_kernel_compile_time_args_to_make_cmd(make_src_args, kernel_compile_time_args);
    // TODO: commonize this with runtime_common.hpp?
    uint32_t TRISC_BASE = l1_mem::address_map::TRISC_BASE;
    uint32_t TRISC_L1_MAILBOX_OFFSET = l1_mem::address_map::TRISC_L1_MAILBOX_OFFSET;

    uint32_t trisc_sizes[3] = {
        l1_mem::address_map::TRISC0_SIZE, l1_mem::address_map::TRISC1_SIZE, l1_mem::address_map::TRISC2_SIZE};

    uint32_t trisc_mailbox_addresses[3] = {
        TRISC_BASE + TRISC_L1_MAILBOX_OFFSET,
        TRISC_BASE + trisc_sizes[0] + TRISC_L1_MAILBOX_OFFSET,
        TRISC_BASE + trisc_sizes[0] + trisc_sizes[1] + TRISC_L1_MAILBOX_OFFSET};

    string make_args = "-C ";
    make_args += make_ckernels_link_dir;
    // Add Trisc sizes to Make arg command
    make_args += " TRISC0_SIZE=" + to_string(trisc_sizes[0]) + " TRISC1_SIZE=" + to_string(trisc_sizes[1]) +
                 " TRISC2_SIZE=" + to_string(trisc_sizes[2]);
    make_args += " ARCH_NAME=" + arch_name;
    // make_args += " DEVICE_RUNNER=" + (model|versim|silicon);
    make_args += " TRISC_BASE=" + to_string(TRISC_BASE);

    string used_kernels[3] = {"chlkc_unpack", "chlkc_math", "chlkc_pack"};

    std::vector<std::thread> ths(3);
    for (int thread_id = 0; thread_id < 3; thread_id++) {
        stringstream ckernels_compile_output_dir;
        ckernels_compile_output_dir << op_path << "/tensix_thread" << (uint)thread_id;

        ths[thread_id] = std::thread(
            compile_ckernels_for_trisc,
            op_path,
            ckernels_compile_output_dir.str(),
            make_src_args,
            make_args,
            trisc_mailbox_addresses[thread_id],
            thread_id,
            used_kernels[thread_id]
            );
    }
    for (int thread_id = 0; thread_id < 3; thread_id++) {
        ths[thread_id].join();
    }
}

void compile_ckernels_for_all_triscs(string arch_name, string root, string chlkc_src_dir, vector<uint32_t> kernel_compile_time_args) {
    fs::remove("hlk_ckernels_compile.log");  // clean the log file

    const string make_ckernels_compile_dir = root + "/src/ckernels/" + arch_name + "/common";
    const string make_ckernels_link_dir = root + "/src/ckernels";

    string make_src_args = "-C ";
    make_src_args += make_ckernels_compile_dir;
    make_src_args += " HLKC_KERNELS=1 ";
    make_src_args += " -j8 ";

    add_kernel_compile_time_args_to_make_cmd(make_src_args, kernel_compile_time_args);


    // TODO: commonize this with runtime_common.hpp?
    uint32_t TRISC_BASE = l1_mem::address_map::TRISC_BASE;
    uint32_t TRISC_L1_MAILBOX_OFFSET = l1_mem::address_map::TRISC_L1_MAILBOX_OFFSET;

    uint32_t trisc_sizes[3] = {
        l1_mem::address_map::TRISC0_SIZE, l1_mem::address_map::TRISC1_SIZE, l1_mem::address_map::TRISC2_SIZE};

    uint32_t trisc_mailbox_addresses[3] = {
        TRISC_BASE + TRISC_L1_MAILBOX_OFFSET,
        TRISC_BASE + trisc_sizes[0] + TRISC_L1_MAILBOX_OFFSET,
        TRISC_BASE + trisc_sizes[0] + trisc_sizes[1] + TRISC_L1_MAILBOX_OFFSET};

    string make_args = "-C ";
    make_args += make_ckernels_link_dir;
    // Add Trisc sizes to Make arg command
    make_args += " TRISC0_SIZE=" + to_string(trisc_sizes[0]) + " TRISC1_SIZE=" + to_string(trisc_sizes[1]) +
                 " TRISC2_SIZE=" + to_string(trisc_sizes[2]);
    make_args += " ARCH_NAME=" + arch_name;
    // make_args += " DEVICE_RUNNER=" + (model|versim|silicon);
    make_args += " TRISC_BASE=" + to_string(TRISC_BASE);

    string used_kernels[3] = {"chlkc_unpack", "chlkc_math", "chlkc_pack"};

    std::vector<std::thread> ths(3);
    for (int thread_id = 0; thread_id < 3; thread_id++) {
        stringstream ckernels_compile_output_dir;
        ckernels_compile_output_dir << chlkc_src_dir << "/tensix_thread" << (uint)thread_id;

        ths[thread_id] = std::thread(
            compile_ckernels_for_trisc,
            chlkc_src_dir,
            ckernels_compile_output_dir.str(),
            make_src_args,
            make_args,
            trisc_mailbox_addresses[thread_id],
            thread_id,
            used_kernels[thread_id]
            );
    }
    for (int thread_id = 0; thread_id < 3; thread_id++) {
        ths[thread_id].join();
    }
}

void compile_ckernels_for_trisc(string chlkc_src_dir, string output_dir, string make_src_args, string make_args, uint32_t trisc_mailbox_addr, int thread_id, string kernel_file_name)
{
    stringstream make_cmd;
    stringstream make_gen_cmd;
    stringstream make_src_cmd;
    string full_output_dir = fs::absolute(output_dir).string();
    string common_output_dir = fs::absolute(output_dir).string();
    string log_file = chlkc_src_dir + "/" + "hlk_ckernels_compile_thread" + to_string(thread_id) + ".log";

    //bool is_perf_dump_en = perf_desc.device_perf_mode != perf::PerfDumpMode::Disable;
    //bool is_perf_spill_dram = perf_desc.device_perf_mode == perf::PerfDumpMode::IntermediateDump;
    bool is_perf_dump_en = false;
    bool is_perf_spill_dram = false;

    // Build ckernels/src
    make_src_cmd << " make " << make_src_args;
    make_src_cmd << " KERNELS='" << kernel_file_name << '\'';
    make_src_cmd << " PERF_DUMP=" << to_string(is_perf_dump_en);
    make_src_cmd << " INTERMED_DUMP=" << to_string(is_perf_spill_dram);
    //make_src_cmd << " PERF_DUMP_LEVEL=" << to_string(perf_desc.perf_dump_level);
    make_src_cmd << " PERF_DUMP_LEVEL=" << to_string(0);
    make_src_cmd << " MAILBOX_ADDR=" << trisc_mailbox_addr;
    make_src_cmd << " HLKC_INC=" << fs::absolute(chlkc_src_dir).string();
    make_src_cmd << " OUTPUT_DIR=" << common_output_dir;

    // std::cout is not thread-safe
    // cout << "    Make compile cmd: " << make_src_cmd.str() << "\n";
    log_debug(tt::LogBuildKernels, "    TRISC {} Make compile cmd: {}", thread_id, make_src_cmd.str());
    if (!tt::utils::run_command(make_src_cmd.str(), log_file, false)) {
        string err_msg = "Build ckernels/src failed for a thread " + to_string(thread_id) + " with CKernels '" + kernel_file_name + "'";
        throw std::runtime_error(err_msg);
    }

    // FIXME: This doesn't sound right --> For the HLKC generated kernel dont send the kernel list. inclusion is controlled through defines???
    string no_kernels = "";
    bool is_hlkc_kernel = true;

    make_cmd << " make " << make_args;
    make_cmd << " FIRMWARE_NAME=tensix_thread" << (uint32_t)thread_id;
    make_cmd << " KERNELS=''";
    make_cmd << " LINKER_SCRIPT_NAME=trisc" << (uint32_t)thread_id << ".ld";
    make_cmd << " TEST='chlkc'";
    make_cmd << " OUTPUT_DIR=" << full_output_dir;
    make_cmd << " CKERNELS_COMMON_OUT_DIR=" << common_output_dir;
    make_cmd << " CLEAN_OUTPUT_DIR=0";

    // std::cout is not thread-safe
    // cout << "    Make link cmd: " << make_cmd.str() << "\n";
    log_debug(tt::LogBuildKernels, "    TRISC {} Make link cmd: {}", thread_id, make_cmd.str());
    if (!tt::utils::run_command(make_cmd.str(), log_file, false)) {
        string err_msg = "Link ckernels/src failed for a thread " + to_string(thread_id) + " with CKernels '" + kernel_file_name + "'";
        throw std::runtime_error(err_msg);
    }
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
generate_pack_data_formats(tt_hlk_desc& desc, DataFormat unpack_conditional_dst_format, bool fp32_dest_acc_en) {
    vector<DataFormat> src_formats = tt::get_pack_src_formats(
        desc.input_buf_dataformat_arr, desc.param_buf_dataformat_arr, desc.intermediate_buf_dataformat_arr,
        desc.output_buf_dataformat_arr, unpack_conditional_dst_format, fp32_dest_acc_en);

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

void generate_data_format_descriptors(build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path) {
    string out_file_name_base = "chlkc_";
    string out_file_name_suffix = "_data_format.h";
    string unpack_data_format_descs = out_dir_path + "/" + out_file_name_base + "unpack" + out_file_name_suffix;
    string pack_data_format_descs = out_dir_path + "/" + out_file_name_base + "pack" + out_file_name_suffix;

    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = build_kernel_for_riscv_options->hlk_desc;

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

    if (tt::is_all_fp32_formats(desc.input_buf_dataformat_arr) && build_kernel_for_riscv_options->fp32_dest_acc_en){
        unpack_conditional_dst_format = DataFormat::Tf32;
    }

    tt::check_valid_in_out_data_formats(
        desc.input_buf_dataformat_arr,
        desc.output_buf_dataformat_arr,
        desc.param_buf_dataformat_arr,
        desc.intermediate_buf_dataformat_arr);

    vector<DataFormat> unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs;
    tie(unpack_src_formats_all_cbs, unpack_dst_formats_all_cbs) = generate_unpack_data_formats(desc, unpack_conditional_dst_format, build_kernel_for_riscv_options->fp32_dest_acc_en);

    vector<DataFormat> pack_src_formats_all_cbs, pack_dst_formats_all_cbs;
    tie(pack_src_formats_all_cbs, pack_dst_formats_all_cbs) = generate_pack_data_formats(desc, unpack_conditional_dst_format, build_kernel_for_riscv_options->fp32_dest_acc_en);

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

void generate_math_fidelity_descriptor(build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path) {
    string math_fidelity_descriptor = out_dir_path + "/" + "chlkc_math_fidelity.h";

    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = build_kernel_for_riscv_options->hlk_desc;

    ofstream file_stream;

    file_stream.open(math_fidelity_descriptor);
    file_stream << "constexpr std::int32_t MATH_FIDELITY = " << (int)desc.get_hlk_math_fidelity() << ";" << endl;
    file_stream.close();
}

void generate_math_approx_mode_descriptor(build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path) {
    string approx_descriptor = out_dir_path + "/" + "chlkc_math_approx_mode.h";

    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = build_kernel_for_riscv_options->hlk_desc;

    ofstream file_stream;

    file_stream.open(approx_descriptor);
    file_stream << "constexpr bool APPROX = " << std::boolalpha << desc.get_hlk_math_approx_mode() << ";" << endl;
    file_stream.close();
}

void generate_loop_count_file(build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, int input_count, string out_dir_path) {
    string loop_count_file = out_dir_path + "/" + "loop_count.h";

    ofstream file_stream;

    file_stream.open(loop_count_file);
    file_stream << "constexpr std::int32_t arg_loop_count = " << input_count << ";" << endl;
    file_stream.close();
}

void generate_binaries_all_riscs(
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, const std::string& out_dir_path, const std::string& arch_name,
    generate_binaries_params_t p)
{
    std::vector<std::thread*> threads;
    std::function<void()> lambdas[] = {
        [build_kernel_for_riscv_options, out_dir_path, arch_name, p] () { generate_binaries_for_triscs(build_kernel_for_riscv_options, out_dir_path, arch_name, p.skip_hlkc, p.parallel_hlk, p.compute_kernel_compile_time_args); },
        [build_kernel_for_riscv_options, out_dir_path, arch_name, p] () { generate_binary_for_ncrisc(build_kernel_for_riscv_options, out_dir_path, arch_name, p.nc_noc_index, p.nc_kernel_compile_time_args); },
        [build_kernel_for_riscv_options, out_dir_path, arch_name, p] () { generate_binary_for_brisc(build_kernel_for_riscv_options, out_dir_path, arch_name, p.br_noc_index, p.br_kernel_compile_time_args); },
    };

    bool flags[3] = { p.compile_trisc, p.compile_ncrisc, p.compile_brisc };

    // compile all 3 in parallel if requested, otherwise compile just NC,BR in parallel
    for (int j = 0; j < 3; j++) {
        if (flags[j]) {
            if (j == 0 && !p.parallel_trncbr)
                lambdas[0](); // launch trisc sequentially
            else
                threads.push_back( new std::thread(lambdas[j]) );
        }
    }

    for (auto th: threads) {
        th->join();
        delete th;
    }
}
