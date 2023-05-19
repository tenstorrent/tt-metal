
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <string>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "dev_mem_map.h"
#include "l1_address_map.h"
#include "hostdevcommon/common_runtime_address_map.h"

using namespace std;
using namespace tt;

std::string RISCID_to_string(RISCID id) {
    switch (id) {
        case NC: return "NC";
        case BR: return "BR";
        case TR0: return "TR0";
        case TR1: return "TR1";
        case TR2: return "TR2";
        default: TT_ASSERT(false);
    }
    return string();
}

struct CompileDefines {
    RISCID           hwthread               { RISCID::NC }; // 0=NC, 1=UNPACK, BR=4
    uint32_t         mailbox_addr           { 0 };
    uint32_t         perf_dump_level        { 0 };
    uint32_t         noc_index              { 0 };
    bool             firmware               { false };
    bool             profile_kernel         { false };
    vector<uint32_t> compile_time_args;
    string           kernel_inc;
    ARCH             arch                   { ARCH::GRAYSKULL };
    map<std::string, std::string> kernel_defines;

    bool is_trisc() const { return (hwthread >= RISCID::TR0) && (hwthread <= RISCID::TR2); }
    bool is_brisc() const { return hwthread == RISCID::BR; }
    bool is_ncrisc() const { return hwthread == RISCID::NC; }
};

struct TriscParams {
    // TODO: commonize this with runtime_common.hpp?
    uint32_t TRISC_BASE { MEM_TRISC0_BASE };
    uint32_t TRISC_L1_MAILBOX_OFFSET { MEM_TEST_MAILBOX_ADDRESS };
    uint32_t trisc_sizes[3] = { MEM_TRISC0_SIZE, MEM_TRISC1_SIZE, MEM_TRISC2_SIZE };
    uint32_t trisc_mailbox_addresses[3] = {
        TRISC_BASE + TRISC_L1_MAILBOX_OFFSET,
        TRISC_BASE + trisc_sizes[0] + TRISC_L1_MAILBOX_OFFSET,
        TRISC_BASE + trisc_sizes[0] + trisc_sizes[1] + TRISC_L1_MAILBOX_OFFSET};
    int32_t get_trisc_size(RISCID id) const {
        switch (id) {
            case RISCID::TR0:
            case RISCID::TR1:
            case RISCID::TR2: return trisc_sizes[id-RISCID::TR0];
            default:;
        }
        return 0;
    }
    int32_t get_mailbox_addr(RISCID id) const {
        switch (id) {
            case RISCID::TR0:
            case RISCID::TR1:
            case RISCID::TR2: return trisc_mailbox_addresses[id-RISCID::TR0];
            default:;
        }
        return 0;
    }
};

struct CompileContext {
    string home_;
    string gpp_           { "/tt_metal/src/ckernels/sfpi/compiler/bin/riscv32-unknown-elf-g++ " };
    string gcc_           { "/tt_metal/src/ckernels/sfpi/compiler/bin/riscv32-unknown-elf-gcc " }; // TODO(AP): this wasn't really necessary for assembler
    string objcopy_       { "/tt_metal/src/ckernels/sfpi/compiler/bin/riscv32-unknown-elf-objcopy " };
    string kernel_subdir_; // full path to kernel subdir

    CompileContext(const string& kernel_subdir) {
        home_ = std::getenv("TT_METAL_HOME");
        TT_ASSERT(home_.size() > 0);
        if (home_.back() != '/')
            home_.push_back('/');
        kernel_subdir_ = kernel_subdir; // example: "eltwise_binary_writer_unary_8bank_reader_dual_8bank/7036516950107541145"
        gpp_ = home_ + gpp_;
        gcc_ = home_ + gcc_;
        objcopy_ = home_ + objcopy_;
    }

    ~CompileContext() {
    }

    string generate_includes(const CompileDefines& defs) {
        vector<string> includes; // relative to home_
        vector<string> includes_abs; // absolute
        if (defs.is_trisc()) {
            // TODO(AP): allocating the vec every time is suboptimal
            includes = move(vector<string>({
                "",
                "tt_metal/src/ckernels/" + get_string_lowercase(defs.arch) + "/llk_lib/",
                "tt_metal/src/ckernels/" + get_string_lowercase(defs.arch) + "/common/inc",
                "tt_metal/src/ckernels/sfpi/include",
                "tt_metal/src/ckernels/sfpi/include/" + get_string_aliased_arch_lowercase(defs.arch),
                "tt_metal/src/firmware/riscv/common",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/noc",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch),
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/" + get_string_lowercase(defs.arch) + "_defines",
                "tt_metal",
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/noc", // TODO(AP): this looks wrong, but nonetheless matches the old makefile for TRISCS
                "build/src/ckernels/gen/out", // TODO(AP): same as above - point into build where there's no src/
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch),
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/" + get_string_lowercase(defs.arch) + "_defines",
            }));
            includes_abs.push_back(kernel_subdir_);
        } else if (defs.is_brisc()) {
            includes = move(vector<string>({
                "",
                "tt_metal/src/ckernels/" + get_string_lowercase(defs.arch) + "/common/inc",
                "tt_metal/src/ckernels/sfpi/include",
                "tt_metal/src/ckernels/sfpi/include/" + get_string_aliased_arch_lowercase(defs.arch),
                "tt_metal/src/firmware/riscv/common",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch),
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/" + get_string_lowercase(defs.arch) + "_defines",
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch),
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/" + get_string_lowercase(defs.arch) + "_defines",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/noc",
                "tt_metal/src/firmware/riscv/targets/ncrisc",
                "tt_metal"}
            ));
            includes_abs.push_back(kernel_subdir_ + "/brisc");
        } else {
            includes = move(vector<string>({
                "",
                "tt_metal/src/ckernels/" + get_string_lowercase(defs.arch) + "/common/inc",
                "tt_metal/src/ckernels/sfpi/include",
                "tt_metal/src/ckernels/sfpi/include/" + get_string_aliased_arch_lowercase(defs.arch),
                "tt_metal/src/firmware/riscv/common",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch),
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/" + get_string_lowercase(defs.arch) + "_defines",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/noc",
                "tt_metal/src/firmware/riscv/targets/ncrisc",
                "tt_metal"}
            ));
            includes_abs.push_back(kernel_subdir_ + "/ncrisc");
        }

        string result = "";
        for (auto s: includes)
            result += " -I" + home_ + s; // convert to absolute path
        for (auto s: includes_abs)
            result += " -I" + s; // already absolute path
        result += " ";

        // TODO(AP): this seems odd, Makefile had both relative iquote and absolute
        // in general this first pass conversion to c++ compiler driver just replicates the Makefile cmdlines verbatim
        // but it looks like a lot of the include paths in the original Makefile were intermixed in unintended ways
        vector<string> iquote_includes_abs;
        if (defs.is_brisc() || defs.is_ncrisc()) {
            iquote_includes_abs.push_back("/tt_metal/src/firmware/riscv/common/");
            iquote_includes_abs.push_back("/tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/");
            iquote_includes_abs.push_back("/tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/" + get_string_lowercase(defs.arch) + "_defines/");
            if (!defs.is_ncrisc()) // TODO(AP): cleanup, looks like some quirk of original Makefile
                iquote_includes_abs.push_back("/tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/noc/");
        }
        for (auto s: iquote_includes_abs)
            result += " -iquote " + home_ + s;

        vector<string> iquote_includes;
        if (defs.is_trisc()) {
            iquote_includes.push_back("tt_metal/src/");
            iquote_includes.push_back(".");
        } else if (defs.is_brisc() || defs.is_ncrisc()) {
            iquote_includes.push_back(".");
        }
        for (auto s: iquote_includes)
            result += " -iquote " + s;
        return result;
    }

    string generate_gpp_options(const CompileDefines& defs, bool is_asm) {
        string options_string =
            " -march=rv32i -mabi=ilp32 \"-m" + get_string_aliased_arch_lowercase(defs.arch) + "\" -MD -MP -flto -ffast-math -g -Wall -Werror";
        if (!is_asm) // TODO(AP): wasn't necessary to split for assembler
            options_string +=
                " -std=c++17 -Wno-unknown-pragmas -fno-use-cxa-atexit "
                " -Wno-error=multistatement-macros -Wno-error=parentheses "
                " -Wno-error=unused-but-set-variable -Wno-unused-variable -fno-exceptions ";
        string result = "";
        switch  (defs.hwthread) {
            case RISCID::NC:
                result += " -Os";
            break;
            case RISCID::BR:
                result += " -Os";
                result += " -fno-tree-loop-distribute-patterns";
            break;
            default:
                //result += " -finline-limit=1 --no-inline -fno-inline-functions -fno-inline-small-functions -fno-inline-functions-called-once ";
                result += " -O3";
            break;
        }
        result += options_string;
        return result;
    }

    string generate_defines(const CompileDefines& defs) {
        string result = "";
        string arch_define = "";
        switch (defs.arch) {
            case ARCH::GRAYSKULL:
                arch_define = " -DARCH_GRAYSKULL";
                break;
            case ARCH::WORMHOLE:
            case ARCH::WORMHOLE_B0:
                arch_define = " -DARCH_WORMHOLE";
                break;
            default:
                break;
        }
        switch (defs.hwthread) {
            case RISCID::NC:
                result += " -DCOMPILE_FOR_NCRISC ";
            break;
            case RISCID::TR0:
                result += " -DUCK_CHLKC_UNPACK ";
                result += " -DNAMESPACE=chlkc_unpack ";
                result += arch_define;
            break;
            case RISCID::TR1:
                result += " -DUCK_CHLKC_MATH ";
                result += " -DNAMESPACE=chlkc_math ";
                result += arch_define;
            break;
            case RISCID::TR2:
                result += " -DUCK_CHLKC_PACK ";
                result += " -DNAMESPACE=chlkc_pack ";
                result += arch_define;
            break;
            case RISCID::BR:
                result += " -DCOMPILE_FOR_BRISC ";
            break;
            default: break;
        }

        if (defs.is_ncrisc() or defs.is_brisc()) {
            for (const auto &[def, val]: defs.kernel_defines)
                result += " -D" + def + "=" + val + " ";
        }

        if (defs.mailbox_addr != 0)
            result += " -DMAILBOX_ADDR=" + to_string(defs.mailbox_addr);
        if (defs.perf_dump_level != 0 || defs.is_trisc()) // TODO(AP): double check
            result += " -DPERF_DUMP_LEVEL=" + to_string(defs.perf_dump_level);
        result += " -DTENSIX_FIRMWARE"; // TODO(AP): verify where firmware flag comes from
        if (defs.profile_kernel) {
            result += " -DPROFILE_KERNEL=1";
        }
        for (int j = 0; j < defs.compile_time_args.size(); j++)
            result += " -DKERNEL_COMPILE_TIME_ARG_" + to_string(j) + "=" + to_string(defs.compile_time_args[j]);
        if (!defs.is_trisc())
            result += " -DNOC_INDEX=" + to_string(defs.noc_index);
        if (defs.firmware)
            result += " -DTENSIX_FIRMWARE";
        result += " -DLOCAL_MEM_EN=0 ";
        return result;
    }

    string get_compile_cmd(const CompileDefines& defs, const string& hwthread_name, const string& obj_name, const string& cpp_name)
    {
        //"-c -o /home/andrei/git/gp.ai/built_kernels/"
        //ctx.kernel_name_has_str_ = eltwise_binary_writer_unary_8bank_reader_dual_8bank/7036516950107541145/
        //"tensix_thread1/ckernel.o"
        //src/ckernel.cc
        string gpp_str;
        bool is_asm = (cpp_name.find(".S") != std::string::npos);
        if (is_asm) // TODO(AP): wasn't necessary to split for assembler
            gpp_str = gcc_;
        else
            gpp_str = gpp_;

        gpp_str += generate_gpp_options(defs, is_asm);
        gpp_str += generate_includes(defs);
        gpp_str += generate_defines(defs);
        gpp_str += "-c -o " + kernel_subdir_ + hwthread_name + obj_name + " " + cpp_name;
        return gpp_str;
    }

    vector<string> get_verilog_cmd(const CompileDefines& defs, const string& thread_name, const string& elfname) {
        string hk = kernel_subdir_ + thread_name;
        string result = objcopy_ + " -O verilog " + hk + elfname + ".elf " + hk + elfname + ".hex.tmp";
        vector<string> results;
        results.push_back(result);
        result = string("python3 ") + home_ + "tt_metal/src/firmware/riscv/toolchain/hex8tohex32.py " + hk+elfname+".hex.tmp" + " " + hk+elfname + ".hex";
        results.push_back(result);
        return results;
    }

    vector<string> get_link_cmd(const CompileDefines& defs, const string& hwthread_name, const vector<string>& obj_names)
    {
        string linkopts = " -march=rv32i -mabi=ilp32 -m" + get_string_aliased_arch_lowercase(defs.arch) + " -flto -ffast-math -Wl,--gc-sections"
                          " -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -Wl,--defsym=__firmware_start=0 "
                          " -nostartfiles -g";
        if (defs.is_trisc()) {
            linkopts += " -fno-exceptions"; // TODO(AP): odd that this was not present for brisc in the Makefile
        } else if (defs.is_brisc()) {
            // TODO(AP): not on ncrisc, why?
            linkopts += " -fno-tree-loop-distribute-patterns";
        }

        if (getenv("TT_KERNEL_LINKER_MAP") != nullptr) {
            linkopts += " -Wl,-Map=" + kernel_subdir_ + hwthread_name + "linker.map";
        }

        string hk = string(" ") + kernel_subdir_;
        string link_str = gpp_;
        link_str += " -L" + home_ + "/tt_metal/src/firmware/riscv/toolchain ";
        link_str += linkopts;
        switch  (defs.hwthread) {
            case RISCID::NC:
            link_str += " -Os";
            link_str += " -T" + home_ + "build/src/firmware/riscv/targets/ncrisc/out/ncrisc.ld "; break;
            case RISCID::TR0:
            link_str += " -O3";
            link_str += " -T" + home_ + "build/src/ckernels/out/trisc0.ld "; break;
            case RISCID::TR1:
            link_str += " -O3";
            link_str += " -T" + home_ + "build/src/ckernels/out/trisc1.ld "; break;
            case RISCID::TR2:
            link_str += " -O3";
            link_str += " -T" + home_ + "build/src/ckernels/out/trisc2.ld "; break;
            default:
            TT_ASSERT(defs.hwthread == RISCID::BR);
            link_str += " -Os";
            link_str += " -T" + home_ + "build/src/firmware/riscv/targets/brisc/out/brisc.ld "; break;
        }
        for (auto oname: obj_names)
            link_str += hk + hwthread_name + oname;

        string elfname;
        switch (defs.hwthread) {
            case RISCID::NC:
                elfname = "ncrisc";
            break;
            case RISCID::BR:
                elfname = "brisc";
            break;
            case RISCID::TR0:
                elfname = "tensix_thread0";
            break;
            case RISCID::TR1:
                elfname = "tensix_thread1";
            break;
            case RISCID::TR2:
                elfname = "tensix_thread2";
            break;
            default: TT_ASSERT(false); break;
        }
        // add -o target.elf
        link_str += " -o " + hk + hwthread_name + elfname + ".elf";
        return vector<string>({link_str, elfname});
    }
};

void generate_binary_for_risc(
    RISCID risc_id,
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    const std::string &out_dir_path,
    const std::string& arch_name,
    const std::uint8_t noc_index,
    const std::vector<std::uint32_t>& kernel_compile_time_args,
    bool profile_kernel) {

    // default ARCH_NAME is grayskull in Makefile
    TT_ASSERT( (arch_name.compare("grayskull") == 0) || (arch_name.compare("wormhole") == 0) || (arch_name.compare("wormhole_b0") == 0) );

    log_trace(tt::LogBuildKernels, "Compiling RISCID={}", risc_id);

    // this is in the top-level outdir because "/ncrisc" is deleted by make clean, but we log the output of make clean as well
    string log_file = fs::absolute(out_dir_path).string() + "/risc_build_" + RISCID_to_string(risc_id) + ".log";
    utils::create_file(log_file);

    CompileContext ctx(out_dir_path);
    string kernel_dir = out_dir_path;
    string thread_bin_subdir;
    switch (risc_id) {
        case RISCID::NC: thread_bin_subdir = "/ncrisc/"; break;
        case RISCID::BR: thread_bin_subdir = "/brisc/"; break;
        case RISCID::TR0: thread_bin_subdir = "/tensix_thread0/"; break;
        case RISCID::TR1: thread_bin_subdir = "/tensix_thread1/"; break;
        case RISCID::TR2: thread_bin_subdir = "/tensix_thread2/"; break;
    }
    kernel_dir += "/";
    kernel_dir += thread_bin_subdir;
    fs::create_directories(kernel_dir);

    CompileDefines defs;
    defs.hwthread = risc_id;
    defs.arch = get_arch_from_string(arch_name);

    // Only modifying dataflow paths, we can make a separate
    // isuue for the compute paths
    if (defs.is_brisc()) {
        //cout << "BRISC NOC_INDEX=" << uint32_t(noc_index) << endl;
        defs.kernel_defines = build_kernel_for_riscv_options->brisc_defines;
    } else if (defs.is_ncrisc()) {
        //cout << "NCRISC NOC_INDEX=" << uint32_t(noc_index) << endl;
        defs.kernel_defines = build_kernel_for_riscv_options->ncrisc_defines;
    }

    defs.noc_index = noc_index;
    defs.profile_kernel = profile_kernel;
    defs.compile_time_args = kernel_compile_time_args;
    defs.kernel_inc = fs::absolute(kernel_dir).string();
    defs.mailbox_addr = TriscParams().get_mailbox_addr(risc_id);

    // copy the NCRISC/BRISC kernel to that directory, w/a generic filename kernel.cpp (this is what ncrisc.cc includes)
    // Note that for TRISCS this is not needed because they are currently generated in a previous pass and included
    // indirectly from ckernel_main.cc
    // ckernel_main.cc then includes "chlkc_list.h" which in turn includes one of previously generated cpps for each trisc thread
    string kernel_file_name;
    if (defs.is_ncrisc()) {
        kernel_file_name = build_kernel_for_riscv_options->ncrisc_kernel_file_name;
        fs::copy(ctx.home_+ kernel_file_name, kernel_dir + "/kernel.cpp", fs::copy_options::overwrite_existing);
    } else if (defs.is_brisc()) {
        kernel_file_name = build_kernel_for_riscv_options->brisc_kernel_file_name;
        fs::copy(ctx.home_+ kernel_file_name, kernel_dir + "/kernel.cpp", fs::copy_options::overwrite_existing);
    }

    // copy unpack/pack data formats to the kernel dir
    if (fs::exists(out_dir_path + "/chlkc_unpack_data_format.h")) {
        fs::copy(out_dir_path + "/chlkc_unpack_data_format.h", kernel_dir, fs::copy_options::overwrite_existing);
        fs::copy(out_dir_path + "/chlkc_pack_data_format.h", kernel_dir, fs::copy_options::overwrite_existing);
    }

    vector<string> bcpps = {"brisc.cc", "risc_common.cc", "tdma_xmov.c", "noc.c", "substitutes.cpp", "tmu-crt0.S"};
    vector<string> bobjs = {"brisc.o", "risc_common.o", "tdma_xmov.o", "noc.o", "substitutes.o",     "tmu-crt0.o"};
    vector<string> bobjl = {"brisc.o", "risc_common.o", "tdma_xmov.o", "noc.o", "substitutes.o",     "tmu-crt0.o"};
    vector<string> bcwds = {"",        "",              "",            "",      "",                  ""};

    vector<string> ncpps = {"ncrisc.cc", "context.cc", "risc_common.cc", "risc_chip_specific.c", "substitutes.cpp", "tmu-crt0.S"};
    vector<string> nobjs = {"ncrisc.o",  "context.o",  "risc_common.o",  "risc_chip_specific.o", "substitutes.o",   "tmu-crt0.o"};
    vector<string> nobjl = {"ncrisc.o",  "context.o",  "risc_common.o",  "risc_chip_specific.o", "substitutes.o",   "tmu-crt0.o"};
    vector<string> ncwds = {"",  "",  "", "",  "", ""};

    vector<string> tcpps = {"src/ckernel.cc", "src/ckernel_template.cc", "src/ckernel_main.cc", "substitutes.cpp", "tmu-crt0.S" };
    vector<string> tobjs = {"ckernel.o",      "ckernel_template.o",      "ckernel_main.o",      "substitutes.o",   "tmu-crt0.o" };
    // TODO(AP): reorder link objects
    vector<string> tobjl = {"ckernel_main.o", "substitutes.o",           "ckernel_template.o",  "ckernel.o",       "tmu-crt0.o" };
    vector<string> tcwds = {"",               "",                        "",                    "",                "" };

    vector<string> cpps, objs, cwds, objls;
    string compile_cwd;
    switch (risc_id) {
        case RISCID::NC:
            ncwds[0] = "tt_metal/src/firmware/riscv/targets/ncrisc";
            ncwds[2] = "tt_metal/src/firmware/riscv/common";
            ncwds[3] = "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "";
            ncwds[4] = "tt_metal/src/firmware/riscv/toolchain";
            cpps = move(ncpps); objs = move(nobjs); cwds = move(ncwds);
            objls = move(nobjl);
        break;
        case RISCID::BR:
            bcwds[0] = "tt_metal/src/firmware/riscv/targets/brisc";
            bcwds[1] = "tt_metal/src/firmware/riscv/common";
            bcwds[2] = "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "";
            bcwds[3] = "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(defs.arch) + "/noc";
            bcwds[4] = "tt_metal/src/firmware/riscv/toolchain";
            cpps = move(bcpps); objs = move(bobjs); cwds = move(bcwds);
            objls = move(bobjl);
        break;
        case RISCID::TR0:
        case RISCID::TR1:
        case RISCID::TR2:
            tcwds[0] = "tt_metal/src/ckernels/" + get_string_lowercase(defs.arch) + "/common";
            tcwds[3] = "tt_metal/src/firmware/riscv/toolchain"; // TODO(AP): refactor
            cpps = move(tcpps); objs = move(tobjs); cwds = move(tcwds);
            objls = move(tobjl);
        break;
    }

    string pushd_cmd;

    vector<thread> compile_threads;
    for (int i = 0; i < cpps.size(); i++) {
        if (cwds[i] != "")
            pushd_cmd = "cd " + ctx.home_ + cwds[i] + " && ";
        string gpp_cmd = pushd_cmd + ctx.get_compile_cmd(defs, thread_bin_subdir, objs[i], cpps[i]);
        auto lambda = [gpp_cmd, log_file, defs]() {
            log_debug(tt::LogBuildKernels, "    g++ compile cmd: {}", gpp_cmd);
            if (!tt::utils::run_command(gpp_cmd, log_file, false)) {
                log_fatal(tt::LogBuildKernels, "{}RISC Build failed -- cmd: {}", RISCID_to_string(defs.hwthread), gpp_cmd);
                exit(1);
            }
        };
        std::thread t(lambda);
        compile_threads.push_back(std::move(t));
    }
    for (auto& t: compile_threads) t.join();

    pushd_cmd = string("cd ") + ctx.home_ + "tt_metal/src/ckernels && "; // TODO(AP): Optimize

    vector<string> link = ctx.get_link_cmd(defs, thread_bin_subdir, objls);
    log_debug(tt::LogBuildKernels, "    g++ link cmd: {}", pushd_cmd + link[0]);
    if (!tt::utils::run_command(pushd_cmd + link[0], log_file, false)) {
        log_fatal(tt::LogBuildKernels, "{}RISC link failed -- cmd: {}", RISCID_to_string(defs.hwthread), link[0]);
        exit(1);
    }

    pushd_cmd = string("cd ") + ctx.kernel_subdir_ + thread_bin_subdir + " && "; // TODO(AP): Optimize
    auto verilogcmds = ctx.get_verilog_cmd(defs, thread_bin_subdir, link[1]);
    tt::utils::run_command(pushd_cmd + verilogcmds[0], log_file, false);
    tt::utils::run_command(pushd_cmd + verilogcmds[1], log_file, false);
}


//////////////////
// TRISCs       //
//////////////////
void generate_data_format_descriptors(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path);
void generate_math_approx_mode_descriptor(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path);
void generate_math_fidelity_descriptor(tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path);

void gen_trisc_cpp(const string& src_name, const string& dst_name, vector<string>& prolog)
{
    std::ofstream out(dst_name);
    for (auto s: prolog)
        out << s;
    out << "#include \"" << src_name << "\"\n";
}

std::string gen_trisc_cpps(
    string input_hlk_file_path,
    string out_dir_path,
    string device_name,
    const std::map<std::string, std::string>& defines,
    bool dump_perf_events,
    bool untilize_output,
    bool enable_cache,
    bool pack_microblocks,
    bool fp32_dest_acc_en
)
{
    string hlkc_path;
    string tt_metal_home;
    if (!getenv("TT_METAL_HOME")) {
        fs::path cwd = fs::current_path();
        tt_metal_home = cwd.string();
    } else {
        tt_metal_home = string(getenv("TT_METAL_HOME"));
    }

    string out_file_name_base = "chlkc";
    string unpack_base        = out_dir_path + "/" + out_file_name_base + "_unpack";
    string math_base          = out_dir_path + "/" + out_file_name_base   + "_math";
    string pack_base          = out_dir_path + "/" + out_file_name_base  + "_pack";
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

    std::thread t0( [&]() { gen_trisc_cpp(input_hlk_file_path, unpack_cpp, unpack_prolog); } );
    std::thread t1( [&]() { gen_trisc_cpp(input_hlk_file_path, math_cpp, math_prolog); } );
    std::thread t2( [&]() { gen_trisc_cpp(input_hlk_file_path, pack_cpp, pack_prolog); } );
    t0.join(); t1.join(); t2.join();

    string input_hlk_with_defines = input_hlk_file_path;
    {
        // Here we generate an auxiliary header with defines added via add_define() call
        // this header is then included from the kernel
        // We also append the include path to generated dir to hlkc cmldline.
        std::ofstream gen_defines_file;
        string generated_defines_fname = out_dir_path + "/defines_generated.h";
        gen_defines_file.open(generated_defines_fname, std::ios_base::out);

        for (auto it = defines.begin(); it != defines.end(); ++it) {
            gen_defines_file << "#define " << it->first << " " << it->second << endl;
        }

        // this string will be returned from the function to be reused in subsequent calls
        input_hlk_with_defines += " -I" + out_dir_path + " ";
        hlkc_path += " " + input_hlk_with_defines;
    }

    return input_hlk_with_defines;
}

void generate_src_for_triscs(
    tt::build_kernel_for_riscv_options_t* topts,
    const string &out_dir_path,
    const string& arch_name,
    std::vector<uint32_t> kernel_compile_time_args) {

    // Note: Dirs (and descriptors) must be created by a prior call of generate_descriptors()
    // So, this call currently doesn't work in isolation

    log_trace(tt::LogBuildKernels, "Generating defines for TRISCs");

    string hlk_file_name = topts->hlk_desc.get_hlk_file_name();
    auto hlk_defines = topts->hlk_defines;

    gen_trisc_cpps(
        hlk_file_name,
        out_dir_path,
        arch_name,
        hlk_defines,
        false, // is_perf_dump_en,
        false, // untilize
        false, // enable_cache,
        false,  // pack_microblocks -- not supported
        topts->fp32_dest_acc_en);
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

void generate_binaries_all_riscs(
    tt::build_kernel_for_riscv_options_t* opts, const std::string& out_dir_path, const std::string& arch_name,
    generate_binaries_params_t p, bool profile_kernel)
{
    generate_descriptors(opts, out_dir_path);

    std::vector<std::thread> threads;
    std::function<void()> lambdas[] = {
        [opts, out_dir_path, arch_name, p, profile_kernel] () {
            generate_binaries_for_triscs(
                opts, out_dir_path, arch_name, p.compute_kernel_compile_time_args, profile_kernel);
        },
        [opts, out_dir_path, arch_name, p, profile_kernel] () {
            generate_binary_for_ncrisc(
                opts, out_dir_path, arch_name, p.nc_noc_index, p.nc_kernel_compile_time_args, profile_kernel);
        },
        [opts, out_dir_path, arch_name, p, profile_kernel] () {
            generate_binary_for_brisc(
                opts, out_dir_path, arch_name, p.br_noc_index, p.br_kernel_compile_time_args, profile_kernel);
        },
    };

    bool flags[3] = { p.compile_trisc, p.compile_ncrisc, p.compile_brisc };

    // compile all 3 in parallel if requested, otherwise compile just NC,BR in parallel
    // TODO(AP): re-paralllelize
    for (int j = 0; j < 3; j++)
        if (flags[j])
            threads.push_back( thread(lambdas[j]) );

    for (auto& th: threads)
        th.join();
}

void generate_binaries_for_triscs(
    tt::build_kernel_for_riscv_options_t* topts,
    const std::string &dir,
    const std::string& arch_name,
    const std::vector<std::uint32_t>& kernel_compile_time_args,
    bool profile_kernel)
{

    generate_src_for_triscs(topts, dir, arch_name, kernel_compile_time_args);
    auto lambda0 = [=]() { generate_binary_for_risc(RISCID::TR0, topts, dir, arch_name, 0, kernel_compile_time_args, profile_kernel); };
    auto lambda1 = [=]() { generate_binary_for_risc(RISCID::TR1, topts, dir, arch_name, 0, kernel_compile_time_args, profile_kernel); };
    auto lambda2 = [=]() { generate_binary_for_risc(RISCID::TR2, topts, dir, arch_name, 0, kernel_compile_time_args, profile_kernel); };
    if (0) {
        lambda0();
        lambda1();
        lambda2();
    } else {
        auto t0 = std::thread(lambda0);
        auto t1 = std::thread(lambda1);
        auto t2 = std::thread(lambda2);
        t0.join(); t1.join(); t2.join();
    }
}

// TODO(AP): can move joins for these threads to happen later (to compiler launch)
void generate_descriptors(
    tt::build_kernel_for_riscv_options_t* opts, const std::string &op_dir)
{
    fs::create_directories(op_dir);

    std::thread td( [=]() { generate_data_format_descriptors(opts, op_dir); } );
    std::thread tm( [=]() { generate_math_fidelity_descriptor(opts, op_dir); } );
    std::thread ta( [=]() { generate_math_approx_mode_descriptor(opts, op_dir); } );
    td.join();
    tm.join();
    ta.join();
}

//! wormhole/wormhole_b0 are aliased for firmwares...
// TODO: (kk) remove these exceptions?
std::string get_string_aliased_arch_lowercase(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "grayskull"; break;
        case tt::ARCH::WORMHOLE: return "wormhole"; break;
        case tt::ARCH::WORMHOLE_B0: return "wormhole"; break;
        default: return "invalid"; break;
    }
}
