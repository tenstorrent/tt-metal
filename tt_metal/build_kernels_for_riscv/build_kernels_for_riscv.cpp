
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <string>

#include "build_kernels_for_riscv/build_kernels_for_riscv.hpp"
#include "dev_mem_map.h"
#include "hostdevcommon/common_runtime_address_map.h"
#include "tools/profiler/profiler_state.hpp"

using namespace std;
using namespace tt;

namespace tt::tt_metal {

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

struct CompileState {
    RISCID           hwthread               { RISCID::NC }; // 0=NC, 1=UNPACK, BR=4
    uint32_t         perf_dump_level        { 0 };
    uint32_t         noc_index              { 0 };
    bool             firmware               { false };
    bool             profile_kernel         { false };
    vector<uint32_t> compile_time_args;
    string           kernel_inc;
    ARCH             arch                   { ARCH::GRAYSKULL };
    map<std::string, std::string> kernel_defines;
    string home_;
    string gpp_           { "/tt_metal/src/ckernels/sfpi/compiler/bin/riscv32-unknown-elf-g++ " };
    string gcc_           { "/tt_metal/src/ckernels/sfpi/compiler/bin/riscv32-unknown-elf-gcc " }; // TODO(AP): this wasn't really necessary for assembler
    string objcopy_       { "/tt_metal/src/ckernels/sfpi/compiler/bin/riscv32-unknown-elf-objcopy " };
    int pcie_slot;
    string kernel_subdir_;
    string thread_bin_subdir;
    string log_file;
    bool is_fw_build_;

    CompileState(RISCID risc_id,
                 const string& in_kernel_subdir,
                 tt::build_kernel_for_riscv_options_t* build_opts) {

        is_fw_build_ = build_opts->fw_build_;
        home_ = tt::utils::get_root_dir();
        if (home_.back() != '/')
            home_.push_back('/');
        kernel_subdir_ = build_opts->outpath + in_kernel_subdir;
        pcie_slot = build_opts->pcie_slot;
        gpp_ = home_ + gpp_;
        gcc_ = home_ + gcc_;
        objcopy_ = home_ + objcopy_;

        // this is in the top-level outdir because "/ncrisc" is deleted by make clean, but we log the output of make clean as well
        log_file = fs::absolute(kernel_subdir_).string() + "/risc_build_" + RISCID_to_string(risc_id) + ".log";
        utils::create_file(log_file);

        switch (risc_id) {
            case RISCID::NC: thread_bin_subdir = "/ncrisc/"; break;
            case RISCID::BR: thread_bin_subdir = "/brisc/"; break;
            case RISCID::TR0: thread_bin_subdir = "/tensix_thread0/"; break;
            case RISCID::TR1: thread_bin_subdir = "/tensix_thread1/"; break;
            case RISCID::TR2: thread_bin_subdir = "/tensix_thread2/"; break;
        }
    }

    ~CompileState() {
    }

    bool is_trisc() const { return (hwthread >= RISCID::TR0) && (hwthread <= RISCID::TR2); }
    bool is_brisc() const { return hwthread == RISCID::BR; }
    bool is_ncrisc() const { return hwthread == RISCID::NC; }

    string generate_includes() const {
        vector<string> includes; // relative to home_
        vector<string> includes_abs; // absolute
        if (is_trisc()) {
            // TODO(AP): allocating the vec every time is suboptimal
            includes = move(vector<string>({
                "",
                "tt_metal/include",
                "tt_metal/src/ckernels/" + get_string_lowercase(arch) + "/llk_lib/",
                "tt_metal/src/ckernels/" + get_string_lowercase(arch) + "/common/inc",
                "tt_metal/src/ckernels/sfpi/include",
                "tt_metal/src/ckernels/sfpi/include/" + get_string_aliased_arch_lowercase(arch),
                "tt_metal/src/firmware/riscv/common",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/noc",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch),
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/" + get_string_lowercase(arch) + "_defines",
                "tt_metal",
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/noc", // TODO(AP): this looks wrong, but nonetheless matches the old makefile for TRISCS
                "build/src/ckernels/gen/out", // TODO(AP): same as above - point into build where there's no src/
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch),
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/" + get_string_lowercase(arch) + "_defines",
            }));
            includes_abs.push_back(kernel_subdir_);
        } else if (is_brisc()) {
            includes = move(vector<string>({
                "",
                "tt_metal/include",
                "tt_metal/src/ckernels/" + get_string_lowercase(arch) + "/common/inc",
                "tt_metal/src/ckernels/sfpi/include",
                "tt_metal/src/ckernels/sfpi/include/" + get_string_aliased_arch_lowercase(arch),
                "tt_metal/src/firmware/riscv/common",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch),
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/" + get_string_lowercase(arch) + "_defines",
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch),
                "build/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/" + get_string_lowercase(arch) + "_defines",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/noc",
                "tt_metal/src/firmware/riscv/targets/brisc",
                "tt_metal"}
            ));
            includes_abs.push_back(kernel_subdir_ + "/brisc");
        } else {
            includes = move(vector<string>({
                "",
                "tt_metal/include",
                "tt_metal/src/ckernels/" + get_string_lowercase(arch) + "/common/inc",
                "tt_metal/src/ckernels/sfpi/include",
                "tt_metal/src/ckernels/sfpi/include/" + get_string_aliased_arch_lowercase(arch),
                "tt_metal/src/firmware/riscv/common",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch),
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/" + get_string_lowercase(arch) + "_defines",
                "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/noc",
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
        if (is_brisc() || is_ncrisc()) {
            iquote_includes_abs.push_back("/tt_metal/src/firmware/riscv/common/");
            iquote_includes_abs.push_back("/tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/");
            iquote_includes_abs.push_back("/tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/" + get_string_lowercase(arch) + "_defines/");
            if (!is_ncrisc()) // TODO(AP): cleanup, looks like some quirk of original Makefile
                iquote_includes_abs.push_back("/tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(arch) + "/noc/");
        }
        for (auto s: iquote_includes_abs)
            result += " -iquote " + home_ + s;

        vector<string> iquote_includes;
        if (is_trisc()) {
            iquote_includes.push_back("tt_metal/src/");
            iquote_includes.push_back(".");
        } else if (is_brisc() || is_ncrisc()) {
            iquote_includes.push_back(".");
        }
        for (auto s: iquote_includes)
            result += " -iquote " + s;
        return result;
    }

    string generate_gpp_options(bool is_asm) const {

        string options_string;

        if (arch == tt::ARCH::GRAYSKULL) {
            options_string = " -mgrayskull -march=rv32iy -mtune=rvtt-b1 ";
        } else if (arch == tt::ARCH::WORMHOLE_B0) {
            options_string = " -mwormhole -march=rv32imw -mtune=rvtt-b1 ";
        } else {
            TT_ASSERT(false, "Invalid arch");
        }

        options_string +=
            "-mabi=ilp32 \"-m" + get_string_aliased_arch_lowercase(arch) + "\" -MD -MP -flto -ffast-math -g -Wall -Werror";
        if (!is_asm) // TODO(AP): wasn't necessary to split for assembler
            options_string +=
                " -std=c++17 -Wno-unknown-pragmas -fno-use-cxa-atexit "
                " -Wno-error=multistatement-macros -Wno-error=parentheses "
                " -Wno-error=unused-but-set-variable -Wno-unused-variable -fno-exceptions ";
        string result = "";
        switch  (hwthread) {
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

    string generate_defines() const {
        string result = "";
        string arch_define = "";
        switch (arch) {
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
        switch (hwthread) {
            case RISCID::NC:
                result += " -DCOMPILE_FOR_NCRISC ";
            break;
            case RISCID::TR0:
                result += " -DUCK_CHLKC_UNPACK ";
                result += " -DNAMESPACE=chlkc_unpack ";
                result += " -DCOMPILE_FOR_TRISC=0 ";
                result += arch_define;
            break;
            case RISCID::TR1:
                result += " -DUCK_CHLKC_MATH ";
                result += " -DNAMESPACE=chlkc_math ";
                result += " -DCOMPILE_FOR_TRISC=1 ";
                result += arch_define;
            break;
            case RISCID::TR2:
                result += " -DUCK_CHLKC_PACK ";
                result += " -DNAMESPACE=chlkc_pack ";
                result += " -DCOMPILE_FOR_TRISC=2 ";
                result += arch_define;
            break;
            case RISCID::BR:
                result += " -DCOMPILE_FOR_BRISC ";
            break;
            default: break;
        }

        if (is_ncrisc() or is_brisc()) {
            for (const auto &[def, val]: kernel_defines)
                result += " -D" + def + "=" + val + " ";
        }

        if (perf_dump_level != 0 || is_trisc()) // TODO(AP): double check
            result += " -DPERF_DUMP_LEVEL=" + to_string(perf_dump_level);
        result += " -DTENSIX_FIRMWARE"; // TODO(AP): verify where firmware flag comes from
        if (profile_kernel) {
            result += " -DPROFILE_KERNEL=1";
        }
        for (int j = 0; j < compile_time_args.size(); j++)
            result += " -DKERNEL_COMPILE_TIME_ARG_" + to_string(j) + "=" + to_string(compile_time_args[j]);
        if (!is_trisc())
            result += " -DNOC_INDEX=" + to_string(noc_index);
        if (firmware)
            result += " -DTENSIX_FIRMWARE";
        result += " -DLOCAL_MEM_EN=0 ";
        return result;
    }

    string get_compile_cmd(const string& hwthread_name, const string& obj_name, const string& cpp_name) const
    {
        string gpp_str;
        bool is_asm = (cpp_name.find(".S") != std::string::npos);
        if (is_asm) // TODO(AP): wasn't necessary to split for assembler
            gpp_str = gcc_;
        else
            gpp_str = gpp_;

        gpp_str += generate_gpp_options(is_asm);
        gpp_str += generate_includes();
        gpp_str += generate_defines();
        gpp_str += "-c -o " + kernel_subdir_ + hwthread_name + obj_name + " " + cpp_name;
        return gpp_str;
    }

    vector<string> get_verilog_cmd(const string& elfname) const {
        string hk = kernel_subdir_ + thread_bin_subdir;
        string result = objcopy_ + " -O verilog " + hk + elfname + ".elf " + hk + elfname + ".hex.tmp";
        vector<string> results;
        results.push_back(result);
        result = string("python3 ") + home_ + "tt_metal/src/firmware/riscv/toolchain/hex8tohex32.py " + hk+elfname+".hex.tmp" + " " + hk+elfname + ".hex";
        results.push_back(result);
        return results;
    }

    string get_weaken_cmd(const string& elfname) const {
        // Given this elf (A) and a later elf (B)
        // Weakens symbols in A so that it can be used as a "library" for B.
        // B imports A's weakened symbols, B's symbols of the same name don't
        // result in duplicate symbols but B can reference A's symbols.
        // Force the fw_export symbols to remain strong so to propogate link
        // addresses
        string hk = kernel_subdir_ + thread_bin_subdir;
        return objcopy_ +
            " --wildcard --weaken-symbol \"*\" --weaken-symbol \"!__fw_export_*\" " +
            hk + elfname + ".elf " + hk + elfname + "_weakened.elf";
    }

    vector<string> get_link_cmd(const vector<string>& obj_names) const
    {
        string linkopts;

        if (arch == tt::ARCH::GRAYSKULL) {
            linkopts = " -mgrayskull -march=rv32iy -mtune=rvtt-b1 ";
        } else if (arch == tt::ARCH::WORMHOLE_B0) {
            linkopts = " -mwormhole -march=rv32imw -mtune=rvtt-b1 ";
        } else {
            TT_ASSERT(false, "Invalid arch");
        }

        linkopts += "-mabi=ilp32 -m" + get_string_aliased_arch_lowercase(arch) + " -flto -ffast-math"
                          " -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 "
                          " -nostartfiles -g";

        string elfname;
        switch (hwthread) {
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

        if (is_trisc()) {
            linkopts += " -fno-exceptions"; // TODO(AP): odd that this was not present for brisc in the Makefile
        } else if (is_brisc()) {
            // TODO(AP): not on ncrisc, why?
            linkopts += " -fno-tree-loop-distribute-patterns";
        }
        if (!is_fw_build_) {
            string weakened_elf_name = tt::get_firmware_compile_outpath(pcie_slot) + elfname + "/" + elfname + "_weakened.elf";
            if (!fs::exists(weakened_elf_name)) {
                log_fatal(tt::LogBuildKernels, "File {} does not exist, link failed\n", weakened_elf_name);
            }
            linkopts += " -Xlinker \"--just-symbols=" + weakened_elf_name + "\"";
        }

        if (getenv("TT_KERNEL_LINKER_MAP") != nullptr) {
            linkopts += " -Wl,-Map=" + kernel_subdir_ + thread_bin_subdir + "linker.map";
        }

        string hk = string(" ") + kernel_subdir_;
        string link_str = gpp_;
        link_str += " -L" + home_ + "/tt_metal/src/firmware/riscv/toolchain ";
        link_str += linkopts;
        switch  (hwthread) {
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
            TT_ASSERT(hwthread == RISCID::BR);
            link_str += " -Os";
            link_str += " -T" + home_ + "build/src/firmware/riscv/targets/brisc/out/brisc.ld "; break;
        }
        for (auto oname: obj_names)
            link_str += hk + thread_bin_subdir + oname;

        // add -o target.elf
        link_str += " -o " + hk + thread_bin_subdir + elfname + ".elf";
        return vector<string>({link_str, elfname});
    }
};

static CompileState pre_compile_for_risc(
    RISCID risc_id,
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    const std::string &out_dir_path,
    const std::string& arch_name,
    const std::uint8_t noc_index,
    const std::vector<std::uint32_t>& kernel_compile_time_args)
{
    ZoneScoped;

    // default ARCH_NAME is grayskull in Makefile
    TT_ASSERT( (arch_name.compare("grayskull") == 0) || (arch_name.compare("wormhole") == 0) || (arch_name.compare("wormhole_b0") == 0) );

    log_trace(tt::LogBuildKernels, "Compiling RISCID={}", risc_id);

    CompileState ctx(risc_id, out_dir_path, build_kernel_for_riscv_options);
    string kernel_dir = ctx.kernel_subdir_ + "/" + ctx.thread_bin_subdir;
    fs::create_directories(kernel_dir);
    ctx.hwthread = risc_id;
    ctx.arch = get_arch_from_string(arch_name);

    // Only modifying dataflow paths, we can make a separate
    // isuue for the compute paths
    if (ctx.is_brisc()) {
        //cout << "BRISC NOC_INDEX=" << uint32_t(noc_index) << endl;
        ctx.kernel_defines = build_kernel_for_riscv_options->brisc_defines;
    } else if (ctx.is_ncrisc()) {
        //cout << "NCRISC NOC_INDEX=" << uint32_t(noc_index) << endl;
        ctx.kernel_defines = build_kernel_for_riscv_options->ncrisc_defines;
    }

    ctx.is_fw_build_ = build_kernel_for_riscv_options->fw_build_;
    ctx.noc_index = noc_index;
    ctx.profile_kernel = tt::tt_metal::getDeviceProfilerState();
    ctx.compile_time_args = kernel_compile_time_args;
    ctx.kernel_inc = fs::absolute(kernel_dir).string();

    // copy the NCRISC/BRISC kernel to that directory, w/a generic filename kernel.cpp (this is what ncrisc.cc includes)
    // Note that for TRISCS this is not needed because they are currently generated in a previous pass and included
    // indirectly from ckernel_main.cc
    // ckernel_main.cc then includes "chlkc_list.h" which in turn includes one of previously generated cpps for each trisc thread
    string kernel_file_name;
    if (!ctx.is_fw_build_) {
        if (ctx.is_ncrisc()) {
            kernel_file_name = build_kernel_for_riscv_options->ncrisc_kernel_file_name;
            fs::copy(ctx.home_+ kernel_file_name, kernel_dir + "/kernel.cpp", fs::copy_options::overwrite_existing);
        } else if (ctx.is_brisc()) {
            kernel_file_name = build_kernel_for_riscv_options->brisc_kernel_file_name;
            fs::copy(ctx.home_+ kernel_file_name, kernel_dir + "/kernel.cpp", fs::copy_options::overwrite_existing);
        }

        // copy unpack/pack data formats to the kernel dir
        if (fs::exists(ctx.kernel_subdir_ + "/chlkc_unpack_data_format.h")) {
            fs::copy(ctx.kernel_subdir_ + "/chlkc_unpack_data_format.h", kernel_dir, fs::copy_options::overwrite_existing);
            fs::copy(ctx.kernel_subdir_ + "/chlkc_pack_data_format.h", kernel_dir, fs::copy_options::overwrite_existing);
        }
    }

    return ctx;
}

static void compile_for_risc(
    RISCID risc_id,
    tt::build_kernel_for_riscv_options_t* build_opts,
    const CompileState& ctx) {
    ZoneScoped;

    struct build_files_t {
        const vector<string> cpps, objs;
    };
    static const build_files_t bf[3][2] = {
        {   // ncrisc
            {   // kernel
                {"ncrisck.cc", "risc_common.cc", "risc_chip_specific.c", "substitutes.cpp", "tmu-crt0k.S"},
                {"ncrisck.o",  "risc_common.o",  "risc_chip_specific.o", "substitutes.o",   "tmu-crt0k.o"},
            },
            {   // firmware
                {"ncrisc.cc", "risc_common.cc", "risc_chip_specific.c", "substitutes.cpp", "tmu-crt0.S"},
                {"ncrisc.o",  "risc_common.o",  "risc_chip_specific.o", "substitutes.o",   "tmu-crt0.o"},
            },
        },
        {   // trisc
            {   // kernel
                {"src/ckernel_template.cc", "src/ckernel_main.cc", "substitutes.cpp", "tmu-crt0k.S" },
                {"ckernel_template.o",      "ckernel_main.o",      "substitutes.o",   "tmu-crt0k.o" },
            },
            {   // firmware
                {"src/ckernel.cc", "substitutes.cpp", "tmu-crt0.S" },
                {"ckernel.o",      "substitutes.o",   "tmu-crt0.o" },
            },
        },
        {   // brisc
            {   // kernel
                {"brisck.cc", "risc_common.cc", "tdma_xmov.c", "noc.c", "substitutes.cpp", "tmu-crt0k.S"},
                {"brisck.o", "risc_common.o", "tdma_xmov.o", "noc.o", "substitutes.o",     "tmu-crt0k.o"},
            },
            {   // firmware
                {"brisc.cc", "risc_common.cc", "tdma_xmov.c", "noc.c", "substitutes.cpp", "tmu-crt0.S"},
                {"brisc.o", "risc_common.o", "tdma_xmov.o", "noc.o", "substitutes.o",     "tmu-crt0.o"},
            },
        },
    };

    // TODO(pgk): have the build system copy these files into a build dir w/
    // arch at the top level below root, move below into table above
    vector<string> cwds;
    int risc_type = 0;
    switch (risc_id) {
        case RISCID::NC:
            cwds.resize(5);
            cwds[0] = "tt_metal/src/firmware/riscv/targets/ncrisc";
            cwds[1] = "tt_metal/src/firmware/riscv/common";
            cwds[2] = "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(ctx.arch) + "";
            cwds[3] = "tt_metal/src/firmware/riscv/toolchain";
            risc_type = 0;
        break;
        case RISCID::BR:
            cwds.resize(6);
            cwds[0] = "tt_metal/src/firmware/riscv/targets/brisc";
            cwds[1] = "tt_metal/src/firmware/riscv/common";
            cwds[2] = "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(ctx.arch) + "";
            cwds[3] = "tt_metal/src/firmware/riscv/" + get_string_aliased_arch_lowercase(ctx.arch) + "/noc";
            cwds[4] = "tt_metal/src/firmware/riscv/toolchain";
            risc_type = 2;
        break;
        case RISCID::TR0:
        case RISCID::TR1:
        case RISCID::TR2:
            if (build_opts->fw_build_) {
                cwds.resize(3);
                cwds[0] = "tt_metal/src/ckernels/" + get_string_lowercase(ctx.arch) + "/common";
                cwds[1] = "tt_metal/src/firmware/riscv/toolchain"; // TODO(AP): refactor
            } else {
                cwds.resize(4);
                cwds[0] = "tt_metal/src/ckernels/" + get_string_lowercase(ctx.arch) + "/common";
                cwds[2] = "tt_metal/src/firmware/riscv/toolchain"; // TODO(AP): refactor
            }
            risc_type = 1;
        break;
    }

    const vector<string> &cpps = bf[risc_type][build_opts->fw_build_].cpps;
    const vector<string> &objs = bf[risc_type][build_opts->fw_build_].objs;

    string pushd_cmd;

    vector<thread> compile_threads;
    for (int i = 0; i < cpps.size(); i++) {
        if (cwds[i] != "")
            pushd_cmd = "cd " + ctx.home_ + cwds[i] + " && ";
        string gpp_cmd = pushd_cmd + ctx.get_compile_cmd(ctx.thread_bin_subdir, objs[i], cpps[i]);
        auto lambda = [gpp_cmd, ctx]() {
            log_debug(tt::LogBuildKernels, "    g++ compile cmd: {}", gpp_cmd);
            if (!tt::utils::run_command(gpp_cmd, ctx.log_file, false)) {
                log_fatal(tt::LogBuildKernels, "{}RISC Build failed -- cmd: {}", RISCID_to_string(ctx.hwthread), gpp_cmd);
                exit(1);
            }
        };
        std::thread t(lambda);
        compile_threads.push_back(std::move(t));
    }
    for (auto& t: compile_threads) t.join();
}

void link_for_risc(RISCID risc_id,
                   tt::build_kernel_for_riscv_options_t* build_opts,
                   const CompileState& ctx) {
    ZoneScoped;

    string pushd_cmd = string("cd ") + ctx.home_ + "tt_metal/src/ckernels && "; // TODO(AP): Optimize

    vector<string> bobjl, nobjl, tobjl;

    if (!build_opts->fw_build_) {
        bobjl = {"brisck.o", "risc_common.o", "tdma_xmov.o", "noc.o", "substitutes.o", "tmu-crt0k.o"};
        nobjl = {"ncrisck.o", "risc_common.o", "risc_chip_specific.o", "substitutes.o", "tmu-crt0k.o"};
        tobjl = {"ckernel_main.o", "substitutes.o", "ckernel_template.o", "tmu-crt0k.o" };
    } else {
        bobjl = {"brisc.o", "risc_common.o", "tdma_xmov.o", "noc.o", "substitutes.o", "tmu-crt0.o"};
        nobjl = {"ncrisc.o", "risc_common.o", "risc_chip_specific.o", "substitutes.o", "tmu-crt0.o"};
        tobjl = {"substitutes.o", "ckernel.o", "tmu-crt0.o" };
    }

    vector<string> objls;
    switch (risc_id) {
        case RISCID::NC:
            objls = move(nobjl);
        break;
        case RISCID::BR:
            objls = move(bobjl);
        break;
        case RISCID::TR0:
        case RISCID::TR1:
        case RISCID::TR2:
            objls = move(tobjl);
        break;
    }

    vector<string> link = ctx.get_link_cmd(objls);
    log_debug(tt::LogBuildKernels, "    g++ link cmd: {}", pushd_cmd + link[0]);
    if (!tt::utils::run_command(pushd_cmd + link[0], ctx.log_file, false)) {
        log_fatal(tt::LogBuildKernels, "{}RISC link failed -- cmd: {}", RISCID_to_string(ctx.hwthread), link[0]);
        exit(1);
    }

    pushd_cmd = string("cd ") + ctx.kernel_subdir_ + ctx.thread_bin_subdir + " && "; // TODO(AP): Optimize
    auto verilogcmds = ctx.get_verilog_cmd(link[1]);
    tt::utils::run_command(pushd_cmd + verilogcmds[0], ctx.log_file, false);
    tt::utils::run_command(pushd_cmd + verilogcmds[1], ctx.log_file, false);

    string weaken_cmd = ctx.get_weaken_cmd(link[1]);
    log_debug(tt::LogBuildKernels, "    objcopy cmd: {}", weaken_cmd);
    if (!tt::utils::run_command(weaken_cmd, ctx.log_file, false)) {
        log_fatal(tt::LogBuildKernels, "{}RISC objcopy failed -- cmd: {}", RISCID_to_string(ctx.hwthread), weaken_cmd);
        exit(1);
    }
}

void generate_binary_for_risc(RISCID risc_id,
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    const std::string &out_dir_path,
    const std::string& arch_name,
    const std::uint8_t noc_index,
    const std::vector<std::uint32_t>& kernel_compile_time_args)
{

    CompileState state = pre_compile_for_risc(
        risc_id,
        build_kernel_for_riscv_options,
        out_dir_path,
        arch_name,
        noc_index,
        kernel_compile_time_args);

    compile_for_risc(risc_id, build_kernel_for_riscv_options, state);

    link_for_risc(risc_id, build_kernel_for_riscv_options, state);
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
    tt::build_kernel_for_riscv_options_t* build_opts,
    string out_kernel_path,
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

    string out_dir_path = build_opts->outpath + out_kernel_path + "/";
    string out_file_name_base = "chlkc";
    string unpack_base        = out_dir_path + out_file_name_base + "_unpack";
    string math_base          = out_dir_path + out_file_name_base + "_math";
    string pack_base          = out_dir_path + out_file_name_base + "_pack";
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
        topts,
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
    string unpack_data_format_descs = build_kernel_for_riscv_options->outpath + out_dir_path + "/" + out_file_name_base + "unpack" + out_file_name_suffix;
    string pack_data_format_descs = build_kernel_for_riscv_options->outpath + out_dir_path + "/" + out_file_name_base + "pack" + out_file_name_suffix;

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
    string math_fidelity_descriptor = build_kernel_for_riscv_options->outpath + out_dir_path + "/" + "chlkc_math_fidelity.h";
    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = build_kernel_for_riscv_options->hlk_desc;

    ofstream file_stream;

    file_stream.open(math_fidelity_descriptor);
    file_stream << "constexpr std::int32_t MATH_FIDELITY = " << (int)desc.get_hlk_math_fidelity() << ";" << endl;
    file_stream.close();
}

void generate_math_approx_mode_descriptor(build_kernel_for_riscv_options_t* build_kernel_for_riscv_options, string out_dir_path) {
    string approx_descriptor = build_kernel_for_riscv_options->outpath + out_dir_path + "/" + "chlkc_math_approx_mode.h";

    // assuming all cores within a op have the same desc
    tt_hlk_desc& desc = build_kernel_for_riscv_options->hlk_desc;

    ofstream file_stream;

    file_stream.open(approx_descriptor);
    file_stream << "constexpr bool APPROX = " << std::boolalpha << desc.get_hlk_math_approx_mode() << ";" << endl;
    file_stream.close();
}

std::string generate_bank_to_noc_coord_descriptor_string(
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<i32>& dram_bank_offset_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<i32>& l1_bank_offset_map
) {
    stringstream ss;
    bool is_dram_pow2 = ceil(log2(dram_bank_map.size())) == log2(dram_bank_map.size());
    bool is_l1_pow2 = ceil(log2(l1_bank_map.size())) == log2(l1_bank_map.size());
    ss << "#pragma once" << endl;
    ss << "/*" << endl;
    ss << " * This file is autogenerated by tt-metal API" << endl;
    ss << " * This file contains values that are visible to both host and device compiled code." << endl;
    ss << " */" << endl;
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
    ss << "// This is to store offsets for any banks that share a noc node (dram on wh) or core (storage core), so we can view all banks similarly using banks" << endl;
    ss << "// for cores with only 1 bank, this is set to 0 and ignored" << endl;
    ss << "int32_t bank_to_l1_offset[NUM_L1_BANKS];" << endl;
    ss << "int32_t bank_to_dram_offset[NUM_DRAM_BANKS];" << endl;
    ss << endl;
    ss << "//! Assumes consumer of the bank coords are always sequential" << endl;
    ss << "inline void init_dram_bank_coords(uint8_t dram_bank_to_noc_x[], uint8_t dram_bank_to_noc_y[]) {" << endl;
    for (unsigned int bank_id = 0; bank_id < dram_bank_map.size(); bank_id++) {
        ss << "    dram_bank_to_noc_x[" << bank_id << "] = " << dram_bank_map[bank_id].x << ";" << endl;
        ss << "    dram_bank_to_noc_y[" << bank_id << "] = " << dram_bank_map[bank_id].y << ";" << endl;
        ss << "    bank_to_dram_offset[" << bank_id << "] = " << dram_bank_offset_map[bank_id]<< ";" << endl;
    }
    ss << "}" << endl;
    ss << "//! Assumes consumer of the bank coords are always sequential" << endl;
    ss << "inline void init_l1_bank_coords(uint8_t l1_bank_to_noc_x[], uint8_t l1_bank_to_noc_y[], int32_t bank_to_l1_offset[]) {" << endl;
    for (unsigned int bank_id = 0; bank_id < l1_bank_map.size(); bank_id++) {
        ss << "    l1_bank_to_noc_x[" << bank_id << "] = " << l1_bank_map[bank_id].x << ";" << endl;
        ss << "    l1_bank_to_noc_y[" << bank_id << "] = " << l1_bank_map[bank_id].y << ";" << endl;
        ss << "    bank_to_l1_offset[" << bank_id << "] = " << l1_bank_offset_map[bank_id]<< ";" << endl;
    }
    ss << "}" << endl;
    return ss.str();
}
void generate_bank_to_noc_coord_descriptor(
    build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    string out_dir_path,
    std::vector<CoreCoord>& dram_bank_map,
    std::vector<i32>& dram_bank_offset_map,
    std::vector<CoreCoord>& l1_bank_map,
    std::vector<i32>& l1_bank_offset_map
) {
    string output_string = generate_bank_to_noc_coord_descriptor_string(dram_bank_map, dram_bank_offset_map, l1_bank_map, l1_bank_offset_map);

    string full_path = build_kernel_for_riscv_options->outpath;
    full_path += out_dir_path;
    fs::create_directories(full_path + "/brisc");
    ofstream file_stream_br(full_path + "/brisc/generated_bank_to_noc_coord_mapping.h");
    file_stream_br << output_string;
    file_stream_br.close();
    fs::create_directories(full_path + "/ncrisc");
    ofstream file_stream_nc(full_path + "/ncrisc/generated_bank_to_noc_coord_mapping.h");
    file_stream_nc << output_string;
    file_stream_nc.close();
}

void generate_binaries_all_riscs(
    tt::build_kernel_for_riscv_options_t* opts, const std::string& out_dir_path, const std::string& arch_name,
    generate_binaries_params_t p)
{
    ZoneScoped;
    const std::string tracyPrefix = "generate_binaries_all_riscs_";
    ZoneName( (tracyPrefix + out_dir_path).c_str(), out_dir_path.length() + tracyPrefix.length());

    generate_descriptors(opts, out_dir_path);

    std::vector<std::thread> threads;
    std::function<void()> lambdas[] = {
        [opts, out_dir_path, arch_name, p] () {
            generate_binaries_for_triscs(
                opts, out_dir_path, arch_name, p.compute_kernel_compile_time_args);
        },
        [opts, out_dir_path, arch_name, p] () {
            generate_binary_for_ncrisc(
                opts, out_dir_path, arch_name, p.nc_noc_index, p.nc_kernel_compile_time_args);
        },
        [opts, out_dir_path, arch_name, p] () {
            generate_binary_for_brisc(
                opts, out_dir_path, arch_name, p.br_noc_index, p.br_kernel_compile_time_args);
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
    const std::vector<std::uint32_t>& kernel_compile_time_args)
{
    ZoneScoped;
    const std::string tracyPrefix = "generate_binaries_for_triscs_";
    ZoneName( (tracyPrefix + dir).c_str(), dir.length() + tracyPrefix.length());
    generate_src_for_triscs(topts, dir, arch_name, kernel_compile_time_args);
    auto lambda0 = [=]() { generate_binary_for_risc(RISCID::TR0, topts, dir, arch_name, 0, kernel_compile_time_args); };
    auto lambda1 = [=]() { generate_binary_for_risc(RISCID::TR1, topts, dir, arch_name, 0, kernel_compile_time_args); };
    auto lambda2 = [=]() { generate_binary_for_risc(RISCID::TR2, topts, dir, arch_name, 0, kernel_compile_time_args); };
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
    ZoneScoped;
    const std::string tracyPrefix = "generate_descriptors_";
    ZoneName( (tracyPrefix + op_dir).c_str(), op_dir.length() + tracyPrefix.length());
    string full_path = opts->outpath + op_dir;
    fs::create_directories(full_path);
    try {
        std::thread td( [=]() { generate_data_format_descriptors(opts, op_dir); } );
        std::thread tm( [=]() { generate_math_fidelity_descriptor(opts, op_dir); } );
        std::thread ta( [=]() { generate_math_approx_mode_descriptor(opts, op_dir); } );
        td.join();
        tm.join();
        ta.join();
    } catch (std::runtime_error &ex) {
        std::cerr << "EXCEPTION FROM THREADING IN GENERATE_DESCRIPTORS: " << ex.what() << std::endl;
    }
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

//
namespace __internal {
void generate_default_bank_to_noc_coord_descriptor(
    tt::build_kernel_for_riscv_options_t* build_kernel_for_riscv_options,
    string out_dir_path,
    tt::ARCH arch
) {
    std::vector<CoreCoord> dram_bank_map;
    std::vector<int32_t> dram_bank_offset_map;
    std::vector<CoreCoord> l1_bank_map;
    std::vector<int32_t> l1_bank_offset_map;
    unsigned int bank_id = 0;
    switch (arch) {
        case tt::ARCH::GRAYSKULL:
            dram_bank_map = std::vector<CoreCoord> ({
                {.x = 1, .y = 0},
                {.x = 1, .y = 6},
                {.x = 4, .y = 0},
                {.x = 4, .y = 6},
                {.x = 7, .y = 0},
                {.x = 7, .y = 6},
                {.x = 10, .y = 0},
                {.x = 10, .y = 6},
            });
            dram_bank_offset_map = std::vector<int32_t>(8, 0);
            l1_bank_map = std::vector<CoreCoord> (128, {.x=0, .y=0});
            for (unsigned int r = 0; r < 10; r++) {
                for (unsigned int c = 0; c < 12; c++) {
                    l1_bank_map.at(bank_id) = {.x = c, .y = r};
                    bank_id++;
                }
            }
            l1_bank_offset_map = std::vector<int32_t> (l1_bank_map.size(), 0);
            break;
        case tt::ARCH::WORMHOLE_B0:
            dram_bank_map = std::vector<CoreCoord> ({
                {.x = 0, .y = 11},
                {.x = 0, .y = 1},
                {.x = 0, .y = 5},
                {.x = 0, .y = 7},
                {.x = 5, .y = 1},
                {.x = 5, .y = 11},
                {.x = 5, .y = 2},
                {.x = 5, .y = 9},
                {.x = 5, .y = 8},
                {.x = 5, .y = 3},
                {.x = 5, .y = 5},
                {.x = 5, .y = 7},
            });
            dram_bank_offset_map = std::vector<int32_t>({
                0,
                1 * 1024 *1024,
                0,
                1 * 1024 *1024,
                0,
                1 * 1024 *1024,
                0,
                1 * 1024 *1024,
                0,
                1 * 1024 *1024,
                0,
                1 * 1024 *1024,
            });
            l1_bank_map = std::vector<CoreCoord> (128, {.x=0, .y=0});
            for (unsigned int r = 0; r < 10; r++) {
                for (unsigned int c = 0; c < 8; c++) {
                    l1_bank_map.at(bank_id) = {.x = c, .y = r};
                    bank_id++;
                }
            }
            l1_bank_offset_map = std::vector<int32_t> (l1_bank_map.size(), 0);
            break;
        default:
            log_fatal("Unsupported arch in generate_default_bank_to_noc_coord_descriptor");
            break;
    }

    generate_bank_to_noc_coord_descriptor(build_kernel_for_riscv_options, out_dir_path, dram_bank_map, dram_bank_offset_map, l1_bank_map, l1_bank_offset_map);
}
}
}
