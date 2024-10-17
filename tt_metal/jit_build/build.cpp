// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "jit_build/build.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "common/executor.hpp"
#include "jit_build/genfiles.hpp"
#include "jit_build/kernel_args.hpp"
#include "tools/profiler/common.hpp"
#include "tools/profiler/profiler_state.hpp"
#include "tt_metal/impl/kernels/kernel.hpp"
#include "tt_metal/impl/dispatch/command_queue_interface.hpp"

namespace fs = std::filesystem;

using namespace std;
using namespace tt;

namespace tt::tt_metal {

static std::string get_string_aliased_arch_lowercase(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "grayskull"; break;
        case tt::ARCH::WORMHOLE: return "wormhole"; break;
        case tt::ARCH::WORMHOLE_B0: return "wormhole"; break;
        case tt::ARCH::BLACKHOLE: return "blackhole"; break;
        default: return "invalid"; break;
    }
}

JitBuildEnv::JitBuildEnv() {}

void JitBuildEnv::init(uint32_t build_key, tt::ARCH arch) {
    // Paths
    this->root_ = llrt::OptionsG.get_root_dir();
    this->out_root_ = this->root_ + "built/";
    this->arch_ = arch;
    this->arch_name_ = get_string_lowercase(arch);
    this->aliased_arch_name_ = get_string_aliased_arch_lowercase(arch);

    this->out_firmware_root_ = this->out_root_ + to_string(build_key) + "/firmware/";
    this->out_kernel_root_ = this->out_root_ + to_string(build_key) + "/kernels/";

    // Tools
    this->gpp_ = this->root_ + "tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-g++ ";
    this->objcopy_ = this->root_ + "tt_metal/third_party/sfpi/compiler/bin/riscv32-unknown-elf-objcopy ";

    // Flags
    string common_flags;
    switch (arch) {
        case ARCH::GRAYSKULL: common_flags = "-mgrayskull -march=rv32iy -mtune=rvtt-b1 -mabi=ilp32 "; break;
        case ARCH::WORMHOLE_B0: common_flags = "-mwormhole -march=rv32imw -mtune=rvtt-b1 -mabi=ilp32 "; break;
        case ARCH::BLACKHOLE: common_flags = "-mblackhole -march=rv32iml -mtune=rvtt-b1 -mabi=ilp32 "; break;
        default: TT_ASSERT(false, "Invalid arch"); break;
    }
    common_flags += "-std=c++17 -flto -ffast-math ";

    if (tt::llrt::OptionsG.get_riscv_debug_info_enabled()) {
        common_flags += "-g ";
    }

    this->cflags_ = common_flags;
    this->cflags_ +=
        "-fno-use-cxa-atexit -fno-exceptions "
        "-Wall -Werror -Wno-unknown-pragmas "
        "-Wno-error=multistatement-macros -Wno-error=parentheses "
        "-Wno-error=unused-but-set-variable -Wno-unused-variable "
        "-Wno-unused-function ";

    // Defines
    switch (arch) {
        case ARCH::GRAYSKULL: this->defines_ = "-DARCH_GRAYSKULL "; break;
        case ARCH::WORMHOLE_B0: this->defines_ = "-DARCH_WORMHOLE "; break;
        case ARCH::BLACKHOLE: this->defines_ = "-DARCH_BLACKHOLE "; break;
        default: break;
    }
    this->defines_ += "-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 ";

    if (tt::tt_metal::getDeviceProfilerState()) {
        if (tt::llrt::OptionsG.get_profiler_do_dispatch_cores()) {
            //TODO(MO): Standard bit mask for device side profiler options
            this->defines_ += "-DPROFILE_KERNEL=2 ";
        } else {
            this->defines_ += "-DPROFILE_KERNEL=1 ";
        }
    }

    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        this->defines_ += "-DWATCHER_ENABLED ";
    }
    if (tt::llrt::OptionsG.get_watcher_noinline()) {
        this->defines_ += "-DWATCHER_NOINLINE ";
    }
    for (auto& feature : tt::llrt::OptionsG.get_watcher_disabled_features()) {
        this->defines_ += "-DWATCHER_DISABLE_" + feature + " ";
    }

    if (tt::llrt::OptionsG.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        this->defines_ += "-DDEBUG_PRINT_ENABLED -DL1_UNRESERVED_BASE=" + to_string(hal.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED)) + " ";
    }

    if (tt::llrt::OptionsG.get_record_noc_transfers()) {
        this->defines_ += "-DNOC_LOGGING_ENABLED ";
    }

    if (tt::llrt::OptionsG.get_kernels_nullified()) {
        this->defines_ += "-DDEBUG_NULL_KERNELS ";
    }

    if (tt::llrt::OptionsG.get_watcher_debug_delay()) {
        this->defines_ += "-DWATCHER_DEBUG_DELAY=" + to_string(tt::llrt::OptionsG.get_watcher_debug_delay()) + " ";
    }

    // Includes
    // TODO(pgk) this list is insane
    this->includes_ = string("") + "-I. " + "-I.. " + "-I" + this->root_ + " " + "-I" + this->root_ + "tt_metal " +
                      "-I" + this->root_ + "tt_metal/include " + "-I" + this->root_ + "tt_metal/hw/inc " + "-I" +
                      this->root_ + "tt_metal/hw/inc/debug " + "-I" + this->root_ + "tt_metal/hw/inc/" +
                      this->aliased_arch_name_ + " " + "-I" + this->root_ + "tt_metal/hw/inc/" +
                      this->aliased_arch_name_ + "/" + this->arch_name_ + "_defines " + "-I" + this->root_ +
                      "tt_metal/hw/inc/" + this->aliased_arch_name_ + "/noc " + "-I" + this->root_ +
                      "tt_metal/third_party/umd/device/" + this->arch_name_ + " " +  // TODO(fixme)
                      "-I" + this->root_ + "tt_metal/hw/ckernels/" + this->arch_name_ + "/metal/common " + "-I" +
                      this->root_ + "tt_metal/hw/ckernels/" + this->arch_name_ + "/metal/llk_io " + "-I" + this->root_ +
                      "tt_metal/third_party/tt_llk_" + this->arch_name_ +
                      "/common/inc " +  // TODO(fixme) datamovement fw shouldn't read this
                      "-I" + this->root_ + "tt_metal/third_party/tt_llk_" + this->arch_name_ + "/llk_lib ";

    this->lflags_ = common_flags;
    this->lflags_ += "-fno-exceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";
}

JitBuildState::JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig &build_config) :
    env_(env), core_id_(build_config.processor_id), is_fw_(build_config.is_fw), dispatch_message_addr_(build_config.dispatch_message_addr) {}

// Fill in common state derived from the default state set up in the constructors
void JitBuildState::finish_init() {
    if (this->is_fw_) {
        this->defines_ += "-DFW_BUILD ";
    } else {
        this->defines_ += "-DKERNEL_BUILD ";
    }
    this->defines_ += "-DDISPATCH_MESSAGE_ADDR=" + to_string(this->dispatch_message_addr_) + " ";

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

    // Append hw build objects compiled offline
    std::string build_dir = llrt::OptionsG.get_root_dir() + "runtime/hw/lib/" + get_alias(env_.arch_) + "/";
    if (this->is_fw_) {
        if (this->target_name_ == "brisc") {
            this->link_objs_ += build_dir + "tdma_xmov.o ";
        }
        if (this->target_name_ != "erisc") {
            this->link_objs_ += build_dir + "tmu-crt0.o ";
        }
        if (this->target_name_ == "ncrisc" and
            ((this->env_.arch_ == tt::ARCH::GRAYSKULL or this->env_.arch_ == tt::ARCH::WORMHOLE_B0))) {
            this->link_objs_ += build_dir + "ncrisc-halt.o ";
        }
    } else {
        this->link_objs_ += build_dir + "tmu-crt0k.o ";
    }
    if (this->target_name_ == "brisc" or this->target_name_ == "idle_erisc") {
        this->link_objs_ += build_dir + "noc.o ";
    }
    this->link_objs_ += build_dir + "substitutes.o ";

    // Note the preceding slash which defies convention as this gets appended to
    // the kernel name used as a path which doesn't have a slash
    this->target_full_path_ = "/" + this->target_name_ + "/" + this->target_name_ + ".elf";
}

JitBuildDataMovement::JitBuildDataMovement(const JitBuildEnv& env, const JitBuiltStateConfig &build_config) :
    JitBuildState(env, build_config) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 2, "Invalid data movement processor");

    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->cflags_ = env_.cflags_ + "-Os " + "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops
    this->includes_ = env_.includes_ + "-I " + env_.root_ + "tt_metal/hw/firmware/src " + "-I " + env_.root_ +
                      "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/common " + "-I " + env_.root_ +
                      "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_io ";

    this->defines_ = env_.defines_;

    uint32_t l1_cache_disable_mask =
        tt::llrt::OptionsG.get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDisableL1DataCache);

    this->lflags_ = env_.lflags_ + "-Os ";

    switch (this->core_id_) {
        case 0:
            this->target_name_ = "brisc";

            this->defines_ += "-DCOMPILE_FOR_BRISC ";
            if ((l1_cache_disable_mask & tt::llrt::DebugHartFlags::RISCV_BR) == tt::llrt::DebugHartFlags::RISCV_BR) {
                this->defines_ += "-DDISABLE_L1_DATA_CACHE ";
            }
            if (this->is_fw_) {
                this->srcs_.push_back("tt_metal/hw/firmware/src/brisc.cc");
            } else {
                this->srcs_.push_back("tt_metal/hw/firmware/src/brisck.cc");
            }

            if (this->is_fw_) {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_brisc.ld ";
            } else {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_brisc.ld ";
            }

            break;

        case 1:
            this->target_name_ = "ncrisc";

            this->defines_ += "-DCOMPILE_FOR_NCRISC ";
            if ((l1_cache_disable_mask & tt::llrt::DebugHartFlags::RISCV_NC) == tt::llrt::DebugHartFlags::RISCV_NC) {
                this->defines_ += "-DDISABLE_L1_DATA_CACHE ";
            }

            if (this->is_fw_) {
                this->srcs_.push_back("tt_metal/hw/firmware/src/ncrisc.cc");
            } else {
                this->srcs_.push_back("tt_metal/hw/firmware/src/ncrisck.cc");
            }

            if (this->is_fw_) {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_ncrisc.ld ";
            } else {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_ncrisc.ld ";
            }

            break;
    }

    this->process_defines_at_compile = true;

    finish_init();
}

JitBuildCompute::JitBuildCompute(const JitBuildEnv& env, const JitBuiltStateConfig &build_config) : JitBuildState(env, build_config) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 3, "Invalid compute processor");

    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->cflags_ = env_.cflags_ + "-O3 ";

    this->defines_ = env_.defines_;
    uint32_t l1_cache_disable_mask =
        tt::llrt::OptionsG.get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDisableL1DataCache);
    uint32_t debug_compute_mask =
        (tt::llrt::DebugHartFlags::RISCV_TR0 | tt::llrt::DebugHartFlags::RISCV_TR1 |
         tt::llrt::DebugHartFlags::RISCV_TR2);
    if ((l1_cache_disable_mask & debug_compute_mask) == debug_compute_mask) {
        this->defines_ += "-DDISABLE_L1_DATA_CACHE ";
    }

    this->includes_ = env_.includes_ + "-I" + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/inc " + "-I" +
                      env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/common " + "-I" + env_.root_ +
                      "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_io " + "-I" + env_.root_ +
                      "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_api " + "-I" + env_.root_ +
                      "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_api/llk_sfpu " + "-I" + env_.root_ +
                      "tt_metal/third_party/sfpi/include " + "-I" + env_.root_ + "tt_metal/hw/firmware/src " + "-I" +
                      env_.root_ + "tt_metal/third_party/tt_llk_" + env.arch_name_ + "/llk_lib ";

    if (this->is_fw_) {
        this->srcs_.push_back("tt_metal/hw/firmware/src/trisc.cc");
    } else {
        this->srcs_.push_back("tt_metal/hw/firmware/src/trisck.cc");
    }

    this->lflags_ = env_.lflags_ + "-O3 ";

    switch (this->core_id_) {
        case 0:
            this->target_name_ = "trisc0";

            this->defines_ += "-DUCK_CHLKC_UNPACK ";
            this->defines_ += "-DNAMESPACE=chlkc_unpack ";
            this->defines_ += "-DCOMPILE_FOR_TRISC=0 ";

            if (this->is_fw_) {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_trisc0.ld ";
            } else {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_trisc0.ld ";
            }

            break;

        case 1:
            this->target_name_ = "trisc1";

            this->defines_ += "-DUCK_CHLKC_MATH ";
            this->defines_ += "-DNAMESPACE=chlkc_math ";
            this->defines_ += "-DCOMPILE_FOR_TRISC=1 ";

            if (this->is_fw_) {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_trisc1.ld ";
            } else {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_trisc1.ld ";
            }

            break;

        case 2:
            this->target_name_ = "trisc2";

            this->defines_ += "-DUCK_CHLKC_PACK ";
            this->defines_ += "-DNAMESPACE=chlkc_pack ";
            this->defines_ += "-DCOMPILE_FOR_TRISC=2 ";

            if (this->is_fw_) {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_trisc2.ld ";
            } else {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_trisc2.ld ";
            }

            break;
    }

    this->process_defines_at_compile = false;

    finish_init();
}

JitBuildEthernet::JitBuildEthernet(const JitBuildEnv& env, const JitBuiltStateConfig &build_config) : JitBuildState(env, build_config) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 2, "Invalid ethernet processor");
    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->includes_ = env_.includes_ + "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ +
                      "/metal/common " + "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ +
                      "/metal/llk_io ";

    this->defines_ = env_.defines_;
    uint32_t l1_cache_disable_mask =
        tt::llrt::OptionsG.get_feature_riscv_mask(tt::llrt::RunTimeDebugFeatureDisableL1DataCache);
    if ((l1_cache_disable_mask & tt::llrt::DebugHartFlags::RISCV_ER) == tt::llrt::DebugHartFlags::RISCV_ER) {
        this->defines_ += "-DDISABLE_L1_DATA_CACHE ";
    }

    switch (this->core_id_) {
        case 0: {
            this->target_name_ = "erisc";
            this->cflags_ = env_.cflags_ + "-Os -fno-delete-null-pointer-checks ";

            this->defines_ +=
                "-DCOMPILE_FOR_ERISC "
                "-DERISC "
                "-DRISC_B0_HW ";
            if (this->is_fw_) {
                this->defines_ += "-DLOADING_NOC=0 ";
            }

            this->includes_ += "-I " + env_.root_ + "tt_metal/hw/inc/ethernet ";

            if (this->is_fw_) {
                this->srcs_.push_back("tt_metal/hw/firmware/src/erisc.cc");
                this->srcs_.push_back("tt_metal/hw/firmware/src/erisc-crt0.cc");
            } else {
                this->srcs_.push_back("tt_metal/hw/firmware/src/erisck.cc");
            }

            string linker_str;
            if (this->is_fw_) {
                linker_str = "tt_metal/hw/toolchain/erisc-b0-app.ld ";
            } else {
                linker_str = "tt_metal/hw/toolchain/erisc-b0-kernel.ld ";
            }
            this->lflags_ = env_.lflags_ +
                            "-Os "
                            "-L" +
                            env_.root_ +
                            "/tt_metal/hw/toolchain "
                            "-T" +
                            env_.root_ + linker_str;
            break;
        }
        case 1:
            this->target_name_ = "idle_erisc";
            this->cflags_ =
                env_.cflags_ + "-Os " + "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops

            this->defines_ +=
                "-DCOMPILE_FOR_IDLE_ERISC "
                "-DERISC "
                "-DRISC_B0_HW ";

            this->includes_ += "-I " + env_.root_ + "tt_metal/hw/firmware/src ";

            if (this->is_fw_) {
                this->srcs_.push_back("tt_metal/hw/firmware/src/idle_erisc.cc");
            } else {
                this->srcs_.push_back("tt_metal/hw/firmware/src/idle_erisck.cc");
            }
            this->lflags_ = env_.lflags_ + "-Os ";

            if (this->is_fw_) {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_ierisc.ld ";
            } else {
                this->lflags_ += "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_ierisc.ld ";
            }

            break;
    }
    this->process_defines_at_compile = true;

    finish_init();
}

static void build_failure(const string& target_name, const string& op, const string& cmd, const string& log_file) {
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

void JitBuildState::compile_one(
    const string& log_file,
    const string& out_dir,
    const JitBuildSettings* settings,
    const string& src,
    const string& obj) const {
    ZoneScoped;
    fs::create_directories(out_dir);

    // Add kernel specific defines
    string defines = this->defines_;
    if (settings != nullptr) {
        if (process_defines_at_compile) {
            settings->process_defines([&defines](const string& define, const string& value) {
                defines += "-D" + define + "=" + value + " ";
            });
        }

        settings->process_compile_time_args([&defines](int i, uint32_t value) {
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

    if (tt::llrt::OptionsG.get_watcher_enabled() && settings) {
        log_kernel_defines_and_args(out_dir, settings->get_full_kernel_name(), defines);
    }

    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "compile", cmd, log_file);
    }
}

void JitBuildState::compile(const string& log_file, const string& out_dir, const JitBuildSettings* settings) const {
    ZoneScoped;
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < this->srcs_.size(); ++i) {
        launch_build_step(
            [this, &log_file, &out_dir, settings, i] {
                this->compile_one(log_file, out_dir, settings, this->srcs_[i], this->objs_[i]);
            },
            events);
    }

    sync_build_step(events);
    if (tt::llrt::OptionsG.get_watcher_enabled()) {
        dump_kernel_defines_and_args(env_.get_out_kernel_root_path());
    }
}

void JitBuildState::link(const string& log_file, const string& out_dir) const {
    ZoneScoped;
    string lflags = this->lflags_;
    if (tt::llrt::OptionsG.get_build_map_enabled()) {
        lflags += "-Wl,-Map=" + out_dir + "linker.map ";
    }

    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.gpp_;
    cmd += lflags;
    cmd += this->link_objs_;

    if (!this->is_fw_) {
        string weakened_elf_name =
            env_.out_firmware_root_ + this->target_name_ + "/" + this->target_name_ + "_weakened.elf";
        cmd += " -Xlinker \"--just-symbols=" + weakened_elf_name + "\" ";
    }

    cmd += "-o " + out_dir + this->target_name_ + ".elf";
    log_debug(tt::LogBuildKernels, "    g++ link cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "link", cmd, log_file);
    }
}

// Given this elf (A) and a later elf (B):
// weakens symbols in A so that it can be used as a "library" for B. B imports A's weakened symbols, B's symbols of the
// same name don't result in duplicate symbols but B can reference A's symbols. Force the fw_export symbols to remain
// strong so to propogate link addresses
void JitBuildState::weaken(const string& log_file, const string& out_dir) const {
    ZoneScoped;
    string cmd;
    cmd = "cd " + out_dir + " && ";
    cmd += env_.objcopy_;
    cmd += " --wildcard --weaken-symbol \"*\" --weaken-symbol \"!__fw_export_*\" " + this->target_name_ + ".elf " +
           this->target_name_ + "_weakened.elf";

    log_debug(tt::LogBuildKernels, "    objcopy cmd: {}", cmd);
    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "objcopy weaken", cmd, log_file);
    }
}

void JitBuildState::extract_zone_src_locations(const string& log_file) const {
    ZoneScoped;
    static std::atomic<bool> new_log = true;
    if (tt::tt_metal::getDeviceProfilerState()) {
        if (new_log.exchange(false) && std::filesystem::exists(tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG)) {
            std::remove(tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG.c_str());
        }

        if (!std::filesystem::exists(tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG)) {
            tt::utils::create_file(tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG);
        }

        // Only interested in log entries with KERNEL_PROFILER inside them as device code
        // tags source location info with it using pragma messages
        string cmd = "cat " + log_file + " | grep KERNEL_PROFILER";
        tt::utils::run_command(cmd, tt::tt_metal::PROFILER_ZONE_SRC_LOCATIONS_LOG, false);
    }
}

void JitBuildState::build(const JitBuildSettings* settings) const {
    ZoneScoped;
    string out_dir = (settings == nullptr)
                         ? this->out_path_ + this->target_name_ + "/"
                         : this->out_path_ + settings->get_full_kernel_name() + this->target_name_ + "/";

    string log_file = out_dir + "build.log";
    if (fs::exists(log_file)) {
        std::remove(log_file.c_str());
    }

    compile(log_file, out_dir, settings);
    link(log_file, out_dir);
    if (this->is_fw_) {
        weaken(log_file, out_dir);
    }

    extract_zone_src_locations(log_file);
}

void jit_build(const JitBuildState& build, const JitBuildSettings* settings) {
    ZoneScoped;

    build.build(settings);
}

void jit_build_set(const JitBuildStateSet& build_set, const JitBuildSettings* settings) {
    ZoneScoped;
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < build_set.size(); ++i) {
        // Capture the necessary objects by reference
        auto& build = build_set[i];
        launch_build_step([build, settings] { build->build(settings); }, events);
    }
    sync_build_step(events);
}

void jit_build_subset(const JitBuildStateSubset& build_subset, const JitBuildSettings* settings) {
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < build_subset.size; ++i) {
        // Capture the necessary objects by reference
        auto& build = build_subset.build_ptr[i];
        launch_build_step([build, settings] { build->build(settings); }, events);
    }
    sync_build_step(events);
}

}  // namespace tt::tt_metal
