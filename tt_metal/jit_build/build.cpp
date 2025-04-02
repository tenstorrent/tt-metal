// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "build.hpp"

#include <taskflow/core/async.hpp>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <span>
#include <string>
#include <string_view>

#include "assert.hpp"
#include "common/executor.hpp"
#include "env_lib.hpp"
#include "fmt/base.h"
#include "hal_types.hpp"
#include "jit_build/kernel_args.hpp"
#include "jit_build_settings.hpp"
#include "llrt/hal.hpp"
#include "logger.hpp"
#include "profiler_paths.hpp"
#include "profiler_state.hpp"
#include "rtoptions.hpp"
#include "tt_backend_api_types.hpp"
#include "tt_metal/llrt/tt_elffile.hpp"
#include <umd/device/types/arch.h>

namespace fs = std::filesystem;

using namespace std;
using namespace tt;

namespace {

void sync_events(auto& events) {
    for (auto& f : events) {
        f.get();
    }
}

}  // namespace

namespace tt::tt_metal {

static std::string get_string_aliased_arch_lowercase(tt::ARCH arch) {
    switch (arch) {
        case tt::ARCH::GRAYSKULL: return "grayskull"; break;
        case tt::ARCH::WORMHOLE_B0: return "wormhole"; break;
        case tt::ARCH::BLACKHOLE: return "blackhole"; break;
        default: return "invalid"; break;
    }
}

static void build_failure(const string& target_name, const string& op, const string& cmd, const string& log_file) {
    log_error(tt::LogBuildKernels, "{} {} failure -- cmd: {}", target_name, op, cmd);
    std::ifstream file{log_file};
    if (file.is_open()) {
        std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        TT_THROW("{} build failed. Log: {}", target_name, log_contents);
    } else {
        TT_THROW("Failed to open {} failure log file {}", op, log_file);
    }
}

static void write_successful_jit_build_marker(const JitBuildState& build, const JitBuildSettings* settings) {
    const string out_dir = (settings == nullptr) ? build.get_out_path() + "/"
                                                 : build.get_out_path() + settings->get_full_kernel_name() + "/";
    std::ofstream file(out_dir + SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME);
}

static void check_built_dir(const std::filesystem::path& dir_path, const std::filesystem::path& git_hash_path) {
    if (dir_path.compare(git_hash_path) != 0) {
        std::filesystem::remove_all(dir_path);
    }
}

std::string get_default_root_path() {
    const std::string emptyString("");
    const std::string home_path = parse_env<std::string>("HOME", emptyString);
    if (!home_path.empty() && std::filesystem::exists(home_path)) {
        return home_path + "/.cache/tt-metal-cache/";
    } else {
        return "/tmp/tt-metal-cache/";
    }
}
JitBuildEnv::JitBuildEnv() {}

void JitBuildEnv::init(
    uint32_t build_key, tt::ARCH arch, const std::map<std::string, std::string>& device_kernel_defines) {
    // Paths
    this->root_ = llrt::RunTimeOptions::get_instance().get_root_dir();
    this->out_root_ = llrt::RunTimeOptions::get_instance().is_cache_dir_specified()
                          ? llrt::RunTimeOptions::get_instance().get_cache_dir()
                          : get_default_root_path();

    this->arch_ = arch;
    this->arch_name_ = get_string_lowercase(arch);
    this->aliased_arch_name_ = get_string_aliased_arch_lowercase(arch);

#ifndef GIT_COMMIT_HASH
    log_info(tt::LogBuildKernels, "GIT_COMMIT_HASH not found");
#else
    std::string git_hash(GIT_COMMIT_HASH);

    std::filesystem::path git_hash_path(this->out_root_ + git_hash);
    std::filesystem::path root_path(this->out_root_);
    if ((not llrt::RunTimeOptions::get_instance().get_skip_deleting_built_cache()) &&
        std::filesystem::exists(root_path)) {
        std::ranges::for_each(
            std::filesystem::directory_iterator{root_path},
            [&git_hash_path](const auto& dir_entry) { check_built_dir(dir_entry.path(), git_hash_path); });
    } else {
        log_info(tt::LogBuildKernels, "Skipping deleting built cache");
    }

    this->out_root_ = this->out_root_  + git_hash + "/";
#endif

    this->out_firmware_root_ = this->out_root_ + to_string(build_key) + "/firmware/";
    this->out_kernel_root_ = this->out_root_ + to_string(build_key) + "/kernels/";

    const static bool use_ccache = std::getenv("TT_METAL_CCACHE_KERNEL_SUPPORT") != nullptr;
    if (use_ccache) {
        this->gpp_ = "ccache ";
    } else {
        this->gpp_ = "";
    }

    // Tools
    this->gpp_ += this->root_ + "runtime/sfpi/compiler/bin/riscv32-unknown-elf-g++ ";

    // Flags
    string common_flags;
    switch (arch) {
        case ARCH::GRAYSKULL: common_flags = "-mcpu=tt-gs "; break;
        case ARCH::WORMHOLE_B0: common_flags = "-mcpu=tt-wh "; break;
        case ARCH::BLACKHOLE: common_flags = "-mcpu=tt-bh -fno-rvtt-sfpu-replay "; break;
        default: TT_ASSERT(false, "Invalid arch"); break;
    }
    common_flags += "-std=c++17 -flto -ffast-math ";

    if (tt::llrt::RunTimeOptions::get_instance().get_riscv_debug_info_enabled()) {
        common_flags += "-g ";
    }

    this->cflags_ = common_flags;
    this->cflags_ +=
        "-fno-use-cxa-atexit -fno-exceptions "
        "-Wall -Werror -Wno-unknown-pragmas "
        "-Wno-deprecated-declarations "
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
    for (auto it = device_kernel_defines.begin(); it != device_kernel_defines.end(); ++it) {
        this->defines_ += "-D" + it->first + "=" + it->second + " ";
    }
    this->defines_ += "-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 ";

    if (tt::tt_metal::getDeviceProfilerState()) {
        if (tt::llrt::RunTimeOptions::get_instance().get_profiler_do_dispatch_cores()) {
            // TODO(MO): Standard bit mask for device side profiler options
            this->defines_ += "-DPROFILE_KERNEL=2 ";
        } else {
            this->defines_ += "-DPROFILE_KERNEL=1 ";
        }
    }
    if (tt::llrt::RunTimeOptions::get_instance().get_profiler_noc_events_enabled()) {
        // force profiler on if noc events are being profiled
        if (not tt::tt_metal::getDeviceProfilerState()) {
            this->defines_ += "-DPROFILE_KERNEL=1 ";
        }
        this->defines_ += "-DPROFILE_NOC_EVENTS=1 ";
    }

    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled()) {
        this->defines_ += "-DWATCHER_ENABLED ";
    }
    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_noinline()) {
        this->defines_ += "-DWATCHER_NOINLINE ";
    }
    for (auto& feature : tt::llrt::RunTimeOptions::get_instance().get_watcher_disabled_features()) {
        this->defines_ += "-DWATCHER_DISABLE_" + feature + " ";
    }

    if (tt::llrt::RunTimeOptions::get_instance().get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        this->defines_ +=
            "-DDEBUG_PRINT_ENABLED -DL1_UNRESERVED_BASE=" +
            to_string(hal_ref.get_dev_addr(HalProgrammableCoreType::TENSIX, HalL1MemAddrType::UNRESERVED)) + " ";
    }

    if (tt::llrt::RunTimeOptions::get_instance().get_record_noc_transfers()) {
        this->defines_ += "-DNOC_LOGGING_ENABLED ";
    }

    if (tt::llrt::RunTimeOptions::get_instance().get_kernels_nullified()) {
        this->defines_ += "-DDEBUG_NULL_KERNELS ";
    }

    if (tt::llrt::RunTimeOptions::get_instance().get_kernels_early_return()) {
        this->defines_ += "-DDEBUG_EARLY_RETURN_KERNELS ";
    }

    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_debug_delay()) {
        this->defines_ +=
            "-DWATCHER_DEBUG_DELAY=" + to_string(tt::llrt::RunTimeOptions::get_instance().get_watcher_debug_delay()) +
            " ";
    }

    if (tt::llrt::RunTimeOptions::get_instance().get_hw_cache_invalidation_enabled()) {
        this->defines_ += "-DENABLE_HW_CACHE_INVALIDATION ";
    }

    // Includes
    // TODO(pgk) this list is insane
    std::vector<std::string> includeDirs = {
        ".",
        "..",
        root_,
        root_ + "ttnn",
        root_ + "tt_metal",
        root_ + "tt_metal/include",
        root_ + "tt_metal/hw/inc",
        root_ + "tt_metal/hostdevcommon/api",
        root_ + "tt_metal/hw/inc/debug",
        root_ + "tt_metal/hw/inc/" + this->aliased_arch_name_,
        root_ + "tt_metal/hw/inc/" + this->aliased_arch_name_ + "/" + this->arch_name_ + "_defines",
        root_ + "tt_metal/hw/inc/" + this->aliased_arch_name_ + "/noc",
        root_ + "tt_metal/hw/ckernels/" + this->arch_name_ + "/metal/common",
        root_ + "tt_metal/hw/ckernels/" + this->arch_name_ + "/metal/llk_io",
        // TODO: datamovement fw shouldn't read this
        root_ + "tt_metal/third_party/tt_llk/tt_llk_" + this->arch_name_ + "/common/inc",
        root_ + "tt_metal/api/",
        root_ + "tt_metal/api/tt-metalium/",
        root_ + "tt_metal/third_party/tt_llk/tt_llk_" + this->arch_name_ + "/llk_lib"
    };

    std::ostringstream oss;
    for (size_t i = 0; i < includeDirs.size(); ++i) {
        oss << "-I" << includeDirs[i] << " ";
    }
    this->includes_ = oss.str();

    this->lflags_ = common_flags;
    this->lflags_ += "-fno-exceptions -Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";
}

JitBuildState::JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig& build_config) :
    env_(env),
    core_id_(build_config.processor_id),
    is_fw_(build_config.is_fw),
    dispatch_message_addr_(build_config.dispatch_message_addr) {}

// Fill in common state derived from the default state set up in the constructors
void JitBuildState::finish_init() {
    if (this->is_fw_) {
        this->defines_ += "-DFW_BUILD ";
    } else {
        this->defines_ += "-DKERNEL_BUILD ";
    }
    this->defines_ += "-DDISPATCH_MESSAGE_ADDR=" + to_string(this->dispatch_message_addr_) + " ";

    // Create the objs from the srcs
    for (const string& src : srcs_) {
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
    std::string build_dir =
        llrt::RunTimeOptions::get_instance().get_root_dir() + "runtime/hw/lib/" + get_alias(env_.arch_) + "/";
    if (this->is_fw_) {
        if (this->target_name_ != "erisc") {
            this->link_objs_ += build_dir + "tmu-crt0.o ";
        }
        if (this->target_name_ == "ncrisc" and this->env_.arch_ == tt::ARCH::WORMHOLE_B0) {
            this->link_objs_ += build_dir + "ncrisc-halt-wormhole.o ";
            this->link_objs_ += build_dir + "tdma_xmov.o ";
        }
    } else {
        if (this->target_name_ == "ncrisc" and this->env_.arch_ == tt::ARCH::WORMHOLE_B0) {
            this->link_objs_ += build_dir + "tmu-crt0k-ncrisc.o ";
        } else if (this->target_name_ != "erisc") {
            this->link_objs_ += build_dir + "tmu-crt0k.o ";
        }
    }
    if (this->target_name_ == "brisc" or this->target_name_ == "idle_erisc") {
        this->link_objs_ += build_dir + "noc.o ";
    }
    this->link_objs_ += build_dir + "substitutes.o ";

    // Note the preceding slash which defies convention as this gets appended to
    // the kernel name used as a path which doesn't have a slash
    this->target_full_path_ = "/" + this->target_name_ + "/" + this->target_name_ + ".elf";

    if (not this->is_fw_) {
        // Emit relocations, so we can relocate the resulting binary
        this->lflags_ += "-Wl,--emit-relocs ";
    }
}

JitBuildDataMovement::JitBuildDataMovement(const JitBuildEnv& env, const JitBuiltStateConfig& build_config) :
    JitBuildState(env, build_config) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 2, "Invalid data movement processor");
    this->lflags_ = env.lflags_;
    this->cflags_ = env.cflags_;
    this->default_compile_opt_level_ = "Os";
    this->default_linker_opt_level_ = "Os";
    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;
    this->cflags_ = env_.cflags_ + "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops

    // clang-format off
    this->includes_ = env_.includes_ +
        "-I " + env_.root_ + "tt_metal/hw/firmware/src " +
        "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/common " +
        "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_io ";
    // clang-format on

    this->defines_ = env_.defines_;

    uint32_t l1_cache_disable_mask = tt::llrt::RunTimeOptions::get_instance().get_feature_riscv_mask(
        tt::llrt::RunTimeDebugFeatureDisableL1DataCache);

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
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_brisc.ld ";
            } else {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_brisc.ld ";
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
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_ncrisc.ld ";
            } else {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_ncrisc.ld ";
            }

            break;
    }

    this->process_defines_at_compile = true;

    finish_init();
}

JitBuildCompute::JitBuildCompute(const JitBuildEnv& env, const JitBuiltStateConfig& build_config) :
    JitBuildState(env, build_config) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 3, "Invalid compute processor");
    this->lflags_ = env.lflags_;
    this->cflags_ = env.cflags_;
    this->default_compile_opt_level_ = "O3";
    this->default_linker_opt_level_ = "O3";
    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    this->defines_ = env_.defines_;
    uint32_t l1_cache_disable_mask = tt::llrt::RunTimeOptions::get_instance().get_feature_riscv_mask(
        tt::llrt::RunTimeDebugFeatureDisableL1DataCache);
    uint32_t debug_compute_mask =
        (tt::llrt::DebugHartFlags::RISCV_TR0 | tt::llrt::DebugHartFlags::RISCV_TR1 |
         tt::llrt::DebugHartFlags::RISCV_TR2);
    if ((l1_cache_disable_mask & debug_compute_mask) == debug_compute_mask) {
        this->defines_ += "-DDISABLE_L1_DATA_CACHE ";
    }

    // clang-format off
    this->includes_ = env_.includes_ +
        "-I" + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/common " +
        "-I" + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_io " +
        "-I" + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_api " +
        "-I" + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_api/llk_sfpu " +
        "-I" + env_.root_ + "runtime/sfpi/include " +
        "-I" + env_.root_ + "tt_metal/hw/firmware/src " +
        "-I" + env_.root_ + "tt_metal/third_party/tt_llk/tt_llk_" + env.arch_name_ + "/llk_lib ";
    // clang-format on

    if (this->is_fw_) {
        this->srcs_.push_back("tt_metal/hw/firmware/src/trisc.cc");
    } else {
        this->srcs_.push_back("tt_metal/hw/firmware/src/trisck.cc");
    }

    switch (this->core_id_) {
        case 0:
            this->target_name_ = "trisc0";

            this->defines_ += "-DUCK_CHLKC_UNPACK ";
            this->defines_ += "-DNAMESPACE=chlkc_unpack ";
            this->defines_ += "-DCOMPILE_FOR_TRISC=0 ";

            if (this->is_fw_) {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_trisc0.ld ";
            } else {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_trisc0.ld ";
            }

            break;

        case 1:
            this->target_name_ = "trisc1";

            this->defines_ += "-DUCK_CHLKC_MATH ";
            this->defines_ += "-DNAMESPACE=chlkc_math ";
            this->defines_ += "-DCOMPILE_FOR_TRISC=1 ";

            if (this->is_fw_) {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_trisc1.ld ";
            } else {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_trisc1.ld ";
            }

            break;

        case 2:
            this->target_name_ = "trisc2";

            this->defines_ += "-DUCK_CHLKC_PACK ";
            this->defines_ += "-DNAMESPACE=chlkc_pack ";
            this->defines_ += "-DCOMPILE_FOR_TRISC=2 ";

            if (this->is_fw_) {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_trisc2.ld ";
            } else {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_trisc2.ld ";
            }

            break;
    }

    this->process_defines_at_compile = false;

    finish_init();
}

JitBuildActiveEthernet::JitBuildActiveEthernet(const JitBuildEnv& env, const JitBuiltStateConfig& build_config) :
    JitBuildState(env, build_config) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 1, "Invalid active ethernet processor");
    this->lflags_ = env.lflags_;
    this->cflags_ = env.cflags_;
    this->default_compile_opt_level_ = "Os";
    this->default_linker_opt_level_ = "Os";
    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    // clang-format off
    this->includes_ = env_.includes_ +
        "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/common " +
        "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_io " +
        "-I " + env_.root_ + "tt_metal/hw/inc/ethernet ";
    // clang-format on

    this->defines_ = env_.defines_;
    uint32_t l1_cache_disable_mask = tt::llrt::RunTimeOptions::get_instance().get_feature_riscv_mask(
        tt::llrt::RunTimeDebugFeatureDisableL1DataCache);
    uint32_t erisc_mask = (tt::llrt::DebugHartFlags::RISCV_ER0 | tt::llrt::DebugHartFlags::RISCV_ER1);
    if ((l1_cache_disable_mask & erisc_mask) == erisc_mask) {
        this->defines_ += "-DDISABLE_L1_DATA_CACHE ";
    }

    // 0: core_id = 0 and not cooperative
    // 1: core_id = 0 and cooperative
    uint32_t build_class = (this->core_id_ << 1) | uint32_t(build_config.is_cooperative);

    switch (build_class) {
        case 0: {
            this->target_name_ = "active_erisc";
            this->cflags_ = env_.cflags_ + "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops

            this->defines_ +=
                "-DCOMPILE_FOR_ERISC "
                "-DERISC "
                "-DRISC_B0_HW ";

            this->includes_ += "-I " + env_.root_ + "tt_metal/hw/firmware/src ";

            if (this->is_fw_) {
                this->srcs_.push_back("tt_metal/hw/firmware/src/active_erisc.cc");
            } else {
                this->srcs_.push_back("tt_metal/hw/firmware/src/active_erisck.cc");
            }

            if (this->is_fw_) {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_aerisc.ld ";
            } else {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_aerisc.ld ";
            }

            break;
        }
        case 1: {
            this->target_name_ = "erisc";
            this->cflags_ = env_.cflags_ + " -fno-delete-null-pointer-checks ";

            if (tt::llrt::RunTimeOptions::get_instance().get_erisc_iram_enabled()) {
                this->defines_ += "-DENABLE_IRAM ";
            }
            this->defines_ +=
                "-DCOMPILE_FOR_ERISC "
                "-DERISC "
                "-DRISC_B0_HW "
                "-DCOOPERATIVE_ERISC ";

            this->includes_ += "-I " + env_.root_ + "tt_metal/hw/inc/ethernet ";

            if (this->is_fw_) {
                this->srcs_.push_back("tt_metal/hw/firmware/src/erisc.cc");
                this->srcs_.push_back("tt_metal/hw/firmware/src/erisc-crt0.cc");
            } else {
                this->srcs_.push_back("tt_metal/hw/firmware/src/erisck.cc");
            }

            string linker_str;
            if (this->is_fw_) {
                if (tt::llrt::RunTimeOptions::get_instance().get_erisc_iram_enabled()) {
                    linker_str = "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/erisc-b0-app_iram.ld ";
                } else {
                    linker_str = "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/erisc-b0-app.ld ";
                }
            } else {
                if (tt::llrt::RunTimeOptions::get_instance().get_erisc_iram_enabled()) {
                    linker_str = "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/erisc-b0-kernel_iram.ld ";
                } else {
                    linker_str = "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/erisc-b0-kernel.ld ";
                }
            }
            this->lflags_ = env_.lflags_ + "-L" + env_.root_ +
                            "/tt_metal/hw/toolchain "
                            "-T" +
                            env_.root_ + linker_str;

            break;
        }
        default:
            TT_THROW(
                "Invalid processor ID {} and cooperative scheme {} for Active Ethernet core.",
                this->core_id_,
                build_config.is_cooperative);
    }
    this->process_defines_at_compile = true;

    finish_init();
}

JitBuildIdleEthernet::JitBuildIdleEthernet(const JitBuildEnv& env, const JitBuiltStateConfig& build_config) :
    JitBuildState(env, build_config) {
    TT_ASSERT(this->core_id_ >= 0 && this->core_id_ < 2, "Invalid idle ethernet processor");
    this->lflags_ = env.lflags_;
    this->cflags_ = env.cflags_;
    this->default_compile_opt_level_ = "Os";
    this->default_linker_opt_level_ = "Os";
    this->out_path_ = this->is_fw_ ? env_.out_firmware_root_ : env_.out_kernel_root_;

    // clang-format off
    this->includes_ = env_.includes_ +
        "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/common " +
        "-I " + env_.root_ + "tt_metal/hw/ckernels/" + env.arch_name_ + "/metal/llk_io ";
    // clang-format on

    this->defines_ = env_.defines_;
    uint32_t l1_cache_disable_mask = tt::llrt::RunTimeOptions::get_instance().get_feature_riscv_mask(
        tt::llrt::RunTimeDebugFeatureDisableL1DataCache);
    uint32_t erisc_mask = (tt::llrt::DebugHartFlags::RISCV_ER0 | tt::llrt::DebugHartFlags::RISCV_ER1);
    if ((l1_cache_disable_mask & erisc_mask) == erisc_mask) {
        this->defines_ += "-DDISABLE_L1_DATA_CACHE ";
    }

    switch (this->core_id_) {
        case 0: {
            this->target_name_ = "idle_erisc";
            this->cflags_ = env_.cflags_ + "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops

            this->defines_ +=
                "-DCOMPILE_FOR_IDLE_ERISC=0 "
                "-DERISC "
                "-DRISC_B0_HW ";  // do we need this for BH?

            this->includes_ += "-I " + env_.root_ + "tt_metal/hw/firmware/src ";

            if (this->is_fw_) {
                this->srcs_.push_back("tt_metal/hw/firmware/src/idle_erisc.cc");
            } else {
                this->srcs_.push_back("tt_metal/hw/firmware/src/idle_erisck.cc");
            }

            if (this->is_fw_) {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_ierisc.ld ";
            } else {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_ierisc.ld ";
            }

            break;
        }
        case 1: {
            this->target_name_ = "slave_idle_erisc";
            this->cflags_ = env_.cflags_ + "-fno-tree-loop-distribute-patterns ";  // don't use memcpy for cpy loops
            this->defines_ +=
                "-DCOMPILE_FOR_IDLE_ERISC=1 "
                "-DERISC "
                "-DRISC_B0_HW ";
            this->includes_ += "-I " + env_.root_ + "tt_metal/hw/firmware/src ";
            if (this->is_fw_) {
                this->srcs_.push_back("tt_metal/hw/firmware/src/slave_idle_erisc.cc");
            } else {
                this->srcs_.push_back("tt_metal/hw/firmware/src/idle_erisck.cc");
            }
            if (this->is_fw_) {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/firmware_slave_ierisc.ld ";
            } else {
                this->lflags_ +=
                    "-T" + env_.root_ + "runtime/hw/toolchain/" + get_alias(env_.arch_) + "/kernel_slave_ierisc.ld ";
            }
            break;
        }
        default: TT_THROW("Invalid processor ID {} for Idle Ethernet core.", this->core_id_);
    }
    this->process_defines_at_compile = true;

    finish_init();
}

void JitBuildState::compile_one(
    const string& log_file,
    const string& out_dir,
    const JitBuildSettings* settings,
    const string& src,
    const string& obj) const {
    // ZoneScoped;
    fs::create_directories(out_dir);

    string cmd{"cd " + out_dir + " && " + env_.gpp_};
    string defines = this->defines_;

    if (settings) {
        // Append user args
        if (process_defines_at_compile) {
            settings->process_defines([&defines](const string& define, const string& value) {
                defines += "-D" + define + "='" + value + "' ";
            });
        }

        settings->process_compile_time_args([&defines](const std::vector<uint32_t>& values) {
            if (values.empty()) {
                return;
            }
            std::ostringstream ss;
            ss << "-DKERNEL_COMPILE_TIME_ARGS=";
            for (uint32_t i = 0; i < values.size(); ++i) {
                ss << values[i] << ",";
            }
            std::string args = ss.str();
            args.pop_back();  // Remove the trailing comma
            defines += args + " ";
        });

        cmd += fmt::format("-{} ", settings->get_compiler_opt_level());
    } else {
        cmd += fmt::format("-{} ", this->default_compile_opt_level_);
    }

    // Append common args provided by the build state
    cmd += this->cflags_;
    cmd += this->includes_;
    cmd += "-c -o " + obj + " " + src + " ";
    cmd += defines;

    log_debug(tt::LogBuildKernels, "    g++ compile cmd: {}", cmd);

    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled() && settings) {
        log_kernel_defines_and_args(out_dir, settings->get_full_kernel_name(), defines);
    }

    if (!tt::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "compile", cmd, log_file);
    }
}

void JitBuildState::compile(const string& log_file, const string& out_dir, const JitBuildSettings* settings) const {
    // ZoneScoped;
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < this->srcs_.size(); ++i) {
        launch_build_step(
            [this, &log_file, &out_dir, settings, i] {
                this->compile_one(log_file, out_dir, settings, this->srcs_[i], this->objs_[i]);
            },
            events);
    }

    sync_events(events);

    if (tt::llrt::RunTimeOptions::get_instance().get_watcher_enabled()) {
        dump_kernel_defines_and_args(env_.get_out_kernel_root_path());
    }
}

void JitBuildState::link(const string& log_file, const string& out_dir, const JitBuildSettings* settings) const {
    // ZoneScoped;
    string cmd{"cd " + out_dir + " && " + env_.gpp_};
    string lflags = this->lflags_;
    if (tt::llrt::RunTimeOptions::get_instance().get_build_map_enabled()) {
        lflags += "-Wl,-Map=" + out_dir + "linker.map ";
    }

    // Append user args
    cmd += fmt::format("-{} ", settings ? settings->get_linker_opt_level() : this->default_linker_opt_level_);

    if (!this->is_fw_) {
        string weakened_elf_name =
            env_.out_firmware_root_ + this->target_name_ + "/" + this->target_name_ + "_weakened.elf";
        cmd += "-Wl,--just-symbols=" + weakened_elf_name + " ";
    }

    // Append common args provided by the build state
    cmd += lflags;
    cmd += this->link_objs_;
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
    // ZoneScoped;

    std::string pathname_in = out_dir + target_name_ + ".elf";
    std::string pathname_out = out_dir + target_name_ + "_weakened.elf";

    ll_api::ElfFile elf;
    elf.ReadImage(pathname_in);
    static std::string_view const strong_names[] = {"__fw_export_*", "__global_pointer$"};
    elf.WeakenDataSymbols(strong_names);
    elf.WriteImage(pathname_out);
}

void JitBuildState::extract_zone_src_locations(const string& log_file) const {
    // ZoneScoped;
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
    // ZoneScoped;
    string out_dir = (settings == nullptr)
                         ? this->out_path_ + this->target_name_ + "/"
                         : this->out_path_ + settings->get_full_kernel_name() + this->target_name_ + "/";

    string log_file = out_dir + "build.log";
    if (fs::exists(log_file)) {
        std::remove(log_file.c_str());
    }
    compile(log_file, out_dir, settings);
    link(log_file, out_dir, settings);
    if (this->is_fw_) {
        weaken(log_file, out_dir);
    }

    extract_zone_src_locations(log_file);
}

void jit_build(const JitBuildState& build, const JitBuildSettings* settings) {
    // ZoneScoped;

    build.build(settings);
    write_successful_jit_build_marker(build, settings);
}

void jit_build_set(const JitBuildStateSet& build_set, const JitBuildSettings* settings) {
    // ZoneScoped;
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < build_set.size(); ++i) {
        // Capture the necessary objects by reference
        auto& build = build_set[i];
        launch_build_step([build, settings] { build->build(settings); }, events);
    }

    sync_events(events);
    for (size_t i = 0; i < build_set.size(); ++i) {
        auto& build = build_set[i];
        write_successful_jit_build_marker(*build, settings);
    }
}

void jit_build_subset(const JitBuildStateSubset& build_subset, const JitBuildSettings* settings) {
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < build_subset.size; ++i) {
        // Capture the necessary objects by reference
        auto& build = build_subset.build_ptr[i];
        launch_build_step([build, settings] { build->build(settings); }, events);
    }

    sync_events(events);
    for (size_t i = 0; i < build_subset.size; ++i) {
        auto& build = build_subset.build_ptr[i];
        write_successful_jit_build_marker(*build, settings);
    }
}

void launch_build_step(const std::function<void()>& build_func, std::vector<std::shared_future<void>>& events) {
    events.emplace_back(detail::async(build_func));
}

}  // namespace tt::tt_metal
