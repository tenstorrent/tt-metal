// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "build.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include <enchantum/enchantum.hpp>
#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <taskflow/core/async.hpp>

#include <tt_stl/assert.hpp>
#include "common/executor.hpp"
#include "env_lib.hpp"
#include "hal_types.hpp"
#include "impl/context/metal_context.hpp"
#include "jit_build/kernel_args.hpp"
#include "jit_build/depend.hpp"
#include "jit_build_settings.hpp"
#include "jit_build_utils.hpp"
#include <tt-logger/tt-logger.hpp>
#include "profiler_paths.hpp"
#include "profiler_state.hpp"
#include "tt_cluster.hpp"
#include "tt_metal/llrt/tt_elffile.hpp"
#include <umd/device/types/arch.hpp>

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

namespace {

void build_failure(const string& target_name, const string& op, const string& cmd, const string& log_file) {
    log_error(tt::LogBuildKernels, "{} {} failure -- cmd: {}", target_name, op, cmd);
    std::ifstream file{log_file};
    if (file.is_open()) {
        std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        TT_THROW("{} build failed. Log: {}", target_name, log_contents);
    } else {
        TT_THROW("Failed to open {} failure log file {}", op, log_file);
    }
}

void write_successful_jit_build_marker(const JitBuildState& build, const JitBuildSettings* settings) {
    const string out_dir = (settings == nullptr) ? build.get_out_path() + "/"
                                                 : build.get_out_path() + settings->get_full_kernel_name() + "/";
    std::ofstream file(out_dir + SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME);
}

void check_built_dir(const std::filesystem::path& dir_path, const std::filesystem::path& git_hash_path) {
    if (dir_path.compare(git_hash_path) != 0) {
        std::filesystem::remove_all(dir_path);
    }
}

}  // namespace

std::string get_default_root_path() {
    const std::string emptyString;
    const std::string home_path = parse_env<std::string>("HOME", emptyString);
    if (!home_path.empty() && std::filesystem::exists(home_path)) {
        return home_path + "/.cache/tt-metal-cache/";
    } else {
        return "/tmp/tt-metal-cache/";
    }
}

JitBuildEnv::JitBuildEnv() = default;

void JitBuildEnv::init(
    uint64_t build_key,
    size_t fw_compile_hash,
    tt::ARCH arch,
    const std::map<std::string, std::string>& device_kernel_defines) {
    // Paths
    const auto& rtoptions = tt_metal::MetalContext::instance().rtoptions();
    this->root_ = rtoptions.get_root_dir();
    this->out_root_ = rtoptions.is_cache_dir_specified() ? rtoptions.get_cache_dir() : get_default_root_path();

    this->arch_ = arch;

#ifndef GIT_COMMIT_HASH
    log_info(tt::LogBuildKernels, "GIT_COMMIT_HASH not found");
#else
    std::string git_hash(GIT_COMMIT_HASH);

    std::filesystem::path git_hash_path(this->out_root_ + git_hash);
    std::filesystem::path root_path(this->out_root_);
    if ((not rtoptions.get_skip_deleting_built_cache()) && std::filesystem::exists(root_path)) {
        std::ranges::for_each(
            std::filesystem::directory_iterator{root_path},
            [&git_hash_path](const auto& dir_entry) { check_built_dir(dir_entry.path(), git_hash_path); });
    } else {
        log_info(tt::LogBuildKernels, "Skipping deleting built cache");
    }

    this->out_root_ = this->out_root_  + git_hash + "/";
#endif

    // Tools
    const static bool use_ccache = std::getenv("TT_METAL_CCACHE_KERNEL_SUPPORT") != nullptr;
    if (use_ccache) {
        this->gpp_ = "ccache ";
    } else {
        this->gpp_ = "";
    }

    // Use local sfpi for development
    // Use system sfpi for production to avoid packaging it
    // Ordered by precedence
    const std::array<std::string, 2> sfpi_roots = {
        this->root_ + "runtime/sfpi",
        "/opt/tenstorrent/sfpi"
    };

    bool sfpi_found = false;
    for (unsigned i = 0; i < 2; ++i) {
        auto gxx = sfpi_roots[i] + "/compiler/bin/riscv-tt-elf-g++";
        if (std::filesystem::exists(gxx)) {
            this->gpp_ += gxx + " ";
            this->gpp_include_dir_ = sfpi_roots[i] + "/include";
            log_debug(tt::LogBuildKernels, "Using {} sfpi at {}", i ? "system" : "local", sfpi_roots[i]);
            sfpi_found = true;
            break;
        }
    }
    if (!sfpi_found) {
        TT_THROW("sfpi not found at {} or {}", sfpi_roots[0], sfpi_roots[1]);
    }

    // Check for LLVM compiler override (will be used for kernels only, not firmware)
    const char* compiler_override = std::getenv("TT_METAL_KERNEL_COMPILER");
    bool llvm_requested =
        (compiler_override != nullptr &&
         (std::string(compiler_override) == "llvm" || std::string(compiler_override) == "clang"));

    if (llvm_requested) {
        std::string llvm_path = "/usr/lib/llvm-17/bin/clang++";
        if (std::filesystem::exists(llvm_path)) {
            // Store LLVM path for later use with kernels (not firmware)
            this->gpp_llvm_ = (use_ccache ? "ccache " : "") + llvm_path + " ";
            log_info(tt::LogBuildKernels, "LLVM/Clang available at {} for compilation", llvm_path);
            log_warning(
                tt::LogBuildKernels,
                "LLVM mode: Will use LLVM for firmware and dispatch kernels. SFPU/compute kernels will use GCC. (ELF loader debugging in progress)");
        } else {
            TT_THROW("LLVM compiler requested but not found at {}", llvm_path);
        }
    }

    // Flags
    string common_flags = "-std=c++17 -flto=auto -ffast-math -fno-exceptions ";

    if (rtoptions.get_riscv_debug_info_enabled()) {
        common_flags += "-g ";
    }

    this->cflags_ = common_flags;
    this->cflags_ +=
        "-MMD "
        "-fno-use-cxa-atexit "
        "-Wall -Werror -Wno-unknown-pragmas "
        "-Wno-deprecated-declarations "
        "-Wno-error=multistatement-macros -Wno-error=parentheses "
        "-Wno-error=unused-but-set-variable -Wno-unused-variable "
        "-Wno-unused-function ";

    // Set up LLVM flags if requested (for both kernels and firmware)
    if (llvm_requested && !this->gpp_llvm_.empty()) {
        // Use LTO to reduce code size (needed to stay within firmware size limits with -mno-relax)
        // Firmware exports symbols to kernels, so both must use the same compiler/ABI
        // Use -Os (optimize for size) to compensate for -mno-relax bloat
        string llvm_common_flags = "-std=c++17 -Os -flto -ffast-math -fno-exceptions ";
        // Match GCC's -mcpu=tt-bh which uses rv32im (without 'c' compressed extension)
        // This avoids relocation alignment issues caused by 2-byte compressed instructions
        llvm_common_flags += "-target riscv32 -march=rv32im -mabi=ilp32 ";
        llvm_common_flags += "-ffreestanding -nostdlib ";
        // Disable PIE and PLT to avoid dynamic relocations (R_RISCV_CALL_PLT)
        llvm_common_flags += "-fno-pic -fno-pie -fno-plt ";
        // Disable linker relaxation - the ELF loader expects lui instructions for R_RISCV_HI20 relocations
        llvm_common_flags += "-mno-relax ";
        // Ensure volatile semantics are preserved and prevent strict aliasing optimizations
        // This is critical for dispatch kernels that poll volatile semaphore addresses
        llvm_common_flags += "-fno-strict-aliasing ";
        // Enable function/data sections so linker can garbage-collect unused/empty sections
        llvm_common_flags += "-ffunction-sections -fdata-sections ";

        // Add GCC's RISC-V toolchain headers so LLVM can find <cstdint>, <unistd.h>, etc.
        // Use the same SFPI path we found for GCC
        if (!this->gpp_include_dir_.empty()) {
            // Extract the SFPI root from gpp_include_dir_ (removes "/include" suffix)
            std::string sfpi_root = this->gpp_include_dir_.substr(0, this->gpp_include_dir_.rfind("/include"));
            llvm_common_flags += "-isystem " + sfpi_root + "/compiler/riscv-tt-elf/include/c++/15.1.0 ";
            llvm_common_flags += "-isystem " + sfpi_root + "/compiler/riscv-tt-elf/include/c++/15.1.0/riscv-tt-elf ";
            llvm_common_flags += "-isystem " + sfpi_root + "/compiler/riscv-tt-elf/include ";
        }

        if (rtoptions.get_riscv_debug_info_enabled()) {
            llvm_common_flags += "-g ";
        }

        this->cflags_llvm_ = llvm_common_flags;
        this->cflags_llvm_ +=
            "-MMD "
            "-fno-use-cxa-atexit "
            "-Wall -Werror -Wno-unknown-pragmas "
            "-Wno-deprecated-declarations "
            "-Wno-error=parentheses "
            "-Wno-unused-variable "
            "-Wno-unused-function "
            "-Wno-unknown-attributes "  // LLVM doesn't know GCC's custom attributes like rvtt_l1_ptr
            "-Wno-microsoft-anon-tag "
            "-Wno-empty-body "       // LLVM complains about empty while loops
            "-Wno-unused-but-set-variable "  // Handled with [[maybe_unused]] in source code
            "-Wno-constant-logical-operand "  // Dispatch kernels use constant && in conditionals
            "-falign-functions=4 ";  // Force 4-byte alignment to avoid misaligned relocations

        // Linker uses the same base flags (with LTO to resolve internal calls)
        this->lflags_llvm_ = llvm_common_flags;
        this->lflags_llvm_ += "-Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";
        // Use --gc-sections to reduce code size
        this->lflags_llvm_ += "-Wl,--gc-sections ";
        // Force the linker to keep the .data and .bss sections by marking their symbols as undefined/required
        // This ensures empty sections still get PT_LOAD segments as the ELF loader expects
        this->lflags_llvm_ += "-Wl,--undefined=__ldm_data_start -Wl,--undefined=__ldm_bss_start ";
        // Force static linking mode and resolve all calls directly (no PLT)
        this->lflags_llvm_ += "-static -Wl,--no-dynamic-linker -Wl,--no-undefined -Wl,-Bstatic ";
        // Link with GCC's compiler runtime library for functions like __ashldi3
        // Use the blackhole variant (bh-ilp32) to match our target
        // gpp_include_dir_ is /path/to/sfpi/include, so go to ../compiler/lib/gcc/...
        std::string sfpi_root = this->gpp_include_dir_.substr(0, this->gpp_include_dir_.rfind("/include"));
        std::string libgcc_path = sfpi_root + "/compiler/lib/gcc/riscv-tt-elf/15.1.0/bh-ilp32";
        // Use LLVM's own linker (ld.lld) which properly handles LLVM-compiled objects and libraries
        // This avoids ABI/symbol incompatibilities when mixing GCC and LLVM
        this->lflags_llvm_ += "-fuse-ld=lld ";
        // Link with libgcc and libc for runtime support
        // Note: watcher's atomic operations (__atomic_load_1) are not available with LLVM currently
        // Order matters: substitutes.o provides atexit/exit, libc provides setjmp
        // We link libc AFTER object files so linker prefers substitutes.o implementations
        std::string libc_path = sfpi_root + "/compiler/riscv-tt-elf/lib/bh-ilp32";
        this->lflags_llvm_ += "-L" + libgcc_path + " -L" + libc_path + " -lgcc -Wl,--strip-debug ";
    }

    // Defines
    this->defines_ = "";
    for (auto it = device_kernel_defines.begin(); it != device_kernel_defines.end(); ++it) {
        this->defines_ += "-D" + it->first + "=" + it->second + " ";
    }
    this->defines_ += "-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 ";

    if (tt::tt_metal::getDeviceProfilerState()) {
        uint32_t profiler_options = 1;
        if (rtoptions.get_profiler_do_dispatch_cores()) {
            profiler_options |= PROFILER_OPT_DO_DISPATCH_CORES;
        }
        if (rtoptions.get_profiler_trace_only()) {
            profiler_options |= PROFILER_OPT_DO_TRACE_ONLY;
        }
        this->defines_ += "-DPROFILE_KERNEL=" + std::to_string(profiler_options) + " ";
    }
    if (rtoptions.get_profiler_noc_events_enabled()) {
        // force profiler on if noc events are being profiled
        if (not tt::tt_metal::getDeviceProfilerState()) {
            this->defines_ += "-DPROFILE_KERNEL=1 ";
        }
        this->defines_ += "-DPROFILE_NOC_EVENTS=1 ";
    }

    if (rtoptions.get_watcher_enabled()) {
        this->defines_ += "-DWATCHER_ENABLED ";
    }
    if (rtoptions.get_watcher_noinline()) {
        this->defines_ += "-DWATCHER_NOINLINE ";
    }
    if (rtoptions.get_watcher_noc_sanitize_linked_transaction()) {
        this->defines_ += "-DWATCHER_ENABLE_NOC_SANITIZE_LINKED_TRANSACTION ";
    }
    for (auto& feature : rtoptions.get_watcher_disabled_features()) {
        this->defines_ += "-DWATCHER_DISABLE_" + feature + " ";
    }

    if (rtoptions.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        this->defines_ += "-DDEBUG_PRINT_ENABLED ";
    }

    if (rtoptions.get_record_noc_transfers()) {
        this->defines_ += "-DNOC_LOGGING_ENABLED ";
    }

    if (rtoptions.get_kernels_nullified()) {
        this->defines_ += "-DDEBUG_NULL_KERNELS ";
    }

    if (rtoptions.get_kernels_early_return()) {
        this->defines_ += "-DDEBUG_EARLY_RETURN_KERNELS ";
    }

    if (rtoptions.get_watcher_debug_delay()) {
        this->defines_ += "-DWATCHER_DEBUG_DELAY=" + to_string(rtoptions.get_watcher_debug_delay()) + " ";
    }

    if (rtoptions.get_hw_cache_invalidation_enabled()) {
        this->defines_ += "-DENABLE_HW_CACHE_INVALIDATION ";
    }

    if (rtoptions.get_relaxed_memory_ordering_disabled()) {
        this->defines_ += "-DDISABLE_RELAXED_MEMORY_ORDERING ";
    }

    if (rtoptions.get_gathering_enabled()) {
        this->defines_ += "-DENABLE_GATHERING ";
    }

    if (tt::tt_metal::MetalContext::instance().get_cluster().is_base_routing_fw_enabled()) {
        this->defines_ += "-DROUTING_FW_ENABLED ";
    }

    // Includes
    // TODO(pgk) this list is insane
    std::vector<std::string> includeDirs = {
        ".",
        "..",
        root_,
        root_ + "ttnn",
        root_ + "ttnn/cpp",
        root_ + "tt_metal",
        root_ + "tt_metal/include",
        root_ + "tt_metal/hw/inc",
        root_ + "tt_metal/hostdevcommon/api",
        root_ + "tt_metal/hw/inc/debug",
        root_ + "tt_metal/api/",
        root_ + "tt_metal/api/tt-metalium/"};

    std::ostringstream oss;
    for (size_t i = 0; i < includeDirs.size(); ++i) {
        oss << "-I" << includeDirs[i] << " ";
    }
    this->includes_ = oss.str();

    this->lflags_ = common_flags;
    this->lflags_ += "-Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";

    // Need to capture more info in build key to prevent stale binaries from being reused.
    jit_build::utils::FNV1a hasher;
    hasher.update(build_key);
    hasher.update(enchantum::to_underlying(this->arch_));
    hasher.update(cflags_.begin(), cflags_.end());
    hasher.update(lflags_.begin(), lflags_.end());
    hasher.update(defines_.begin(), defines_.end());
    build_key_ = hasher.digest();

    // Firmware build path is a combination of build_key and fw_compile_hash
    // If either change, the firmware build path will change and FW will be rebuilt
    // if it's not already in MetalContext::firmware_built_keys_
    this->out_firmware_root_ = fmt::format("{}{}/firmware/{}/", this->out_root_, build_key_, fw_compile_hash);
    this->out_kernel_root_ = fmt::format("{}{}/kernels/", this->out_root_, build_key_);
}

JitBuildState::JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig& build_config) :
    env_(env),
    is_fw_(build_config.is_fw),
    process_defines_at_compile_(true),
    dispatch_message_addr_(build_config.dispatch_message_addr),
    out_path_(build_config.is_fw ? env_.out_firmware_root_ : env_.out_kernel_root_),
    cflags_(env.cflags_),
    defines_(env.defines_),
    includes_(env.includes_),
    lflags_(env.lflags_),
    default_compile_opt_level_("Os"),
    default_linker_opt_level_("Os") {
    // Anything that is arch-specific should be added to HalJitBuildQueryInterface instead of here.
    if (build_config.core_type == HalProgrammableCoreType::TENSIX &&
        build_config.processor_class == HalProcessorClassType::COMPUTE) {
        this->default_compile_opt_level_ = "O3";
        this->default_linker_opt_level_ = "O3";
        this->includes_ += "-I" + env_.gpp_include_dir_ + " ";
        this->process_defines_at_compile_ = false;
    } else if (build_config.core_type == HalProgrammableCoreType::ACTIVE_ETH && build_config.is_cooperative) {
        // Only cooperative active ethernet needs "-L <root>/tt_metal/hw/toolchain",
        // because its linker script depends on some files in that directory.
        // Maybe we should move the dependencies to runtime/hw/toolchain/<arch>/?
        fmt::format_to(std::back_inserter(this->lflags_), "-L{}/tt_metal/hw/toolchain/ ", env_.root_);
    }

    HalJitBuildQueryInterface::Params params{
        this->is_fw_, build_config.core_type, build_config.processor_class, build_config.processor_id};
    const auto& jit_build_query = tt_metal::MetalContext::instance().hal().get_jit_build_query();

    this->target_name_ = jit_build_query.target_name(params);
    // Includes
    {
        auto it = std::back_inserter(this->includes_);
        for (const auto& include : jit_build_query.includes(params)) {
            fmt::format_to(it, "-I{}{} ", env_.root_, include);
        }
    }
    // Defines
    {
        auto it = std::back_inserter(this->defines_);
        for (const auto& define : jit_build_query.defines(params)) {
            fmt::format_to(it, "-D{} ", define);
        }
        fmt::format_to(it, "-DDISPATCH_MESSAGE_ADDR={} ", this->dispatch_message_addr_);
    }
    if (this->is_fw_) {
        this->defines_ += "-DFW_BUILD ";
    } else {
        this->defines_ += "-DKERNEL_BUILD ";
    }
    // Flags
    {
        auto common_flags = jit_build_query.common_flags(params);
        this->cflags_ += common_flags;
        this->lflags_ += common_flags;
    }
    this->linker_script_ = env_.root_ + jit_build_query.linker_script(params);
    this->lflags_ += fmt::format("-T{} ", this->linker_script_);
    // Source files
    {
        auto srcs = jit_build_query.srcs(params);
        this->srcs_.insert(this->srcs_.end(), std::move_iterator(srcs.begin()), std::move_iterator(srcs.end()));
    }

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
    {
        auto it = std::back_inserter(this->link_objs_);
        for (const auto& obj : jit_build_query.link_objs(params)) {
            fmt::format_to(it, "{}{} ", env_.root_, obj);
        }
    }

    // Note the preceding slash which defies convention as this gets appended to
    // the kernel name used as a path which doesn't have a slash
    this->target_full_path_ = "/" + this->target_name_ + "/" + this->target_name_ + ".elf";

    if (not this->is_fw_) {
        // Emit relocations, so we can relocate the resulting binary
        this->lflags_ += "-Wl,--emit-relocs ";
    }
}

void JitBuildState::compile_one(
    const string& out_dir, const JitBuildSettings* settings, const string& src, const string& obj) const {
    // ZoneScoped;

    // Choose compiler: LLVM for firmware and dispatch kernels
    // SFPU/compute kernels (trisc) require GCC SFPU builtins
    [[maybe_unused]] bool is_firmware_build = this->is_fw_;
    bool is_trisc_kernel = (src.find("trisck.cc") != std::string::npos) || (src.find("trisc.cc") != std::string::npos);
    // Use LLVM for all non-SFPU builds (firmware + dispatch + user kernels)
    bool use_llvm_for_this_build = !is_trisc_kernel && !env_.gpp_llvm_.empty();
    string compiler = use_llvm_for_this_build ? env_.gpp_llvm_ : env_.gpp_;

    // For LLVM, start with LLVM base flags and add HAL flags (filtering GCC-specific ones)
    // For GCC, use the already-configured cflags
    string cflags;
    if (use_llvm_for_this_build) {
        cflags = env_.cflags_llvm_;
        // Add HAL-specific flags from this->cflags_, but filter out GCC-only flags
        string hal_flags = this->cflags_;
        // Remove GCC-specific flags that LLVM doesn't support
        // Need to handle flags with values (e.g. -Werror=stack-usage=1912) by finding the full flag
        std::vector<std::string> gcc_only_flag_prefixes = {
            "-mcpu=tt-bh",
            "-mcpu=tt-wh",
            "-mcpu=tt-gs",
            "-mno-tt",              // GCC-specific Tensix flags (e.g., -mno-tt-tensix-optimize-replay)
            "-fno-rvtt-sfpu-replay",
            "-fno-tree-loop-distribute-patterns",
            "-flto=auto",  // LLVM uses -flto without =auto
            "-Werror=multistatement-macros",
            "-Wno-error=multistatement-macros",
            "-Werror=unused-but-set-variable",
            "-Wno-error=unused-but-set-variable",
            "-Werror=stack-usage",  // LLVM doesn't support stack-usage warning (may have =value)
            "-tensix"               // GCC-specific Tensix architecture flag
        };
        for (const auto& prefix : gcc_only_flag_prefixes) {
            size_t pos = hal_flags.find(prefix);
            while (pos != std::string::npos) {
                // Find the end of this flag (next space or end of string)
                size_t end_pos = hal_flags.find(' ', pos);
                if (end_pos == std::string::npos) {
                    end_pos = hal_flags.length();
                }
                hal_flags.erase(pos, end_pos - pos + 1);  // +1 to include the space
                pos = hal_flags.find(prefix);
            }
        }
        cflags += hal_flags;
    } else {
        cflags = this->cflags_;
    }

    string cmd{"cd " + out_dir + " && " + compiler};
    string defines = this->defines_;

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_build_map_enabled()) {
        cmd += "-save-temps=obj -fdump-tree-all -fdump-rtl-all ";
    }

    if (settings) {
        // Append user args
        if (process_defines_at_compile_) {
            settings->process_defines([&defines](const string& define, const string& value) {
                defines += fmt::format("-D{}='{}' ", define, value);
            });
        }

        settings->process_compile_time_args([&defines](const std::vector<uint32_t>& values) {
            if (values.empty()) {
                return;
            }
            defines += fmt::format("-DKERNEL_COMPILE_TIME_ARGS={} ", fmt::join(values, ","));
        });

        // This creates a command-line define for named compile time args
        // Ex. for named_args like {"buffer_size": 1024, "num_tiles": 64}
        // This generates:
        // -DKERNEL_COMPILE_TIME_ARG_MAP="{{\"buffer_size\",1024}, {\"num_tiles\",64}} "
        // The macro expansion is defined in tt_metal/hw/inc/compile_time_args.h
        settings->process_named_compile_time_args(
            [&defines](const std::unordered_map<std::string, uint32_t>& named_args) {
                if (named_args.empty()) {
                    return;
                }
                std::ostringstream ss;
                ss << "-DKERNEL_COMPILE_TIME_ARG_MAP=\"";
                for (const auto& [name, value] : named_args) {
                    ss << "{\\\"" << name << "\\\"," << value << "}, ";
                }
                ss << "\"";
                defines += ss.str() + " ";
            });

        cmd += fmt::format("-{} ", settings->get_compiler_opt_level());
    } else {
        cmd += fmt::format("-{} ", this->default_compile_opt_level_);
    }

    // Append common args provided by the build state
    cmd += cflags;
    cmd += this->includes_;
    cmd += fmt::format("-c -o {} {} ", obj, src);
    cmd += defines;

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_log_kernels_compilation_commands()) {
        log_info(tt::LogBuildKernels, "    g++ compile cmd: {}", cmd);
    }

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled() && settings) {
        log_kernel_defines_and_args(out_dir, settings->get_full_kernel_name(), defines);
    }

    std::string log_file = out_dir + obj + ".log";
    fs::remove(log_file);
    if (!tt::jit_build::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "compile", cmd, log_file);
    }
    jit_build::write_dependency_hashes(out_dir, obj);
}

bool JitBuildState::need_compile(const string& out_dir, const string& obj) const {
    return MetalContext::instance().rtoptions().get_force_jit_compile() || !fs::exists(out_dir + obj) ||
           !jit_build::dependencies_up_to_date(out_dir, obj);
}

size_t JitBuildState::compile(const string& out_dir, const JitBuildSettings* settings) const {
    // ZoneScoped;
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < this->srcs_.size(); ++i) {
        if (need_compile(out_dir, this->objs_[i])) {
            launch_build_step(
                [this, &out_dir, settings, i] { this->compile_one(out_dir, settings, this->srcs_[i], this->objs_[i]); },
                events);
        } else {
            log_debug(tt::LogBuildKernels, "JIT build cache hit: {}{}", out_dir, this->objs_[i]);
        }
    }

    sync_events(events);

    if (tt::tt_metal::MetalContext::instance().rtoptions().get_watcher_enabled()) {
        dump_kernel_defines_and_args(env_.get_out_kernel_root_path());
    }
    return events.size();
}

bool JitBuildState::need_link(const string& out_dir) const {
    std::string elf_path = out_dir + this->target_name_ + ".elf";
    return !fs::exists(elf_path) || !jit_build::dependencies_up_to_date(out_dir, elf_path);
}

void JitBuildState::link(const string& out_dir, const JitBuildSettings* settings) const {
    // ZoneScoped;
    // Choose linker: MUST match compile_one logic!
    // Use LLVM linker for firmware and dispatch kernels, GCC for SFPU/compute (trisc)
    [[maybe_unused]] bool is_firmware_build = this->is_fw_;
    bool is_trisc_kernel = (out_dir.find("/trisc") != std::string::npos);
    // Use LLVM linker if we used LLVM compiler (for firmware and dispatch kernels)
    bool use_llvm_for_this_build = !is_trisc_kernel && !env_.gpp_llvm_.empty();
    string compiler = use_llvm_for_this_build ? env_.gpp_llvm_ : env_.gpp_;

    // Choose linker flags based on compiler
    string lflags;
    if (use_llvm_for_this_build) {
        // LLVM linker flags: use LLVM's own linker (ld.lld) via clang++ driver
        lflags = env_.lflags_llvm_;
        // Add the linker script (not included in env_.lflags_llvm_ by default)
        lflags += fmt::format("-T{} ", this->linker_script_);
        // For kernels (not firmware), emit relocations so we can relocate the binary
        if (!this->is_fw_) {
            lflags += "-Wl,--emit-relocs ";
        }
    } else {
        // GCC linker flags (already includes linker script)
        lflags = this->lflags_;
    }

    string cmd{"cd " + out_dir + " && " + compiler};
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_build_map_enabled()) {
        lflags += "-Wl,-Map=" + out_dir + this->target_name_ + ".map ";
        lflags += "-save-temps=obj -fdump-tree-all -fdump-rtl-all ";
    }

    // Append user args
    cmd += fmt::format("-{} ", settings ? settings->get_linker_opt_level() : this->default_linker_opt_level_);

    // Elf file has dependencies other than object files:
    // 1. Linker script
    // 2. Weakened firmware elf (for kernels)
    std::vector<std::string> link_deps = {this->linker_script_};
    if (!this->is_fw_) {
        std::string weakened_elf = weakened_firmeware_elf_name();
        cmd += "-Wl,--just-symbols=" + weakened_elf + " ";
        link_deps.push_back(weakened_elf);
    }

    // Append common args provided by the build state
    cmd += lflags;
    cmd += this->link_objs_;
    // For LLVM, add -lc after object files so linker prefers substitutes.o implementations
    if (use_llvm_for_this_build) {
        cmd += "-lc ";
    }
    std::string elf_name = out_dir + this->target_name_ + ".elf";
    cmd += "-o " + elf_name;
    if (tt::tt_metal::MetalContext::instance().rtoptions().get_log_kernels_compilation_commands()) {
        log_info(tt::LogBuildKernels, "    g++ link cmd: {}", cmd);
    }
    std::string log_file = elf_name + ".log";
    fs::remove(log_file);
    if (!tt::jit_build::utils::run_command(cmd, log_file, false)) {
        build_failure(this->target_name_, "link", cmd, log_file);
    }
    std::string hash_path = elf_name + ".dephash";
    std::ofstream hash_file(hash_path);
    jit_build::write_dependency_hashes({{elf_name, std::move(link_deps)}}, out_dir, elf_name, hash_file);
    hash_file.close();
    if (hash_file.fail()) {
        // Don't leave incomplete hash file
        std::filesystem::remove(hash_path);
    }
}

// Given this elf (A) and a later elf (B):
// weakens symbols in A so that it can be used as a "library" for B. B imports A's weakened symbols, B's symbols of the
// same name don't result in duplicate symbols but B can reference A's symbols. Force the fw_export symbols to remain
// strong so to propogate link addresses
void JitBuildState::weaken(const string& out_dir) const {
    // ZoneScoped;

    std::string pathname_in = out_dir + target_name_ + ".elf";
    std::string pathname_out = out_dir + target_name_ + "_weakened.elf";

    ll_api::ElfFile elf;
    elf.ReadImage(pathname_in);
    static std::string_view const strong_names[] = {"__fw_export_*", "__global_pointer$"};
    elf.WeakenDataSymbols(strong_names);
    elf.WriteImage(pathname_out);
}

std::string JitBuildState::weakened_firmeware_elf_name() const {
    return fmt::format("{}{}/{}_weakened.elf", this->env_.out_firmware_root_, this->target_name_, this->target_name_);
}

void JitBuildState::extract_zone_src_locations(const std::string& out_dir) const {
    // ZoneScoped;
    static std::atomic<bool> new_log = true;
    if (tt::tt_metal::getDeviceProfilerState()) {
        if (new_log.exchange(false) && std::filesystem::exists(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG)) {
            std::remove(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG.c_str());
        }

        if (!std::filesystem::exists(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG)) {
            tt::jit_build::utils::create_file(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG);
        }

        auto cmd = fmt::format("grep KERNEL_PROFILER {}*.o.log", out_dir);
        tt::jit_build::utils::run_command(cmd, tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG, false);
    }
}

void JitBuildState::build(const JitBuildSettings* settings) const {
    // ZoneScoped;
    string out_dir = (settings == nullptr)
                         ? this->out_path_ + this->target_name_ + "/"
                         : this->out_path_ + settings->get_full_kernel_name() + this->target_name_ + "/";

    fs::create_directories(out_dir);
    if (compile(out_dir, settings) > 0 || need_link(out_dir)) {
        link(out_dir, settings);
        if (this->is_fw_) {
            weaken(out_dir);
        }
    }

    // `extract_zone_src_locations` must be called every time, because it writes to a global file
    // that gets cleared in each run.
    extract_zone_src_locations(out_dir);
}

void jit_build(const JitBuildState& build, const JitBuildSettings* settings) {
    // ZoneScoped;

    build.build(settings);
    write_successful_jit_build_marker(build, settings);
}

void jit_build_subset(JitBuildStateSubset build_subset, const JitBuildSettings* settings) {
    std::vector<std::shared_future<void>> events;
    for (auto& build : build_subset) {
        // Capture the necessary objects by reference
        launch_build_step([&build, settings] { build.build(settings); }, events);
    }

    sync_events(events);
    for (auto& build : build_subset) {
        write_successful_jit_build_marker(build, settings);
    }
}

void launch_build_step(const std::function<void()>& build_func, std::vector<std::shared_future<void>>& events) {
    events.emplace_back(detail::async(build_func));
}

void sync_build_steps(std::vector<std::shared_future<void>>& events) {
    for (auto& event : events) {
        event.wait();
    }
}

}  // namespace tt::tt_metal
