// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "build.hpp"

#include "build_cache_telemetry.hpp"
#include "jit_build_cache.hpp"
#include "jit_device_config.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <vector>

#include <unistd.h>

#include <enchantum/enchantum.hpp>
#include <fmt/base.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <taskflow/core/async.hpp>

#include <tt_stl/assert.hpp>
#include "common/executor.hpp"
#include "common/filesystem_utils.hpp"
#include "common/stable_hash.hpp"
#include "env_lib.hpp"
#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "hostdevcommon/profiler_common.h"
#include "llrt/rtoptions.hpp"
#include "jit_build/kernel_args.hpp"
#include "jit_build/depend.hpp"
#include "jit_build_settings.hpp"
#include "jit_build_utils.hpp"
#include <tt-logger/tt-logger.hpp>
#include "profiler_paths.hpp"
#include "tt_metal/llrt/tt_elffile.hpp"
#include <umd/device/types/arch.hpp>

namespace fs = std::filesystem;

using namespace std;

namespace tt::tt_metal {

namespace {

void build_failure(const string& target_name, const string& op, const string& cmd, const fs::path& log_file) {
    log_error(tt::LogBuildKernels, "{} {} failure -- cmd: {}", target_name, op, cmd);
    std::ifstream file{log_file};
    if (file.is_open()) {
        std::string log_contents((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        TT_THROW("{} build failed. Log: {}", target_name, log_contents);
    } else {
        TT_THROW("Failed to open {} failure log file {}", op, log_file);
    }
}

void write_successful_jit_build_marker(const JitBuildState& build, const JitBuildSettings* settings, bool did_work) {
    fs::path out_dir = (settings == nullptr) ? build.get_out_path() / ""
                                             : build.get_out_path() / settings->get_full_kernel_name() / "";
    fs::path marker_path = out_dir / SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME;

    if (!did_work && tt::filesystem::safe_exists(marker_path).value_or(false)) {
        return;
    }

    fs::path tmp_path = jit_build::utils::FileRenamer::generate_temp_path(marker_path);
    std::ofstream file(tmp_path);
    file.close();
    if (!file.fail()) {
        if (!tt::filesystem::safe_rename(tmp_path, marker_path, false)) {
            TT_THROW("Failed to publish successful JIT marker from {} to {}", tmp_path, marker_path);
        }
        if (tt::filesystem::nfs_safety_enabled()) {
            tt::filesystem::fsync_file(marker_path);
        }
    }
}

// NFS write safety
// -----------------
// safe_hard_link_or_copy writes directly to the destination path.  When the
// destination is on a shared NFS filesystem, callers MUST write to a unique
// temporary path first, then atomically rename to the final name.  This
// ensures concurrent multi-client writes are safe: each client writes to its
// own temp file, and the last rename wins.  Last-writer-wins is acceptable
// because all clients produce equivalent content (same source, same flags).
//
// No two clients ever call hard_link_or_copy with the same destination path.
// Each call goes to a process-unique temp path from generate_temp_path(),
// which embeds a 64-bit random per-process ID (collision probability ~2^-64).
//
// All NFS cache writes in this file follow this pattern:
//   1. generate_temp_path(final_path) -> unique temp path per process
//   2. hard_link_or_copy(source, temp_path)
//   3. safe_rename(temp_path, final_path)
bool hard_link_or_copy(const std::filesystem::path& target, const std::filesystem::path& link) {
    return tt::filesystem::safe_hard_link_or_copy(target, link);
}

// Merge build artifacts from a local scratch directory to the NFS cache.
// Copies each regular file atomically: write to a temp name, then rename.
// Returns the number of files merged, or nullopt on error.
std::optional<size_t> merge_scratch_to_cache(
    const std::filesystem::path& scratch_dir, const std::filesystem::path& cache_dir) {
    std::error_code ec;
    size_t count = 0;
    for (const auto& entry : fs::directory_iterator(scratch_dir, ec)) {
        if (ec) {
            log_warning(tt::LogBuildKernels, "Failed to iterate scratch directory {}: {}", scratch_dir, ec.message());
            return std::nullopt;
        }
        if (!entry.is_regular_file(ec)) {
            if (ec) {
                log_warning(tt::LogBuildKernels, "Failed to inspect scratch entry {}: {}", entry.path(), ec.message());
                return std::nullopt;
            }
            continue;
        }
        const auto& src = entry.path();
        fs::path dst = cache_dir / src.filename();

        auto tmp_path = jit_build::utils::FileRenamer::generate_temp_path(dst);
        if (!tt::filesystem::safe_hard_link_or_copy(src, tmp_path)) {
            log_warning(tt::LogBuildKernels, "Failed to copy scratch artifact {} to {}", src, tmp_path);
            return std::nullopt;
        }
        if (!tt::filesystem::safe_rename(tmp_path, dst, false)) {
            log_warning(tt::LogBuildKernels, "Failed to atomically rename scratch artifact {} to {}", tmp_path, dst);
            return std::nullopt;
        }
        ++count;
    }
    if (ec) {
        log_warning(tt::LogBuildKernels, "Failed to iterate scratch directory {}: {}", scratch_dir, ec.message());
        return std::nullopt;
    }
    if (count > 0) {
        BuildCacheTelemetry::inst().record_merge(count);
        log_debug(
            tt::LogBuildKernels, "Merged {} artifact(s) from scratch {} to cache {}", count, scratch_dir, cache_dir);
    }
    return count;
}

// Copy generated header/source files from the scratch kernel directory to the
// NFS cache so future cache-hit checks and non-scratch builds can find them.
// Recursively copies all regular files, preserving subdirectory structure.
// Returns the number of files merged, or nullopt on error.
std::optional<size_t> copy_genfiles_to_cache(
    const std::filesystem::path& scratch_dir, const std::filesystem::path& nfs_dir) {
    if (!tt::filesystem::safe_create_directories(nfs_dir)) {
        log_warning(tt::LogBuildKernels, "Failed to create destination genfiles directory {}", nfs_dir);
        return std::nullopt;
    }
    std::error_code ec;
    size_t count = 0;
    for (const auto& entry : fs::recursive_directory_iterator(scratch_dir, ec)) {
        if (ec) {
            log_warning(tt::LogBuildKernels, "Failed to iterate genfiles directory {}: {}", scratch_dir, ec.message());
            return std::nullopt;
        }
        if (!entry.is_regular_file(ec)) {
            if (ec) {
                log_warning(tt::LogBuildKernels, "Failed to inspect genfiles entry {}: {}", entry.path(), ec.message());
                return std::nullopt;
            }
            continue;
        }
        // Compute relative path from scratch_dir to preserve subdirectory structure
        fs::path relative_path = fs::relative(entry.path(), scratch_dir, ec);
        if (ec) {
            log_warning(tt::LogBuildKernels, "Failed to compute relative path for {}: {}", entry.path(), ec.message());
            return std::nullopt;
        }
        fs::path nfs_dest_path = nfs_dir / relative_path;
        // Create parent directories in NFS cache if needed
        if (!tt::filesystem::safe_create_directories(nfs_dest_path.parent_path())) {
            log_warning(
                tt::LogBuildKernels, "Failed to create parent destination directory {}", nfs_dest_path.parent_path());
            return std::nullopt;
        }
        auto tmp_path = jit_build::utils::FileRenamer::generate_temp_path(nfs_dest_path);
        if (!tt::filesystem::safe_hard_link_or_copy(entry.path(), tmp_path)) {
            log_warning(tt::LogBuildKernels, "Failed to copy generated file {} to {}", entry.path(), tmp_path);
            return std::nullopt;
        }
        if (!tt::filesystem::safe_rename(tmp_path, nfs_dest_path, false)) {
            log_warning(
                tt::LogBuildKernels, "Failed to atomically rename generated file {} to {}", tmp_path, nfs_dest_path);
            return std::nullopt;
        }
        ++count;
    }
    if (count > 0) {
        BuildCacheTelemetry::inst().record_genfile_merge(count);
        log_debug(tt::LogBuildKernels, "Merged {} genfile(s) from scratch {} to NFS {}", count, scratch_dir, nfs_dir);
    }
    return count;
}

}  // namespace

std::filesystem::path get_default_root_path() {
    const std::string emptyString;
    const std::string home_path = parse_env<std::string>("HOME", emptyString);
    if (!home_path.empty() && tt::filesystem::safe_exists(home_path).value_or(false)) {
        return std::filesystem::path(home_path) / ".cache" / "tt-metal-cache";
    }
    return std::filesystem::path("/tmp/tt-metal-cache");
}

JitBuildEnv::JitBuildEnv() = default;

void JitBuildEnv::init(
    uint64_t build_key,
    const JitDeviceConfig& config,
    const tt::llrt::RunTimeOptions& rtoptions,
    const std::map<std::string, std::string>& device_kernel_defines) {
    this->rtoptions_ = &rtoptions;
    // Paths
    this->root_ = fs::path(rtoptions.get_root_dir());
    this->out_root_ =
        rtoptions.is_cache_dir_specified() ? fs::path(rtoptions.get_cache_dir()) : get_default_root_path();

    this->arch_ = config.arch;
    this->max_cbs_ = config.max_cbs;

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
    const std::array<fs::path, 2> sfpi_roots = {this->root_ / "runtime/sfpi", "/opt/tenstorrent/sfpi"};

    bool sfpi_found = false;
    for (unsigned i = 0; i < 2; ++i) {
        auto gxx = (sfpi_roots[i] / "compiler/bin/riscv-tt-elf-g++").string();
        if (tt::filesystem::safe_exists(gxx).value_or(false)) {
            this->gpp_ += gxx + " ";
            this->gpp_include_dir_ = (sfpi_roots[i] / "include").string();
            log_debug(tt::LogBuildKernels, "Using {} sfpi at {}", i ? "system" : "local", sfpi_roots[i]);
            sfpi_found = true;
            break;
        }
    }
    if (!sfpi_found) {
        TT_THROW("sfpi not found at {} or {}", sfpi_roots[0], sfpi_roots[1]);
    }

    // Read the sfpi version file tracked in the repo.  This captures the
    // toolchain version (e.g. "sfpi_version='7.25.0'", "sfpi_build='252'")
    // so that upgrading sfpi invalidates the build cache.
    std::string sfpi_version_contents;
    {
        auto sfpi_version_path = this->root_ / "tt_metal/sfpi-version";
        std::ifstream ifs(sfpi_version_path);
        if (ifs.is_open()) {
            std::ostringstream oss;
            oss << ifs.rdbuf();
            sfpi_version_contents = oss.str();
        }
    }

    // Flags
    string common_flags = "-std=c++17 -flto=auto -ffast-math -fno-exceptions ";

    if (rtoptions.get_jit_analytics_enabled()) {
        common_flags += "-fdump-rtl-all -fdump-tree-original ";
    }

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

    // Defines
    this->defines_ = "";
    for (const auto& device_kernel_define : device_kernel_defines) {
        this->defines_ += "-D" + device_kernel_define.first + "=" + device_kernel_define.second + " ";
    }
    this->defines_ += "-DTENSIX_FIRMWARE -DLOCAL_MEM_EN=0 ";

    if (rtoptions.get_profiler_enabled()) {
        uint32_t profiler_options = 1;
        if (rtoptions.get_profiler_do_dispatch_cores()) {
            profiler_options |= PROFILER_OPT_DO_DISPATCH_CORES;
        }
        if (rtoptions.get_profiler_trace_only()) {
            profiler_options |= PROFILER_OPT_DO_TRACE_ONLY;
        }
        if (rtoptions.get_profiler_sum()) {
            profiler_options |= PROFILER_OPT_DO_SUM;
        }
        this->defines_ += "-DPROFILE_KERNEL=" + std::to_string(profiler_options) + " ";

        this->defines_ += "-DPROFILER_FULL_HOST_BUFFER_SIZE_PER_RISC=" +
                          std::to_string(config.profiler_dram_bank_size_per_risc_bytes) + " ";
    }
    if (rtoptions.get_profiler_noc_events_enabled()) {
        // force profiler on if noc events are being profiled
        if (!rtoptions.get_profiler_enabled()) {
            this->defines_ += "-DPROFILE_KERNEL=1 ";
        }
        this->defines_ += "-DPROFILE_NOC_EVENTS=1 ";
    }
    if (rtoptions.get_experimental_noc_debug_dump_enabled()) {
        this->defines_ += "-DDEVICE_DEBUG_DUMP=1 ";
    }
    if (rtoptions.get_profiler_perf_counter_mode() != 0) {
        // force profiler on if perf counters are being captured
        TT_ASSERT(rtoptions.get_profiler_enabled());
        this->defines_ += "-DPROFILE_PERF_COUNTERS=" + std::to_string(rtoptions.get_profiler_perf_counter_mode()) + " ";
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
    for (const auto& feature : rtoptions.get_watcher_disabled_features()) {
        this->defines_ += "-DWATCHER_DISABLE_" + feature + " ";
    }

    if (rtoptions.get_feature_enabled(tt::llrt::RunTimeDebugFeatureDprint)) {
        this->defines_ += "-DDEBUG_PRINT_ENABLED ";
        if (rtoptions.get_use_device_print()) {
            this->defines_ += "-DUSE_DEVICE_PRINT ";
        }
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

    if (config.routing_fw_enabled) {
        this->defines_ += "-DROUTING_FW_ENABLED ";
    }

    if (rtoptions.get_lightweight_kernel_asserts()) {
        this->defines_ += "-DLIGHTWEIGHT_KERNEL_ASSERTS ";
    }

    if (rtoptions.get_llk_asserts()) {
        this->defines_ += "-DENABLE_LLK_ASSERT ";
    }

    if (!rtoptions.get_watcher_enabled() && !rtoptions.get_lightweight_kernel_asserts() &&
        rtoptions.get_llk_asserts()) {
        this->defines_ += "-DENV_LLK_INFRA ";
    }

    if (rtoptions.get_disable_sfploadmacro()) {
        this->defines_ += "-DDISABLE_SFPLOADMACRO ";
    }

    // Includes
    // TODO(pgk) this list is insane
    std::vector<fs::path> includeDirs = {
        ".",
        "..",
        root_,
        root_ / "ttnn",
        root_ / "ttnn/cpp",
        root_ / "tt_metal",
        root_ / "tt_metal/hw/inc",
        root_ / "tt_metal/third_party/tt_llk/common",
        root_ / "tt_metal/hostdevcommon/api",
        root_ / "tt_metal/api/"};

    std::ostringstream oss;
    for (const auto& includeDir : includeDirs) {
        oss << "-I" << includeDir.string() << " ";
    }
    this->includes_ = oss.str();

    this->lflags_ = common_flags;
    this->lflags_ += "-Wl,-z,max-page-size=16 -Wl,-z,common-page-size=16 -nostartfiles ";

    // Need to capture more info in build key to prevent stale binaries from being reused.
    tt::FNV1a hasher;
    hasher.update(build_key);
    hasher.update(enchantum::to_underlying(this->arch_));
    hasher.update(cflags_);
    hasher.update(lflags_);
    hasher.update(defines_);
    hasher.update(sfpi_version_contents);
    build_key_ = hasher.digest();

    this->out_firmware_root_ = this->out_root_ / std::to_string(build_key_) / "firmware";
    this->out_kernel_root_ = this->out_root_ / std::to_string(build_key_) / "kernels";
    this->firmware_binary_root_ = this->out_firmware_root_;

    if (rtoptions.is_jit_scratch_dir_specified()) {
        this->scratch_root_ = fs::path(rtoptions.get_jit_scratch_dir());
        this->scratch_firmware_root_ = this->scratch_root_ / std::to_string(build_key_) / "firmware";
        this->scratch_kernel_root_ = this->scratch_root_ / std::to_string(build_key_) / "kernels";
        tt::filesystem::set_nfs_safety(true);
        log_debug(
            tt::LogBuildKernels,
            "JIT scratch enabled (NFS safety on): compiling to {} before merging to {}",
            this->scratch_root_,
            this->out_root_);
    }
}

JitBuildState::JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig& build_config, const Hal& hal) :
    env_(env),
    is_fw_(build_config.is_fw),
    process_defines_at_compile_(true),
    out_path_(build_config.is_fw ? env_.out_firmware_root_ : env_.out_kernel_root_),
    scratch_path_([&]() -> fs::path {
        if (!env_.has_scratch()) {
            return {};
        }
        return build_config.is_fw ? env_.scratch_firmware_root_ : env_.scratch_kernel_root_;
    }()),
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
        fmt::format_to(std::back_inserter(this->lflags_), "-L{} ", (env_.root_ / "tt_metal/hw/toolchain").string());
    }

    HalJitBuildQueryInterface::Params params{
        build_config.is_fw,
        build_config.core_type,
        build_config.processor_class,
        static_cast<uint32_t>(build_config.processor_id),
        env_.get_rtoptions()};
    const auto& jit_build_query = hal.get_jit_build_query();

    this->target_name_ = jit_build_query.target_name(params);
    // Includes
    {
        auto it = std::back_inserter(this->includes_);
        for (const auto& include : jit_build_query.includes(params)) {
            fmt::format_to(it, "-I{} ", (env_.root_ / include).string());
        }
    }
    // Defines
    {
        auto it = std::back_inserter(this->defines_);
        for (const auto& define : jit_build_query.defines(params)) {
            fmt::format_to(it, "-D{} ", define);
        }
        fmt::format_to(it, "-DDISPATCH_MESSAGE_ADDR={} ", build_config.dispatch_message_addr);
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

    this->linker_script_ = env_.root_ / jit_build_query.linker_script(params);

    // Source files
    {
        auto srcs = jit_build_query.srcs(params);
        this->srcs_.insert(this->srcs_.end(), std::move_iterator(srcs.begin()), std::move_iterator(srcs.end()));
    }
    this->firmware_is_kernel_object_ = jit_build_query.firmware_is_kernel_object(params);

    // Create the objs from the srcs
    for (const fs::path& src : srcs_) {
        fs::path obj_path = src.filename().replace_extension(".o");
        this->objs_.push_back(obj_path);
        this->temp_objs_.push_back(jit_build::utils::FileRenamer::generate_temp_path(obj_path));
    }

    // Prepend root path to srcs, but not to outputs (objs) due to device dependency
    for (fs::path& src : this->srcs_) {
        src = env_.root_ / src;
    }

    // Append hw build objects compiled offline
    {
        auto it = std::back_inserter(this->extra_link_objs_);
        for (const auto& obj : jit_build_query.link_objs(params)) {
            fmt::format_to(it, "{} ", (env_.root_ / obj).string());
        }
    }

    // Linker flags
    this->lflags_ += jit_build_query.linker_flags(params);
    this->lflags_ += fmt::format("-T{} ", this->linker_script_.string());
    if (!this->is_fw_) {
        this->lflags_ += "-Wl,--emit-relocs ";
    }

    // Precompute the weakened firmware path
    {
        auto target_name = jit_build_query.weakened_firmware_target_name(params);
        std::string_view suffix = this->firmware_is_kernel_object_ ? "object.o" : "weakened.elf";
        this->weakened_firmware_name_ =
            this->env_.firmware_binary_root_ / target_name / fmt::format("{}_{}", target_name, suffix);
    }

    // Relative path suffix: appended to out_path_ / kernel_name to form the full ELF path
    this->target_full_path_ = fs::path(this->target_name_) / (this->target_name_ + ".elf");

    // Compute a hash of all effective compilation/linking parameters.
    // This captures HAL-populated flags (defines, includes, common_flags, linker_flags, etc.)
    // that are not part of the env-level build_key_.  When any of these change between runs
    // (e.g. after a code change that modifies HAL output), the hash changes and cached
    // objects are invalidated, preventing stale binaries from being reused.
    {
        tt::FNV1a hasher;
        hasher.update(env_.gpp_);
        hasher.update(cflags_);
        hasher.update(defines_);
        hasher.update(includes_);
        hasher.update(lflags_);
        hasher.update(linker_script_.string());
        hasher.update(extra_link_objs_);
        for (const auto& src : srcs_) {
            hasher.update(src.string());
        }
        hasher.update(default_compile_opt_level_);
        hasher.update(default_linker_opt_level_);
        build_state_hash_ = hasher.digest();
    }
}

static constexpr std::string_view BUILD_STATE_HASH_FILE = ".build_state";

void publish_build_state_hash(const std::filesystem::path& out_dir, uint64_t build_state_hash) {
    fs::path hash_path = out_dir / BUILD_STATE_HASH_FILE;
    fs::path tmp_path = jit_build::utils::FileRenamer::generate_temp_path(hash_path);

    std::ofstream file(tmp_path, std::ios::trunc);
    if (!file.is_open()) {
        log_error(tt::LogBuildKernels, "Failed to open build state file for writing: {}", tmp_path);
        return;
    }
    file << build_state_hash;
    file.close();
    if (file.fail()) {
        log_error(tt::LogBuildKernels, "Failed to write build state hash to {}", tmp_path);
        tt::filesystem::safe_remove(tmp_path);
        return;
    }
    if (!tt::filesystem::safe_rename(tmp_path, hash_path, false)) {
        log_error(tt::LogBuildKernels, "Failed to publish build state hash from {} to {}", tmp_path, hash_path);
        tt::filesystem::safe_remove(tmp_path);
        return;
    }
    log_info(tt::LogBuildKernels, "Published build state hash {} to {}", build_state_hash, hash_path);
}

bool JitBuildState::build_state_matches(const std::filesystem::path& out_dir) const {
    fs::path hash_path = out_dir / BUILD_STATE_HASH_FILE;
    uint64_t stored_hash{};
    bool hash_matches = false;

    bool success = tt::filesystem::retry_on_estale([&]() {
        errno = 0;
        std::ifstream file(hash_path);
        if (!file.is_open()) {
            return false;
        }
        file >> stored_hash;
        if (file.fail()) {
            return false;
        }
        hash_matches = (stored_hash == build_state_hash_);
        return true;
    });

    if (!success) {
        log_info(
            tt::LogBuildKernels,
            "Build state file not found or unreadable: {} (current hash={})",
            hash_path,
            build_state_hash_);
        return false;
    }
    if (!hash_matches) {
        log_info(
            tt::LogBuildKernels,
            "Build state hash mismatch in {}: stored={}, current={}",
            out_dir,
            stored_hash,
            build_state_hash_);
        return false;
    }
    return true;
}

void JitBuildState::publish_build_state_hash(const std::filesystem::path& out_dir) const {
    tt::tt_metal::publish_build_state_hash(out_dir, build_state_hash_);
}

void JitBuildState::compile_one(
    const std::filesystem::path& out_dir,
    const JitBuildSettings* settings,
    size_t src_index,
    const std::filesystem::path& canonical_dir) const {
    // ZoneScoped;

    string cmd{"cd " + out_dir.string() + " && " + env_.gpp_};
    string defines = this->defines_;

    if (env_.get_rtoptions().get_build_map_enabled()) {
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
    fs::path obj_path = out_dir / this->objs_[src_index];
    fs::path obj_temp_path = out_dir / this->temp_objs_[src_index];
    fs::path temp_d_path = obj_temp_path;
    temp_d_path.replace_extension("d");
    cmd += this->cflags_;
    cmd += this->includes_;
    // Add kernel-specific include paths (e.g. kernel source directory for relative includes)
    if (settings) {
        settings->process_include_paths([&cmd](const std::string& path) { cmd += fmt::format("-I{} ", path); });
    }
    cmd += fmt::format(
        "-c -o {} {} -MF {} ", obj_temp_path.string(), this->srcs_[src_index].string(), temp_d_path.string());
    cmd += defines;

    if (env_.get_rtoptions().get_log_kernels_compilation_commands()) {
        log_info(tt::LogBuildKernels, "    g++ compile cmd: {}", cmd);
    }

    if (env_.get_rtoptions().get_watcher_enabled() && settings) {
        log_kernel_defines_and_args(out_dir.string(), settings->get_full_kernel_name(), defines);
    }

    // log file and dephash file can be renamed after compilation, but the .o file
    // needs to be renamed after link step to avoid LTO reading inconsistent object files.
    jit_build::utils::FileRenamer log_file(obj_path.concat(".log"));
    tt::filesystem::safe_remove(log_file.path());
    if (!tt::jit_build::utils::run_command(cmd, log_file.path(), env_.get_rtoptions().get_dump_build_commands())) {
        build_failure(this->target_name_, "compile", cmd, log_file.path());
    }
    fs::path dephash_path = obj_temp_path;
    dephash_path.concat(".dephash");
    jit_build::write_dependency_hashes(out_dir, obj_temp_path, dephash_path, canonical_dir);
    tt::filesystem::safe_remove(temp_d_path);  // .d file not needed after hash is written
}

bool JitBuildState::need_compile(const std::filesystem::path& out_dir, const std::filesystem::path& obj) const {
    return env_.get_rtoptions().get_force_jit_compile() ||
           !tt::filesystem::safe_exists(out_dir / obj).value_or(false) ||
           !jit_build::dependencies_up_to_date(out_dir, obj);
}

std::bitset<JitBuildState::kMaxBuildBitset> JitBuildState::compile(
    const std::filesystem::path& out_dir,
    const JitBuildSettings* settings,
    bool state_changed,
    const std::filesystem::path& check_dir) const {
    // ZoneScoped;
    TT_FATAL(
        this->srcs_.size() <= kMaxBuildBitset,
        "Number of source files ({}) exceeds kMaxBuildBitset ({})",
        this->srcs_.size(),
        kMaxBuildBitset);

    // When check_dir is provided (scratch mode), cache-hit decisions use the
    // NFS cache directory while compilation output goes to out_dir (local scratch).
    const fs::path& cache_check_dir = check_dir.empty() ? out_dir : check_dir;

    std::bitset<kMaxBuildBitset> compiled;
    std::vector<std::shared_future<void>> events;
    for (size_t i = 0; i < this->srcs_.size(); ++i) {
        if (state_changed || need_compile(cache_check_dir, this->objs_[i])) {
            compiled.set(i);
            launch_build_step(
                [this, &out_dir, &check_dir, settings, i] { this->compile_one(out_dir, settings, i, check_dir); },
                events);
        } else {
            log_debug(tt::LogBuildKernels, "JIT build cache hit: {}{}", cache_check_dir, this->objs_[i]);
            BuildCacheTelemetry::inst().record_cache_hit();
        }
    }

    sync_build_steps(events);

    {
        auto& telemetry = BuildCacheTelemetry::inst();
        telemetry.record_compile(this->srcs_.size(), compiled.count());
        telemetry.log_compile_summary(state_changed);
    }

    if (env_.get_rtoptions().get_watcher_enabled()) {
        dump_kernel_defines_and_args(env_.get_out_kernel_root_path());
    }
    return compiled;
}

bool JitBuildState::need_link(const std::filesystem::path& out_dir) const {
    fs::path elf_path = out_dir / (this->target_name_ + ".elf");
    return !tt::filesystem::safe_exists(elf_path).value_or(false) ||
           !jit_build::dependencies_up_to_date(out_dir, elf_path);
}

void JitBuildState::link(
    const std::filesystem::path& out_dir, const JitBuildSettings* settings, const std::string& link_objs) const {
    string cmd{"cd " + out_dir.string() + " && " + env_.gpp_};
    string lflags = this->lflags_;
    if (env_.get_rtoptions().get_build_map_enabled()) {
        lflags += "-Wl,-Map=" + (out_dir / (this->target_name_ + ".map")).string() + " ";
        lflags += "-save-temps=obj -fdump-tree-all -fdump-rtl-all ";
    }

    // Append user args
    cmd += fmt::format("-{} ", settings ? settings->get_linker_opt_level() : this->default_linker_opt_level_);

    // Elf file has dependencies other than object files:
    // 1. Linker script
    // 2. Weakened firmware elf (for kernels)
    std::vector<fs::path> link_deps = {this->linker_script_};
    if (!this->is_fw_) {
        link_deps.push_back(this->weakened_firmware_name_);
        if (!this->firmware_is_kernel_object_) {
            cmd += "-Wl,--just-symbols=";
        }
        cmd += this->weakened_firmware_name_.string() + " ";
    }

    // Append common args provided by the build state
    cmd += lflags;
    cmd += this->extra_link_objs_;
    cmd += link_objs;
    fs::path elf_path = out_dir / (this->target_name_ + ".elf");
    jit_build::utils::FileRenamer elf_file(elf_path);
    cmd += "-o " + elf_file.path().string();
    if (env_.get_rtoptions().get_log_kernels_compilation_commands()) {
        log_info(tt::LogBuildKernels, "    g++ link cmd: {}", cmd);
    }
    fs::path log_path = elf_path;
    log_path.concat(".log");
    jit_build::utils::FileRenamer log_file(log_path);
    tt::filesystem::safe_remove(log_file.path());
    if (!tt::jit_build::utils::run_command(cmd, log_file.path(), env_.get_rtoptions().get_dump_build_commands())) {
        build_failure(this->target_name_, "link", cmd, log_file.path());
    }
    fs::path elf_dephash_path = elf_path;
    elf_dephash_path.concat(".dephash");
    jit_build::utils::FileRenamer dephash_file(elf_dephash_path);
    std::ofstream hash_file(dephash_file.path());
    {
        std::vector<std::string> link_dep_strs;
        link_dep_strs.reserve(link_deps.size());
        for (const auto& dep : link_deps) {
            link_dep_strs.push_back(dep.string());
        }
        jit_build::write_dependency_hashes(
            {{elf_path.string(), std::move(link_dep_strs)}}, out_dir, elf_path, hash_file);
    }
    hash_file.close();
    if (hash_file.fail()) {
        // Don't leave incomplete hash file
        tt::filesystem::safe_remove(dephash_file.path());
    }
}

// Given this elf (A) and a later elf (B):
// weakens symbols in A so that it can be used as a "library" for B. B imports A's weakened symbols, B's symbols of the
// same name don't result in duplicate symbols but B can reference A's symbols. Force the fw_export symbols to remain
// strong so to propagate link addresses
//
// NOTE: This function writes directly to the NFS cache directory (out_dir), not to scratch.
// This is intentional because:
//   - Weaken is called as part of the link stage, which already uses atomic temp+rename
//     via FileRenamer for the output file
//   - The input file (pathname_in) is already the final ELF in the cache directory
//   - Unlike compilation which generates new objects, this is a transform of an existing file
//   - The operation is idempotent - re-weakening the same ELF produces the same result
void JitBuildState::weaken(const std::filesystem::path& out_dir) const {
    // ZoneScoped;

    fs::path pathname_in = out_dir / (target_name_ + ".elf");
    jit_build::utils::FileRenamer out_file(this->weakened_firmware_name_);

    ll_api::ElfFile elf;
    elf.ReadImage(pathname_in.string());
    static const std::string_view strong_names[] = {"__fw_export_*", "__global_pointer$"};
    elf.WeakenDataSymbols(strong_names);
    if (this->firmware_is_kernel_object_) {
        elf.ObjectifyExecutable();
    }
    elf.WriteImage(out_file.path().string());
}

void JitBuildState::extract_zone_src_locations(const std::filesystem::path& out_dir) const {
    // ZoneScoped;
    static std::atomic<bool> new_log{true};
    if (env_.get_rtoptions().get_profiler_enabled()) {
        if (new_log.exchange(false) &&
            tt::filesystem::safe_exists(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG).value_or(false)) {
            std::remove(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG.c_str());
        }

        if (!tt::filesystem::safe_exists(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG).value_or(false)) {
            tt::jit_build::utils::create_file(tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG);
        }

        auto cmd = fmt::format("grep KERNEL_PROFILER {}/*.o.log", out_dir.string());
        tt::jit_build::utils::run_command(
            cmd, tt::tt_metal::NEW_PROFILER_ZONE_SRC_LOCATIONS_LOG, env_.get_rtoptions().get_dump_build_commands());
    }
}

void JitBuildState::merge_genfiles_to_cache(const JitBuildSettings* settings) const {
    if (scratch_path_.empty() || !settings) {
        return;
    }
    const auto& kernel_name = settings->get_full_kernel_name();
    fs::path scratch_genfiles_dir = this->scratch_path_ / kernel_name;
    fs::path nfs_genfiles_dir = this->out_path_ / kernel_name;
    if (!copy_genfiles_to_cache(scratch_genfiles_dir, nfs_genfiles_dir)) {
        TT_THROW("Failed to merge generated files from {} to {}", scratch_genfiles_dir, nfs_genfiles_dir);
    }
}

bool JitBuildState::build(const JitBuildSettings* settings, std::span<const JitBuildState* const> link_targets) const {
    // ZoneScoped;
    auto t0_build = std::chrono::steady_clock::now();
    auto kernel_name = settings ? std::string_view{settings->get_full_kernel_name()} : "";

    fs::path cache_dir = this->out_path_ / kernel_name / this->target_name_;

    log_info(tt::LogBuildKernels, "JIT build check: cache_dir={}, build_state_hash={}", cache_dir, build_state_hash_);

    const bool use_scratch = !scratch_path_.empty();
    fs::path work_dir = cache_dir;
    if (use_scratch) {
        work_dir = this->scratch_path_ / kernel_name / this->target_name_;
        tt::filesystem::safe_create_directories(work_dir);
    }

    const JitBuildState* self = this;
    if (link_targets.empty()) {
        link_targets = std::span<const JitBuildState* const>(&self, 1);
    }
    const size_t num_objs = this->objs_.size();

    tt::filesystem::safe_create_directories(cache_dir);

    bool state_changed = !build_state_matches(cache_dir);

    auto compiled = compile(work_dir, settings, state_changed, use_scratch ? cache_dir : fs::path{});

    bool did_link = false;
    string link_objs;
    auto populate_link_objs = [&] {
        if (!link_objs.empty()) {
            return;
        }
        for (size_t i = 0; i < num_objs; ++i) {
            auto temp_obj = (work_dir / this->temp_objs_[i]).string();
            if (!compiled.test(i)) {
                const auto& src_dir = use_scratch ? cache_dir : work_dir;
                if (!hard_link_or_copy(src_dir / this->objs_[i], temp_obj)) {
                    TT_THROW("Failed to prepare link object {} from source directory {}", this->objs_[i], src_dir);
                }
            }
            link_objs += temp_obj;
            link_objs += " ";
        }
    };

    std::vector<fs::path> build_state_publish_dirs;
    build_state_publish_dirs.reserve(link_targets.size());
    auto enqueue_build_state_publish = [&build_state_publish_dirs](const fs::path& target_cache_dir) {
        if (std::find(build_state_publish_dirs.begin(), build_state_publish_dirs.end(), target_cache_dir) ==
            build_state_publish_dirs.end()) {
            build_state_publish_dirs.push_back(target_cache_dir);
        }
    };

    for (const auto* target : link_targets) {
        fs::path target_cache_dir = target->out_path_ / kernel_name / target->target_name_;
        fs::path target_work_dir = target_cache_dir;
        if (use_scratch) {
            target_work_dir = target->scratch_path_ / kernel_name / target->target_name_;
            tt::filesystem::safe_create_directories(target_work_dir);
        }
        if (!use_scratch || target_cache_dir != target_work_dir) {
            tt::filesystem::safe_create_directories(target_cache_dir);
        }

        if (state_changed || compiled.any() || target->need_link(target_cache_dir)) {
            populate_link_objs();
            target->link(target_work_dir, settings, link_objs);
            if (target->is_fw_) {
                target->weaken(target_work_dir);
            }
            if (use_scratch) {
                if (!merge_scratch_to_cache(target_work_dir, target_cache_dir)) {
                    TT_THROW("Failed to merge scratch artifacts from {} to {}", target_work_dir, target_cache_dir);
                }
            }
            enqueue_build_state_publish(target_cache_dir);
            did_link = true;
        }
    }

    if (!link_objs.empty()) {
        if (use_scratch) {
            for (size_t i = 0; i < num_objs; ++i) {
                fs::path scratch_temp = work_dir / this->temp_objs_[i];
                fs::path cache_final = cache_dir / this->objs_[i];
                if (compiled.test(i)) {
                    auto tmp_obj = jit_build::utils::FileRenamer::generate_temp_path(cache_final);
                    if (!hard_link_or_copy(scratch_temp, tmp_obj)) {
                        TT_THROW("Failed to stage object {} for cache merge", scratch_temp);
                    }
                    if (!tt::filesystem::safe_rename(tmp_obj, cache_final, false)) {
                        TT_THROW("Failed to merge object {} into cache {}", tmp_obj, cache_final);
                    }

                    fs::path scratch_dephash = scratch_temp;
                    scratch_dephash.concat(".dephash");
                    fs::path cache_dephash = cache_final;
                    cache_dephash.concat(".dephash");
                    auto tmp_dephash = jit_build::utils::FileRenamer::generate_temp_path(cache_dephash);
                    if (!hard_link_or_copy(scratch_dephash, tmp_dephash)) {
                        TT_THROW("Failed to stage dependency hash {} for cache merge", scratch_dephash);
                    }
                    if (!tt::filesystem::safe_rename(tmp_dephash, cache_dephash, false)) {
                        TT_THROW("Failed to merge dependency hash {} into cache {}", tmp_dephash, cache_dephash);
                    }
                }
                tt::filesystem::safe_remove(scratch_temp);
                fs::path scratch_dephash = scratch_temp;
                scratch_dephash.concat(".dephash");
                tt::filesystem::safe_remove(scratch_dephash);
            }
            if (!merge_scratch_to_cache(work_dir, cache_dir)) {
                TT_THROW("Failed to merge remaining scratch artifacts from {} to {}", work_dir, cache_dir);
            }
            for (const auto& target_cache_dir : build_state_publish_dirs) {
                publish_build_state_hash(target_cache_dir);
            }
            // Single syncfs at end of all scratch merges (covers the entire NFS filesystem).
            // Use async so it can overlap with the next kernel's JIT build.
            tt::filesystem::async_sync_filesystem(cache_dir);
        } else {
            // Use operator/ to build paths, NOT replace_filename.  cache_dir has no
            // trailing separator (e.g. ".../brisc"), so replace_filename() would replace
            // the last component ("brisc") instead of appending inside the directory,
            // placing the renamed files one level up and silently breaking the cache.
            for (size_t i = 0; i < num_objs; ++i) {
                fs::path src_path = cache_dir / this->temp_objs_[i];
                fs::path dst_path = cache_dir / this->objs_[i];
                if (compiled.test(i)) {
                    if (!tt::filesystem::safe_rename(src_path, dst_path, true)) {
                        TT_THROW("Failed to finalize object {} to {}", src_path, dst_path);
                    }
                    fs::path src_dephash = src_path;
                    src_dephash.concat(".dephash");
                    fs::path dst_dephash = dst_path;
                    dst_dephash.concat(".dephash");
                    if (!tt::filesystem::safe_rename(src_dephash, dst_dephash, true)) {
                        TT_THROW("Failed to finalize dependency hash {} to {}", src_dephash, dst_dephash);
                    }
                } else {
                    tt::filesystem::safe_remove(src_path);
                }
            }
            for (const auto& target_cache_dir : build_state_publish_dirs) {
                publish_build_state_hash(target_cache_dir);
            }
        }
    } else {
        for (const auto& target_cache_dir : build_state_publish_dirs) {
            publish_build_state_hash(target_cache_dir);
        }
    }

    extract_zone_src_locations(use_scratch ? work_dir : cache_dir);

    bool did_work = compiled.any() || did_link;
    auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_build).count();
    static auto& tok_build = BuildCacheTelemetry::inst().register_metric("JitBuildState::build");
    tok_build.record(elapsed_ms);
    return did_work;
}

void jit_build(const JitBuildState& build, const JitBuildSettings* settings) {
    // ZoneScoped;
    auto t0 = std::chrono::steady_clock::now();

    build.merge_genfiles_to_cache(settings);
    bool did_work = build.build(settings);
    tt::filesystem::wait_for_pending_sync();
    write_successful_jit_build_marker(build, settings, did_work);

    auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static auto& tok = BuildCacheTelemetry::inst().register_metric("jit_build");
    tok.record(elapsed_ms);
}

void jit_build_for_processors(std::span<const JitBuildState* const> targets, const JitBuildSettings* settings) {
    TT_ASSERT(!targets.empty());
    auto t0 = std::chrono::steady_clock::now();

    const JitBuildState& primary = *targets[0];
    primary.merge_genfiles_to_cache(settings);
    bool did_work = primary.build(settings, targets);
    tt::filesystem::wait_for_pending_sync();
    write_successful_jit_build_marker(primary, settings, did_work);

    auto elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    static auto& tok = BuildCacheTelemetry::inst().register_metric("jit_build_for_processors");
    tok.record(elapsed_ms);
}

void jit_build_subset(JitBuildStateSubset build_subset, const JitBuildSettings* settings) {
    if (!build_subset.empty()) {
        build_subset.begin()->merge_genfiles_to_cache(settings);
    }

    // Use vector<char> instead of vector<bool> to avoid bit-packing data races.
    std::vector<char> results(build_subset.size(), 0);
    std::vector<std::shared_future<void>> events;
    for (size_t idx = 0; idx < build_subset.size(); ++idx) {
        launch_build_step(
            [&build_subset, settings, idx, &results] { results[idx] = build_subset[idx].build(settings); }, events);
    }

    sync_build_steps(events);
    // Drain any pending async sync from the parallel builds before writing markers.
    tt::filesystem::wait_for_pending_sync();
    for (size_t idx = 0; idx < build_subset.size(); ++idx) {
        write_successful_jit_build_marker(build_subset[idx], settings, results[idx] != 0);
    }
}

void launch_build_step(const std::function<void()>& build_func, std::vector<std::shared_future<void>>& events) {
    events.emplace_back(detail::async(build_func));
}

void sync_build_steps(std::vector<std::shared_future<void>>& events) {
    for (auto& event : events) {
        event.get();
    }
}

void jit_build_once(size_t hash, const std::function<void()>& build_fn) {
    if (JitBuildCache::inst().build_once(hash, build_fn)) {
        BuildCacheTelemetry::inst().record_jit_once_dedup();
    }
}

void jit_build_cache_clear() { JitBuildCache::inst().clear(); }

}  // namespace tt::tt_metal
