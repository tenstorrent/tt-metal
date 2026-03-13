// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>
#include <bitset>
#include <span>
#include <tt_stl/aligned_allocator.hpp>
#include <functional>
#include <future>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/hal_types.hpp>
#include "jit_build_options.hpp"
#include <umd/device/types/arch.hpp>

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::tt_metal {

struct JitDeviceConfig;
class Hal;

static constexpr uint32_t CACHE_LINE_ALIGNMENT = 64;

static const std::string SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME = ".SUCCESS";

template <typename T>
using vector_cache_aligned = std::vector<T, tt::stl::aligned_allocator<T, CACHE_LINE_ALIGNMENT>>;

class JitBuildSettings;

struct JitBuiltStateConfig {
    HalProgrammableCoreType core_type{};
    HalProcessorClassType processor_class{};
    int processor_id = 0;
    bool is_fw = false;
    uint32_t dispatch_message_addr = 0;
    // Set `is_cooperative` when Metal FW/Kernel code is loaded on risc with some base FW running.
    // In this case Metal FW will need to facilitate context switching to base FW (e.g. code running on WH active
    // eriscs)
    bool is_cooperative = false;
};

// The build environment
// Includes the path to the src/output and global defines, flags, etc
// Device specific
class JitBuildEnv {
    friend class JitBuildState;

public:
    JitBuildEnv();
    void init(
        uint64_t build_key,
        const JitDeviceConfig& config,
        const tt::llrt::RunTimeOptions& rtoptions,
        const std::map<std::string, std::string>& device_kernel_defines);

    tt::ARCH get_arch() const { return arch_; }
    uint32_t get_max_cbs() const { return max_cbs_; };
    const tt::llrt::RunTimeOptions& get_rtoptions() const { return *rtoptions_; }
    const std::string& get_root_path() const { return root_; }
    const std::string& get_out_root_path() const { return out_root_; }
    const std::string& get_out_kernel_root_path() const { return out_kernel_root_; }
    // Returns scratch kernel root when scratch is configured, NFS kernel root otherwise.
    // Use for genfile writes so they happen on local disk in scratch mode.
    const std::string& get_genfiles_kernel_root_path() const {
        return has_scratch() ? scratch_kernel_root_ : out_kernel_root_;
    }
    const std::string& get_out_firmware_root_path() const {
        return out_firmware_root_;
    }  // Path to the firmware directory for this device
    uint64_t get_build_key() const { return build_key_; }

    bool has_scratch() const { return !scratch_root_.empty(); }
    const std::string& get_scratch_firmware_root() const { return scratch_firmware_root_; }
    const std::string& get_scratch_kernel_root() const { return scratch_kernel_root_; }

    // Where firmware binaries are loaded/linked from. Defaults to out_firmware_root_.
    // May differ when binaries are provided from an external source.
    const std::string& get_firmware_binary_root() const { return firmware_binary_root_; }
    void set_firmware_binary_root(const std::string& path) { firmware_binary_root_ = path; }

private:
    const tt::llrt::RunTimeOptions* rtoptions_{nullptr};

    tt::ARCH arch_{tt::ARCH::Invalid};
    uint32_t max_cbs_{};

    // Paths
    std::string root_;
    std::string out_root_;
    std::string out_firmware_root_;
    std::string out_kernel_root_;
    std::string firmware_binary_root_;

    // Local scratch paths for hybrid map-reduce JIT builds.
    // When set, SFPI compilation happens here (local disk) and results are
    // atomically merged to out_*_root_ (NFS cache) after success via
    // temp-file + rename.
    //
    // Cache-sharing model: multiple ranks and hosts intentionally share a
    // single NFS cache directory (out_*_root_).  This is safe because:
    //   - All writes go through local scratch first, then atomic merge
    //   - .dephash files record NFS-canonical paths (scratch paths are
    //     rewritten by write_dependency_hashes), so cache-hit checks work
    //     regardless of which host compiled the artifact
    //   - Sharing avoids redundant N-way compilation across ranks
    // TT_METAL_CACHE is therefore NOT rank-scoped; only scratch and log
    // directories are per-rank.
    std::string scratch_root_;
    std::string scratch_firmware_root_;
    std::string scratch_kernel_root_;

    // Tools
    std::string gpp_;
    std::string gpp_include_dir_;

    // Compilation options
    std::string cflags_;
    std::string defines_;
    std::string includes_;
    std::string lflags_;

    std::uint64_t build_key_{};
};

// All the state used for a build in an abstract base class
// Contains everything needed to do a build (all settings, methods, etc)
class alignas(CACHE_LINE_ALIGNMENT) JitBuildState {
protected:
    const JitBuildEnv& env_;

    bool is_fw_;
    bool process_defines_at_compile_{};
    bool firmware_is_kernel_object_{};

    std::string out_path_;
    std::string scratch_path_;  // Empty when scratch is not configured
    std::string target_name_;
    std::string target_full_path_;

    std::string cflags_;
    std::string defines_;
    std::string includes_;
    std::string lflags_;
    std::string linker_script_;

    vector_cache_aligned<std::string> srcs_;
    vector_cache_aligned<std::string> objs_;
    vector_cache_aligned<std::string> temp_objs_;

    std::string extra_link_objs_;
    std::string weakened_firmware_name_;

    // Default compiler optimization setting
    // Used when JitBuildSettings is not provided
    std::string default_compile_opt_level_;

    // Default linker optimization setting
    // Used when JitBuildSettings is not provided
    std::string default_linker_opt_level_;

    // Hash of all effective compilation/linking parameters (including HAL-populated flags).
    // Used to detect when build flags change between runs so that stale cached objects
    // are not reused.  Written to a ".build_state" file in the output directory.
    uint64_t build_state_hash_{};

    // Upper bound for compile objects.
    // Current max obj count is 2 -- very sufficient for now.
    static constexpr size_t kMaxBuildBitset = 64;

    bool build_state_matches(const std::string& out_dir) const;
    void write_build_state_hash(const std::string& out_dir) const;

    bool need_compile(const std::string& out_dir, const std::string& obj) const;
    // When check_dir is non-empty, cache-hit checks (need_compile) use check_dir
    // while actual compilation writes to out_dir.  This enables scratch-mode:
    // check NFS cache, compile to local disk.
    std::bitset<kMaxBuildBitset> compile(
        const std::string& out_dir,
        const JitBuildSettings* settings,
        bool state_changed,
        const std::string& check_dir = {}) const;
    void compile_one(
        const std::string& out_dir,
        const JitBuildSettings* settings,
        size_t src_index,
        const std::string& canonical_dir = {}) const;
    bool need_link(const std::string& out_dir) const;
    void link(const std::string& out_dir, const JitBuildSettings* settings, const std::string& link_objs) const;
    void weaken(const std::string& out_dir) const;
    void extract_zone_src_locations(const std::string& out_dir) const;

public:
    JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig& build_config, const Hal& hal);

    void build(const JitBuildSettings* settings, std::span<const JitBuildState* const> link_targets = {}) const;

    // When scratch is configured, merge generated headers (chlkc_descriptors.h, etc.)
    // from the scratch genfiles directory to the NFS cache so future cache-hit checks work.
    // Must be called once per kernel AFTER generate_binaries() and BEFORE parallel build() calls.
    void merge_genfiles_to_cache(const JitBuildSettings* settings) const;

    const std::string& get_out_path() const { return this->out_path_; }
    const std::string& get_target_name() const { return this->target_name_; }
    const std::string& get_target_full_path() const { return this->target_full_path_; }
    std::string get_target_out_path(const std::string& kernel_name) const {
        return this->out_path_ + kernel_name + target_full_path_;
    }
};

// Extracts a slice of builds from JitBuildStates
// Used for parallel building a subset of the builds, builds all members in one call
using JitBuildStateSubset = std::span<const JitBuildState>;

void jit_build(const JitBuildState& build, const JitBuildSettings* settings);
void jit_build_subset(JitBuildStateSubset build_subset, const JitBuildSettings* settings);

// Build for multiple processors that share the same source: the first target compiles,
// and all targets (including the first) are linked. Writes the success marker once after all succeed.
void jit_build_for_processors(std::span<const JitBuildState* const> targets, const JitBuildSettings* settings);

void launch_build_step(const std::function<void()>& build_func, std::vector<std::shared_future<void>>& events);
void sync_build_steps(std::vector<std::shared_future<void>>& events);

// Execute build_fn exactly once for a given hash.
// Concurrent callers with the same hash block until the build completes.
// Returns immediately if hash was already built.
// If build_fn throws, subsequent callers will retry.
void jit_build_once(size_t hash, const std::function<void()>& build_fn);

// Clear the JIT build cache so that subsequent jit_build_once() calls re-execute.
void jit_build_cache_clear();

}  // namespace tt::tt_metal
