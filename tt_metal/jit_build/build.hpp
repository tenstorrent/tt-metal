// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
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

namespace tt::tt_metal {

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
        size_t fw_compile_hash,
        tt::ARCH arch,
        uint32_t max_cbs,
        const std::map<std::string, std::string>& device_kernel_defines);

    tt::ARCH get_arch() const { return arch_; }
    uint32_t get_max_cbs() const { return max_cbs_; };
    const std::string& get_root_path() const { return root_; }
    const std::string& get_out_root_path() const { return out_root_; }
    const std::string& get_out_kernel_root_path() const { return out_kernel_root_; }
    const std::string& get_out_firmware_root_path() const {
        return out_firmware_root_;
    }  // Path to the firmware directory for this device
    uint64_t get_build_key() const { return build_key_; }

private:
    tt::ARCH arch_{tt::ARCH::Invalid};
    uint32_t max_cbs_{};

    // Paths
    std::string root_;
    std::string out_root_;
    std::string out_firmware_root_;
    std::string out_kernel_root_;

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

    HalProgrammableCoreType core_type_;
    HalProcessorClassType processor_class_;
    uint32_t processor_id_;

    std::string out_path_;
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

    // Default compiler optimization setting
    // Used when JitBuildSettings is not provided
    std::string default_compile_opt_level_;

    // Default linker optimization setting
    // Used when JitBuildSettings is not provided
    std::string default_linker_opt_level_;

    bool need_compile(const std::string& out_dir, const std::string& obj) const;
    size_t compile(const std::string& out_dir, const JitBuildSettings* settings) const;
    void compile_one(const std::string& out_dir, const JitBuildSettings* settings, size_t src_index) const;
    bool need_link(const std::string& out_dir) const;
    void link(const std::string& out_dir, const JitBuildSettings* settings, const std::string& link_objs) const;
    void weaken(const std::string& out_dir) const;
    std::string weakened_firmware_name() const;
    void extract_zone_src_locations(const std::string& out_dir) const;

public:
    JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig& build_config);

    void build(const JitBuildSettings* settings) const;

    // Links object files from a previously compiled processor build to create a binary for this processor.
    // Used for Quasar when multiple processors share the same kernel code to avoid redundant compilation.
    void link_to_processor(const JitBuildState& processor_build_state, const JitBuildSettings* settings) const;

    const std::string& get_out_path() const { return this->out_path_; }
    const std::string& get_target_name() const { return this->target_name_; }
    std::string get_target_out_path(const std::string& kernel_name) const {
        return this->out_path_ + kernel_name + target_full_path_;
    }
};

// Exracts a slice of builds from JitBuildStates
// Used for parallel building a subset of the builds, builds all members in one call
using JitBuildStateSubset = std::span<const JitBuildState>;

void jit_build(const JitBuildState& build, const JitBuildSettings* settings);
void jit_build_subset(JitBuildStateSubset build_subset, const JitBuildSettings* settings);

// Takes compiled object files from orig_processor_build_state and links them to produce a binary for
// additional_processor_build_state.
// Used for Quasar to share compiled objects across processors.
void jit_link_additional_processor(
    const JitBuildState& orig_processor_build_state,
    const JitBuildState& additional_processor_build_state,
    const JitBuildSettings* additional_processor_settings);

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
