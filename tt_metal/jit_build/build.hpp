// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
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

#include "hal_types.hpp"
#include "llrt/hal.hpp"
#include "jit_build_options.hpp"

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
    void init(uint32_t build_key, tt::ARCH arch, const std::map<std::string, std::string>& device_kernel_defines);

    tt::ARCH get_arch() const { return arch_; }
    const std::string& get_root_path() const { return root_; }
    const std::string& get_out_root_path() const { return out_root_; }
    const std::string& get_out_kernel_root_path() const { return out_kernel_root_; }

private:
    tt::ARCH arch_{tt::ARCH::Invalid};

    // Paths
    std::string root_;
    std::string out_root_;
    std::string out_firmware_root_;
    std::string out_kernel_root_;

    // Tools
    std::string gpp_ = "";
    std::string gpp_include_dir_ = "";

    // Compilation options
    std::string cflags_;
    std::string defines_;
    std::string includes_;
    std::string lflags_;
};

// All the state used for a build in an abstract base class
// Contains everything needed to do a build (all settings, methods, etc)
class alignas(CACHE_LINE_ALIGNMENT) JitBuildState {
protected:
    const JitBuildEnv& env_;

    int core_id_;
    int is_fw_;
    uint32_t dispatch_message_addr_;
    bool process_defines_at_compile_{};

    std::string out_path_;
    std::string target_name_;
    std::string target_full_path_;

    std::string cflags_;
    std::string defines_;
    std::string includes_;
    std::string lflags_;

    vector_cache_aligned<std::string> srcs_;
    vector_cache_aligned<std::string> objs_;

    std::string link_objs_;

    // Default compiler optimization setting
    // Used when JitBuildSettings is not provided
    std::string default_compile_opt_level_;

    // Default linker optimization setting
    // Used when JitBuildSettings is not provided
    std::string default_linker_opt_level_;

    void compile(const std::string& log_file, const std::string& out_path, const JitBuildSettings* settings) const;
    void compile_one(
        const std::string& log_file,
        const std::string& out_path,
        const JitBuildSettings* settings,
        const std::string& src,
        const std::string& obj) const;
    void link(const std::string& log_file, const std::string& out_path, const JitBuildSettings* settings) const;
    void weaken(const std::string& log_file, const std::string& out_path) const;
    void copy_kernel(const std::string& kernel_in_path, const std::string& op_out_path) const;
    void extract_zone_src_locations(const std::string& log_file) const;
    void finish_init(HalProgrammableCoreType core_type, HalProcessorClassType processor_class);

public:
    JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig& build_config);

    void build(const JitBuildSettings* settings) const;
    const std::string& get_out_path() const { return this->out_path_; }
    const std::string& get_target_name() const { return this->target_name_; };
    ;
    std::string get_target_out_path(const std::string& kernel_name) const {
        return this->out_path_ + kernel_name + target_full_path_;
    }
};

// Exracts a slice of builds from JitBuildStates
// Used for parallel building a subset of the builds, builds all members in one call
using JitBuildStateSubset = std::span<const JitBuildState>;

void jit_build(const JitBuildState& build, const JitBuildSettings* settings);
void jit_build_subset(JitBuildStateSubset builds, const JitBuildSettings* settings);

void launch_build_step(const std::function<void()>& build_func, std::vector<std::shared_future<void>>& events);
void sync_build_steps(std::vector<std::shared_future<void>>& events);

}  // namespace tt::tt_metal
