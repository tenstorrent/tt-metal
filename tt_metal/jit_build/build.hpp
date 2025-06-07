// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <stdint.h>
#include <tt_stl/aligned_allocator.hpp>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "core_coord.hpp"
#include "data_format.hpp"
#include "hostdevcommon/common_values.hpp"
#include "jit_build_options.hpp"
#include "tracy/Tracy.hpp"
#include "tt_backend_api_types.hpp"
#include "utils.hpp"

namespace tt {
enum class ARCH;
}  // namespace tt

namespace tt::tt_metal {

static constexpr uint32_t CACHE_LINE_ALIGNMENT = 64;

static const string SUCCESSFUL_JIT_BUILD_MARKER_FILE_NAME = ".SUCCESS";

template <typename T>
using vector_cache_aligned = std::vector<T, tt::stl::aligned_allocator<T, CACHE_LINE_ALIGNMENT>>;

class JitBuildSettings;

struct JitBuiltStateConfig {
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
    friend class JitBuildDataMovement;
    friend class JitBuildCompute;
    friend class JitBuildActiveEthernet;
    friend class JitBuildIdleEthernet;

public:
    JitBuildEnv();
    void init(uint32_t build_key, tt::ARCH arch, const std::map<std::string, std::string>& device_kernel_defines);

    tt::ARCH get_arch() const { return arch_; }
    const string& get_root_path() const { return root_; }
    const string& get_out_root_path() const { return out_root_; }
    const string& get_out_kernel_root_path() const { return out_kernel_root_; }

private:
    tt::ARCH arch_;
    string arch_name_;
    string aliased_arch_name_;

    // Paths
    string root_;
    string out_root_;
    string out_firmware_root_;
    string out_kernel_root_;

    // Tools
    string gpp_ = "";
    string gpp_include_dir_ = "";

    // Compilation options
    string cflags_;
    string defines_;
    string includes_;
    string lflags_;
};

// All the state used for a build in an abstract base class
// Contains everything needed to do a build (all settings, methods, etc)
class alignas(CACHE_LINE_ALIGNMENT) JitBuildState {
protected:
    const JitBuildEnv& env_;

    int core_id_;
    int is_fw_;
    uint32_t dispatch_message_addr_;
    bool process_defines_at_compile;

    string out_path_;
    string target_name_;
    string target_full_path_;

    string cflags_;
    string defines_;
    string includes_;
    string lflags_;

    vector_cache_aligned<std::string> srcs_;
    vector_cache_aligned<std::string> objs_;

    string link_objs_;

    // Default compiler optimization setting
    // Used when JitBuildSettings is not provided
    string default_compile_opt_level_;

    // Default linker optimization setting
    // Used when JitBuildSettings is not provided
    string default_linker_opt_level_;

    void compile(const string& log_file, const string& out_path, const JitBuildSettings* settings) const;
    void compile_one(
        const string& log_file,
        const string& out_path,
        const JitBuildSettings* settings,
        const string& src,
        const string& obj) const;
    void link(const string& log_file, const string& out_path, const JitBuildSettings* settings) const;
    void weaken(const string& log_file, const string& out_path) const;
    void copy_kernel(const string& kernel_in_path, const string& op_out_path) const;
    void extract_zone_src_locations(const string& log_file) const;

public:
    JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig& build_config);
    virtual ~JitBuildState() = default;
    void finish_init();

    void build(const JitBuildSettings* settings) const;

    const string& get_out_path() const { return this->out_path_; };
    const string& get_target_name() const { return this->target_name_; };
    string get_target_out_path(const string& kernel_name) const {
        return this->out_path_ + kernel_name + target_full_path_;
    }
};

// Set of build states
// Used for parallel builds, builds all members in one call
using JitBuildStateSet = std::vector<std::shared_ptr<JitBuildState>>;

// Exracts a slice of builds from a JitBuildState
// Used for parallel building a subset of the builds in a JitBuildStateSet
struct JitBuildStateSubset {
    const std::shared_ptr<JitBuildState>* build_ptr;
    int size;
};

// Specific build types
// These specialize a JitBuildState with everything need to build for a target
class JitBuildDataMovement : public JitBuildState {
private:
public:
    JitBuildDataMovement(const JitBuildEnv& env, const JitBuiltStateConfig& build_config);
};

class JitBuildCompute : public JitBuildState {
private:
public:
    JitBuildCompute(const JitBuildEnv& env, const JitBuiltStateConfig& build_config);
};

class JitBuildActiveEthernet : public JitBuildState {
private:
public:
    JitBuildActiveEthernet(const JitBuildEnv& env, const JitBuiltStateConfig& build_config);
};

class JitBuildIdleEthernet : public JitBuildState {
private:
public:
    JitBuildIdleEthernet(const JitBuildEnv& env, const JitBuiltStateConfig& build_config);
};

void jit_build(const JitBuildState& build, const JitBuildSettings* settings);
void jit_build_set(const JitBuildStateSet& builds, const JitBuildSettings* settings);
void jit_build_subset(const JitBuildStateSubset& builds, const JitBuildSettings* settings);

void launch_build_step(const std::function<void()>& build_func, std::vector<std::shared_future<void>>& events);

}  // namespace tt::tt_metal
