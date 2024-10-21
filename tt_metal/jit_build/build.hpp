// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <thread>
#include <string>
#include <future>

#include "common/tt_backend_api_types.hpp"
#include "common/executor.hpp"
#include "common/utils.hpp"
#include "common/core_coord.h"
#include "jit_build/data_format.hpp"
#include "jit_build/settings.hpp"
#include "hostdevcommon/common_values.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tt_stl/aligned_allocator.hpp"
#include "llrt/rtoptions.hpp"


namespace tt::tt_metal {

static constexpr uint32_t CACHE_LINE_ALIGNMENT = 64;

template <typename T>
using vector_cache_aligned = std::vector<T, tt::stl::aligned_allocator<T, CACHE_LINE_ALIGNMENT>>;

class JitBuildSettings;

enum class JitBuildProcessorType {
    DATA_MOVEMENT,
    COMPUTE,
    ETHERNET
};

struct JitBuiltStateConfig {
    int processor_id = 0;
    bool is_fw = false;
    uint32_t dispatch_message_addr = 0;
};

// The build environment
// Includes the path to the src/output and global defines, flags, etc
// Device specific
class JitBuildEnv {
    friend class JitBuildState;
    friend class JitBuildDataMovement;
    friend class JitBuildCompute;
    friend class JitBuildEthernet;

  public:
    JitBuildEnv();
    void init(uint32_t build_key, tt::ARCH arch);

    tt::ARCH get_arch() const { return arch_; }
    const string& get_root_path() const { return root_; }
    const string& get_out_root_path() const { return out_root_; }
    const string& get_out_firmware_root_path() const { return out_firmware_root_; }
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
    string gpp_;

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

    void compile(const string& log_file, const string& out_path, const JitBuildSettings *settings) const;
    void compile_one(const string& log_file, const string& out_path, const JitBuildSettings *settings, const string& src, const string &obj) const;
    void link(const string& log_file, const string& out_path) const;
    void weaken(const string& log_file, const string& out_path) const;
    void copy_kernel( const string& kernel_in_path, const string& op_out_path) const;
    void extract_zone_src_locations(const string& log_file) const;

  public:
    JitBuildState(const JitBuildEnv& env, const JitBuiltStateConfig &build_config);
    virtual ~JitBuildState() = default;
    void finish_init();

    void build(const JitBuildSettings *settings) const;

    const string& get_out_path() const { return this->out_path_; };
    const string& get_target_name() const { return this->target_name_; };
    const string get_target_out_path(const string& kernel_name) const { return this->out_path_ + kernel_name + target_full_path_; }
};

// Set of build states
// Used for parallel builds, builds all members in one call
typedef vector<std::shared_ptr<JitBuildState>> JitBuildStateSet;

// Exracts a slice of builds from a JitBuildState
// Used for parallel building a subset of the builds in a JitBuildStateSet
struct JitBuildStateSubset {
    const std::shared_ptr<JitBuildState> * build_ptr;
    int size;
};

// Specific build types
// These specialize a JitBuildState with everything need to build for a target
class JitBuildDataMovement : public JitBuildState {
  private:

  public:
    JitBuildDataMovement(const JitBuildEnv& env, const JitBuiltStateConfig &build_config);
};

class JitBuildCompute : public JitBuildState {
  private:
  public:
    JitBuildCompute(const JitBuildEnv& env, const JitBuiltStateConfig &build_config);
};

class JitBuildEthernet : public JitBuildState {
  private:
  public:
    JitBuildEthernet(const JitBuildEnv& env, const JitBuiltStateConfig &build_config);
};

// Abstract base class for kernel specialization
// Higher levels of the SW derive from this and fill in build details not known to the build system
// (eg, API specified settings)
class JitBuildSettings {
  public:
    virtual const string& get_full_kernel_name() const = 0;
    virtual void process_defines(const std::function<void (const string& define, const string &value)>) const = 0;
    virtual void process_compile_time_args(const std::function<void (int i, uint32_t value)>) const = 0;
  private:
    bool use_multi_threaded_compile = true;
};

void jit_build(const JitBuildState& build, const JitBuildSettings* settings);
void jit_build_set(const JitBuildStateSet& builds, const JitBuildSettings* settings);
void jit_build_subset(const JitBuildStateSubset& builds, const JitBuildSettings* settings);

inline const string jit_build_get_kernel_compile_outpath(int build_key) {
    // TODO(pgk), get rid of this
    // The test infra needs the output dir.  Could put this in the device, but we plan
    // to remove the device dependence in the future, so putting this here for now
    return llrt::OptionsG.get_root_dir() + "/built/" + std::to_string(build_key) + "/kernels/";
}

inline void launch_build_step(const std::function<void()> build_func, std::vector<std::shared_future<void>>& events) {
  events.emplace_back(detail::async(build_func));
}

inline void sync_build_step(std::vector<std::shared_future<void>>& events) {
  for (auto & f : events) {
    f.get();
  }
}
} // namespace tt::tt_metal
