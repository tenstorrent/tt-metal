// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace tt::tt_metal {

// Abstract base class for kernel specialization
// Higher levels of the SW derive from this and fill in build details not known to the build system
// (eg, API specified settings)
class JitBuildSettings {
public:
    // Returns the full kernel name
    virtual const std::string& get_full_kernel_name() const = 0;
    // Returns the compiler optimization level
    virtual std::string_view get_compiler_opt_level() const = 0;
    // Returns the linker optimization level
    virtual std::string_view get_linker_opt_level() const = 0;

    // Called to process the user defines
    virtual void process_defines(std::function<void(const std::string& define, const std::string& value)>) const = 0;
    // Called to process the user compile time args
    virtual void process_compile_time_args(std::function<void(const std::vector<uint32_t>& values)>) const = 0;
    // Called to process the user named compile time args
    virtual void process_named_compile_time_args(
        std::function<void(const std::unordered_map<std::string, uint32_t>& named_args)>) const = 0;

    // Called to process the user kernel resource bindings (Metal 2.0 APIs)
    //  - DFB accessors
    //  - Semaphore accessors
    //  - Tensor accessors (TODO)
    virtual void process_dataflow_buffer_local_accessor_handles(
        std::function<void(const std::string& accessor_name, uint16_t logical_dfb_id)>) const {}
    virtual void process_semaphore_local_accessor_handles(
        std::function<void(const std::string& accessor_name, uint16_t semaphore_id)>) const {}

    // Named RTA/CRTA schema (Metal 2.0 APIs).
    // The order of names determines the byte offset of each arg within the named-args
    // section of the dispatch buffer.
    // Returned by const-ref rather than via a process_* callback because the concrete storage
    // is already an ordered vector — the callback indirection would just force a copy.
    virtual const std::vector<std::string>& get_named_runtime_args() const {
        static const std::vector<std::string> k_empty;
        return k_empty;
    }
    virtual const std::vector<std::string>& get_named_common_runtime_args() const {
        static const std::vector<std::string> k_empty;
        return k_empty;
    }

    // Called to process additional include paths (e.g., kernel source directory for relative includes)
    virtual void process_include_paths(const std::function<void(const std::string& path)>&) const {}

    // Fence for Metal 2.0 kernel machinery. When false, the JIT build path must not emit or
    // reference any Metal 2.0 generated headers (kernel_bindings_generated.h, kernel_args_generated.h).
    // Legacy kernels created via the old host API default to false; kernels created via
    // MakeProgramFromSpec set this to true. This flag exists to prevent cross-contamination
    // between the two API paths during the deprecation window of the legacy API.
    virtual bool is_metal2_kernel() const { return false; }

    virtual ~JitBuildSettings() = default;
};

}  // namespace tt::tt_metal
