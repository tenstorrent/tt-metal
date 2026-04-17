// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

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
    // (Initially just DFB local accessor names, but will be extended and refactored as needed.)
    virtual void process_dataflow_buffer_local_accessor_handles(
        std::function<void(const std::string& accessor_name, uint16_t logical_dfb_id)>) const {}

    // Called to process additional include paths (e.g., kernel source directory for relative includes)
    virtual void process_include_paths(const std::function<void(const std::string& path)>&) const {}

    virtual ~JitBuildSettings() = default;
};

}  // namespace tt::tt_metal
