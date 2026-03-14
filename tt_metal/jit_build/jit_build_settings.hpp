// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <functional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tt::tt_metal {

// Dispatch type for named runtime args — determines which device-side accessor to use.
enum class RuntimeArgDispatch : uint8_t {
    COMMON,   // get_common_arg_val (shared across all cores)
    PER_CORE  // get_arg_val (unique per core)
};

// Entry in the named runtime arg namespace map.
struct NamedRuntimeArgEntry {
    std::string field;
    uint32_t index;
    RuntimeArgDispatch dispatch;
};

// Namespace → [entries] map for named runtime arg header generation.
using NamedRuntimeArgNamespaces = std::map<std::string, std::vector<NamedRuntimeArgEntry>>;

// Namespace → [(field, value)] map for named compile-time arg header generation.
using NamedCTArgNamespaces = std::map<std::string, std::vector<std::pair<std::string, uint32_t>>>;

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
    // Called to process named runtime arg namespaces for generated header (rt:: namespace)
    virtual void process_named_runtime_args(std::function<void(const NamedRuntimeArgNamespaces&)>) const = 0;
    // Called to process named compile-time arg namespaces for generated header (ct:: namespace)
    virtual void process_named_ct_arg_namespaces(std::function<void(const NamedCTArgNamespaces&)>) const = 0;
    // Called to process additional include paths (e.g., kernel source directory for relative includes)
    virtual void process_include_paths(const std::function<void(const std::string& path)>&) const {}

    virtual ~JitBuildSettings() = default;
};

}  // namespace tt::tt_metal
