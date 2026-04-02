// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <string_view>
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
// length == 1: emits constexpr Arg (scalar).
// length > 1:  emits constexpr ArrayArg (array of contiguous slots).
struct NamedRuntimeArgEntry {
    std::string field;
    uint32_t index;
    uint32_t length = 1;
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

    // Called to process the user kernel resource bindings (Metal 2.0 APIs)
    //  - DFB accessors
    //  - Semaphore accessors
    //  - Tensor accessors
    virtual void process_dataflow_buffer_local_accessor_handles(
        std::function<void(const std::string& accessor_name, uint16_t logical_dfb_id)>) const {}
    virtual void process_semaphore_local_accessor_handles(
        std::function<void(const std::string& accessor_name, uint16_t semaphore_id)>) const {}

    // TensorBinding callback emits the codegen-relevant fields only:
    //  - accessor_name: kernel-side identifier, used as the symbol name in the `ta::` namespace
    //  - cta_offset: starting word index of this binding's CTA payload in the kernel's
    //    positional compile-time-args buffer
    //  - addr_crta_offset: byte offset of the implicit base-address CRTA within the kernel's
    //    common-runtime-args section
    // (The tensor_parameter_name is also part of TensorBindingHandle, but we don't need it for codegen.)
    virtual void process_tensor_binding_handles(
        std::function<void(const std::string& accessor_name, uint32_t cta_offset, uint32_t addr_crta_offset)>) const {}

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

    // Called to process named runtime arg namespaces for generated header (rt:: namespace)
    virtual void process_named_runtime_args(std::function<void(const NamedRuntimeArgNamespaces&)>) const = 0;
    // Called to process named compile-time arg namespaces for generated header (ct:: namespace)
    virtual void process_named_ct_arg_namespaces(std::function<void(const NamedCTArgNamespaces&)>) const = 0;
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
