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

// Metal 2.0: precomputed layout of a kernel's common runtime args (CRTA) buffer.
//
// The CRTA buffer is laid out as three back-to-back sections:
//   [ user-named CRTAs | TensorBinding section | vararg CRTAs ]
//
// Sections 1 and 2 are fixed-size at spec-resolution time. This struct records their
// sizes (and the resulting vararg section start offset) so consumers don't have to
// re-derive them by walking the binding handles.
//
// Section 2 (TensorBinding) is variable-size: each binding contributes
// (1 + num_runtime_field_crta_words) words — the always-present base-address word, plus
// any runtime accessor fields the TensorParameter opted into (currently: shape, for
// sharded TensorParameters with dynamic_tensor_shape=true).
struct KernelCrtaLayout {
    // Section 1 size, in words. Equals the number of user-named CRTAs.
    uint32_t num_named_words = 0;
    // Section 2 size, in words. Equals the sum-over-bindings of (1 + num_runtime_field_crta_words).
    uint32_t binding_section_words = 0;
    // Start offset of section 3 (varargs), in words.
    // Stored (not computed on demand) so it can be set from a known value at spec resolution
    // and asserted against the derived sum if a consumer wants belt-and-suspenders verification.
    uint32_t vararg_section_offset = 0;
};

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
    //  - accessor_name: kernel-side identifier, used as the symbol name in the `tensor::` namespace
    //  - cta_offset: starting word index of this binding's CTA payload in the kernel's
    //    positional compile-time-args buffer
    //  - addr_crta_offset: byte offset of the implicit base-address CRTA within the kernel's
    //    common-runtime-args section
    //  - num_runtime_field_crta_words: number of CRTA words that immediately follow the address
    //    slot for runtime accessor fields (currently: shape, for sharded TensorParameters with
    //    dynamic_tensor_shape=true). The binding occupies (1 + num_runtime_field_crta_words)
    //    CRTA words in total.
    // (The tensor_parameter_name is also part of TensorBindingHandle, but we don't need it for codegen.)
    virtual void process_tensor_binding_handles(std::function<void(
                                                    const std::string& accessor_name,
                                                    uint32_t cta_offset,
                                                    uint32_t addr_crta_offset,
                                                    uint32_t num_runtime_field_crta_words)>) const {}

    // Named RTA/CRTA schema (Metal 2.0 APIs).
    // The order of names determines the byte offset of each arg within the named-args
    // section of the dispatch buffer.
    // Returned by const-ref rather than via a process_* callback because the concrete storage
    // is already an ordered vector — the callback indirection would just force a copy.
    virtual const std::vector<std::string>& get_runtime_arg_names() const {
        static const std::vector<std::string> k_empty;
        return k_empty;
    }
    virtual const std::vector<std::string>& get_common_runtime_arg_names() const {
        static const std::vector<std::string> k_empty;
        return k_empty;
    }

    // Metal 2.0: full CRTA buffer layout, precomputed at spec resolution time.
    // Default is the all-zero layout (no named CRTAs, no bindings, varargs start at offset 0),
    // which matches the legacy-kernel case where the buffer has only varargs.
    virtual KernelCrtaLayout get_crta_layout() const { return {}; }

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
