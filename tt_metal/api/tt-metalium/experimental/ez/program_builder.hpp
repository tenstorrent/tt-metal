// SPDX-FileCopyrightText: (c) 2026 Olof Johansson <olof@lixom.net>
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>

namespace tt::tt_metal::experimental::ez {

using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

class ProgramBuilder;

// Handle to a kernel added via ProgramBuilder, supporting fluent runtime_args configuration.
class KernelRef {
public:
    // Set runtime args applied uniformly across all cores in the kernel's core spec.
    KernelRef& runtime_args(std::initializer_list<uint32_t> args);
    KernelRef& runtime_args(const std::vector<uint32_t>& args);

    // Set per-core runtime args via a lambda called for each core in the kernel's core spec.
    KernelRef& runtime_args(std::function<std::vector<uint32_t>(const CoreCoord&)> fn);

    // Set runtime args for a specific core.
    KernelRef& runtime_args_at(const CoreCoord& core, const std::vector<uint32_t>& args);

    // Return to the ProgramBuilder for chaining.
    ProgramBuilder& done();

    // Access the underlying kernel handle.
    KernelHandle handle() const;

private:
    friend class ProgramBuilder;
    KernelRef(ProgramBuilder& builder, KernelHandle handle, CoreSpec core_spec);

    ProgramBuilder& builder_;
    KernelHandle handle_;
    CoreSpec core_spec_;
};

// Fluent builder for constructing a Program with circular buffers and kernels.
class ProgramBuilder {
public:
    explicit ProgramBuilder(const CoreSpec& core_spec);

    // Non-copyable and non-movable: KernelRef objects hold references back to
    // this builder, so moving it would leave dangling references.
    ProgramBuilder(const ProgramBuilder&) = delete;
    ProgramBuilder& operator=(const ProgramBuilder&) = delete;
    ProgramBuilder(ProgramBuilder&&) = delete;
    ProgramBuilder& operator=(ProgramBuilder&&) = delete;

    // Add a circular buffer with sensible defaults (bfloat16, tile-sized pages).
    ProgramBuilder& cb(tt::CBIndex index, uint32_t num_tiles = 2,
                       tt::DataFormat fmt = tt::DataFormat::Float16_b);

    // Add a circular buffer with explicit format, tile count, and page size.
    ProgramBuilder& cb(tt::CBIndex index, tt::DataFormat fmt, uint32_t num_tiles, uint32_t page_size);

    // Add a circular buffer backed by an existing L1 MeshBuffer (no extra allocation).
    ProgramBuilder& cb(tt::CBIndex index, const std::shared_ptr<distributed::MeshBuffer>& l1_buffer,
                       uint32_t num_tiles = 0, tt::DataFormat fmt = tt::DataFormat::Float16_b);

    // Add a circular buffer with a raw CircularBufferConfig for advanced use cases
    // such as shared circular buffer indices (multiple CB indices aliasing the same L1 memory).
    ProgramBuilder& cb(const CircularBufferConfig& config);

    // Add a reader data movement kernel (RISCV_1, architecture-preferred read NOC).
    // compile_args are placed first, followed by auto-generated TensorAccessorArgs from buffers.
    // This matches the dominant codebase convention (e.g. CB indices before accessor metadata).
    //
    // Note: buffers must use interleaved layout. Sharded buffers require runtime args
    // (via SetCommonRuntimeArgs) which this API does not support. For sharded buffers,
    // use the lower-level CreateKernel/SetRuntimeArgs APIs directly.
    KernelRef& reader(
        const std::string& path,
        const std::vector<std::shared_ptr<distributed::MeshBuffer>>& buffers = {},
        const std::vector<uint32_t>& compile_args = {});

    // Add a writer data movement kernel (RISCV_0, architecture-preferred write NOC).
    // compile_args are placed first, followed by auto-generated TensorAccessorArgs from buffers.
    //
    // Note: buffers must use interleaved layout. Sharded buffers require runtime args
    // (via SetCommonRuntimeArgs) which this API does not support. For sharded buffers,
    // use the lower-level CreateKernel/SetRuntimeArgs APIs directly.
    KernelRef& writer(
        const std::string& path,
        const std::vector<std::shared_ptr<distributed::MeshBuffer>>& buffers = {},
        const std::vector<uint32_t>& compile_args = {});

    // Add a compute kernel with the given math fidelity.
    KernelRef& compute(
        const std::string& path,
        MathFidelity fidelity = MathFidelity::HiFi4,
        const std::vector<uint32_t>& compile_args = {});

    // Add a compute kernel with a full ComputeConfig.
    KernelRef& compute(const std::string& path, const ComputeConfig& config);

    // Add a kernel with an arbitrary config (full control).
    KernelRef& kernel(
        const std::string& path,
        const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config);

    // Override core spec for the next kernel or CB operation only.
    ProgramBuilder& on(const CoreSpec& core_spec);

    // Set preprocessor defines for the next kernel only (consumed by the next
    // reader/writer/compute call, similar to how on() works for core spec).
    // Each define produces a #define in the kernel's generated header file.
    ProgramBuilder& defines(const std::map<std::string, std::string>& defs);

    // Set named compile-time args for the next kernel only (consumed by the next
    // reader/writer/compute call, similar to how on() works for core spec).
    // Named args let device kernels access compile-time values by string name
    // via get_named_compile_time_arg_val("name") instead of by positional index.
    ProgramBuilder& named_args(const std::unordered_map<std::string, uint32_t>& args);

    // Create a semaphore on the given cores (or default core spec) and return its address.
    uint32_t semaphore(uint32_t initial_value = 0);
    uint32_t semaphore(const CoreSpec& cores, uint32_t initial_value = 0);

    // Finalize and return the built Program. Can only be called once.
    Program build();

private:
    friend class KernelRef;

    Program program_;
    CoreSpec default_core_spec_;
    std::optional<CoreSpec> override_core_spec_;
    std::optional<std::map<std::string, std::string>> override_defines_;
    std::optional<std::unordered_map<std::string, uint32_t>> override_named_compile_args_;
    std::vector<std::unique_ptr<KernelRef>> kernel_refs_;
    bool built_ = false;

    CoreSpec active_core_spec();
    std::map<std::string, std::string> active_defines();
    std::unordered_map<std::string, uint32_t> active_named_compile_args();
};

}  // namespace tt::tt_metal::experimental::ez
