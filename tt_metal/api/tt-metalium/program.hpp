// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>
#include <tt-metalium/workload/utils/table.hpp>

namespace tt::tt_metal {

// Fwd declares
struct ProgramDescriptor;
struct ProgramSpec;
class CircularBuffer;
class Program;

namespace detail {
class ProgramImpl;
}  // namespace detail

namespace distributed {
class MeshDevice;
}  // namespace distributed

using ProgramId = std::uint64_t;

class Program {
public:
    Program();

    // Alternative "ProgramDescriptor" API, created for TTNN generic op
    explicit Program(const ProgramDescriptor& descriptor);

    // Create a Program object from a ProgramSpec
    //
    // INVARIANT: A successfully constructed Program is semantically valid
    // and compiled, but has not yet been finalized for dispatch.
    // Kernel config layout and capacity are validated when the Program is
    // finalized as part of a MeshWorkload.
    //
    // PRE-CONDITION: If skip_validation is true, the caller guarantees that
    // the ProgramSpec satisfies all semantic validation requirements.
    //
    // Runtime arguments for Programs created from a ProgramSpec must be configured
    // through set_run_args or update_run_args, not the legacy host APIs.
    explicit Program(distributed::MeshDevice& mesh_device, const ProgramSpec& spec, bool skip_validation = false);

    // Internal: construct from an already-built ProgramImpl.
    explicit Program(std::shared_ptr<detail::ProgramImpl> impl);

    ~Program() noexcept;

    Program(const Program& other) = delete;
    Program& operator=(const Program& other) = delete;

    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    //////////////////////////////
    // ID related functions:
    // These are often used in tracing and testing.
    //////////////////////////////

    void set_runtime_id(ProgramId id);
    ProgramId get_runtime_id() const;

    //////////////////////////////
    // Run-args related functions:
    //////////////////////////////

    // Configure the arguments (mutable parameters) of an existing Program
    // This performs a copy from the ProgramRunArgs to the Program's internal data structures.
    //
    // COMPLETENESS: You must specify runtime args (named and vararg alike) for every
    // (kernel, node) pair that requires runtime arguments. Missing entries will cause an error.
    //
    // For high-performance inner loops, prefer the power user APIs below.
    // If stateful behavior of parameters is required, use the power user APIs.
    void set_run_args(const ProgramRunArgs& params, bool skip_validation = false);

    // Fast-path partial update: refresh an ARBITRARY subset of arguments for an existing Program.
    // Any argument not supplied exhibits STATEFUL behavior: it retains the value it was most recently
    // set to.
    //
    // PRE-CONDITION: set_run_args must have been called previously.
    //
    // CAUTION: It is the caller's responsibility to ensure that the stateful arguments being retained
    // are still valid in the new execution context. This is especially important for retained tensor
    // arguments (and other user-managed resource arguments): if the previously-bound object has gone
    // out of scope (and its device memory has been deallocated), re-enqueuing with the now-stale
    // binding is undefined behavior.
    //
    // USE CASE: Program re-enqueue loops where only a subset of ProgramRunArgs change per iteration.
    // This saves the host overhead of re-computing, re-specifying and re-validating the full
    // ProgramRunArgs when only a few arguments change per iteration.
    //
    // NOTE: DFB size overrides follow the same stateful rule — a DFB whose size is not overridden here
    //       retains its current size (as last set), rather than reverting to the ProgramSpec default.
    void update_run_args(const ProgramRunArgs& params, bool skip_validation = false);

    // Fast-path partial update: refresh ONLY the TensorArgs of an existing Program.
    // All other ProgramRunArgs (named/vararg RTAs and CRTAs, DFB params) retain their current values
    // (as most recently set by set_run_args or update_run_args).
    //
    // PRE-CONDITION: set_run_args must have been called previously.
    //
    // COMPLETENESS: A TensorArgument must be specified for every TensorParameter declared in the
    // ProgramSpec, exactly once. The supplied MeshTensor's TensorSpec must match the
    // TensorParameter's declared spec (subject to any declared TensorSpecRelaxations).
    //
    // PRE-CONDITION: If skip_validation is true, the caller guarantees the completeness and
    // spec-match requirements above.
    //
    // USE CASE: Program re-enqueue loops where the only per-enqueue ProgramRunArgs variation
    // is in the tensor args (i.e. which specific MeshTensors are operated on by the Program).
    void update_tensor_args(
        const Table<TensorParamName, ProgramRunArgs::TensorArgument>& tensor_args, bool skip_validation = false);

    //////////////////////////////
    // Buffer related functions:
    //////////////////////////////

    // Used in ops.
    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers() const;

    // debug/test/internal usage.
    detail::ProgramImpl& impl() { return *internal_; }
    const detail::ProgramImpl& impl() const { return *internal_; }

private:
    // The internal ProgramImpl may outlive the Program object if it's in-use by a command queue.
    std::shared_ptr<detail::ProgramImpl> internal_;
};

// Only Used in op_profiler, we might want to expose this via a tooling interface instead of through here.
class IDevice;
namespace detail {
struct KernelMeta;
// Collects the meta data of kernels in a program, and the metadata of the binaries within the kernel if device is non-null
// Note: device is nullable
[[deprecated(
    "Use experimental::collect_kernel_meta from <tt-metalium/experimental/op_profiler_support.hpp> instead. This "
    "overload will be removed after 2026-11-14.")]]
std::vector<detail::KernelMeta> collect_kernel_meta(const Program& program, IDevice* device);
}; //namespace detail

}  // namespace tt::tt_metal
