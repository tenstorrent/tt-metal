// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <tt-metalium/workload/run_args/validation_mode.hpp>
#include <tt-metalium/workload/specification/tensor_parameter.hpp>
#include <tt-metalium/workload/utils/table.hpp>

namespace tt::tt_metal {

// Fwd declares
struct ProgramDescriptor;
struct ProgramSpec;
struct ProgramRunArgs;
class CircularBuffer;
class Program;

namespace distributed {
class MeshDevice;
}  // namespace distributed

namespace detail {
class ProgramImpl;
}  // namespace detail

using ProgramId = std::uint64_t;

class Program {
public:
    Program();

    // Create a Program object from a ProgramSpec
    //
    // INVARIANT: A successfully constructed Program is semantically valid
    // and compiled, but has not yet been finalized for dispatch.
    // Kernel config layout and capacity are validated when the Program is
    // finalized as part of a MeshWorkload.
    //
    // Runtime arguments for Programs created from a ProgramSpec must be configured through
    // set_program_run_args or update_program_run_args, not the legacy host APIs.
    explicit Program(distributed::MeshDevice& mesh_device, const ProgramSpec& spec);

    // Alternative "ProgramDescriptor" API, created for TTNN generic op
    explicit Program(const ProgramDescriptor& descriptor);

    // Same, but binds the Program to the MetalEnv that owns `mesh_device` (see CreateProgram(MeshDevice&)).
    // Transitional: goes away together with the ProgramDescriptor path.
    Program(distributed::MeshDevice& mesh_device, const ProgramDescriptor& descriptor);

    // Adds a descriptor's circular buffers, semaphores (at their descriptor ids) and kernels to this program, as the
    // descriptor constructor does; its cores must not overlap what the program already uses.
    void append(const ProgramDescriptor& descriptor);

    // Internal: construct from an already-built ProgramImpl.
    explicit Program(std::shared_ptr<detail::ProgramImpl> impl);

    ~Program() noexcept;

    Program(const Program& other) = delete;
    Program& operator=(const Program& other) = delete;

    Program(Program&& other) noexcept;
    Program& operator=(Program&& other) noexcept;

    //////////////////////////////
    // Functions to configure Program arguments
    // (Only applies when constructed from a ProgramSpec)
    //////////////////////////////

    // Configure the arguments (mutable parameters) of an existing Program
    // This performs a copy from the ProgramRunArgs to the Program's internal data structures.
    //
    // COMPLETENESS: You must specify runtime args (named and vararg alike) for every
    // (kernel, node) pair that requires runtime arguments. Missing entries will cause an error.
    //
    // For high-performance inner loops, prefer the power user APIs below.
    // If stateful behavior of parameters is required, use the power user APIs.
    //
    // PER-DEVICE VALUES: the args are set for every device the Program runs on. Every per-device
    // value (MeshWorkload::update_program_run_args_for_device) of the Program is dropped.
    void set_program_run_args(const ProgramRunArgs& params, ValidationMode validation = ValidationMode::FULL);

    // Fast-path partial update: refresh an ARBITRARY subset of arguments for an existing Program.
    // Any argument not supplied exhibits STATEFUL behavior: it retains the value it was most recently
    // set to.
    //
    // PRE-CONDITION: set_program_run_args must have been called previously.
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
    //
    // PER-DEVICE VALUES: the args supplied are set for every device the Program runs on, and their
    // per-device values (MeshWorkload::update_program_run_args_for_device) are dropped. Args not
    // supplied keep their per-device values.
    void update_program_run_args(const ProgramRunArgs& params, ValidationMode validation = ValidationMode::FULL);

    // Fast-path partial update: refresh ONLY the TensorArgs of an existing Program.
    // All other ProgramRunArgs (named/vararg RTAs and CRTAs, DFB params) retain their current values
    // (as most recently set by set_program_run_args or update_program_run_args).
    //
    // PRE-CONDITION: set_program_run_args must have been called previously.
    //
    // COMPLETENESS: A TensorArgument must be specified for every TensorParameter declared in the
    // ProgramSpec, exactly once. The supplied MeshTensor's TensorSpec must match the
    // TensorParameter's declared spec (subject to any declared TensorSpecRelaxations).
    //
    // PRE-CONDITION: If validation is ValidationMode::SKIP, the caller guarantees the completeness and
    // spec-match requirements above.
    //
    // USE CASE: Program re-enqueue loops where the only per-enqueue ProgramRunArgs variation
    // is in the tensor args (i.e. which specific MeshTensors are operated on by the Program).
    //
    // PER-DEVICE VALUES: tensor args cannot have per-device values, so every per-device value
    // (MeshWorkload::update_program_run_args_for_device) is kept.
    void update_tensor_args(
        const Table<TensorParamName, TensorArgument>& tensor_args, ValidationMode validation = ValidationMode::FULL);

    //////////////////////////////
    // ID related functions:
    // These are often used in tracing and testing.
    //////////////////////////////

    void set_runtime_id(ProgramId id);
    ProgramId get_runtime_id() const;

    //////////////////////////////
    // Buffer related functions:
    //////////////////////////////

    // Used in ops.
    std::vector<std::shared_ptr<CircularBuffer>> circular_buffers() const;

    // True once this Program's kernels have been JIT-compiled for at least one build
    // key. Exposed because ProgramImpl is not visible outside tt_metal, and callers
    // that compile programs in bulk (e.g. ttnn's up-front precompile) need to tell
    // "I compiled this" from "it arrived already compiled".
    bool is_compiled() const;

    // debug/test/internal usage.
    detail::ProgramImpl& impl() { return *internal_; }
    const detail::ProgramImpl& impl() const { return *internal_; }

private:
    // Shared by both descriptor constructors; `internal_` must already be set.
    void populate_from_descriptor(const ProgramDescriptor& descriptor);

    // The internal ProgramImpl may outlive the Program object if it's in-use by a command queue.
    std::shared_ptr<detail::ProgramImpl> internal_;
};

}  // namespace tt::tt_metal
