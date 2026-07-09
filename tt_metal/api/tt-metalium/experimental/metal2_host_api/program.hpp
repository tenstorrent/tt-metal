// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>

#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <unordered_map>

namespace tt::tt_metal::experimental {

//------------------------------------------------
// Temporary Metal 2.0 APIs
// (experimental namespace free functions)
//------------------------------------------------

// Create a Program object from a ProgramSpec
// (This will become a constructor for the Program class)
//
// INVARIANT: A successfully constructed Program is always valid.
//
Program MakeProgramFromSpec(
    const distributed::MeshDevice& mesh_device, const ProgramSpec& spec, bool skip_validation = false);

// Create a MeshWorkload object from a set of region-mapped ProgramSpecs
// (This will become a constructor for the MeshWorkload class)
//
// INVARIANT: A successfully constructed MeshWorkload is always valid.
//
distributed::MeshWorkload MakeMeshWorkloadFromSpecs(
    distributed::MeshDevice& mesh_device,
    const std::unordered_map<distributed::MeshCoordinateRange, ProgramSpec>& program_specs,
    bool skip_validation = false);

// Create a MeshWorkload object from single ProgramSpec,
// to be applied mesh-wide (SPMD)
// (This will become a constructor for the MeshWorkload class)
//
// INVARIANT: A successfully constructed MeshWorkload is always valid.
//
distributed::MeshWorkload MakeMeshWorkloadFromSpec(
    distributed::MeshDevice& mesh_device, const ProgramSpec& program_spec, bool skip_validation = false);

// Configure the arguments (mutable parameters) of an existing Program
// (This will become a member function for the Program class)
// This performs a copy from the ProgramRunArgs to the Program's internal data structures.
//
// COMPLETENESS: You must specify runtime args (named and vararg alike) for every
// (kernel, node) pair that requires runtime arguments. Missing entries will cause an error.
//
// For high-performance inner loops, prefer the power user APIs below.
// If stateful behavior of parameters is required, use the power user APIs.
void SetProgramRunArgs(Program& program, const ProgramRunArgs& params, bool skip_validation = false);

// Fast-path partial update: refresh ONLY a subset of arguments for an existing Program.
// All other arguments exhibit STATEFUL behavior: they retain their values from whatever they
// were most recently set to.
//
// PRE-CONDITION: SetProgramRunArgs must have been called previously.
//
// CAUTION: It is the caller's responsibility to ensure that the stateful, enqueue-invariant
// tensor and runtime arguments being retained are still valid in the new execution context.
//
// COMPLETENESS: Only those tensor and runtime arguments that have been specified in the
// ProgramSpec as enqueue-loop invariant (via the AdvancedOptions fields) may be omitted
// when calling UpdateProgramRunArgs. All regular arguments must be specified. This is enforced
// by runtime validation checks.
//
// USE CASE: Program re-enqueue loops where only a subset of ProgramRunArgs need to be mutated
// per iteration. This saves the host overhead of re-computing, re-specifying and re-validating
// the full ProgramRunArgs if only a few arguments change per iteration. The onus is on the
// programmer to ensure that the retained arguments remain valid across iterations.
//
// NOTE: DFB size overrides follow the same stateful rule — a DFB whose size is not overridden here
//       retains its current size (as last set), rather than reverting to the ProgramSpec default.
void UpdateProgramRunArgs(Program& program, const ProgramRunArgs& params, bool skip_validation = false);

// Fast-path partial update: refresh ONLY the TensorArgs of an existing Program.
// All other ProgramRunArgs (named/vararg RTAs and CRTAs, DFB params) retain their values
// from the most recent SetProgramRunArgs call.
//
// PRE-CONDITION: SetProgramRunArgs must have been called previously.
//
// COMPLETENESS: A TensorArgument must be specified for every TensorParameter declared in the
// ProgramSpec, exactly once. The supplied MeshTensor's TensorSpec must match the
// TensorParameter's declared spec.
//
// USE CASE: Program re-enqueue loops where the only per-enqueue ProgramRunArgs variation
// is in the tensor args (i.e. which specific MeshTensors are operated on by the Program).
void UpdateTensorArgs(Program& program, const Table<TensorParamName, ProgramRunArgs::TensorArgument>& tensor_args);

}  // namespace tt::tt_metal::experimental

// The code below is not compiled!
// This is a placeholder for the "post-experimental" desired Program object semantics.
// For now, it's here as documentation only.
#if 0

// Desired Program object semantics
//   - unique ownership
//   - moveable, non-copyable
//   - RAII semantics
//
//  The RAII "resource" here is the host-side dispatch command buffers.
//
// Invariant: A constructed Program is always valid.
//  - All legality checks are performed at construction time
//    (including JIT compilation of kernels)
//
// Tentative plan:
//   Current implementation (broken semantics):
//     The Program object looks like a user-managed RAII object... but only for 10 seconds.
//     - When you add your Program to a MeshWorkload, you transfer ownership to the MeshWorkload.
//     - The Program object's lifetime is actually that of its containing MeshWorkload.
//     - The dispatch command buffers aren't actually created until you enqueue the Program.
//     - But since you can only enqueue a MeshWorkload, the initial Program object (that you
//       temporarily hold) never actually owns anything while you hold it. Huh??
//
//   Proposed new semantics:
//     Create a MeshWorkload constructor that takes a vector/set of ProgramSpecs.
//     - Clear ownership model: MeshWorkload owns the creates Programs.
//     - Dispatch command buffers are created at construction, not at first enqueue.
//     - Still preserves the MeshWorkload multi-threaded Program creation optimization.
//
//     Note: We will still need a standalone Program constructor from a ProgramSpec for the
//     slow dispatch use case. But, it will TT_FATAL unless slow dispatch is enabled.
//     No fast dispatch data structures are created via this API.


class Program {
public:
    ///////////////////////////////////////////////////////////////
    // Special member functions
    ///////////////////////////////////////////////////////////////

    // Program constructor:
    //   - Program attributes (defined by the ProgramSpec) are immutable after construction.
    //   - Performs legality checks on the ProgramSpec
    //   - All Program creation work (including kernel JIT compilation) is performed at construction time.
    //     (Not at enqueue time, as previously done.)
    //   - Throws if construction fails
    explicit Program(const ProgramSpec& spec); // slow dispatch only

    // Destructor frees all resources
    // Program object owns host-side, device-mapped memory resources only.
    ~Program();

    // Program is moveable, but non-copyable
    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&) noexcept = default;
    Program& operator=(Program&&) noexcept = default;


    ///////////////////////////////////////////////////////////////
    // Parameterization
    ///////////////////////////////////////////////////////////////

    // Single descriptor API for program execution arguments
    void set_run_args(const ProgramRunArgs& params, bool skip_validation = false);

    // Partial update API for program execution arguments
    void update_run_args(const ProgramRunArgs& params, bool skip_validation = false);


    ///////////////////////////////////////////////////////////////
    // Program ID assignment
    // Optional; can be used in tracing and testing.
    ///////////////////////////////////////////////////////////////
    using ProgramId = std::uint64_t;
    void set_runtime_id(ProgramId id);
    ProgramId get_runtime_id() const;


private:
    std::shared_ptr<detail::ProgramImpl> internal_;
};

#endif
