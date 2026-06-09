// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>

#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace tt::tt_metal::experimental {

//------------------------------------------------
// Temporary Metal 2.0 APIs
// (experimental namespace free functions)
//------------------------------------------------

// Create a Program object from a ProgramSpec
// (This will become a constructor for the Program class)
Program MakeProgramFromSpec(
    const distributed::MeshDevice& mesh_device, const ProgramSpec& spec, bool skip_validation = false);

// Configure the mutable parameters of an existing Program
// (This will become a member function for the Program class)
// This performs a copy from the ProgramRunArgs to the Program's internal data structures.
//
// COMPLETENESS: You must specify runtime args (named and vararg alike) for every
// (kernel, node) pair that requires runtime arguments. Missing entries will cause an error.
//
// For high-performance inner loops, prefer the power user APIs below.
// If stateful behavior of parameters is required, use the power user APIs.
void SetProgramRunArgs(Program& program, const ProgramRunArgs& params);

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
void UpdateTensorArgs(Program& program, std::span<const ProgramRunArgs::TensorArgument> tensor_args);

// Partial in-place update of a cached Program: re-applies ONLY the named RTAs (per node) and named
// CRTAs passed here, writing them straight into the dispatch buffers SetProgramRunArgs already laid
// out. Everything else is left as-is. PRE-CONDITION: SetProgramRunArgs ran once on this Program.
//
// Why this exists (it is not a convenience wrapper around SetProgramRunArgs):
// The only other way to refresh args on a cached Program is SetProgramRunArgs, which is
// all-or-nothing — it re-serializes the COMPLETE arg set for every node the program runs on. An op
// that legitimately caches across calls but varies a few hash-excluded scalars per dispatch (an RNG
// seed, a sampling [from,to), an optimizer lr/step) would then rebuild and rewrite every per-core
// work-split arg on every cache hit, none of which changed. On a multi-core op that re-serialization
// dominates the steady-state cost: measured on rand (WH B0, 512x512 bf16, 500 warm trials), the full
// SetProgramRunArgs re-apply is ~132us/dispatch versus ~40us for the descriptor op it replaces — a
// 3.3x regression against main on the cache-hit path, i.e. every dispatch after the first. Writing
// only the handful of changed scalars in place brings it back to ~39us, at parity with descriptor.
// Without a partial-update primitive, any multi-core op with per-dispatch dynamic scalars regresses
// against descriptor the moment its program is cached.
void ApplyDynamicArgs(Program& program, const ProgramRunArgs& dynamic_args);

// Power-user API for updating the mutable parameters of a Program in-place.
// ProgramRunArgsView is a non-owning view into the Program's command buffers,
// enabling in-place modification of mutable Program parameters.
// (Sketch only; not yet implemented. TBD if needed at all.)
ProgramRunArgsView& GetProgramRunArgsView(Program& program);

// Useful? Might want to expose a const view for debug/test use?
// ProgramRunArgsConstView GetProgramRunArgsConstView(const Program& program);

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

    // Single descriptor API for program execution parameters
    void set_run_args(const ProgramRunArgs& params);

    // Alternative API for setting the program execution parameters in-place,
    // modifying the underlying dispatch command buffers directly.
    ProgramRunArgsView get_run_args_view();


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
