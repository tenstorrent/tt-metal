// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

//------------------------------------------------
// Temporary Metal 2.0 APIs
// (experimental namespace free functions)
//------------------------------------------------

// Create a Program object from a ProgramSpec
// (This will become a constructor for the Program class)
Program MakeProgramFromSpec(const ProgramSpec& spec, bool skip_validation = false);

// Configure the mutable parameters of an existing Program
// (This will become a member function for the Program class)
// This performs a copy from the ProgramRunParams to the Program's internal data structures.
//
// COMPLETENESS: You must specify runtime_args for every (kernel, node) pair that
// requires runtime arguments. Missing entries will cause an error.
//
// For high-performance inner loops, prefer the in-place power user API below.
// If stateful behavior of parameters is required, use the power user API.
void SetProgramRunParameters(Program& program, const ProgramRunParams& parameters);

// Power-user API for updating the mutable parameters of a Program in-place.
// ProgramRunParamsView is a non-owning view into the Program's command buffers,
// enabling in-place modification of mutable Program parameters.
// (Sketch only; not yet implemented)
ProgramRunParamsView GetProgramRunParamsView(Program& program);

// Useful? Might want to expose a const view for debug/test use?
// ProgramRunParamsConstView GetProgramRunParamsConstView(const Program& program);

}  // namespace tt::tt_metal::experimental::metal2_host_api

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
    void set_run_parameters(const ProgramRunParams& parameters);

    // Alternative API for setting the program execution parameters in-place,
    // modifying the underlying dispatch command buffers directly.
    ProgramRunParamsView get_run_params_view();


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
