// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace tt::tt_metal::experimental::metal2_host_api {

// Experimental Metal 2.0 API for creating a Program from a ProgramSpec (Metal 2.0 host API).
// This will become a constructor for the Program class.
Program MakeProgramFromSpec(const ProgramSpec& spec);

}  // namespace tt::tt_metal::experimental::metal2_host_api



#if 0

// Program object semantics
//   - unique ownership
//   - moveable, non-copyable
//   - RAII semantics
//
// Invariant: A constructed Program is always valid.
//  - All legality checks are performed at construction time
//    (including JIT compilation of kernels)


// Future plan:
//   The Program object is a user-managed RAII object... but only for 10 seconds.
//   When you add your Program to a MeshWorkload, you transfer ownership to the MeshWorkload.
//   The Program object's lifetime is that of its containing MeshWorkload.
//   The current Program has a broken ownership model. It only pretends that you can own it.
// 
//   Instead, the MeshWorkload API should directly take the ProgramSpec, and construct the 
//   Program object internally:
// 
//      my_mesh_workload.add_program(device_range, ProgramSpec{...});
// 
//   Then the ownership model is clean and obvious.
//   All the Program public APIs below would then be accessed via the MeshWorkload.
//   And, we could potentially implement an automatic TTNN-style program cache.



class Program {
public:
    ///////////////////////////////////////////////////////////////
    // Construction
    ///////////////////////////////////////////////////////////////

    // Program constructor:
    //   - Program attributes (defined by the ProgramSpec) are immutable after construction.
    //   - Performs legality checks on the ProgramSpec
    //   - All Program creation work (including kernel JIT compilation) is performed at construction time.
    //     (Not at enqueue time, as previously done.)
    //   - Throws if construction fails
    explicit Program(const ProgramSpec& spec);

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

    // Alternative APIs for setting the program execution parameters
    // These enable a minor optimization -- arguments can be constructed in-place,
    // within the existing Program's dispatch data structures (e.g., kernel arguments).
    // This is a slightly clunkier API, but it saves copying data at runtime.
    void set_runtime_arguments(...);
    void set_common_runtime_arguments(...);
    void set_dfb_size_parameters(...);
    void set_dfb_borrowed_memory_parameters(...);


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