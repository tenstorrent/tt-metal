// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KernelSpec fields that must change the kernel's JIT cache key (compute_hash).

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestQuasar;

// ----------------------------------------------------------------------------
// Kernel hash sensitivity to scratchpad bindings
// ----------------------------------------------------------------------------
//
// The kernel's JIT cache key is its compute_hash(); a scratchpad binding flows into the device-side
// codegen (the scratch:: namespace + the CRTA-injected base address) and into the kernel's
// ScratchpadBindingHandles, so a kernel that binds a scratchpad must NOT hash equal to one that
// doesn't — otherwise it would silently reuse a stale cached binary.

TEST_F(ProgramSpecTestQuasar, CPU_ScratchpadBindingAffectsKernelHash) {
    // Same kernel source, differing only in whether the kernel binds a scratchpad. The bound variant
    // carries an extra ScratchpadBindingHandle, so the hashes must differ.
    auto make_bound_spec = [] {
        ProgramSpec spec;
        spec.name = "scratchpad_hash_bound";
        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        dm_kernel.scratchpad_bindings.push_back(KernelSpec::ScratchpadBinding{
            .scratchpad_spec_name = ScratchpadSpecName{"scratch"}, .accessor_name = "scratch"});
        spec.kernels = {dm_kernel};
        spec.scratchpads = {ScratchpadSpec{.unique_id = ScratchpadSpecName{"scratch"}, .size_per_node = 1024}};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
        return spec;
    };
    auto make_unbound_spec = [] {
        ProgramSpec spec;
        spec.name = "scratchpad_hash_unbound";
        auto dm_kernel = MakeMinimalGen2DMKernel("dm_kernel");
        spec.kernels = {dm_kernel};
        spec.work_units = std::vector<WorkUnitSpec>{MakeMinimalWorkUnit("work_unit", NodeCoord{0, 0}, {"dm_kernel"})};
        return spec;
    };

    Program prog_bound = MakeProgramFromSpec(*mesh_device_, make_bound_spec());
    Program prog_unbound = MakeProgramFromSpec(*mesh_device_, make_unbound_spec());

    auto hash_bound = prog_bound.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    auto hash_unbound = prog_unbound.impl().get_kernel_by_spec_name("dm_kernel")->compute_hash();
    EXPECT_NE(hash_bound, hash_unbound)
        << "A kernel that binds a scratchpad must not share a JIT cache slot with one that doesn't.";
}

}  // namespace
}  // namespace tt::tt_metal::experimental
