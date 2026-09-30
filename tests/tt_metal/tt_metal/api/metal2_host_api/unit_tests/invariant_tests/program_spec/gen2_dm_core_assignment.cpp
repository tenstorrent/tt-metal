// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// ProgramSpec structural invariant on Gen2 data-movement core assignment (program_spec.hpp): every DM
// kernel gets the same DM cores on every node it runs on, disjoint within a WorkUnitSpec and shared
// across one side of a DFB.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>

#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeMinimalDFB;
using test_helpers::MakeMinimalGen2ComputeKernel;
using test_helpers::MakeMinimalGen2DMKernel;
using test_helpers::MakeMinimalWorkUnit;
using test_helpers::ProgramSpecTestQuasar;

// Here, we test several edge cases:
//
// A) ALGORITHM FAILURE
//    The original naive greedy algorithm could either pass or fail on logically
//    equivalent ProgramSpecs, depending on the order of kernels and work_units.
//    The ProgramSpec was legal and solvable (under the simplifying assumption),
//    but the algorithm failed to find a solution.
//    This class of failure should not occur with the backtracking solver.
//
// B) SIMPLIFYING ASSUMPTION VIOLATION
//    Some strictly legal ProgramSpecs are unsolvable if we assume that a kernel
//    must uses the same processor indices on all nodes it runs on.
//
//    NOTE: Our plan is to keep the simplifying assumption for now.
//    We issue a clear message if the assumption is ever violated in the real world.
//
//   C) KERNELS COUPLED THROUGH MULTI-DFB BINDINGS
//    When a DFBSpec is bound by multiple KernelSpecs (which is legal, provided
//    that the invariant that any given DFB instance has one one producer kernel
//    instance and only one consumer kernel instance), the resulting cross-kernel
//    coupling through the shared DFB binding induces additional DM solver constraints.
//
//    NOTE: The original plan called for lifting this artificial constraint once LLK adopted
//    DFBBindingToken using implicit RTAs. However, to realize performance gains, we've
//    chosen instead to GUARANTEE using implicit CTAs for DFBBindingToken. This
//    constraint is therefore permanent.
//
//    If we were ever to start encountering serious unsolvable-Program issues as a result
//    we might consider revisiting this decision. However, an unsolvable Program could
//    can always be worked around by artificially dividing a KernelSpec (at the expense of
//    dispatch overhead).

// Category A: Order-Independence Test
// This test verifies that the backtracking solver finds valid assignments,
// regardless of kernel and work_unit orderings.
TEST_F(ProgramSpecTestQuasar, CPU_BacktrackingSolverFindsAssignment_RegardlessOfOrder) {
    // This test verifies that semantically identical ProgramSpecs succeed
    // regardless of the order of:
    //  - work_units in spec.work_units
    //  - kernels within each work_unit
    //  - kernel order within the ProgramSpec
    //
    // Scenario:
    //   - K_a:  3 threads on node A only (single-node)
    //   - K_ab: 3 threads on nodes A and B (multi-node)
    //   - K_bc: 3 threads on nodes B and C (multi-node)
    //   - K_c:  3 threads on node C only (single-node)
    //
    // Budget check (6 DM cores per node):
    //   Node A: K_a(3) + K_ab(3) = 6
    //   Node B: K_ab(3) + K_bc(3) = 6
    //   Node C: K_bc(3) + K_c(3) = 6
    //
    // SOLUTION EXISTS:
    //   K_ab uses [2-4] on A and B
    //   K_a  uses [5-7] on A
    //   K_bc uses [5-7] on B and C
    //   K_c  uses [2-4] on C
    //
    // The original naive greedy algorithm would fail on certain orderings:
    //   - Order [K_ab, K_bc, K_a, K_c] would succeed
    //   - Order [K_a, K_c, K_ab, K_bc] would fail
    //
    // The backtracking solver should be order-independent.
    // (Though it still makes the simplifying assumption.)

    NodeCoord node_a{0, 0};
    NodeCoord node_b{0, 1};
    NodeCoord node_c{0, 2};

    NodeRangeSet nodes_ab(std::set<NodeRange>{NodeRange{node_a, node_a}, NodeRange{node_b, node_b}});
    NodeRangeSet nodes_bc(std::set<NodeRange>{NodeRange{node_b, node_b}, NodeRange{node_c, node_c}});

    auto k_a = MakeMinimalGen2DMKernel("k_a", 3);
    auto k_ab = MakeMinimalGen2DMKernel("k_ab", 3);
    auto k_bc = MakeMinimalGen2DMKernel("k_bc", 3);
    auto k_c = MakeMinimalGen2DMKernel("k_c", 3);

    auto work_unit_a1 = MakeMinimalWorkUnit("work_unit_a1", node_a, {"k_a", "k_ab"});
    auto work_unit_b1 = MakeMinimalWorkUnit("work_unit_b1", node_b, {"k_ab", "k_bc"});
    auto work_unit_c1 = MakeMinimalWorkUnit("work_unit_c1", node_c, {"k_bc", "k_c"});
    auto work_unit_c2 = MakeMinimalWorkUnit("work_unit_c2", node_c, {"k_c", "k_bc"});

    // Helper to create a ProgramSpec with a given id and work_unit ordering
    auto make_spec =
        [&](const std::string& id, std::vector<KernelSpec> kernels, const std::vector<WorkUnitSpec>& work_units) {
            ProgramSpec spec;
            spec.name = id;
            spec.kernels = std::move(kernels);
            spec.work_units = work_units;
            return spec;
        };

    // All 24 possible permutations of k_a, k_ab, k_bc, k_c
    std::vector<std::vector<KernelSpec>> kernel_permutations = {
        {k_a, k_ab, k_bc, k_c}, {k_a, k_ab, k_c, k_bc}, {k_a, k_bc, k_ab, k_c},
        {k_a, k_bc, k_c, k_ab}, {k_a, k_c, k_ab, k_bc}, {k_a, k_c, k_bc, k_ab},

        {k_ab, k_a, k_bc, k_c}, {k_ab, k_a, k_c, k_bc}, {k_ab, k_bc, k_a, k_c},
        {k_ab, k_bc, k_c, k_a}, {k_ab, k_c, k_a, k_bc}, {k_ab, k_c, k_bc, k_a},

        {k_bc, k_a, k_ab, k_c}, {k_bc, k_a, k_c, k_ab}, {k_bc, k_ab, k_a, k_c},
        {k_bc, k_ab, k_c, k_a}, {k_bc, k_c, k_a, k_ab}, {k_bc, k_c, k_ab, k_a},

        {k_c, k_a, k_ab, k_bc}, {k_c, k_a, k_bc, k_ab}, {k_c, k_ab, k_a, k_bc},
        {k_c, k_ab, k_bc, k_a}, {k_c, k_bc, k_a, k_ab}, {k_c, k_bc, k_ab, k_a}};

    // All 6 possible permutations of work_unit orderings (using c1)
    std::vector<std::vector<WorkUnitSpec>> work_unit_permutations1 = {
        {work_unit_a1, work_unit_b1, work_unit_c1},
        {work_unit_a1, work_unit_c1, work_unit_b1},
        {work_unit_b1, work_unit_c1, work_unit_a1},
        {work_unit_b1, work_unit_a1, work_unit_c1},
        {work_unit_c1, work_unit_a1, work_unit_b1},
        {work_unit_c1, work_unit_b1, work_unit_a1}};

    // All 6 possible permutations of work_unit orderings (using c2)
    std::vector<std::vector<WorkUnitSpec>> work_unit_permutations2 = {
        {work_unit_a1, work_unit_b1, work_unit_c2},
        {work_unit_a1, work_unit_c2, work_unit_b1},
        {work_unit_b1, work_unit_c2, work_unit_a1},
        {work_unit_b1, work_unit_a1, work_unit_c2},
        {work_unit_c2, work_unit_a1, work_unit_b1},
        {work_unit_c2, work_unit_b1, work_unit_a1}};

    // All kernel permutations should succeed with all work_unit permutations.
    for (const auto& kernel_perm : kernel_permutations) {
        for (const auto& work_unit_perm : work_unit_permutations1) {
            EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, make_spec("", kernel_perm, work_unit_perm)));
        }
    }
    for (const auto& kernel_perm : kernel_permutations) {
        for (const auto& work_unit_perm : work_unit_permutations2) {
            EXPECT_NO_THROW(MakeProgramFromSpec(*mesh_device_, make_spec("", kernel_perm, work_unit_perm)));
        }
    }
}

// Category B: True Simplifying Assumption Violation
// This test is UNSOLVABLE with the simplifying assumption - no algorithm can help.
// When this case arises in production, the assumption must be removed from the codebase.
TEST_F(ProgramSpecTestQuasar, CPU_SimplifyingAssumptionViolation_OverlappingMultiNodeKernels) {
    // This test constructs a valid ProgramSpec that CANNOT work with the
    // "same DM cores on every node" simplifying assumption, regardless of
    // how clever the solver is.
    //
    // Scenario (the "triangle of doom"):
    //   - Kernel A (3 threads) runs on nodes (0,0) and (0,1)
    //   - Kernel B (3 threads) runs on nodes (0,0) and (0,2)
    //   - Kernel C (3 threads) runs on nodes (0,1) and (0,2)
    //
    // Per-node DM budget (6 DM cores available per node):
    //   Node (0,0): A(3) + B(3) = 6 ✓
    //   Node (0,1): A(3) + C(3) = 6 ✓
    //   Node (0,2): B(3) + C(3) = 6 ✓
    //
    // With simplifying assumption (each kernel uses same cores on all its nodes):
    //   Node (0,0): A and B must partition [2-7]. Say A=[2-4], B=[5-7].
    //   Node (0,1): A must be [2-4] (from above). So C=[5-7].
    //   Node (0,2): B must be [5-7], C must be [5-7]. CONFLICT!
    //
    // No matter how we assign, there's always a conflict on one node.
    // The simplifying assumption must be removed to handle this case.
    //
    // When this test starts PASSING: the simplifying assumption has been removed.
    // Change to EXPECT_NO_THROW.

    NodeCoord node_00{0, 0};
    NodeCoord node_01{0, 1};
    NodeCoord node_02{0, 2};

    NodeRangeSet nodes_A(std::set<NodeRange>{NodeRange{node_00, node_00}, NodeRange{node_01, node_01}});
    NodeRangeSet nodes_B(std::set<NodeRange>{NodeRange{node_00, node_00}, NodeRange{node_02, node_02}});
    NodeRangeSet nodes_C(std::set<NodeRange>{NodeRange{node_01, node_01}, NodeRange{node_02, node_02}});

    ProgramSpec spec;
    spec.name = "triangle_of_doom";

    auto kernel_a = MakeMinimalGen2DMKernel("kernel_a", 3);
    auto kernel_b = MakeMinimalGen2DMKernel("kernel_b", 3);
    auto kernel_c = MakeMinimalGen2DMKernel("kernel_c", 3);

    spec.kernels = {kernel_a, kernel_b, kernel_c};

    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("work_unit_00", node_00, {"kernel_a", "kernel_b"}),
        MakeMinimalWorkUnit("work_unit_01", node_01, {"kernel_a", "kernel_c"}),
        MakeMinimalWorkUnit("work_unit_02", node_02, {"kernel_b", "kernel_c"}),
    };

    // EXPECTED BEHAVIOR: FAILS due to simplifying assumption violation.
    EXPECT_THAT(
        [&] { MakeProgramFromSpec(*mesh_device_, spec); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("Failed to find valid processor assignments for DM kernels")));
}

// Category C: Coupling-group constraint exercised
// Multi-bound same-role DM kernels must end up with identical DM RISC masks (the DFB's
// hardware config carries one producer_risc_mask / consumer_risc_mask per side). The
// solver implements this by treating each coupling-group equivalence class as a single
// "super-kernel" with merged node coverage. Without that constraint, an unrelated DM
// kernel competing for lanes on one zone could push the producers to different lanes on
// their respective nodes — passing the greedy assignment but failing per-role mask
// uniformity.
TEST_F(ProgramSpecTestQuasar, CPU_DFBMultiBindingForcesUniformRiscMaskAcrossProducers) {
    // Scenario: zone-specialized DM producers (producer_a on node0, producer_b on node1)
    // both bound as PRODUCER of the same DFB. An unrelated 2-thread DM kernel on node0
    // consumes lanes DM2-DM3 (the lanes the un-constrained solver would greedily hand to
    // producer_a). If the producers weren't coupled, producer_a would get bumped to DM4
    // on node0 while producer_b kept DM2 on node1 — different masks. The coupling-group
    // solver instead picks a lane available on BOTH producer nodes first, then lets the
    // unrelated kernel work around it.
    NodeCoord node0{0, 0};
    NodeCoord node1{1, 0};

    ProgramSpec spec;
    spec.name = "test_program";

    auto producer_a = MakeMinimalGen2DMKernel("producer_a", /*num_threads=*/1);
    auto producer_b = MakeMinimalGen2DMKernel("producer_b", /*num_threads=*/1);
    auto unrelated_dm = MakeMinimalGen2DMKernel("unrelated_dm", /*num_threads=*/2);
    auto consumer = MakeMinimalGen2ComputeKernel("consumer");

    auto dfb = MakeMinimalDFB("dfb");
    dfb.data_format_metadata = tt::DataFormat::Float16_b;

    producer_a.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    producer_b.dfb_bindings.push_back(ProducerOf(DFBSpecName{"dfb"}, "out"));
    consumer.dfb_bindings.push_back(ConsumerOf(DFBSpecName{"dfb"}, "in"));
    // unrelated_dm intentionally has no DFB bindings — it just consumes DM lanes on node0.

    spec.kernels = {producer_a, producer_b, unrelated_dm, consumer};
    spec.dataflow_buffers = {dfb};
    spec.work_units = std::vector<WorkUnitSpec>{
        MakeMinimalWorkUnit("wu_g1", node0, {"producer_a", "unrelated_dm", "consumer"}),
        MakeMinimalWorkUnit("wu_g2", node1, {"producer_b", "consumer"}),
    };

    EXPECT_NO_THROW({ MakeProgramFromSpec(*mesh_device_, spec); });
}

}  // namespace
}  // namespace tt::tt_metal::experimental
