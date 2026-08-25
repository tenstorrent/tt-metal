// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Tests for named args (CT and RT) with generated header.
//
// Verifies:
// 1. Named common runtime args delivered via blaze_rt_args::get<>()
// 2. Named per-core runtime args deliver different values per core
// 3. Named compile-time args accessible via blaze_ct_args:: namespace

#include <gtest/gtest.h>
#include <cstdint>
#include <map>
#include <string>
#include <variant>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "device_fixture.hpp"
#include "multi_device_fixture.hpp"

using namespace tt;
using namespace tt::tt_metal;
using NamedArgsTest = GenericMeshDeviceFixture;

TEST_F(NamedArgsTest, TensixTestNamedCommonAndPerCoreRuntimeArgs) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core0 = {0, 0};
    CoreCoord core1 = {1, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core0, core1)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const uint32_t expected_marker = 0xCAFE;
    const uint32_t core0_idx = 10;
    const uint32_t core1_idx = 20;

    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = cores,
        .named_compile_time_args = {{"my_kernel.param_a", 0}, {"my_kernel.param_b", 0}},
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"my_kernel.marker", expected_marker}},
                .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core0, core0_idx}, {core1, core1_idx}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };

    distributed::MeshWorkload workload;
    Program program(ProgramDescriptor{.kernels = {kernel}});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results_core0;
    detail::ReadFromDeviceL1(device, core0, write_addr, 2 * sizeof(uint32_t), results_core0);
    std::vector<uint32_t> results_core1;
    detail::ReadFromDeviceL1(device, core1, write_addr, 2 * sizeof(uint32_t), results_core1);

    EXPECT_EQ(results_core0[0], expected_marker) << "Core (0,0): marker should be 0xCAFE";
    EXPECT_EQ(results_core1[0], expected_marker) << "Core (1,0): marker should be 0xCAFE";
    EXPECT_EQ(results_core0[1], core0_idx) << "Core (0,0): core_idx should be 10";
    EXPECT_EQ(results_core1[1], core1_idx) << "Core (1,0): core_idx should be 20";
}

TEST_F(NamedArgsTest, TensixTestNamedArrayRuntimeArgs) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core = {0, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core, core)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const uint32_t prefix_val = 10;
    const std::vector<uint32_t> array_data = {100, 200, 300, 400};
    const uint32_t num_elements = static_cast<uint32_t>(array_data.size());

    // Expected: prefix + sum(array_data) = 10 + 100 + 200 + 300 + 400 = 1010
    uint32_t expected_sum = prefix_val;
    for (auto v : array_data) {
        expected_sum += v;
    }

    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_array_runtime_args_kernel.cpp",
        .core_ranges = cores,
        .defines =
            {
                {"WRITE_ADDRESS", std::to_string(write_addr)},
                {"NUM_ELEMENTS", std::to_string(num_elements)},
            },
        // Scalar named common RT arg
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"my_kernel.prefix", prefix_val}},
                // Array named common RT arg
                .named_common_runtime_arg_arrays = {{"my_kernel.data", array_data}},
            },
        .config = DataMovementConfigDescriptor{},
    };

    distributed::MeshWorkload workload;
    Program program(ProgramDescriptor{.kernels = {kernel}});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results;
    detail::ReadFromDeviceL1(device, core, write_addr, sizeof(uint32_t), results);

    EXPECT_EQ(results[0], expected_sum) << "Sum should be prefix(" << prefix_val << ") + sum(data) = " << expected_sum
                                        << ", got " << results[0];
}

TEST_F(NamedArgsTest, TensixTestNamedCompileTimeArgs) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core = {0, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core, core)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const uint32_t param_a = 42;
    const uint32_t param_b = 0xBEEF;

    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = cores,
        .named_compile_time_args = {{"my_kernel.param_a", param_a}, {"my_kernel.param_b", param_b}},
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"my_kernel.marker", 0}},
                .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core, 0}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };

    distributed::MeshWorkload workload;
    Program program(ProgramDescriptor{.kernels = {kernel}});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results;
    detail::ReadFromDeviceL1(device, core, write_addr, 4 * sizeof(uint32_t), results);

    EXPECT_EQ(results[2], param_a) << "blaze_ct_args::my_kernel::param_a should be 42";
    EXPECT_EQ(results[3], param_b) << "blaze_ct_args::my_kernel::param_b should be 0xBEEF";
}

TEST_F(NamedArgsTest, TensixTestNamedPerCoreArrayRuntimeArgs) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core0 = {0, 0};
    CoreCoord core1 = {1, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core0, core1)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const std::vector<uint32_t> core0_weights = {10, 20, 30};
    const std::vector<uint32_t> core1_weights = {100, 200, 300};
    const uint32_t num_elements = static_cast<uint32_t>(core0_weights.size());

    uint32_t expected_sum_core0 = 0;
    for (auto v : core0_weights) {
        expected_sum_core0 += v;
    }
    uint32_t expected_sum_core1 = 0;
    for (auto v : core1_weights) {
        expected_sum_core1 += v;
    }

    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_per_core_array_runtime_args_kernel.cpp",
        .core_ranges = cores,
        .defines =
            {
                {"WRITE_ADDRESS", std::to_string(write_addr)},
                {"NUM_ELEMENTS", std::to_string(num_elements)},
            },
        .blaze_named_args =
            {
                .named_per_core_runtime_arg_arrays =
                    {{"my_kernel.weights", {{core0, core0_weights}, {core1, core1_weights}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };

    distributed::MeshWorkload workload;
    Program program(ProgramDescriptor{.kernels = {kernel}});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results_core0;
    detail::ReadFromDeviceL1(device, core0, write_addr, sizeof(uint32_t), results_core0);
    std::vector<uint32_t> results_core1;
    detail::ReadFromDeviceL1(device, core1, write_addr, sizeof(uint32_t), results_core1);

    EXPECT_EQ(results_core0[0], expected_sum_core0)
        << "Core (0,0): sum should be " << expected_sum_core0 << ", got " << results_core0[0];
    EXPECT_EQ(results_core1[0], expected_sum_core1)
        << "Core (1,0): sum should be " << expected_sum_core1 << ", got " << results_core1[0];
}

// Covers the COMPUTE JIT compile path for the experimental named blaze_ct_args:: header.
// All tests above use DataMovementConfigDescriptor (the BRISC/NCRISC path via
// jit_build_genfiles_kernel_include); this one uses ComputeConfigDescriptor (the
// TRISC path via jit_build_genfiles_triscs_src + build_trisc_prolog). Before the
// genfiles relocation, named_args_generated.h was emitted per-source by build.cpp's
// compile_one and delivered via `-include` — a non-atomic write on a shared path
// (racy under multiprocess, and never carried to the remote/JIT-server path). This
// exercises the relocated, presence-gated prolog #include on the compute path.
TEST_F(NamedArgsTest, TensixTestNamedCompileTimeArgsComputeKernel) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core = {0, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core, core)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const uint32_t param_a = 42;
    const uint32_t param_b = 0xBEEF;

    KernelDescriptor kernel = {
        .kernel_source =
            "tests/tt_metal/tt_metal/test_kernels/compute/blaze_named_compile_time_args_compute_kernel.cpp",
        .core_ranges = cores,
        .named_compile_time_args = {{"my_kernel.param_a", param_a}, {"my_kernel.param_b", param_b}},
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        .config = ComputeConfigDescriptor{},
    };

    distributed::MeshWorkload workload;
    Program program(ProgramDescriptor{.kernels = {kernel}});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results;
    detail::ReadFromDeviceL1(device, core, write_addr, 2 * sizeof(uint32_t), results);

    EXPECT_EQ(results[0], param_a) << "blaze_ct_args::my_kernel::param_a should be 42 (compute path)";
    EXPECT_EQ(results[1], param_b) << "blaze_ct_args::my_kernel::param_b should be 0xBEEF (compute path)";
}

// Test 1: Mixed positional + named args coexist in the same kernel.
// Positional per-core and common RT args are set via runtime_args / common_runtime_args;
// named per-core and common RT args are set via named_*_runtime_args. The merge logic in
// program.cpp appends named values after positional ones. This verifies the index mapping
// is correct: positional at [0..N-1], named at [N..].
TEST_F(NamedArgsTest, TensixTestMixedPositionalAndNamedRuntimeArgs) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core = {0, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core, core)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const uint32_t positional_per_core = 111;
    const uint32_t positional_common = 222;
    const uint32_t named_per_core_val = 333;
    const uint32_t named_common_val = 444;

    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_mixed_positional_named_args_kernel.cpp",
        .core_ranges = cores,
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        // Positional per-core RT arg (index 0)
        .runtime_args = {{{core, {positional_per_core}}}},
        // Positional common RT arg (index 0)
        .common_runtime_args = {positional_common},
        // Named common RT arg (appended after positional common → index 1)
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"my_kernel.named_common", named_common_val}},
                // Named per-core RT arg (appended after positional per-core → index 1)
                .named_per_core_runtime_args = {{"my_kernel.named_per_core", {{core, named_per_core_val}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };

    distributed::MeshWorkload workload;
    Program program(ProgramDescriptor{.kernels = {kernel}});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results;
    detail::ReadFromDeviceL1(device, core, write_addr, 4 * sizeof(uint32_t), results);

    EXPECT_EQ(results[0], positional_per_core) << "Positional per-core RT arg at index 0";
    EXPECT_EQ(results[1], positional_common) << "Positional common RT arg at index 0";
    EXPECT_EQ(results[2], named_per_core_val) << "Named per-core RT arg (after positional)";
    EXPECT_EQ(results[3], named_common_val) << "Named common RT arg (after positional)";
}

// Test 2a: CT arg redefinition with same value succeeds (dedup).
// Two entries with the same name and value should be silently deduplicated.
TEST_F(NamedArgsTest, TensixTestCTArgDedupSameValue) {
    auto mesh_device = get_mesh_device();
    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core = {0, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core, core)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    const uint32_t param_a = 42;
    const uint32_t param_b = 0xBEEF;

    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = cores,
        // Duplicate param_a with the same value — should be silently deduplicated
        .named_compile_time_args =
            {{"my_kernel.param_a", param_a}, {"my_kernel.param_b", param_b}, {"my_kernel.param_a", param_a}},
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"my_kernel.marker", 0}},
                .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core, 0}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };

    distributed::MeshWorkload workload;
    Program program(ProgramDescriptor{.kernels = {kernel}});
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);

    std::vector<uint32_t> results;
    detail::ReadFromDeviceL1(device, core, write_addr, 4 * sizeof(uint32_t), results);

    EXPECT_EQ(results[2], param_a) << "Deduplicated blaze_ct_args::my_kernel::param_a should be 42";
    EXPECT_EQ(results[3], param_b) << "blaze_ct_args::my_kernel::param_b should be 0xBEEF";
}

// Test 2b: CT arg redefinition with conflicting values fails.
// Two entries with the same name but different values should TT_FATAL.
TEST_F(NamedArgsTest, TensixTestCTArgConflictFails) {
    auto mesh_device = get_mesh_device();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core = {0, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core, core)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    KernelDescriptor kernel = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = cores,
        // Same name, conflicting values — should fatal
        .named_compile_time_args = {{"my_kernel.param_a", 42}, {"my_kernel.param_a", 99}},
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"my_kernel.marker", 0}},
                .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core, 0}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };

    EXPECT_THROW(Program program(ProgramDescriptor{.kernels = {kernel}}), std::exception);
}

// Test 3: Invalid named arg identifiers fail during Program construction.
// Names must be valid C++ identifiers (alpha/underscore start, alphanumeric/underscore rest).
TEST_F(NamedArgsTest, TensixTestInvalidIdentifierFails) {
    auto mesh_device = get_mesh_device();
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    CoreCoord core = {0, 0};
    CoreRangeSet cores = std::set<CoreRange>({CoreRange(core, core)});

    const uint32_t write_addr = mesh_device->allocator()->get_base_allocator_addr(tt_metal::HalMemType::L1);

    // Invalid namespace: starts with a digit
    KernelDescriptor kernel_bad_ns = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = cores,
        .named_compile_time_args = {{"123bad.field", 1}},
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"123bad.marker", 0}},
                .named_per_core_runtime_args = {{"123bad.core_idx", {{core, 0}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };

    EXPECT_THROW(Program program(ProgramDescriptor{.kernels = {kernel_bad_ns}}), std::exception);

    // Invalid field: contains a hyphen
    KernelDescriptor kernel_bad_field = {
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = cores,
        .named_compile_time_args = {{"my_kernel.bad-field", 1}},
        .defines = {{"WRITE_ADDRESS", std::to_string(write_addr)}},
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"my_kernel.marker", 0}},
                .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core, 0}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };

    EXPECT_THROW(Program program(ProgramDescriptor{.kernels = {kernel_bad_field}}), std::exception);
}

// ============================================================================
// Descriptor program-cache HIT regression: named VALUES must not go stale.
//
// The descriptor hashers intentionally hash only the named-arg SCHEMA (see
// test_blaze_named_args_hashing.cpp), so two invocations with the same schema
// but different named values land on the SAME cached program.  On a hit the
// framework re-applies runtime args via apply_descriptor_runtime_args()
// (mesh_device_operation_adapter.hpp); before the fix that copied only the
// positional KernelDescriptor::runtime_args/common_runtime_args, silently
// keeping the first invocation's named scalars and arrays.
//
// Reproduced host-side (no device, plain TEST): Program{desc_v1} plays the
// cache miss and apply_descriptor_runtime_args(program, desc_v2) plays the
// hit; GetRuntimeArgs/GetCommonRuntimeArgs then expose exactly the values the
// device would read on the second enqueue.
// ============================================================================

namespace {

// One kernel on two cores with all four named-RT-arg variants plus positional
// args.  The schema (names, array lengths, dispatch kinds, order) is fixed;
// `base` shifts every runtime VALUE, so two invocations share a cache hash
// but carry different data.
KernelDescriptor blaze_cache_hit_kernel(const CoreCoord& core0, const CoreCoord& core1, uint32_t base) {
    return KernelDescriptor{
        .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
        .core_ranges = CoreRangeSet(std::set<CoreRange>({CoreRange(core0, core1)})),
        .runtime_args = {{core0, {base + 1}}, {core1, {base + 2}}},
        .common_runtime_args = {base + 3},
        .blaze_named_args =
            {
                .named_common_runtime_args = {{"my_kernel.marker", base + 4}},
                .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core0, base + 5}, {core1, base + 6}}}},
                .named_common_runtime_arg_arrays = {{"my_kernel.data", {base + 7, base + 8}}},
                .named_per_core_runtime_arg_arrays =
                    {{"my_kernel.weights", {{core0, {base + 9, base + 10}}, {core1, {base + 11, base + 12}}}}},
            },
        .config = DataMovementConfigDescriptor{},
    };
}

// Asserts the merged layout for the descriptor built with `base`: positional
// slots first, then named scalars, then named arrays.
void blaze_expect_cache_hit_values(
    const Program& program, const CoreCoord& core0, const CoreCoord& core1, uint32_t base) {
    const auto& core0_args = GetRuntimeArgs(program, 0, core0);
    const auto& core1_args = GetRuntimeArgs(program, 0, core1);
    const auto& common_args = GetCommonRuntimeArgs(program, 0);
    ASSERT_EQ(core0_args.size(), 4u);
    ASSERT_EQ(core1_args.size(), 4u);
    ASSERT_EQ(common_args.size(), 4u);

    EXPECT_EQ(core0_args[0], base + 1) << "core0 positional per-core arg";
    EXPECT_EQ(core0_args[1], base + 5) << "core0 named per-core scalar (after positional)";
    EXPECT_EQ(core0_args[2], base + 9) << "core0 named per-core array[0]";
    EXPECT_EQ(core0_args[3], base + 10) << "core0 named per-core array[1]";

    EXPECT_EQ(core1_args[0], base + 2) << "core1 positional per-core arg";
    EXPECT_EQ(core1_args[1], base + 6) << "core1 named per-core scalar (after positional)";
    EXPECT_EQ(core1_args[2], base + 11) << "core1 named per-core array[0]";
    EXPECT_EQ(core1_args[3], base + 12) << "core1 named per-core array[1]";

    EXPECT_EQ(common_args[0], base + 3) << "positional common arg";
    EXPECT_EQ(common_args[1], base + 4) << "named common scalar (after positional)";
    EXPECT_EQ(common_args[2], base + 7) << "named common array[0]";
    EXPECT_EQ(common_args[3], base + 8) << "named common array[1]";
}

}  // namespace

TEST(NamedArgsDescriptorCacheHit, SameSchemaDifferentValuesShareProgramHash) {
    // Premise of the staleness bug (and of the regression test below): the two
    // invocations must land on the SAME cache entry.  Values are deliberately
    // excluded from the schema hash; if that ever changes, this test fails first
    // and the tests below lose their meaning.
    const CoreCoord core0{0, 0};
    const CoreCoord core1{1, 0};
    ProgramDescriptor desc_v1{.kernels = {blaze_cache_hit_kernel(core0, core1, 0)}};
    ProgramDescriptor desc_v2{.kernels = {blaze_cache_hit_kernel(core0, core1, 1000)}};
    EXPECT_EQ(std::hash<ProgramDescriptor>{}(desc_v1), std::hash<ProgramDescriptor>{}(desc_v2))
        << "Same named-arg schema with different values must share a program-cache hash";
}

TEST(NamedArgsDescriptorCacheHit, NamedValuesReappliedOnCacheHit) {
    const CoreCoord core0{0, 0};
    const CoreCoord core1{1, 0};
    ProgramDescriptor desc_v1{.kernels = {blaze_cache_hit_kernel(core0, core1, 0)}};
    ProgramDescriptor desc_v2{.kernels = {blaze_cache_hit_kernel(core0, core1, 1000)}};
    ASSERT_EQ(std::hash<ProgramDescriptor>{}(desc_v1), std::hash<ProgramDescriptor>{}(desc_v2))
        << "Test premise: both invocations target the same cache entry";

    // "Cache miss": first invocation constructs the program; process_named_args
    // merges the named values after the positional slots.
    Program program{desc_v1};
    blaze_expect_cache_hit_values(program, core0, core1, 0);

    // "Cache hit": second invocation, same schema, new values — exactly the
    // framework's slow cache-hit path.  The program must observe desc_v2's values.
    apply_descriptor_runtime_args(program, desc_v2);
    blaze_expect_cache_hit_values(program, core0, core1, 1000);
}

TEST(NamedArgsDescriptorCacheHit, NamedOnlyValuesReappliedOnCacheHit) {
    // No positional args at all: before the fix, apply_descriptor_runtime_args had
    // nothing to copy for this descriptor, so EVERY named value stayed frozen at the
    // first invocation's.  Covers all four named-RT-arg variants in this configuration.
    const CoreCoord core{0, 0};
    auto make_kernel =
        [&core](
            uint32_t prefix, uint32_t core_idx, uint32_t weight0, uint32_t weight1, uint32_t data0, uint32_t data1) {
            return KernelDescriptor{
                .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
                .core_ranges = CoreRangeSet(std::set<CoreRange>({CoreRange(core)})),
                .blaze_named_args =
                    {
                        .named_common_runtime_args = {{"my_kernel.prefix", prefix}},
                        .named_per_core_runtime_args = {{"my_kernel.core_idx", {{core, core_idx}}}},
                        .named_common_runtime_arg_arrays = {{"my_kernel.data", {data0, data1}}},
                        .named_per_core_runtime_arg_arrays = {{"my_kernel.weights", {{core, {weight0, weight1}}}}},
                    },
                .config = DataMovementConfigDescriptor{},
            };
        };
    ProgramDescriptor desc_v1{.kernels = {make_kernel(10, 1, 5, 6, 100, 200)}};
    ProgramDescriptor desc_v2{.kernels = {make_kernel(20, 2, 7, 8, 300, 400)}};
    ASSERT_EQ(std::hash<ProgramDescriptor>{}(desc_v1), std::hash<ProgramDescriptor>{}(desc_v2))
        << "Test premise: both invocations target the same cache entry";

    Program program{desc_v1};
    apply_descriptor_runtime_args(program, desc_v2);

    const auto& per_core_args = GetRuntimeArgs(program, 0, core);
    const auto& common_args = GetCommonRuntimeArgs(program, 0);
    ASSERT_EQ(per_core_args.size(), 3u);
    ASSERT_EQ(common_args.size(), 3u);
    EXPECT_EQ(per_core_args[0], 2u) << "named per-core scalar must be desc_v2's value";
    EXPECT_EQ(per_core_args[1], 7u) << "named per-core array[0] must be desc_v2's value";
    EXPECT_EQ(per_core_args[2], 8u) << "named per-core array[1] must be desc_v2's value";
    EXPECT_EQ(common_args[0], 20u) << "named common scalar must be desc_v2's value";
    EXPECT_EQ(common_args[1], 300u) << "named common array[0] must be desc_v2's value";
    EXPECT_EQ(common_args[2], 400u) << "named common array[1] must be desc_v2's value";
}

TEST(NamedArgsDescriptorCacheHit, RepeatedCacheHitsKeepReapplyingValues) {
    // Repeated same-schema invocations are the production norm.  The patch must be a
    // pure function of the current descriptor on EVERY hit — a works-once or otherwise
    // stateful re-application (e.g. one that corrupts sizes or skips later hits) must
    // not slip through.
    const CoreCoord core0{0, 0};
    const CoreCoord core1{1, 0};
    ProgramDescriptor desc_v1{.kernels = {blaze_cache_hit_kernel(core0, core1, 0)}};
    ProgramDescriptor desc_v2{.kernels = {blaze_cache_hit_kernel(core0, core1, 1000)}};
    ProgramDescriptor desc_v3{.kernels = {blaze_cache_hit_kernel(core0, core1, 2000)}};
    ASSERT_EQ(std::hash<ProgramDescriptor>{}(desc_v1), std::hash<ProgramDescriptor>{}(desc_v3))
        << "Test premise: all invocations target the same cache entry";

    Program program{desc_v1};
    apply_descriptor_runtime_args(program, desc_v2);
    blaze_expect_cache_hit_values(program, core0, core1, 1000);

    apply_descriptor_runtime_args(program, desc_v3);
    blaze_expect_cache_hit_values(program, core0, core1, 2000);
}

TEST(NamedArgsDescriptorCacheHit, MultipleNamedArgsPerVariantReappliedOnCacheHit) {
    // Two entries per variant (arrays of different lengths) pack into adjacent slots
    // in declaration order: positional, then scalars, then arrays.  Every packed
    // offset must be re-applied on a cache hit, not just the first named slot.
    const CoreCoord core{0, 0};
    auto make_kernel = [&core](uint32_t base) {
        return KernelDescriptor{
            .kernel_source = "tests/tt_metal/tt_metal/test_kernels/misc/blaze_named_runtime_args_kernel.cpp",
            .core_ranges = CoreRangeSet(std::set<CoreRange>({CoreRange(core)})),
            .runtime_args = {{core, {base + 11}}},
            .common_runtime_args = {base + 1},
            .blaze_named_args =
                {
                    .named_common_runtime_args = {{"my_kernel.a", base + 2}, {"my_kernel.b", base + 3}},
                    .named_per_core_runtime_args =
                        {{"my_kernel.c", {{core, base + 12}}}, {"my_kernel.d", {{core, base + 13}}}},
                    .named_common_runtime_arg_arrays =
                        {{"my_kernel.e", {base + 4, base + 5}}, {"my_kernel.f", {base + 6, base + 7, base + 8}}},
                    .named_per_core_runtime_arg_arrays =
                        {{"my_kernel.g", {{core, {base + 14, base + 15}}}},
                         {"my_kernel.h", {{core, {base + 16, base + 17, base + 18}}}}},
                },
            .config = DataMovementConfigDescriptor{},
        };
    };
    ProgramDescriptor desc_v1{.kernels = {make_kernel(0)}};
    ProgramDescriptor desc_v2{.kernels = {make_kernel(1000)}};
    ASSERT_EQ(std::hash<ProgramDescriptor>{}(desc_v1), std::hash<ProgramDescriptor>{}(desc_v2))
        << "Test premise: both invocations target the same cache entry";

    // Merged layouts (8 slots each):
    //   common:   [pos, a, b, e0, e1, f0, f1, f2] = base + [1..8]
    //   per-core: [pos, c, d, g0, g1, h0, h1, h2] = base + [11..18]
    auto expect_values = [&core](const Program& program, uint32_t base) {
        const auto& per_core_args = GetRuntimeArgs(program, 0, core);
        const auto& common_args = GetCommonRuntimeArgs(program, 0);
        ASSERT_EQ(per_core_args.size(), 8u);
        ASSERT_EQ(common_args.size(), 8u);
        for (uint32_t i = 0; i < 8; ++i) {
            EXPECT_EQ(common_args[i], base + 1 + i) << "common merged slot " << i;
            EXPECT_EQ(per_core_args[i], base + 11 + i) << "per-core merged slot " << i;
        }
    };

    Program program{desc_v1};
    expect_values(program, 0);

    apply_descriptor_runtime_args(program, desc_v2);
    expect_values(program, 1000);
}
