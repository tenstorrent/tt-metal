// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// UpdateTensorArgs: the fast path that patches only tensor binding address slots.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>
#include <tt-metalium/distributed.hpp>

#include "impl/kernels/kernel.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers.hpp"
#include "metal2_host_api/mock_device_fixtures.hpp"
#include "metal2_host_api/run_args_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::BindTensorParameterToKernel;
using test_helpers::MakeKernelRunArgs;
using test_helpers::MakeMinimalGen1ValidProgramSpec;
using test_helpers::MakeMinimalTensorParameter;
using test_helpers::MakeRunArgsForMinimalSpec;
using test_helpers::ProgramRunArgsTestGen1;
using test_helpers::ReadBindingAddressFromCRTA;

// ============================================================================
// UpdateTensorArgs Tests (Gen1 / WH)
// ============================================================================
// UpdateTensorArgs is the partial-update fast path: it patches only the tensor binding
// address slots in the per-kernel CRTA buffer, leaving everything else (named CRTAs,
// vararg CRTAs, all RTAs) statefully unchanged from the most recent SetProgramRunArgs
// call.

// Helper: allocate a MeshTensor matching the given TensorParameter's spec.
inline MeshTensor AllocateTensorForBinding(distributed::MeshDevice& mesh_device, const TensorParameter& binding) {
    return MeshTensor::allocate_on_device(mesh_device, binding.spec);
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateTensorArgs_BeforeSetProgramRunArgsFails) {
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    Table<TensorParamName, TensorArgument> tensor_args{
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };

    // No SetProgramRunArgs call before the partial update.
    EXPECT_THAT(
        [&] { UpdateTensorArgs(program, tensor_args); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("UpdateTensorArgs called on Program before SetProgramRunArgs")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateTensorArgs_MissingTensorArgFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Establish baseline state.
    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    SetProgramRunArgs(program, params);

    // Empty tensor_args fails the completeness check.
    Table<TensorParamName, TensorArgument> empty;
    EXPECT_THAT(
        [&] { UpdateTensorArgs(program, empty); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "TensorParameter 'input_tensor' is declared in the Program but has no TensorArgument entry")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateTensorArgs_UnknownTensorParameterFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    SetProgramRunArgs(program, params);

    Table<TensorParamName, TensorArgument> tensor_args{
        {TensorParamName{"ghost_tensor"}, TensorArgument{tensor}},
    };
    EXPECT_THAT(
        [&] { UpdateTensorArgs(program, tensor_args); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("TensorArgument references unknown TensorParameter 'ghost_tensor'")));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateTensorArgs_TensorSpecMismatchFails) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // shape {1, 32}
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    SetProgramRunArgs(program, params);

    // Different shape: {1, 64} instead of declared {1, 32}.
    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 64}, tensor_layout);
    MeshTensor wrong_tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec);

    Table<TensorParamName, TensorArgument> tensor_args{
        {TensorParamName{"input_tensor"}, TensorArgument{wrong_tensor}},
    };
    EXPECT_THAT(
        [&] { UpdateTensorArgs(program, tensor_args); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
            "TensorArgument for binding 'input_tensor' supplied a MeshTensor whose TensorSpec does not match")));
}

// Locks the gate's polarity: the three tests above prove UpdateTensorArgs validates by default, so
// this one proves skip_validation=true is what turns it off — and still performs the address patch.
TEST_F(ProgramRunArgsTestGen1, CPU_UpdateTensorArgs_SkipValidationBypassesSpecCheck) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");  // shape {1, 32}
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor}},
    };
    SetProgramRunArgs(program, params);

    auto page_config = tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR);
    auto memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    auto tensor_layout = tt::tt_metal::TensorLayout(tt::tt_metal::DataType::BFLOAT16, page_config, memory_config);
    auto wrong_spec = tt::tt_metal::TensorSpec(tt::tt_metal::Shape{1, 64}, tensor_layout);
    MeshTensor wrong_tensor = MeshTensor::allocate_on_device(*mesh_device_, wrong_spec);

    Table<TensorParamName, TensorArgument> tensor_args{
        {TensorParamName{"input_tensor"}, TensorArgument{wrong_tensor}},
    };
    EXPECT_NO_THROW(UpdateTensorArgs(program, tensor_args, /*skip_validation=*/true));
    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(wrong_tensor.address()))
        << "skip_validation should bypass only the checks, not the address patch";
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateTensorArgs_PatchesBindingAddress) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // First enqueue: full SetProgramRunArgs with tensor T1.
    MeshTensor tensor1 = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor1}},
    };
    SetProgramRunArgs(program, params);
    ASSERT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(tensor1.address()));

    // Second enqueue: partial update to tensor T2.
    MeshTensor tensor2 = AllocateTensorForBinding(*mesh_device_, binding);
    ASSERT_NE(tensor1.address(), tensor2.address())
        << "Test pre-condition: two separate allocations should yield distinct addresses";

    Table<TensorParamName, TensorArgument> tensor_args{
        {TensorParamName{"input_tensor"}, TensorArgument{tensor2}},
    };
    EXPECT_NO_THROW(UpdateTensorArgs(program, tensor_args));

    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(tensor2.address()));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateTensorArgs_LeavesNamedCRTAsUnchanged) {
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("input_tensor");
    spec.tensor_parameters = {binding};
    // Schema with a named CRTA preceding the binding-address section.
    spec.kernels[0].runtime_arg_schema.common_runtime_arg_names = {"tile_count"};
    BindTensorParameterToKernel(spec.kernels[0], "input_tensor", "input_ta");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor1 = AllocateTensorForBinding(*mesh_device_, binding);
    constexpr uint32_t kNamedCRTAValue = 0xABCD1234;
    ProgramRunArgs params;
    params.kernel_run_args.push_back(ProgramRunArgs::KernelRunArgs{
        .kernel = KernelSpecName{"dm_kernel"},
        .common_runtime_arg_values = {{"tile_count", kNamedCRTAValue}},
    });
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"compute_kernel"}, node, {}, {}));
    params.tensor_args = {
        {TensorParamName{"input_tensor"}, TensorArgument{tensor1}},
    };
    SetProgramRunArgs(program, params);

    // Capture the named CRTA's slot value before the partial update.
    auto dm_kernel = program.impl().get_kernel_by_spec_name("dm_kernel");
    const auto* crta_data_before = dm_kernel->common_runtime_args_data().data();
    ASSERT_EQ(crta_data_before[0], kNamedCRTAValue) << "named CRTA should occupy slot 0 of the CRTA buffer";

    // Partial update.
    MeshTensor tensor2 = AllocateTensorForBinding(*mesh_device_, binding);
    Table<TensorParamName, TensorArgument> tensor_args{
        {TensorParamName{"input_tensor"}, TensorArgument{tensor2}},
    };
    UpdateTensorArgs(program, tensor_args);

    // Named CRTA preserved; binding address patched.
    const auto* crta_data_after = dm_kernel->common_runtime_args_data().data();
    EXPECT_EQ(crta_data_after[0], kNamedCRTAValue) << "named CRTA must be untouched by UpdateTensorArgs";
    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "input_tensor"), static_cast<uint32_t>(tensor2.address()));
}

TEST_F(ProgramRunArgsTestGen1, CPU_UpdateTensorArgs_PatchesAllKernelsBoundToSameTensor) {
    // Binds the shared tensor to both a DM kernel and the compute kernel (kernels[1]) — now legal,
    // since a compute kernel constructs a LocalTensorAccessor from the binding token. Exercises the
    // host-side address patching reaching every kernel bound to the same tensor.
    NodeCoord node{0, 0};
    ProgramSpec spec = MakeMinimalGen1ValidProgramSpec();
    auto binding = MakeMinimalTensorParameter("shared_tensor");
    spec.tensor_parameters = {binding};
    // Bind the same TensorParameter on both kernels.
    BindTensorParameterToKernel(spec.kernels[0], "shared_tensor", "shared_ta_dm");
    BindTensorParameterToKernel(spec.kernels[1], "shared_tensor", "shared_ta_compute");

    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    MeshTensor tensor1 = AllocateTensorForBinding(*mesh_device_, binding);
    auto params = MakeRunArgsForMinimalSpec(node, {}, {});
    params.tensor_args = {
        {TensorParamName{"shared_tensor"}, TensorArgument{tensor1}},
    };
    SetProgramRunArgs(program, params);

    MeshTensor tensor2 = AllocateTensorForBinding(*mesh_device_, binding);
    Table<TensorParamName, TensorArgument> tensor_args{
        {TensorParamName{"shared_tensor"}, TensorArgument{tensor2}},
    };
    UpdateTensorArgs(program, tensor_args);

    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "dm_kernel", "shared_tensor"), static_cast<uint32_t>(tensor2.address()));
    EXPECT_EQ(
        ReadBindingAddressFromCRTA(program, "compute_kernel", "shared_tensor"),
        static_cast<uint32_t>(tensor2.address()));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
