// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Runtime attach of borrowed-memory DFBs to their backing tensor argument.

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/tensor/mesh_tensor.hpp>

#include "impl/dataflow_buffer/dataflow_buffer_impl.hpp"
#include "impl/program/program_impl.hpp"
#include "metal2_host_api/test_helpers/test_helpers.hpp"
#include "metal2_host_api/test_helpers/mock_device_fixtures.hpp"
#include "metal2_host_api/test_helpers/run_args_test_helpers.hpp"

namespace tt::tt_metal::experimental {
namespace {

using test_helpers::MakeBorrowedDFBProgramSpecForRunArgs;
using test_helpers::MakeKernelRunArgs;
using test_helpers::ProgramRunArgsTestQuasar;

// These tests cover the runtime attach path for DataflowBuffers that borrow their L1 storage
// from a TensorParameter (DataflowBufferSpec::borrowed_from). The flow:
//
//   1. MakeProgramFromSpec    → DFB added with config.borrows_memory = true; groups not yet
//                                created; no L1 allocated.
//   2. finalize_dataflow_buffer_configs (would normally run inside the dispatch pipeline at
//                                first enqueue) populates per-DFB `groups[].l1_by_core`
//                                entries with placeholder addr = 0.
//   3. SetProgramRunArgs / UpdateTensorArgs → AttachBorrowedDFBBuffers resolves the
//                                bound MeshTensor, extracts its reference-buffer address, and
//                                calls dfb->set_borrowed_memory_base_addr(addr), which
//                                overwrites every `groups[].l1_by_core` entry (and any
//                                populated `core_lookup_` entries) with that address.
//
// On mock device we don't run the full dispatch pipeline, so we trigger
// finalize_dataflow_buffer_configs() explicitly and verify state directly off the
// DataflowBufferImpl. allocate_dataflow_buffers requires a configured device and isn't needed
// for these checks — set_borrowed_memory_base_addr iterates `groups[].l1_by_core` regardless.

// Helper: build minimal ProgramRunArgs for the borrowed-DFB spec above. Both kernels are
// MakeMinimalGen2DMKernel (no per-node or common args required), so kernel_run_args entries
// are empty schemas. Caller supplies the tensor_args entry separately.
inline ProgramRunArgs MakeBorrowedDFBRunArgs() {
    NodeCoord node{0, 0};
    ProgramRunArgs params;
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"producer"}, node, {}, {}));
    params.kernel_run_args.push_back(MakeKernelRunArgs(KernelSpecName{"consumer"}, node, {}, {}));
    return params;
}

// Helper: peek at the DFB's per-core L1 base address as written into groups[].l1_by_core.
// All cores of a single DFB share the same address (lockstep allocation invariant), so
// returning the first entry is representative.
inline uint32_t PeekBorrowedDFBAddress(Program& program, const std::string& dfb_name) {
    const uint32_t dfb_id = program.impl().get_dfb_handle(dfb_name);
    const auto dfb = program.impl().get_dataflow_buffer(dfb_id);
    TT_FATAL(!dfb->groups.empty(), "DFB '{}' has no groups; finalize_dataflow_buffer_configs() not called?", dfb_name);
    TT_FATAL(!dfb->groups[0].l1_by_core.empty(), "DFB '{}' group 0 is empty", dfb_name);
    return dfb->groups[0].l1_by_core[0].second;
}

TEST_F(ProgramRunArgsTestQuasar, CPU_BorrowedDFB_BorrowsFlagPropagatesToConfig) {
    // MakeProgramFromSpec should set config.borrows_memory = true on the device-side DFB
    // config when (and only when) DataflowBufferSpec::borrowed_from is set.
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunArgs();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    const uint32_t dfb_id = program.impl().get_dfb_handle("dfb");
    const auto dfb = program.impl().get_dataflow_buffer(dfb_id);
    EXPECT_TRUE(dfb->borrows_memory()) << "DFB declared borrowed_from should have config.borrows_memory = true";
}

TEST_F(ProgramRunArgsTestQuasar, CPU_BorrowedDFB_AttachWritesTensorAddressToDFB) {
    // After SetProgramRunArgs, the bound MeshTensor's address should appear in the
    // DFB's per-core L1 base address tables (overwriting Almeet's 0 placeholder).
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunArgs();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);

    // Populate groups[].l1_by_core with placeholder addr = 0. (Normally invoked inside the
    // dispatch pipeline at first enqueue; we don't enqueue in unit tests.)
    program.impl().finalize_dataflow_buffer_configs();
    ASSERT_EQ(PeekBorrowedDFBAddress(program, "dfb"), 0u) << "DFB base addr before attach should be the placeholder 0";

    // Allocate the borrowed tensor and attach via SetProgramRunArgs.
    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec);
    ProgramRunArgs params = MakeBorrowedDFBRunArgs();
    params.tensor_args = {
        {TensorParamName{"borrowed_tensor"}, TensorArgument{tensor}},
    };
    SetProgramRunArgs(program, params);

    EXPECT_EQ(PeekBorrowedDFBAddress(program, "dfb"), static_cast<uint32_t>(tensor.address()))
        << "DFB base addr should match the borrowed MeshTensor's address after attach";
}

TEST_F(ProgramRunArgsTestQuasar, CPU_BorrowedDFB_UpdateTensorArgsRefreshesAddress) {
    // The cache-hit path: UpdateTensorArgs should re-attach the borrowed Buffer, refreshing
    // the DFB's base address to the new MeshTensor's address.
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunArgs();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    program.impl().finalize_dataflow_buffer_configs();

    MeshTensor tensor1 = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec);
    ProgramRunArgs params = MakeBorrowedDFBRunArgs();
    params.tensor_args = {
        {TensorParamName{"borrowed_tensor"}, TensorArgument{tensor1}},
    };
    SetProgramRunArgs(program, params);
    ASSERT_EQ(PeekBorrowedDFBAddress(program, "dfb"), static_cast<uint32_t>(tensor1.address()));

    MeshTensor tensor2 = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec);
    ASSERT_NE(tensor1.address(), tensor2.address())
        << "Test pre-condition: two separate allocations should yield distinct addresses";

    Table<TensorParamName, TensorArgument> tensor_args{
        {TensorParamName{"borrowed_tensor"}, TensorArgument{tensor2}},
    };
    EXPECT_NO_THROW(UpdateTensorArgs(program, tensor_args));
    EXPECT_EQ(PeekBorrowedDFBAddress(program, "dfb"), static_cast<uint32_t>(tensor2.address()))
        << "DFB base addr should refresh to the new MeshTensor's address after UpdateTensorArgs";
}

// Guard: resizing a borrowed-memory DFB on the partial-update path without supplying its backing
// tensor is rejected — otherwise the per-bank fit check in AttachBorrowedDFBBuffers never re-runs
// against the new size, and a grown DFB could silently overflow its borrowed buffer at execution.
TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_ResizingBorrowedDFBWithoutTensorFails) {
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunArgs();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    program.impl().finalize_dataflow_buffer_configs();

    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec);
    ProgramRunArgs setup = MakeBorrowedDFBRunArgs();
    setup.tensor_args = {{TensorParamName{"borrowed_tensor"}, TensorArgument{tensor}}};
    SetProgramRunArgs(program, setup);

    // Partial update resizes the borrowed DFB but omits its backing tensor.
    ProgramRunArgs upd;
    upd.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb"}, .num_entries = 4});
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, upd); },
        ::testing::ThrowsMessage<std::runtime_error>(
            ::testing::HasSubstr("its backing TensorParameter 'borrowed_tensor' was not supplied")));
}

// Supplying the backing tensor alongside the resize is accepted: the fit check re-runs and the new
// size (48 B) still fits the 64 B backing.
TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_ResizingBorrowedDFBWithTensorSucceeds) {
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunArgs();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    program.impl().finalize_dataflow_buffer_configs();

    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec);
    ProgramRunArgs setup = MakeBorrowedDFBRunArgs();
    setup.tensor_args = {{TensorParamName{"borrowed_tensor"}, TensorArgument{tensor}}};
    SetProgramRunArgs(program, setup);

    // num_entries must stay divisible by this DFB's txn/producer/tc divisor (2 here); 4 grows the DFB
    // to 16 * 4 = 64 B, which still fits the 64 B backing exactly.
    ProgramRunArgs upd;
    upd.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb"}, .num_entries = 4});
    upd.tensor_args = {{TensorParamName{"borrowed_tensor"}, TensorArgument{tensor}}};
    EXPECT_NO_THROW(UpdateProgramRunArgs(program, upd));

    auto dfb = program.impl().get_dataflow_buffer(program.impl().get_dfb_handle("dfb"));
    EXPECT_EQ(dfb->config.num_entries, 4u);
}

// The payoff: with the backing tensor supplied, an over-large resize is now caught by the per-bank fit
// check on the partial path (16 * 64 = 1024 B >> the 64 B backing) — the overflow the guard keeps checkable.
TEST_F(ProgramRunArgsTestQuasar, CPU_UpdateProgramRunArgs_ResizingBorrowedDFBBeyondBackingFails) {
    ProgramSpec spec = MakeBorrowedDFBProgramSpecForRunArgs();
    Program program = MakeProgramFromSpec(*mesh_device_, spec);
    program.impl().finalize_dataflow_buffer_configs();

    MeshTensor tensor = MeshTensor::allocate_on_device(*mesh_device_, spec.tensor_parameters[0].spec);
    ProgramRunArgs setup = MakeBorrowedDFBRunArgs();
    setup.tensor_args = {{TensorParamName{"borrowed_tensor"}, TensorArgument{tensor}}};
    SetProgramRunArgs(program, setup);

    ProgramRunArgs upd;
    upd.dfb_run_overrides.push_back({.dfb = DFBSpecName{"dfb"}, .num_entries = 64});
    upd.tensor_args = {{TensorParamName{"borrowed_tensor"}, TensorArgument{tensor}}};
    EXPECT_THAT(
        [&] { UpdateProgramRunArgs(program, upd); },
        ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr("exceeds the borrowed")));
}

}  // namespace
}  // namespace tt::tt_metal::experimental
