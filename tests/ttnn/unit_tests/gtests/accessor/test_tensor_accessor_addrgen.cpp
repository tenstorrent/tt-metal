// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Phase 1: TensorAccessor HW addrgen for interleaved tensors (peek-only).
//
// Single DM kernel pops hardware addrgen addresses and DPRINTs them alongside
// CoreLocalMem L1 scratch slots. No NOC transactions (ATT not programmed).
//
// Tests:
//   InterleavedDramRead / InterleavedDramWrite
//   InterleavedL1Read   / InterleavedL1Write

#include <gtest/gtest.h>
#include <cstdint>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/shape.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_params.hpp>
#include <tt-metalium/experimental/metal2_host_api/addrgen_support.hpp>
#include <tt-metalium/experimental/tensor/mesh_tensor.hpp>

#include "tests/tt_metal/tt_metal/common/device_fixture.hpp"
#include "tests/tt_metal/tt_metal/api/metal2_host_api/test_helpers.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/api/ttnn/distributed/api.hpp"

namespace {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental::metal2_host_api;
using namespace tt::tt_metal::experimental::metal2_host_api::test_helpers;
using namespace ttnn;

constexpr const char* kInterleavedAddrgenPeekPath =
    "tests/ttnn/unit_tests/gtests/accessor/kernels/interleaved_addrgen_peek.cpp";

constexpr uint32_t kL1ScratchBase = 0x10000;

void set_addrgen_mode(KernelSpec& kernel, KernelSpec::TensorBinding::AddrgenMode mode) {
    TT_FATAL(!kernel.tensor_bindings.empty(), "No tensor bindings to set addrgen_mode on");
    kernel.tensor_bindings.back().addrgen_mode = mode;
}

void run_addrgen_peek_test(
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    const TensorSpec& tensor_spec,
    KernelSpec::TensorBinding::AddrgenMode addrgen_mode,
    uint32_t is_read,
    uint32_t num_pages) {
    const AddrgenSupport support = addrgen_support_for(tensor_spec);
    ASSERT_EQ(support, AddrgenSupport::kSupported) << describe_skip_reason(support);

    IDevice* device = mesh_device->get_devices().at(0);
    const NodeCoord node{0, 0};

    auto tensor = MeshTensor::allocate_on_device(*mesh_device, tensor_spec, TensorTopology{});

    ProgramSpec spec;
    spec.program_id = "ta_addrgen_interleaved_peek";

    KernelSpec dm = MakeMinimalDMKernel("dm", /*num_threads=*/1);
    dm.source = KernelSpec::SourceFilePath{kInterleavedAddrgenPeekPath};
    dm.compile_time_arg_bindings = {{"is_read", is_read}, {"l1_scratch_base", kL1ScratchBase}};
    dm.runtime_arguments_schema.num_runtime_varargs = 1;
    BindTensorParameterToKernel(dm, "tensor", "tensor");
    set_addrgen_mode(dm, addrgen_mode);

    spec.kernels = {dm};
    spec.tensor_parameters = {TensorParameter{.unique_id = "tensor", .spec = tensor_spec}};
    spec.work_units = {MakeMinimalWorkUnit("work_unit_0", node, {"dm"})};

    Program program = MakeProgramFromSpec(*mesh_device, spec);

    ProgramRunParams run_params;
    run_params.kernel_run_params = {
        ProgramRunParams::KernelRunParams{
            .kernel_spec_name = "dm",
            .runtime_varargs = {{node, {num_pages}}},
        },
    };
    run_params.tensor_args = {
        ProgramRunParams::TensorArg{.tensor_parameter_name = "tensor", .tensor = std::cref(tensor)},
    };
    SetProgramRunParameters(program, run_params);

    detail::LaunchProgram(device, program);
}

class TensorAddrgenTest : public tt::tt_metal::MeshDeviceFixture {
protected:
    void SetUp() override {
        MeshDeviceFixture::SetUp();
        if (this->IsSkipped()) {
            return;
        }
        if (devices_.at(0)->arch() != tt::ARCH::QUASAR) {
            GTEST_SKIP() << "TensorAccessor addrgen tests require Quasar";
        }
    }

    static TensorSpec interleaved_dram_tensor_spec(uint32_t num_pages = 8, uint32_t page_width = 512) {
        auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
        return TensorSpec(
            Shape{num_pages, page_width},
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), memory_config));
    }

    static TensorSpec interleaved_l1_tensor_spec(uint32_t num_pages = 8, uint32_t page_width = 512) {
        auto memory_config = MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::L1};
        return TensorSpec(
            Shape{num_pages, page_width},
            TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), memory_config));
    }
};

TEST_F(TensorAddrgenTest, InterleavedDramRead) {
    using AddrgenMode = KernelSpec::TensorBinding::AddrgenMode;
    constexpr uint32_t num_pages = 8;
    run_addrgen_peek_test(
        devices_.at(0).get(),
        interleaved_dram_tensor_spec(num_pages),
        AddrgenMode::READ,
        /*is_read=*/1,
        num_pages);
}

TEST_F(TensorAddrgenTest, InterleavedDramWrite) {
    using AddrgenMode = KernelSpec::TensorBinding::AddrgenMode;
    constexpr uint32_t num_pages = 8;
    run_addrgen_peek_test(
        devices_.at(0).get(),
        interleaved_dram_tensor_spec(num_pages),
        AddrgenMode::WRITE,
        /*is_read=*/0,
        num_pages);
}

TEST_F(TensorAddrgenTest, InterleavedL1Read) {
    using AddrgenMode = KernelSpec::TensorBinding::AddrgenMode;
    constexpr uint32_t num_pages = 8;
    run_addrgen_peek_test(
        devices_.at(0).get(),
        interleaved_l1_tensor_spec(num_pages),
        AddrgenMode::READ,
        /*is_read=*/1,
        num_pages);
}

TEST_F(TensorAddrgenTest, InterleavedL1Write) {
    using AddrgenMode = KernelSpec::TensorBinding::AddrgenMode;
    constexpr uint32_t num_pages = 8;
    run_addrgen_peek_test(
        devices_.at(0).get(),
        interleaved_l1_tensor_spec(num_pages),
        AddrgenMode::WRITE,
        /*is_read=*/0,
        num_pages);
}

}  // namespace
