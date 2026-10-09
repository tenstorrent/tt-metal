// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Tensor prefetcher op signals: a Tensix kernel's experimental::tensor_prefetcher_signal must land one
// increment on the signal's slot of each DRAM bank's free sender, from either NoC, and on no other slot or
// core. That rests on the per-device firmware table of those cores, which these tests read back through the
// slots. The free sender passes each count it waits for on to the bank's other sender, which the waits
// that Stop drains exercise.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/tensor_prefetcher.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>

#include "tests/tt_metal/tt_metal/api/dram_sender_fixture.hpp"
#include "distributed/mesh_device_impl.hpp"
#include "impl/context/metal_context.hpp"
#include "llrt/hal.hpp"
#include "llrt/tt_cluster.hpp"

namespace tt::tt_metal {

namespace {

class TensorPrefetcherSignalFixture : public DramSenderFixture {};

// Which RISC raises the signal, in which NoC mode, and on which NoC.
struct SignalSource {
    const char* name;
    DataMovementProcessor processor;
    NOC kernel_noc;
    NOC_MODE noc_mode;
    uint32_t signal_noc;
};

class TensorPrefetcherSignalNocFixture : public TensorPrefetcherSignalFixture,
                                         public ::testing::WithParamInterface<SignalSource> {};

constexpr const char* kSignalKernel = "tests/tt_metal/tt_metal/test_kernels/misc/tensor_prefetcher_signal.cpp";

// The slot at `addr` on both Tensor prefetcher senders of every bank of every device in the mesh.
struct BankSlots {
    uint32_t free_sender = 0;  // holds the bank's op signal counters
    uint32_t noc1_endpoint_sender = 0;
};

std::vector<BankSlots> read_slot_on_prefetcher_cores(distributed::MeshDevice& mesh_device, uint32_t addr) {
    const auto& ctx = MetalContext::instance(mesh_device.impl().get_context_id());
    const uint64_t dram_l1_noc_offset = ctx.hal().get_l1_noc_offset(HalProgrammableCoreType::DRAM);
    const auto read = [&](IDevice* device, const CoreCoord& sender) {
        const CoreCoord virtual_core = device->virtual_core_from_logical_core(sender, CoreType::DRAM);
        uint32_t value = 0;
        ctx.get_cluster().read_core(
            &value, sizeof(value), tt_cxy_pair(device->id(), virtual_core), dram_l1_noc_offset + addr);
        return value;
    };
    std::vector<BankSlots> slots;
    for (IDevice* device : mesh_device.get_devices()) {
        for (uint32_t bank = 0; bank < mesh_device.dram_grid_size().x; ++bank) {
            const std::vector<CoreCoord> senders = mesh_device.impl().dram_sender_logical_cores(device, bank);
            slots.push_back(
                {.free_sender = read(device, senders[0]), .noc1_endpoint_sender = read(device, senders[1])});
        }
    }
    return slots;
}

void run_signal_kernel(
    distributed::MeshDevice& mesh_device,
    const CoreCoord& core,
    uint32_t signal_addr,
    uint32_t num_signals,
    const SignalSource& source) {
    Program program;
    const KernelHandle kernel = CreateKernel(
        program,
        kSignalKernel,
        core,
        DataMovementConfig{
            .processor = source.processor,
            .noc = source.kernel_noc,
            .noc_mode = source.noc_mode,
            .compile_args = {source.signal_noc}});
    SetRuntimeArgs(program, kernel, core, {signal_addr, num_signals});
    distributed::MeshWorkload workload;
    workload.add_program(distributed::MeshCoordinateRange(mesh_device.shape()), std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device.mesh_command_queue(), workload, /*blocking=*/true);
}

}  // namespace

TEST_P(TensorPrefetcherSignalNocFixture, TensixSignalIncrementsEveryBanksCounter) {
    constexpr uint32_t kSignalId = 3;
    constexpr uint32_t kNumSignals = 3;
    const uint32_t num_banks =
        mesh_device_->dram_grid_size().x * static_cast<uint32_t>(mesh_device_->get_devices().size());

    experimental::StartTensorPrefetcher(*mesh_device_, {});
    const uint32_t addr = experimental::GetTensorPrefetcherSignalAddress(*mesh_device_, kSignalId);
    const uint32_t below = experimental::GetTensorPrefetcherSignalAddress(*mesh_device_, kSignalId - 1);
    const uint32_t above = experimental::GetTensorPrefetcherSignalAddress(*mesh_device_, kSignalId + 1);

    for (const BankSlots& bank : read_slot_on_prefetcher_cores(*mesh_device_, addr)) {
        EXPECT_EQ(bank.free_sender, 0u) << "Start must zero the signal slots";
        EXPECT_EQ(bank.noc1_endpoint_sender, 0u) << "Start must zero the signal slots";
    }

    // From a core other than (0, 0), so nothing about the table depends on where the signal comes from.
    run_signal_kernel(*mesh_device_, CoreCoord{3, 2}, addr, kNumSignals, GetParam());

    const std::vector<BankSlots> slots = read_slot_on_prefetcher_cores(*mesh_device_, addr);
    ASSERT_EQ(slots.size(), num_banks);
    for (size_t b = 0; b < slots.size(); ++b) {
        EXPECT_EQ(slots[b].free_sender, kNumSignals) << "bank " << b << "'s counter missed or doubled a signal";
        EXPECT_EQ(slots[b].noc1_endpoint_sender, 0u) << "a signal landed on bank " << b << "'s NOC1-endpoint sender";
    }
    for (uint32_t neighbour : {below, above}) {
        for (const BankSlots& bank : read_slot_on_prefetcher_cores(*mesh_device_, neighbour)) {
            EXPECT_EQ(bank.free_sender, 0u) << "a signal landed on the slot at " << neighbour;
        }
    }

    // The waits these signals release pass on both senders of every bank, so Stop, which processes every queued
    // request first, returns, and each free sender has passed its count on to the bank's other sender.
    for (uint32_t i = 0; i < kNumSignals; ++i) {
        experimental::QueueTensorPrefetcherWaitForSignal(*mesh_device_, kSignalId);
    }
    experimental::StopTensorPrefetcher(*mesh_device_);
    const std::vector<BankSlots> forwarded = read_slot_on_prefetcher_cores(*mesh_device_, addr);
    for (size_t b = 0; b < forwarded.size(); ++b) {
        EXPECT_EQ(forwarded[b].noc1_endpoint_sender, kNumSignals) << "bank " << b << "'s count was not passed on";
    }

    experimental::StartTensorPrefetcher(*mesh_device_, {});
    for (const BankSlots& bank : read_slot_on_prefetcher_cores(*mesh_device_, addr)) {
        EXPECT_EQ(bank.free_sender, 0u) << "a restart must zero the signal slots again";
        EXPECT_EQ(bank.noc1_endpoint_sender, 0u) << "a restart must zero the signal slots again";
    }
    experimental::StopTensorPrefetcher(*mesh_device_);
}

INSTANTIATE_TEST_SUITE_P(
    TensorPrefetcherSignal,
    TensorPrefetcherSignalNocFixture,
    ::testing::Values(
        SignalSource{"BriscNoc0", DataMovementProcessor::RISCV_0, NOC::NOC_0, NOC_MODE::DM_DEDICATED_NOC, 0},
        SignalSource{"NcriscNoc1", DataMovementProcessor::RISCV_1, NOC::NOC_1, NOC_MODE::DM_DEDICATED_NOC, 1},
        SignalSource{"BriscDynamicNoc1", DataMovementProcessor::RISCV_0, NOC::NOC_0, NOC_MODE::DM_DYNAMIC_NOC, 1}),
    [](const ::testing::TestParamInfo<SignalSource>& info) { return std::string(info.param.name); });

TEST_F(TensorPrefetcherSignalFixture, RejectsOutOfRangeSignal) {
    EXPECT_ANY_THROW(experimental::GetTensorPrefetcherSignalAddress(*mesh_device_, 0));

    experimental::StartTensorPrefetcher(*mesh_device_, {});
    EXPECT_ANY_THROW(
        experimental::GetTensorPrefetcherSignalAddress(*mesh_device_, experimental::kTensorPrefetcherNumSignals));
    EXPECT_ANY_THROW(
        experimental::QueueTensorPrefetcherWaitForSignal(*mesh_device_, experimental::kTensorPrefetcherNumSignals));
    experimental::StopTensorPrefetcher(*mesh_device_);
}

}  // namespace tt::tt_metal
