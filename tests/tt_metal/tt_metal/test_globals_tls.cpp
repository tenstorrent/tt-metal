// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "test_kernels/dataflow/simple_tls_check_defines.h"

#include <cstring>
#include <set>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t NUM_DM_CORES = 8;
constexpr uint32_t TOTAL_RESULT_BYTES = NUM_DM_CORES * TLS_CHECK_RESULT_SLOT_BYTES;

class LegacyVsNonLegacyTest
    : public MeshDeviceSingleCardFixture,
      public testing::WithParamInterface<bool> {
};

// This test requires simulator environment
TEST_P(LegacyVsNonLegacyTest, GlobalsAndTLS) {
    auto is_legacy_kernel = GetParam();
    tt::tt_metal::MetalContext::instance().rtoptions().set_force_jit_compile(true);
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device->get_devices()[0];

    const uint32_t signal_address = 100 * 1024;
    const uint32_t l1_result_addr = 200 * 1024;
    const uint32_t dram_address = 30000 * 1024;

    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    constexpr CoreCoord core = {0, 0};
    const uint32_t dram_channel = mesh_device->dram_channel_from_virtual_core(core);

    // Initialize L1 signal so hart 0 can proceed (then 1, 2, ... in order)
    std::vector<uint32_t> init_signal = {0};
    tt_metal::detail::WriteToDeviceL1(
        device,
        core,
        signal_address,
        std::span(reinterpret_cast<const uint8_t*>(init_signal.data()), sizeof(uint32_t)),
        CoreType::WORKER);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program = CreateProgram();

    KernelHandle data_movement_kernel_0 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_tls_check_1.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 4, .is_legacy_kernel = is_legacy_kernel});

    KernelHandle data_movement_kernel_1 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_tls_check_2.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 3, .is_legacy_kernel = is_legacy_kernel});

    KernelHandle data_movement_kernel_2 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_tls_check_3.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{.num_processors_per_cluster = 1, .is_legacy_kernel = is_legacy_kernel});

    // signal_address, dram_dst_address, dram_dst_bank_id, l1_result_addr, kernel_id
    SetRuntimeArgs(
        program, data_movement_kernel_0, core, {signal_address, dram_address, dram_channel, l1_result_addr});
    SetRuntimeArgs(
        program, data_movement_kernel_1, core, {signal_address, dram_address, dram_channel, l1_result_addr});
    SetRuntimeArgs(
        program, data_movement_kernel_2, core, {signal_address, dram_address, dram_channel, l1_result_addr});

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    distributed::Finish(mesh_device->mesh_command_queue());

    std::vector<uint32_t> dram_data;
    tt_metal::detail::ReadFromDeviceDRAMChannel(device, dram_channel, dram_address, TOTAL_RESULT_BYTES, dram_data);

    // Reference global addresses for non-legacy check 6 (DM 0-3 same, 4-6 same, 7 unique)
    auto slot_global_addr = [&dram_data](uint32_t dm) {
        const uint32_t offset = dm * TLS_CHECK_RESULT_SLOT_WORDS;
        return (uint64_t)dram_data[offset + TLS_CHECK_GLOBAL_ADDR_LO] | ((uint64_t)dram_data[offset + TLS_CHECK_GLOBAL_ADDR_HI] << 32);
    };
    const uint64_t ref_addr_0_3 = slot_global_addr(0);
    const uint64_t ref_addr_4_6 = slot_global_addr(4);
    const uint64_t ref_addr_7 = slot_global_addr(7);

    for (uint32_t dm = 0; dm < NUM_DM_CORES; dm++) {
        const uint32_t offset = dm * TLS_CHECK_RESULT_SLOT_WORDS;
        uint32_t kernel_id = dram_data[offset + TLS_CHECK_KERNEL_ID];
        uint32_t num_kernel_threads = dram_data[offset + TLS_CHECK_NUM_KERNEL_THREADS];
        uint32_t my_thread_id = dram_data[offset + TLS_CHECK_MY_THREAD_ID];
        uint32_t hartid = dram_data[offset + TLS_CHECK_HART_ID];
        uint32_t thread_0_hartid = dram_data[offset + TLS_CHECK_THREAD_0_HART_ID];
        uint32_t global_start = dram_data[offset + TLS_CHECK_GLOBAL_START];
        uint32_t global_end = dram_data[offset + TLS_CHECK_GLOBAL_END];
        uint64_t global_addr = slot_global_addr(dm);

        // 1. Kernel ID: DM 0-3 → 1, DM 4-6 → 2, DM 7 → 3
        uint32_t expected_kernel_id = (dm <= 3) ? 1u : (dm <= 6) ? 2u : 3u;
        EXPECT_EQ(kernel_id, expected_kernel_id) << "dm=" << dm;

        // 2. num_kernel_threads & my_thread_id
        uint32_t expected_num_threads = (dm <= 3) ? 4u : (dm <= 6) ? 3u : 1u;
        uint32_t expected_thread_id = (dm <= 3) ? dm : (dm <= 6) ? (dm - 4) : 0u;
        EXPECT_EQ(num_kernel_threads, expected_num_threads) << "dm=" << dm;
        EXPECT_EQ(my_thread_id, expected_thread_id) << "dm=" << dm;

        // 3. hartid matches DM #
        EXPECT_EQ(hartid, dm) << "dm=" << dm;

        // 4. thread_0_hartid
        if (is_legacy_kernel) {
            EXPECT_EQ(thread_0_hartid, dm) << "dm=" << dm << " (legacy)";
        } else {
            uint32_t expected_t0 = (dm <= 3) ? 0u : (dm <= 6) ? 4u : 7u;
            EXPECT_EQ(thread_0_hartid, expected_t0) << "dm=" << dm << " (non-legacy)";
        }

        // 5. global start & end
        if (is_legacy_kernel) {
            EXPECT_EQ(global_start, 5u) << "dm=" << dm << " (legacy)";
            EXPECT_EQ(global_end, 6u) << "dm=" << dm << " (legacy)";
        } else {
            if (dm <= 3) {
                EXPECT_EQ(global_start, 5u + dm) << "dm=" << dm;
                EXPECT_EQ(global_end, 6u + dm) << "dm=" << dm;
            } else if (dm <= 6) {
                EXPECT_EQ(global_start, 5u + (dm - 4)) << "dm=" << dm;
                EXPECT_EQ(global_end, 6u + (dm - 4)) << "dm=" << dm;
            } else {
                EXPECT_EQ(global_start, 5u) << "dm=" << dm;
                EXPECT_EQ(global_end, 6u) << "dm=" << dm;
            }
        }

        // 6. global address
        if (is_legacy_kernel) {
            (void)global_addr;
        } else {
            if (dm <= 3) {
                EXPECT_EQ(global_addr, ref_addr_0_3) << "dm=" << dm;
            } else if (dm <= 6) {
                EXPECT_EQ(global_addr, ref_addr_4_6) << "dm=" << dm;
                EXPECT_NE(ref_addr_4_6, ref_addr_0_3) << "DM 4-6 addr should differ from DM 0-3";
            } else {
                EXPECT_EQ(global_addr, ref_addr_7) << "dm=" << dm;
                EXPECT_NE(ref_addr_7, ref_addr_0_3) << "DM 7 addr should differ from DM 0-3";
                EXPECT_NE(ref_addr_7, ref_addr_4_6) << "DM 7 addr should differ from DM 4-6";
            }
        }
    }
    if (is_legacy_kernel) {
        std::set<uint64_t> addrs;
        for (uint32_t dm = 0; dm < NUM_DM_CORES; dm++) {
            addrs.insert(slot_global_addr(dm));
        }
        EXPECT_EQ(addrs.size(), NUM_DM_CORES) << "Legacy: all global addresses should be unique";
    }
}

INSTANTIATE_TEST_SUITE_P(
    LegacyVsNonLegacyTest,
    LegacyVsNonLegacyTest,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "Legacy" : "Threaded";
});
