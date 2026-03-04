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
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 4, .is_legacy_kernel = is_legacy_kernel});

    KernelHandle data_movement_kernel_1 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_tls_check_2.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 3, .is_legacy_kernel = is_legacy_kernel});

    KernelHandle data_movement_kernel_2 = experimental::quasar::CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_tls_check_3.cpp",
        core,
        experimental::quasar::QuasarDataMovementConfig{.num_threads_per_cluster = 1, .is_legacy_kernel = is_legacy_kernel});

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
    auto slot_thread_local_addr = [&dram_data](uint32_t dm) {
        const uint32_t offset = dm * TLS_CHECK_RESULT_SLOT_WORDS;
        return (uint64_t)dram_data[offset + TLS_CHECK_THREAD_LOCAL_ADDR_LO] | ((uint64_t)dram_data[offset + TLS_CHECK_THREAD_LOCAL_ADDR_HI] << 32);
    };
    const uint64_t ref_addr_0_3 = slot_global_addr(0);
    const uint64_t ref_addr_4_6 = slot_global_addr(4);
    const uint64_t ref_addr_7 = slot_global_addr(7);

    for (uint32_t dm = 0; dm < NUM_DM_CORES; dm++) {
        const uint32_t offset = dm * TLS_CHECK_RESULT_SLOT_WORDS;
        uint32_t kernel_id = dram_data[offset + TLS_CHECK_KERNEL_ID];
        uint32_t num_sw_threads = dram_data[offset + TLS_CHECK_NUM_THREADS];
        uint32_t my_thread_id = dram_data[offset + TLS_CHECK_MY_THREAD_ID];
        uint32_t hartid = dram_data[offset + TLS_CHECK_HART_ID];
        uint32_t thread_0_hartid = dram_data[offset + TLS_CHECK_THREAD_0_HART_ID];
        uint32_t global_start = dram_data[offset + TLS_CHECK_GLOBAL_START];
        uint32_t global_end = dram_data[offset + TLS_CHECK_GLOBAL_END];
        uint64_t global_addr = slot_global_addr(dm);
        uint32_t uninitialized_global_start = dram_data[offset + TLS_CHECK_UNINITIALIZED_GLOBAL_START];
        uint32_t uninitialized_global_end = dram_data[offset + TLS_CHECK_UNINITIALIZED_GLOBAL_END];
        uint64_t thread_local_start = dram_data[offset + TLS_CHECK_THREAD_LOCAL_START];
        uint64_t thread_local_end = dram_data[offset + TLS_CHECK_THREAD_LOCAL_END];
        uint32_t uninitialized_thread_local_start = dram_data[offset + TLS_CHECK_UNINITIALIZED_THREAD_LOCAL_START];
        uint32_t uninitialized_thread_local_end = dram_data[offset + TLS_CHECK_UNINITIALIZED_THREAD_LOCAL_END];

        // 1. Check that each set is running the correct kernel by verifying the hard-coded kernel ID number.
        // This check assumes that the threaded kernels are assigned sequentially to the DMs, which is how they
        // are currently assigned by CreateKernel(). If this assumption is violated in the future, update this
        // check to find the shared kernel ID and just verify counts. Follow on checks will have to be updated
        // as well as this assumption was made for all the checks for simplicity.
        // Kernel ID: DM 0-3 → 1, DM 4-6 → 2, DM 7 → 3
        uint32_t expected_kernel_id = 0;
        if (dm <= 3) {
            expected_kernel_id = 1;
        } else if (dm <= 6) {
            expected_kernel_id = 2;
        } else {
            expected_kernel_id = 3;
        }
        EXPECT_EQ(kernel_id, expected_kernel_id) << "dm=" << dm;

        // 2. Verify num_sw_threads & my_thread_id
        uint32_t expected_num_threads = 0;
        if (dm <= 3) {
            expected_num_threads = 4;
        } else if (dm <= 6) {
            expected_num_threads = 3;
        } else {
            expected_num_threads = 1;
        }
        uint32_t expected_thread_id = 0;
        if (dm <= 3) {
            expected_thread_id = dm;
        } else if (dm <= 6) {
            expected_thread_id = dm - 4;
        } else {
            expected_thread_id = 0;
        }
        EXPECT_EQ(num_sw_threads, expected_num_threads) << "dm=" << dm;
        EXPECT_EQ(my_thread_id, expected_thread_id) << "dm=" << dm;

        // 3. Verify that hartid matches DM #
        EXPECT_EQ(hartid, dm) << "dm=" << dm;

        // 4. Verify that the DM is pointing to the correct binary. The specific
        // check here is the lowest hartid with the same kernel ID.
        if (is_legacy_kernel) {
            // For legacy kernels, each DM has its own binary.
            EXPECT_EQ(thread_0_hartid, dm) << "dm=" << dm << " (legacy)";
        } else {
            // For threaded kernels, DM 0-3 are in the same binary, DM 4-6 are in the same binary, and DM 7 is in a different binary.
            uint32_t expected_t0 = 0;
            if (dm <= 3) {
                expected_t0 = 0;
            } else if (dm <= 6) {
                expected_t0 = 4;
            } else {
                expected_t0 = 7;
            }
            EXPECT_EQ(thread_0_hartid, expected_t0) << "dm=" << dm << " (non-legacy)";
        }

        // 5. Check that initialized global variables have the correct start and end values.
        // Initialized globals are set to 5, then incremented by 1 for each DM in sequence.
        if (is_legacy_kernel) {
            // For legacy kernels, globals are not shared.
            EXPECT_EQ(global_start, 5u) << "dm=" << dm << " (legacy)";
            EXPECT_EQ(global_end, 6u) << "dm=" << dm << " (legacy)";
        } else {
            // For threaded kernels, globals are shared between DMs in the same set, so values
            // start at 5 with the first DM in the set (DM 0, 4, 7) and increment by 1 for each DM in sequence.
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

        // 6. For threaded kernels, check that the global variable address is shared between DMs in the same set.
        if (!is_legacy_kernel) {
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

        // 7. Check that uninitialized global variables have the correct start and end values.
        // Uninitialized globals are cleared to 0, then incremented by 1 for each DM in sequence.
        if (is_legacy_kernel) {
            EXPECT_EQ(uninitialized_global_start, 0u) << "dm=" << dm;
            EXPECT_EQ(uninitialized_global_end, 1u) << "dm=" << dm;
        } else {
            if (dm <= 3) {
                EXPECT_EQ(uninitialized_global_start, 0u + dm) << "dm=" << dm;
                EXPECT_EQ(uninitialized_global_end, 1u + dm) << "dm=" << dm;
            } else if (dm <= 6) {
                EXPECT_EQ(uninitialized_global_start, 0u + (dm - 4)) << "dm=" << dm;
                EXPECT_EQ(uninitialized_global_end, 1u + (dm - 4)) << "dm=" << dm;
            } else {
                EXPECT_EQ(uninitialized_global_start, 0u) << "dm=" << dm;
                EXPECT_EQ(uninitialized_global_end, 1u) << "dm=" << dm;
            }
        }

        // 8. Check that initialized thread local variables have the correct value.
        // I.e. incrementing the variable in one DM does not affect the value in another DM.
        // TODO: Initializing thread local variables does not work yet. Once they work,
        // update this check with the correct values.
        EXPECT_EQ(thread_local_start, 0u) << "dm=" << dm;
        EXPECT_EQ(thread_local_end, 1u) << "dm=" << dm;

        // 9. Check that uninitialized thread local variables have the correct value.
        // Same as #8, but variables are cleared to 0 at the start.
        EXPECT_EQ(uninitialized_thread_local_start, 0u) << "dm=" << dm;
        EXPECT_EQ(uninitialized_thread_local_end, 1u) << "dm=" << dm;
    }

    // For legacy kernels, check that the global variable addresses are unique.
    if (is_legacy_kernel) {
        std::set<uint64_t> addrs;
        for (uint32_t dm = 0; dm < NUM_DM_CORES; dm++) {
            addrs.insert(slot_global_addr(dm));
        }
        EXPECT_EQ(addrs.size(), NUM_DM_CORES) << "Legacy: all global addresses should be unique";
    }

    // For both legacy & threaded kernels, check that the thread local variable addresses are unique.
    std::set<uint64_t> thread_local_addrs;
    for (uint32_t dm = 0; dm < NUM_DM_CORES; dm++) {
        thread_local_addrs.insert(slot_thread_local_addr(dm));
    }
    EXPECT_EQ(thread_local_addrs.size(), NUM_DM_CORES) << "All thread local addresses should be unique";
}

INSTANTIATE_TEST_SUITE_P(
    LegacyVsNonLegacyTest,
    LegacyVsNonLegacyTest,
    ::testing::Values(true, false),
    [](const ::testing::TestParamInfo<bool>& info) {
        return info.param ? "Legacy" : "Threaded";
});
