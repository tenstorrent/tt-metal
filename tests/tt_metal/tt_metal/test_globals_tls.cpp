// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "common/device_fixture.hpp"
#include "test_kernels/dataflow/simple_tls_check_defines.h"

#include <cstring>
#include <set>

#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/tt_metal.hpp>

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

using namespace tt;
using namespace tt::tt_metal;

// Metal 2.0 reserves DM0/DM1 for runtime; user kernels may use only DM2..DM7 (6 DMs).
// We size the result buffer for all 8 hart slots so that the kernel's hartid-keyed
// writes line up directly without remapping; slots 0/1 are simply unused.
constexpr uint32_t NUM_DM_CORES = 8;
constexpr uint32_t FIRST_USER_DM = 2;
constexpr uint32_t NUM_USER_DMS = 6;
constexpr uint32_t TOTAL_RESULT_BYTES = NUM_DM_CORES * TLS_CHECK_RESULT_SLOT_BYTES;

// This test requires simulator environment
TEST_F(QuasarMeshDeviceSingleCardFixture, GlobalsAndTLS) {
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

    // Initialize L1 signal so hart FIRST_USER_DM (2) can proceed first; the simple_tls_check
    // kernel chains the signal forward (signal_addr := hartid + 1) so hartids 2..7 run in order.
    std::vector<uint32_t> init_signal = {FIRST_USER_DM};
    tt_metal::detail::WriteToDeviceL1(
        device,
        core,
        signal_address,
        std::span(reinterpret_cast<const uint8_t*>(init_signal.data()), sizeof(uint32_t)),
        CoreType::WORKER);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const experimental::KernelSpecName DM_KERNEL_1{"dm_kernel_1"};
    const experimental::KernelSpecName DM_KERNEL_2{"dm_kernel_2"};
    const experimental::KernelSpecName DM_KERNEL_3{"dm_kernel_3"};

    // Three kernels split 6 user DMs as 3 + 2 + 1 to mirror the original 4 + 3 + 1 split
    // (preserving the "shared kernel binary across multiple DMs" + "single-DM kernel" mix).
    auto make_dm_kernel_spec = [](const experimental::KernelSpecName& unique_id,
                                  uint32_t kernel_id,
                                  uint32_t num_threads) {
        return experimental::KernelSpec{
            .unique_id = unique_id,
            .source =

                OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/dataflow/simple_tls_check.cpp",
            .num_threads = num_threads,
            .compile_time_args = {{"kernel_id", kernel_id}},
            .runtime_arg_schema =
                {
                    .runtime_arg_names = {"signal_address", "dram_dst_address", "dram_dst_bank_id", "l1_result_addr"},
                },
            .hw_config = experimental::DataMovementGen2Config{},
        };
    };

    auto k1 = make_dm_kernel_spec(DM_KERNEL_1, /*kernel_id=*/1, /*num_threads=*/3);
    auto k2 = make_dm_kernel_spec(DM_KERNEL_2, /*kernel_id=*/2, /*num_threads=*/2);
    auto k3 = make_dm_kernel_spec(DM_KERNEL_3, /*kernel_id=*/3, /*num_threads=*/1);

    const experimental::NodeCoord node{0, 0};

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {DM_KERNEL_1, DM_KERNEL_2, DM_KERNEL_3},
        .target_nodes = node,
    };

    // Total user-DM threads = 3 + 2 + 1 = 6, fitting within the default DM2..DM7 cap.
    experimental::ProgramSpec spec{
        .name = "globals_and_tls",
        .kernels = {k1, k2, k3},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    auto make_kernel_run_params = [&]() {
        return experimental::ProgramRunArgs::KernelRunArgs{
            .runtime_arg_values =
                {{node,
                  {{"signal_address", signal_address},
                   {"dram_dst_address", dram_address},
                   {"dram_dst_bank_id", dram_channel},
                   {"l1_result_addr", l1_result_addr}}}},
        };
    };

    experimental::ProgramRunArgs params;
    auto kra1 = make_kernel_run_params();
    kra1.kernel = DM_KERNEL_1;
    auto kra2 = make_kernel_run_params();
    kra2.kernel = DM_KERNEL_2;
    auto kra3 = make_kernel_run_params();
    kra3.kernel = DM_KERNEL_3;
    params.kernel_run_args = {kra1, kra2, kra3};
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    distributed::Finish(mesh_device->mesh_command_queue());

    std::vector<uint32_t> dram_data;
    tt_metal::detail::ReadFromDeviceDRAMChannel(device, dram_channel, dram_address, TOTAL_RESULT_BYTES, dram_data);

    // Reference global addresses: DM 2-4 share one binary, DM 5-6 share another, DM 7 alone.
    auto slot_global_addr = [&dram_data](uint32_t dm) {
        const uint32_t offset = dm * TLS_CHECK_RESULT_SLOT_WORDS;
        return (uint64_t)dram_data[offset + TLS_CHECK_GLOBAL_ADDR_LO] | ((uint64_t)dram_data[offset + TLS_CHECK_GLOBAL_ADDR_HI] << 32);
    };
    auto slot_thread_local_addr = [&dram_data](uint32_t dm) {
        const uint32_t offset = dm * TLS_CHECK_RESULT_SLOT_WORDS;
        return (uint64_t)dram_data[offset + TLS_CHECK_THREAD_LOCAL_ADDR_LO] | ((uint64_t)dram_data[offset + TLS_CHECK_THREAD_LOCAL_ADDR_HI] << 32);
    };
    const uint64_t ref_addr_2_4 = slot_global_addr(2);
    const uint64_t ref_addr_5_6 = slot_global_addr(5);
    const uint64_t ref_addr_7 = slot_global_addr(7);

    for (uint32_t dm = FIRST_USER_DM; dm < FIRST_USER_DM + NUM_USER_DMS; dm++) {
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

        // 1. Each set runs the correct kernel by verifying the hard-coded kernel ID.
        // This check assumes that the threaded kernels are assigned sequentially to the DMs, which is how they
        // are currently assigned by CreateKernel(). If this assumption is violated in the future, update this
        // check to find the shared kernel ID and just verify counts. Follow on checks will have to be updated
        // as well as this assumption was made for all the checks for simplicity.
        // Kernel ID: DM 2-4 → 1, DM 5-6 → 2, DM 7 → 3
        uint32_t expected_kernel_id = 0;
        if (dm <= 4) {
            expected_kernel_id = 1;
        } else if (dm <= 6) {
            expected_kernel_id = 2;
        } else {
            expected_kernel_id = 3;
        }
        EXPECT_EQ(kernel_id, expected_kernel_id) << "dm=" << dm;

        // 2. Verify num_sw_threads & my_thread_id
        uint32_t expected_num_threads = 0;
        if (dm <= 4) {
            expected_num_threads = 3;
        } else if (dm <= 6) {
            expected_num_threads = 2;
        } else {
            expected_num_threads = 1;
        }
        uint32_t expected_thread_id = 0;
        if (dm <= 4) {
            expected_thread_id = dm - 2;
        } else if (dm <= 6) {
            expected_thread_id = dm - 5;
        } else {
            expected_thread_id = 0;
        }
        EXPECT_EQ(num_sw_threads, expected_num_threads) << "dm=" << dm;
        EXPECT_EQ(my_thread_id, expected_thread_id) << "dm=" << dm;

        // 3. Verify that hartid matches DM #
        EXPECT_EQ(hartid, dm) << "dm=" << dm;

        // 4. Verify that the DM is pointing to the correct binary (lowest hartid sharing the kernel).
        // For threaded kernels: DM 2-4 share a binary, DM 5-6 share a binary, DM 7 is alone.
        uint32_t expected_t0 = 0;
        if (dm <= 4) {
            expected_t0 = 2;
        } else if (dm <= 6) {
            expected_t0 = 5;
        } else {
            expected_t0 = 7;
        }
        EXPECT_EQ(thread_0_hartid, expected_t0) << "dm=" << dm;

        // 5. Initialized global variables: shared between DMs in the same set, start at 5
        // and increment by 1 for each DM in sequence within the set.
        if (dm <= 4) {
            EXPECT_EQ(global_start, 5u + (dm - 2)) << "dm=" << dm;
            EXPECT_EQ(global_end, 6u + (dm - 2)) << "dm=" << dm;
        } else if (dm <= 6) {
            EXPECT_EQ(global_start, 5u + (dm - 5)) << "dm=" << dm;
            EXPECT_EQ(global_end, 6u + (dm - 5)) << "dm=" << dm;
        } else {
            EXPECT_EQ(global_start, 5u) << "dm=" << dm;
            EXPECT_EQ(global_end, 6u) << "dm=" << dm;
        }

        // 6. Check that the global variable address is shared between DMs in the same set.
        if (dm <= 4) {
            EXPECT_EQ(global_addr, ref_addr_2_4) << "dm=" << dm;
        } else if (dm <= 6) {
            EXPECT_EQ(global_addr, ref_addr_5_6) << "dm=" << dm;
            EXPECT_NE(ref_addr_5_6, ref_addr_2_4) << "DM 5-6 addr should differ from DM 2-4";
        } else {
            EXPECT_EQ(global_addr, ref_addr_7) << "dm=" << dm;
            EXPECT_NE(ref_addr_7, ref_addr_2_4) << "DM 7 addr should differ from DM 2-4";
            EXPECT_NE(ref_addr_7, ref_addr_5_6) << "DM 7 addr should differ from DM 5-6";
        }

        // 7. Check that uninitialized global variables have the correct start and end values.
        // Uninitialized globals are cleared to 0, then incremented by 1 for each DM in sequence.
        if (dm <= 4) {
            EXPECT_EQ(uninitialized_global_start, 0u + (dm - 2)) << "dm=" << dm;
            EXPECT_EQ(uninitialized_global_end, 1u + (dm - 2)) << "dm=" << dm;
        } else if (dm <= 6) {
            EXPECT_EQ(uninitialized_global_start, 0u + (dm - 5)) << "dm=" << dm;
            EXPECT_EQ(uninitialized_global_end, 1u + (dm - 5)) << "dm=" << dm;
        } else {
            EXPECT_EQ(uninitialized_global_start, 0u) << "dm=" << dm;
            EXPECT_EQ(uninitialized_global_end, 1u) << "dm=" << dm;
        }

        // 8. Each DM observes its own independently initialized TLS value.
        EXPECT_EQ(thread_local_start, 10u) << "dm=" << dm;
        EXPECT_EQ(thread_local_end, 11u) << "dm=" << dm;

        // 9. Same as #8 for uninitialized TLS, cleared to 0 at start.
        EXPECT_EQ(uninitialized_thread_local_start, 0u) << "dm=" << dm;
        EXPECT_EQ(uninitialized_thread_local_end, 1u) << "dm=" << dm;
    }

    // For threaded kernels, the thread local variable addresses must all be unique.
    std::set<uint64_t> thread_local_addrs;
    for (uint32_t dm = FIRST_USER_DM; dm < FIRST_USER_DM + NUM_USER_DMS; dm++) {
        thread_local_addrs.insert(slot_thread_local_addr(dm));
    }
    EXPECT_EQ(thread_local_addrs.size(), NUM_USER_DMS) << "All thread local addresses should be unique";
}

// Quasar compute: 4 Tensix engines per cluster, 4 TRISCs per engine = 16 slots.
// Verification mirrors GlobalsAndTLS (DM test) for the same information.
static constexpr uint32_t QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER = 4;
static constexpr uint32_t QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE = 4;
static constexpr uint32_t NUM_COMPUTE_SLOTS =
    QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER * QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE;
static constexpr uint32_t QUASAR_FIRST_COMPUTE_HARTID = 8;  // DM 0-7, compute 8-23

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarComputeKernelTLS) {
    auto mesh_device = devices_[0];
    IDevice* device = mesh_device->get_devices()[0];

    char* env_var = std::getenv("TT_METAL_SIMULATOR");
    if (env_var == nullptr) {
        GTEST_SKIP() << "This test can only be run using a simulator. Set TT_METAL_SIMULATOR environment variable.";
    }

    constexpr CoreCoord core = {0, 0};
    const uint32_t signal_address = 100 * 1024;
    constexpr uint32_t l1_result_addr = 200 * 1024;
    constexpr uint32_t total_result_bytes = NUM_COMPUTE_SLOTS * TLS_CHECK_RESULT_SLOT_BYTES;

    std::vector<uint32_t> init_signal = {QUASAR_FIRST_COMPUTE_HARTID};
    tt_metal::detail::WriteToDeviceL1(
        device,
        core,
        signal_address,
        std::span(reinterpret_cast<const uint8_t*>(init_signal.data()), sizeof(uint32_t)),
        CoreType::WORKER);

    std::vector<uint32_t> init_data(total_result_bytes / sizeof(uint32_t), 0);
    tt_metal::detail::WriteToDeviceL1(
        device,
        core,
        l1_result_addr,
        std::span(reinterpret_cast<const uint8_t*>(init_data.data()), total_result_bytes),
        CoreType::WORKER);

    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    const experimental::KernelSpecName COMPUTE_KERNEL{"compute_tls"};
    const experimental::NodeCoord node{0, 0};

    experimental::KernelSpec compute_kernel_spec{
        .unique_id = COMPUTE_KERNEL,
        .source =

            OVERRIDE_KERNEL_PREFIX "tests/tt_metal/tt_metal/test_kernels/compute/simple_tls_check.cpp",
        .num_threads = QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"signal_address", "l1_result_addr"},
            },
        .hw_config = experimental::ComputeGen2Config{},
    };

    experimental::WorkUnitSpec main_wu{
        .name = "main",
        .kernels = {COMPUTE_KERNEL},
        .target_nodes = node,
    };

    experimental::ProgramSpec spec{
        .name = "compute_kernel_tls",
        .kernels = {compute_kernel_spec},
        .work_units = {main_wu},
    };
    Program program = experimental::MakeProgramFromSpec(*mesh_device, spec);

    experimental::ProgramRunArgs params;
    params.kernel_run_args = {experimental::ProgramRunArgs::KernelRunArgs{
        .kernel = COMPUTE_KERNEL,
        .runtime_arg_values = {{node, {{"l1_result_addr", l1_result_addr}, {"signal_address", signal_address}}}},
    }};
    experimental::SetProgramRunArgs(program, params);

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, true);
    distributed::Finish(mesh_device->mesh_command_queue());

    std::vector<uint32_t> l1_data;
    tt_metal::detail::ReadFromDeviceL1(device, core, l1_result_addr, total_result_bytes, l1_data, CoreType::WORKER);

    auto slot_global_addr = [&l1_data](uint32_t slot) {
        const uint32_t offset = slot * TLS_CHECK_RESULT_SLOT_WORDS;
        return (uint64_t)l1_data[offset + TLS_CHECK_GLOBAL_ADDR_LO] |
               ((uint64_t)l1_data[offset + TLS_CHECK_GLOBAL_ADDR_HI] << 32);
    };

    for (uint32_t engine = 0; engine < QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER; engine++) {
        for (uint32_t trisc = 0; trisc < QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE; trisc++) {
            uint32_t slot = engine * QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE + trisc;
            uint32_t offset = slot * TLS_CHECK_RESULT_SLOT_WORDS;
            uint32_t kernel_id = l1_data[offset + TLS_CHECK_KERNEL_ID];
            uint32_t num_sw_threads = l1_data[offset + TLS_CHECK_NUM_THREADS];
            uint32_t my_thread_id = l1_data[offset + TLS_CHECK_MY_THREAD_ID];
            uint32_t hartid = l1_data[offset + TLS_CHECK_HART_ID];
            uint32_t thread_0_hartid = l1_data[offset + TLS_CHECK_THREAD_0_HART_ID];
            uint32_t global_start = l1_data[offset + TLS_CHECK_GLOBAL_START];
            uint32_t global_end = l1_data[offset + TLS_CHECK_GLOBAL_END];
            uint64_t global_addr = slot_global_addr(slot);
            uint32_t uninitialized_global_start = l1_data[offset + TLS_CHECK_UNINITIALIZED_GLOBAL_START];
            uint32_t uninitialized_global_end = l1_data[offset + TLS_CHECK_UNINITIALIZED_GLOBAL_END];
            uint64_t thread_local_start = l1_data[offset + TLS_CHECK_THREAD_LOCAL_START];
            uint64_t thread_local_end = l1_data[offset + TLS_CHECK_THREAD_LOCAL_END];
            uint32_t uninitialized_thread_local_start = l1_data[offset + TLS_CHECK_UNINITIALIZED_THREAD_LOCAL_START];
            uint32_t uninitialized_thread_local_end = l1_data[offset + TLS_CHECK_UNINITIALIZED_THREAD_LOCAL_END];

            // 1. Single kernel: all slots run kernel_id 1
            EXPECT_EQ(kernel_id, 1u) << "N" << engine << "T" << trisc;

            // 2. num_sw_threads & my_thread_id (engine id 0-3, config indices 8-11)
            EXPECT_EQ(num_sw_threads, QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER) << "N" << engine << "T" << trisc;
            EXPECT_EQ(my_thread_id, engine) << "N" << engine << "T" << trisc;

            // 3. hartid: slot 0 -> 8, slot 1 -> 9, ..., slot 15 -> 23
            EXPECT_EQ(hartid, QUASAR_FIRST_COMPUTE_HARTID + slot) << "N" << engine << "T" << trisc;

            // 4. Verify that the core is pointing to the correct binary. The specific
            // check here is the lowest hartid with the same kernel ID.
            EXPECT_EQ(thread_0_hartid, QUASAR_FIRST_COMPUTE_HARTID + trisc)
                << "N" << engine << "T" << trisc << " (shared)";

            // 5. Check that initialized global variables have the correct start and end values.
            // Initialized globals are set to 5, then incremented by 1 for each core in sequence.
            // For threaded kernels, globals are shared between trisc lanes in each NEO engine, so values
            // start at 5 with the first engine and increment by 1 for each engine in sequence.
            EXPECT_EQ(global_start, 5u + engine) << "N" << engine << "T" << trisc;
            EXPECT_EQ(global_end, 6u + engine) << "N" << engine << "T" << trisc;

            // 6. For threaded kernels, check that the global variable address is shared between slots.
            EXPECT_EQ(global_addr, slot_global_addr(trisc)) << "N" << engine << "T" << trisc;

            // 7. Check that uninitialized global variables have the correct start and end values.
            // Uninitialized globals are cleared to 0, then incremented by 1 for each slot in sequence.
            EXPECT_EQ(uninitialized_global_start, 0u + engine) << "N" << engine << "T" << trisc;
            EXPECT_EQ(uninitialized_global_end, 1u + engine) << "N" << engine << "T" << trisc;

            // 8. Check that initialized thread local variables have the correct value.
            // I.e. incrementing the variable in one DM does not affect the value in another DM.
            // TODO: Initializing thread local variables does not work yet. Once they work,
            // update this check with the correct values.
            EXPECT_EQ(thread_local_start, 10u) << "N" << engine << "T" << trisc;
            EXPECT_EQ(thread_local_end, 11u) << "N" << engine << "T" << trisc;

            // 9. Check that uninitialized thread local variables have the correct value.
            // Same as #8, but variables are cleared to 0 at the start.
            EXPECT_EQ(uninitialized_thread_local_start, 0u) << "N" << engine << "T" << trisc;
            EXPECT_EQ(uninitialized_thread_local_end, 1u) << "N" << engine << "T" << trisc;
        }
    }

    for (uint32_t trisc = 0; trisc < QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE; trisc++) {
        const uint64_t ref = slot_global_addr(trisc);
        for (uint32_t neo = 0; neo < QUASAR_NUM_TENSIX_ENGINES_PER_CLUSTER; neo++) {
            const uint32_t slot = neo * QUASAR_NUM_COMPUTE_PROCESSORS_PER_TENSIX_ENGINE + trisc;
            EXPECT_EQ(slot_global_addr(slot), ref) << "N" << neo << "T" << trisc;
        }
    }
}
