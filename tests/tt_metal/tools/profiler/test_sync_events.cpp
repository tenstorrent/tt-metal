// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Sync events profiler test.
//
// Runs producer on BRISC, consumer on NCRISC (same kernel, different api_id).
// Producer delays DELAY_CYCLES before signaling.
// Consumer waits - the measured wait duration should match the delay.
//
// Usage: test_sync_events_timing [API_ID]
//   0 = CB test
//   1 = Semaphore local test
//   2 = Semaphore remote test
//   3 = Semaphore wait_min test
//   (no arg = run all)

#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/streaming_profiler.hpp>
#include "tt_metal/impl/kernels/kernel.hpp"

using namespace tt;
using namespace tt::tt_metal;
using Risc = tt::tt_metal::experimental::streaming_profiler::Risc;

constexpr uint32_t DELAY_CYCLES = 10000;

void RunApiTest(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    int producer_api_id,
    int consumer_api_id,
    bool use_remote_core,
    const char* test_name,
    Risc producer_risc = Risc::BRISC,
    Risc consumer_risc = Risc::NCRISC) {
    const char* risc_names[] = {"BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2"};
    fmt::print(
        "Running {} (producer api={} on {}, consumer api={} on {}{})...\n",
        test_name,
        producer_api_id,
        risc_names[(int)producer_risc],
        consumer_api_id,
        risc_names[(int)consumer_risc],
        use_remote_core ? ", different core" : "");

    CoreCoord producer_core = {0, 0};
    CoreCoord consumer_core = use_remote_core ? CoreCoord{1, 0} : CoreCoord{0, 0};

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    tt_metal::Program program = tt_metal::CreateProgram();

    // Create CB
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t tile_size = 2048;
    CircularBufferConfig cb_config =
        CircularBufferConfig(tile_size, {{cb_id, tt::DataFormat::Float16_b}}).set_page_size(cb_id, tile_size);
    CreateCircularBuffer(program, producer_core, cb_config);
    if (use_remote_core) {
        CreateCircularBuffer(program, consumer_core, cb_config);
    }

    // Create semaphores - semaphore ID 0 on each core
    auto producer_sem_id = CreateSemaphore(program, producer_core, 0);
    uint32_t consumer_sem_id = producer_sem_id;

    uint32_t remote_noc_x = 0xFFFFFFFF;
    uint32_t remote_noc_y = 0xFFFFFFFF;

    if (use_remote_core) {
        consumer_sem_id = CreateSemaphore(program, consumer_core, 0);
        auto noc_coords = mesh_device->worker_core_from_logical_core(consumer_core);
        remote_noc_x = noc_coords.x;
        remote_noc_y = noc_coords.y;
    }

    std::map<std::string, std::string> defines = {
        {"DELAY_CYCLES", std::to_string(DELAY_CYCLES)},
        {"CB_ID", std::to_string(cb_id)},
    };

    // Create producer kernel
    if (producer_risc == Risc::TRISC0) {
        auto producer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/sync_apis_compute.cpp",
            producer_core,
            tt_metal::ComputeConfig{.defines = defines});
        SetRuntimeArgs(program, producer_kernel, producer_core, {(uint32_t)producer_api_id});
    } else {
        auto producer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/sync_apis_dm.cpp",
            producer_core,
            tt_metal::DataMovementConfig{
                .processor = producer_risc == Risc::BRISC ? tt_metal::DataMovementProcessor::RISCV_0
                                                          : tt_metal::DataMovementProcessor::RISCV_1,
                .noc = producer_risc == Risc::BRISC ? tt_metal::NOC::RISCV_0_default : tt_metal::NOC::RISCV_1_default,
                .defines = defines});
        SetRuntimeArgs(
            program,
            producer_kernel,
            producer_core,
            {(uint32_t)producer_api_id,
             producer_sem_id,
             remote_noc_x,
             remote_noc_y,
             consumer_sem_id,
             use_remote_core ? 1u : 0u});
    }

    // Create consumer kernel
    if (consumer_risc == Risc::TRISC0) {
        auto consumer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/sync_apis_compute.cpp",
            consumer_core,
            tt_metal::ComputeConfig{.defines = defines});
        SetRuntimeArgs(program, consumer_kernel, consumer_core, {(uint32_t)consumer_api_id});
    } else {
        auto consumer_kernel = tt_metal::CreateKernel(
            program,
            "tests/tt_metal/tools/profiler/kernels/sync_apis_dm.cpp",
            consumer_core,
            tt_metal::DataMovementConfig{
                .processor = consumer_risc == Risc::NCRISC ? tt_metal::DataMovementProcessor::RISCV_1
                                                           : tt_metal::DataMovementProcessor::RISCV_0,
                .noc = consumer_risc == Risc::NCRISC ? tt_metal::NOC::RISCV_1_default : tt_metal::NOC::RISCV_0_default,
                .defines = defines});
        SetRuntimeArgs(
            program,
            consumer_kernel,
            consumer_core,
            {(uint32_t)consumer_api_id, consumer_sem_id, 0u, 0u, 0u, 0u});  // is_remote=0 for consumer
    }

    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, false);
}

int main(int argc, char* argv[]) {
    bool pass = true;

    try {
        const char* sync_events_env = std::getenv("TT_METAL_STREAMING_PROFILER_SYNC_EVENTS");
        if (!sync_events_env || std::string(sync_events_env) != "1") {
            fmt::print(stderr, "WARNING: Run with TT_METAL_STREAMING_PROFILER_SYNC_EVENTS=1\n");
        }

        int device_id = 0;
        auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

        int test_api = -1;
        if (argc > 1) {
            test_api = std::atoi(argv[1]);
        }

        // Test configs: (producer_api, consumer_api, remote, name, producer_risc, consumer_risc)
        struct TestConfig {
            int producer_api;
            int consumer_api;
            bool remote;
            const char* name;
            Risc producer_risc;
            Risc consumer_risc;
        };

        TestConfig tests[] = {
            // Raw CB APIs
            {0, 1, false, "CB wait", Risc::BRISC, Risc::NCRISC},
            {7, 8, false, "CB reserve", Risc::BRISC, Risc::NCRISC},

            // Raw Semaphore APIs. The noc 1 cases are produced from NCRISC; every other case is noc 0.
            {2, 3, false, "Raw: sem_set + sem_wait", Risc::BRISC, Risc::NCRISC},
            {4, 5, true, "Raw: sem_inc remote", Risc::BRISC, Risc::NCRISC},
            {2, 6, false, "Raw: sem_set + sem_wait_min", Risc::BRISC, Risc::NCRISC},
            {11, 12, true, "Raw: sem_inc_multicast", Risc::BRISC, Risc::NCRISC},
            {13, 14, true, "Raw: sem_set_multicast", Risc::BRISC, Risc::NCRISC},
            {31, 5, true, "Raw: sem_set_remote", Risc::BRISC, Risc::NCRISC},
            {33, 5, true, "Raw: sem_set_multicast_loopback_src", Risc::BRISC, Risc::NCRISC},
            {4, 5, true, "Raw: sem_inc remote (noc 1)", Risc::NCRISC, Risc::BRISC},
            {11, 12, true, "Raw: sem_inc_multicast (noc 1)", Risc::NCRISC, Risc::BRISC},
            {13, 14, true, "Raw: sem_set_multicast (noc 1)", Risc::NCRISC, Risc::BRISC},

            // Semaphore class APIs (dataflow)
            {20, 21, false, "Class: set() + wait()", Risc::BRISC, Risc::NCRISC},
            {22, 23, false, "Class: up() + wait_min()", Risc::BRISC, Risc::NCRISC},
            {24, 25, true, "Class: up() remote", Risc::BRISC, Risc::NCRISC},
            {20, 26, false, "Class: set() + down()", Risc::BRISC, Risc::NCRISC},
            {27, 28, true, "Class: set_multicast()", Risc::BRISC, Risc::NCRISC},
            {29, 30, true, "Class: inc_multicast()", Risc::BRISC, Risc::NCRISC},

            // Compute RISC (TRISC) CB APIs - all architectures
            {0, 102, false, "Compute CB: BRISC push + TRISC wait+pop", Risc::BRISC, Risc::TRISC0},
            {100, 1, false, "Compute CB: TRISC push + NCRISC wait", Risc::TRISC0, Risc::NCRISC},
        };

        constexpr int num_tests = sizeof(tests) / sizeof(tests[0]);
        if (test_api >= 0 && test_api < num_tests) {
            auto& t = tests[test_api];
            RunApiTest(mesh_device, t.producer_api, t.consumer_api, t.remote, t.name, t.producer_risc, t.consumer_risc);
        } else {
            for (auto& t : tests) {
                RunApiTest(
                    mesh_device, t.producer_api, t.consumer_api, t.remote, t.name, t.producer_risc, t.consumer_risc);
            }
        }

        // Streaming profiler writes the zone CSV automatically on device close
        // (via TT_METAL_STREAMING_PROFILER_ZONE_CSV env var)
        pass &= mesh_device->close();

        fmt::print("\nExpected events & timing:\n");
        fmt::print("  CB:         SYNC-CB-RESERVE/PUSH (BRISC), SYNC-CB-WAIT/POP (NCRISC)\n");
        fmt::print("  Sem local:  SYNC-SEM-SET (BRISC), SYNC-SEM-WAIT (NCRISC)\n");
        fmt::print("  Sem remote: SYNC-SEM-SET-REMOTE (BRISC), SYNC-SEM-WAIT (NCRISC)\n");
        fmt::print("  Wait duration: ~%u cycles\n", DELAY_CYCLES);

    } catch (const std::exception& e) {
        pass = false;
        fmt::print(stderr, "Exception: {}\n", e.what());
    }

    return pass ? 0 : 1;
}
