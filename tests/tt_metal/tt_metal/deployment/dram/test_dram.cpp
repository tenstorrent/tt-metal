#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "dram_base.hpp"

#include <gtest/gtest.h>
#include <tt-logger/tt-logger.hpp>

#include <atomic>
#include <csignal>

namespace tt::tt_metal {

using namespace std;
using namespace tt;

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentDramSingleCoreSingleController) {
    bool all_pass = true;

    const uint32_t repeats = 1u;
    const uint32_t initial_seed = 0x12345678u;
    const uint32_t advance_seed = 1u;
    const bool stop_on_fail = true;

    static const uint32_t kDeploymentPatterns[] = {
        DRAM_PATTERN_COUNTER, DRAM_PATTERN_CHECKERBOARD,
        //      DRAM_PATTERN_ADDRESS,
        //      DRAM_PATTERN_MARCHING_ONES,
        //      DRAM_PATTERN_MARCHING_ZEROES,
        //      DRAM_PATTERN_MARCHING_ONE_BITS,
        //      DRAM_PATTERN_MARCHING_ZERO_BITS,
    };

    SignalGuard(SIGINT, handle_sigint);

    for (const auto& mesh_device : devices_) {
        if (g_stop_requested.load()) {
            break;
        }

        auto* const device = mesh_device->get_devices()[0];
        CoreCoord core = {0, 0};

        for (uint32_t pattern_id : kDeploymentPatterns) {
            if (g_stop_requested.load()) {
                break;
            }

            DramDeploymentConfig cfg{
                .bank_id = 0u,
                .bank_offset = 0u,
                .total_bytes = 256u * 1024u * 1024u,
                .chunk_bytes = 4096u,
                .pattern_id = pattern_id,
                .write_noc = 0u,
                .read_noc = 0u,
                .transfer_len_mode = 0u,
                .max_burst_len = 4096u,
                .skip_writes = 0u,
                .skip_reads = 0u,
            };

            uint32_t seed = initial_seed;

            for (uint32_t repeat_index = 0; repeat_index < repeats; ++repeat_index) {
                if (g_stop_requested.load()) {
                    break;
                }

                const uint32_t num_passes = num_passes_for_pattern(pattern_id);

                for (uint32_t pass_index = 0; pass_index < num_passes; ++pass_index) {
                    if (g_stop_requested.load()) {
                        break;
                    }

                    log_info(
                        tt::LogTest,
                        "Running {} pattern on device {}, core {}, repeat {}, pass {}, seed=0x{:08x}",
                        pattern_name(pattern_id),
                        device->id(),
                        core,
                        repeat_index,
                        pass_index,
                        seed);

                    bool one_pass = run_dram_base_test(
                        static_cast<MeshDispatchFixture*>(this),
                        mesh_device,
                        core,
                        cfg,
                        seed,
                        pass_index,
                        repeat_index,
                        DataMovementProcessor::RISCV_0);

                    all_pass &= one_pass;

                    if (!one_pass && stop_on_fail) {
                        ASSERT_TRUE(false);
                    }
                }

                seed += advance_seed;
            }
        }
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user after current test finished.";
    }

    ASSERT_TRUE(all_pass);
}

static std::vector<CoreCoord> get_worker_cores_for_deployment(IDevice* device) {
    std::vector<CoreCoord> cores;

    const auto grid = device->compute_with_storage_grid_size();
    for (uint32_t y = 0; y < grid.y; ++y) {
        for (uint32_t x = 0; x < grid.x; ++x) {
            cores.emplace_back(x, y);
        }
    }

    return cores;
}

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentDramAllCoresSingleController) {
    bool all_pass = true;

    const uint32_t controller_bank_id = 0u;
    const uint64_t controller_bank_offset = 0u;

    const uint32_t total_bytes_across_controller =
        DRAM_TEST_MAX_BANK_BYTES;  //(4GB-16MB)    // 2 * 1024u * 1024u * 1024u;  // 2 GiB
    const uint32_t chunk_bytes = 4 * 4096u;

    const uint32_t repeats = 1u;
    const uint32_t initial_seed = 0x12345678u;
    const uint32_t advance_seed = 1u;
    const bool stop_on_fail = true;

    static const uint32_t kDeploymentPatterns[] = {
        DRAM_PATTERN_COUNTER,
        DRAM_PATTERN_CHECKERBOARD,
        DRAM_PATTERN_ADDRESS,
        DRAM_PATTERN_MARCHING_ONES,
        DRAM_PATTERN_MARCHING_ZEROES,
        DRAM_PATTERN_MARCHING_ONE_BITS,
        DRAM_PATTERN_MARCHING_ZERO_BITS,
    };

    SignalGuard(SIGINT, handle_sigint);

    for (const auto& mesh_device : devices_) {
        if (g_stop_requested.load()) {
            break;
        }

        auto* const device = mesh_device->get_devices()[0];
        const auto worker_cores = get_worker_cores_for_deployment(device);

        TT_FATAL(!worker_cores.empty(), "No worker cores found");

        for (uint32_t pattern_id : kDeploymentPatterns) {
            if (g_stop_requested.load()) {
                break;
            }

            DramDeploymentConfig cfg{
                .bank_id = controller_bank_id,
                .bank_offset = controller_bank_offset,
                .total_bytes = total_bytes_across_controller,
                .chunk_bytes = chunk_bytes,
                .pattern_id = pattern_id,
                .write_noc = 0u,
                .read_noc = 0u,
                .transfer_len_mode = 0u,
                .max_burst_len = chunk_bytes,
                .skip_writes = 0u,
                .skip_reads = 0u,
            };

            uint32_t seed = initial_seed;

            for (uint32_t repeat_index = 0; repeat_index < repeats; ++repeat_index) {
                if (g_stop_requested.load()) {
                    break;
                }

                const uint32_t num_passes = num_passes_for_pattern(pattern_id);

                for (uint32_t pass_index = 0; pass_index < num_passes; ++pass_index) {
                    if (g_stop_requested.load()) {
                        break;
                    }

                    log_info(
                        tt::LogTest,
                        "Running {} pattern on all cores to controller {}, repeat {}, pass {}, seed=0x{:08x}",
                        pattern_name(pattern_id),
                        controller_bank_id,
                        repeat_index,
                        pass_index,
                        seed);

                    bool one_pass = run_dram_multi_core_single_controller_test(
                        static_cast<MeshDispatchFixture*>(this),
                        mesh_device,
                        worker_cores,
                        cfg,
                        seed,
                        pass_index,
                        repeat_index,
                        DataMovementProcessor::RISCV_0);

                    all_pass &= one_pass;

                    if (!one_pass && stop_on_fail) {
                        ASSERT_TRUE(false);
                    }
                }

                seed += advance_seed;
            }
        }
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user after current test finished.";
    }

    ASSERT_TRUE(all_pass);
}

TEST_F(UnitMeshCQProgramFixture, TensixDeploymentDramAllCoresAllControllers) {
    bool all_pass = true;

    const uint32_t total_bytes_per_controller = 256u * 1024u * 1024u;  // 256 MiB per controller
    const uint32_t chunk_bytes = 4096u;

    const uint32_t repeats = 1u;
    const uint32_t initial_seed = 0x12345678u;
    const uint32_t advance_seed = 1u;
    const bool stop_on_fail = true;

    static const uint32_t kDeploymentPatterns[] = {
        DRAM_PATTERN_COUNTER,
        DRAM_PATTERN_CHECKERBOARD,
        DRAM_PATTERN_ADDRESS,
        DRAM_PATTERN_MARCHING_ONES,
        DRAM_PATTERN_MARCHING_ZEROES,
        DRAM_PATTERN_MARCHING_ONE_BITS,
        DRAM_PATTERN_MARCHING_ZERO_BITS,
    };

    SignalGuard(SIGINT, handle_sigint);

    for (const auto& mesh_device : devices_) {
        if (g_stop_requested.load()) {
            break;
        }

        auto* const device = mesh_device->get_devices()[0];
        const auto worker_cores = get_worker_cores_for_deployment(device);

        TT_FATAL(!worker_cores.empty(), "No worker cores found");

        for (uint32_t pattern_id : kDeploymentPatterns) {
            if (g_stop_requested.load()) {
                break;
            }

            uint32_t seed = initial_seed;

            for (uint32_t repeat_index = 0; repeat_index < repeats; ++repeat_index) {
                if (g_stop_requested.load()) {
                    break;
                }

                const uint32_t num_passes = num_passes_for_pattern(pattern_id);

                for (uint32_t pass_index = 0; pass_index < num_passes; ++pass_index) {
                    if (g_stop_requested.load()) {
                        break;
                    }

                    log_info(
                        tt::LogTest,
                        "Running {} pattern on all cores to all controllers, repeat {}, pass {}, seed=0x{:08x}",
                        pattern_name(pattern_id),
                        repeat_index,
                        pass_index,
                        seed);

                    bool one_pass = run_dram_multi_core_all_controllers_test(
                        static_cast<MeshDispatchFixture*>(this),
                        mesh_device,
                        worker_cores,
                        total_bytes_per_controller,
                        chunk_bytes,
                        pattern_id,
                        0u,  // write_noc
                        0u,  // read_noc
                        0u,  // transfer_len_mode
                        chunk_bytes,
                        0u,  // skip_writes
                        0u,  // skip_reads
                        seed,
                        pass_index,
                        repeat_index,
                        DataMovementProcessor::RISCV_0);

                    all_pass &= one_pass;

                    if (!one_pass && stop_on_fail) {
                        ASSERT_TRUE(false);
                    }
                }

                seed += advance_seed;
            }
        }
    }

    if (g_stop_requested.load()) {
        GTEST_SKIP() << "Test interrupted by user after current test finished.";
    }

    ASSERT_TRUE(all_pass);
}

}  // namespace tt::tt_metal
