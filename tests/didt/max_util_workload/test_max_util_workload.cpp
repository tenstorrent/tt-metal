// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Max-utilization workload test.
// Architecture:
//   1. Pre-fill phase: BRISC runs on each core to read 2 DRAM buffers into L1
//      - Buffer 0: 8 tiles of bfloat16 from DRAM
//      - Buffer 1: 8 tiles of bfloat16 from DRAM
//   2. Main phase: Three decoupled kernels run simultaneously:
//      - BRISC (NOC0): Sends 8KB to right/down neighbors only
//      - NCRISC (NOC1): Sends 8KB to left/up neighbors only
//      - TRISC: Uses pre-filled L1 buffers directly, runs compute at full speed
//   No CB dependencies between data movement and compute.

#include <gtest/gtest.h>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program.hpp>
#include <tt_metal/test_utils/stimulus.hpp>
#include <distributed/mesh_device_impl.hpp>
#include "multi_device_fixture.hpp"

#include <random>
#include <vector>

namespace tt::tt_metal {

using namespace std;
using namespace tt;

namespace unit_tests::didt::max_util_workload {

// ---------------------------------------------------------------------------
// Test configuration
// ---------------------------------------------------------------------------

struct MaxUtilConfig {
    // Core grid used for the workload (logical coordinates).
    CoreCoord grid_start = {0, 0};
    CoreCoord grid_end = {0, 0};

    // Number of tiles for pre-filled buffers.
    uint32_t num_tiles = 8;  // 8 tiles as requested

    // Stress-test loop count.
    uint32_t num_iterations = 100;

    // Data transfer size in bytes.
    uint32_t data_transfer_size = 8192;  // 8KB

    // L1 buffer addresses (filled by pre-fill phase, passed to main phase).
    uint32_t l1_buffer0_addr = 0;   // input 0 bfloat16 data
    uint32_t l1_buffer1_addr = 0;   // input 1 bfloat16 data
    uint32_t l1_buffer2_addr = 0;   // output bfloat16 data
    uint32_t l1_buffer3_addr = 0;   // NOC0 send data pattern A
    uint32_t l1_buffer4_addr = 0;   // NOC0 send data pattern B
    uint32_t l1_buffer5_addr = 0;   // NOC1 send data pattern A
    uint32_t l1_buffer6_addr = 0;   // NOC1 send data pattern B
    uint32_t l1_buffer7_addr = 0;   // rx buffer from left neighbor
    uint32_t l1_buffer8_addr = 0;   // rx buffer from up neighbor
    uint32_t l1_buffer9_addr = 0;   // rx buffer from right neighbor
    uint32_t l1_buffer10_addr = 0;  // rx buffer from down neighbor
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static uint32_t tile_size_bytes(DataFormat fmt, uint32_t h = 32, uint32_t w = 32) {
    if (fmt == DataFormat::Float16_b) {
        return uint32_t(w * h * 2);  // two bytes per float16_b element
    } else {
        throw std::invalid_argument("Invalid data format");
    }
}

static std::vector<uint32_t> rng_bfp16(
    uint32_t num_bytes, uint32_t tile_rows, uint32_t tile_cols, float mean, float stdev, int seed) {
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(mean, stdev);
    std::vector<bfloat16> results(
        num_bytes / tile_size_bytes(DataFormat::Float16_b, tile_rows, tile_cols) * tile_rows * tile_cols);
    std::generate(results.begin(), results.end(), [&]() { return bfloat16(dis(gen)); });
    std::vector<uint32_t> packed_results = tt::test_utils::pack_vector<uint32_t, bfloat16>(results);
    return packed_results;
}

/// Returns a MaxUtilConfig that covers every compute core on @p device.
static MaxUtilConfig full_grid_config(IDevice* device, uint32_t num_tiles, uint32_t num_iterations) {
    auto grid = device->compute_with_storage_grid_size();
    MaxUtilConfig cfg;
    cfg.grid_start = {0, 0};
    cfg.grid_end = {grid.x - 1, grid.y - 1};
    cfg.num_tiles = num_tiles;
    cfg.num_iterations = num_iterations;
    return cfg;
}

// ---------------------------------------------------------------------------
// build_prefill_program – constructs a pre-fill Program
//
// Creates DRAM buffers, fills them with random data, and builds a program
// that reads from DRAM into L1 on all cores.
//   - Buffer 0: 8 tiles of bfloat16 from DRAM
//   - Buffer 1: 8 tiles of bfloat16 from DRAM
// ---------------------------------------------------------------------------

static Program build_prefill_program(IDevice* device, MaxUtilConfig& cfg) {
    const CoreRange core_range(cfg.grid_start, cfg.grid_end);
    const CoreRangeSet core_range_set({core_range});

    // Create DRAM buffers (using 2 bytes per element for bfloat16, 32x32 tiles)
    const uint32_t tile_rows = 32, tile_cols = 32;
    uint32_t tile_bytes_bfloat16 = tile_size_bytes(DataFormat::Float16_b, tile_rows, tile_cols);
    uint32_t buffer_size_bfloat16 = cfg.num_tiles * tile_bytes_bfloat16;
    auto dram_cfg_bfloat16 = InterleavedBufferConfig{
        .device = device,
        .size = buffer_size_bfloat16,
        .page_size = tile_bytes_bfloat16,
        .buffer_type = BufferType::DRAM,
    };

    // Buffers 3,4,5,6: uint32 pattern data
    const uint32_t buffer_size_uint32 = cfg.data_transfer_size;

    // Buffers 7,8,9,10: rx buffers (no init)
    const uint32_t rx_buffer_size = buffer_size_uint32;

    auto dram_buffer0 = CreateBuffer(dram_cfg_bfloat16);
    auto dram_buffer1 = CreateBuffer(dram_cfg_bfloat16);

    // DRAM buffers for uint32 patterns: one for 0xAAAA, one for 0x5555
    auto dram_cfg_uint32 = InterleavedBufferConfig{
        .device = device,
        .size = buffer_size_uint32,
        .page_size = buffer_size_uint32,
        .buffer_type = BufferType::DRAM,
    };
    auto dram_buffer_0xAAAA = CreateBuffer(dram_cfg_uint32);
    auto dram_buffer_0x5555 = CreateBuffer(dram_cfg_uint32);

    uint32_t dram_buffer0_addr = dram_buffer0->address();
    uint32_t dram_buffer1_addr = dram_buffer1->address();
    uint32_t dram_buffer_0xAAAA_addr = dram_buffer_0xAAAA->address();
    uint32_t dram_buffer_0x5555_addr = dram_buffer_0x5555->address();

    // Fill buffer 0 with random data
    std::vector<uint32_t> data0 =
        rng_bfp16(buffer_size_bfloat16, tile_rows, tile_cols, /*mean=*/0.0f, /*stdev=*/1.0f, /*seed=*/42);
    detail::WriteToBuffer(dram_buffer0, data0);

    // Fill buffer 1 with random data
    std::vector<uint32_t> data1 =
        rng_bfp16(buffer_size_bfloat16, tile_rows, tile_cols, /*mean=*/0.0f, /*stdev=*/1.0f, /*seed=*/43);
    detail::WriteToBuffer(dram_buffer1, data1);

    // Fill pattern buffers: 0xAAAAAAAA and 0x55555555 (32-bit patterns)
    std::vector<uint32_t> data_0xAAAA(buffer_size_uint32 / sizeof(uint32_t), 0xAAAAAAAAu);
    std::vector<uint32_t> data_0x5555(buffer_size_uint32 / sizeof(uint32_t), 0x55555555u);
    detail::WriteToBuffer(dram_buffer_0xAAAA, data_0xAAAA);
    detail::WriteToBuffer(dram_buffer_0x5555, data_0x5555);

    // Get L1 base address and pack all buffers densely
    CoreCoord physical_core = device->worker_core_from_logical_core(CoreCoord{0, 0});
    uint64_t l1_base_addr = device->get_dev_addr(physical_core, HalL1MemAddrType::DEFAULT_UNRESERVED);

    uint32_t addr = static_cast<uint32_t>(l1_base_addr);
    cfg.l1_buffer0_addr = addr;
    addr += buffer_size_bfloat16;
    cfg.l1_buffer1_addr = addr;
    addr += buffer_size_bfloat16;
    cfg.l1_buffer2_addr = addr;  // output, 8 float16_b tiles, no init
    addr += buffer_size_bfloat16;
    cfg.l1_buffer3_addr = addr;  // pattern 0xAAAA
    addr += buffer_size_uint32;
    cfg.l1_buffer4_addr = addr;  // pattern 0x5555
    addr += buffer_size_uint32;
    cfg.l1_buffer5_addr = addr;  // pattern 0xAAAA
    addr += buffer_size_uint32;
    cfg.l1_buffer6_addr = addr;  // pattern 0x5555
    addr += buffer_size_uint32;
    cfg.l1_buffer7_addr = addr;  // rx from left, no init
    addr += rx_buffer_size;
    cfg.l1_buffer8_addr = addr;  // rx from up, no init
    addr += rx_buffer_size;
    cfg.l1_buffer9_addr = addr;  // rx from right, no init
    addr += rx_buffer_size;
    cfg.l1_buffer10_addr = addr;  // rx from down, no init

    // Pre-fill: L1 buf0..10=0x1b200, 0x1f200, 0x21400, 0x25400, 0x27400, 0x29400, 0x2b400, 0x2d400, 0x2f400, 0x31400,
    // 0x33400
    log_info(
        LogTest,
        "Pre-fill: L1 buf0..10=0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}, 0x{:x}",
        cfg.l1_buffer0_addr,
        cfg.l1_buffer1_addr,
        cfg.l1_buffer2_addr,
        cfg.l1_buffer3_addr,
        cfg.l1_buffer4_addr,
        cfg.l1_buffer5_addr,
        cfg.l1_buffer6_addr,
        cfg.l1_buffer7_addr,
        cfg.l1_buffer8_addr,
        cfg.l1_buffer9_addr,
        cfg.l1_buffer10_addr);

    Program program = CreateProgram();

    // Pre-fill kernel on BRISC - reads from DRAM to L1
    CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/prefill_l1.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {
                    dram_buffer0_addr,        // 0: dram_buffer0_addr
                    dram_buffer1_addr,        // 1: dram_buffer1_addr
                    dram_buffer_0xAAAA_addr,  // 2: dram_buffer_0xAAAA_addr
                    dram_buffer_0x5555_addr,  // 3: dram_buffer_0x5555_addr
                    cfg.l1_buffer0_addr,      // 4: l1_buffer0_addr
                    cfg.l1_buffer1_addr,      // 5: l1_buffer1_addr
                    cfg.l1_buffer3_addr,      // 6: l1_buffer3_addr
                    cfg.l1_buffer4_addr,      // 7: l1_buffer4_addr
                    cfg.l1_buffer5_addr,      // 8: l1_buffer5_addr
                    cfg.l1_buffer6_addr,      // 9: l1_buffer6_addr
                    tile_bytes_bfloat16,      // 10: tile_size_bytes (bfloat16, 2048)
                    buffer_size_uint32,       // 11: data_transfer_size (8KB)
                    cfg.num_tiles,            // 12: num_tiles (8)
                },
        });

    return program;
}

// ---------------------------------------------------------------------------
// build_program – constructs the main Program
//
// Decoupled kernels: BRISC and NCRISC generate NOC traffic only,
// TRISC uses pre-filled L1 buffers directly without CB waits.
// ---------------------------------------------------------------------------

static Program build_program(IDevice* device, const MaxUtilConfig& cfg) {
    const CoreRange core_range(cfg.grid_start, cfg.grid_end);
    const CoreRangeSet core_range_set({core_range});

    log_info(LogTest, "Main program: {} cores, {} iterations", core_range.size(), cfg.num_iterations);

    Program program = CreateProgram();

    // -- Reader kernel (BRISC / RISCV_0 / NOC0) -----------------------------
    // Only generates NOC traffic - no CBs, no DRAM
    auto reader_kernel = CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/max_util_reader.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args =
                {
                    cfg.num_iterations,      // 0: num_iterations
                    cfg.l1_buffer3_addr,     // 1: l1_buffer_addr for tx pattern A
                    cfg.l1_buffer4_addr,     // 2: l1_buffer_addr for tx pattern B
                    cfg.l1_buffer7_addr,     // 3: l1_buffer_addr for rx from left
                    cfg.l1_buffer8_addr,     // 4: l1_buffer_addr for rx from up
                    cfg.l1_buffer9_addr,     // 5: l1_buffer_addr for rx from right
                    cfg.l1_buffer10_addr,    // 6: l1_buffer_addr for rx from down
                    cfg.data_transfer_size,  // 7: data_transfer_size (8KB)
                },
        });

    // -- Writer kernel (NCRISC / RISCV_1 / NOC1) ----------------------------
    // Only generates NOC traffic - no CBs, no DRAM
    auto writer_kernel = CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/max_util_writer.cpp",
        core_range_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args =
                {
                    cfg.num_iterations,      // 0: num_iterations
                    cfg.l1_buffer5_addr,     // 1: l1_buffer_addr for tx pattern A
                    cfg.l1_buffer6_addr,     // 2: l1_buffer_addr for tx pattern B
                    cfg.l1_buffer7_addr,     // 3: l1_buffer_addr for rx from left
                    cfg.l1_buffer8_addr,     // 4: l1_buffer_addr for rx from up
                    cfg.l1_buffer9_addr,     // 5: l1_buffer_addr for rx from right
                    cfg.l1_buffer10_addr,    // 6: l1_buffer_addr for rx from down
                    cfg.data_transfer_size,  // 7: data_transfer_size (8KB)
                },
        });

    // -- Compute kernel (TRISC) ---------------------------------------------
    // Uses pre-filled L1 buffers directly, no CB waits
    CreateKernel(
        program,
        "tests/didt/max_util_workload/kernels/max_util_compute.cpp",
        core_range_set,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .compile_args =
                {
                    cfg.l1_buffer0_addr,  // 0: l1_buffer0_addr (bfloat16)
                    cfg.l1_buffer1_addr,  // 1: l1_buffer1_addr (bfloat16)
                    cfg.l1_buffer2_addr,  // 2: l1_buffer2_addr (output, 8 float16_b tiles)
                    cfg.num_tiles,        // 3: num_tiles (8)
                    cfg.num_iterations,   // 4: num_iterations
                },
        });

    // -- Set runtime arguments for NOC traffic kernels ----------------------
    // Pass physical coordinates of neighbor cores for NOC addressing
    for (uint32_t y = cfg.grid_start.y; y <= cfg.grid_end.y; ++y) {
        for (uint32_t x = cfg.grid_start.x; x <= cfg.grid_end.x; ++x) {
            CoreCoord core = {x, y};

            // Calculate neighbor logical coordinates with wraparound
            // Reader (NOC0): sends to right and down
            uint32_t target_right_x = (x >= cfg.grid_end.x) ? cfg.grid_start.x : (x + 1);
            uint32_t target_down_y = (y >= cfg.grid_end.y) ? cfg.grid_start.y : (y + 1);
            // Writer (NOC1): sends to left and up
            uint32_t target_left_x = (x == cfg.grid_start.x) ? cfg.grid_end.x : (x - 1);
            uint32_t target_up_y = (y == cfg.grid_start.y) ? cfg.grid_end.y : (y - 1);

            // Convert neighbor logical coordinates to physical coordinates
            CoreCoord right_core = {target_right_x, y};
            CoreCoord down_core = {x, target_down_y};
            CoreCoord left_core = {target_left_x, y};
            CoreCoord up_core = {x, target_up_y};

            CoreCoord physical_right = device->worker_core_from_logical_core(right_core);
            CoreCoord physical_down = device->worker_core_from_logical_core(down_core);
            CoreCoord physical_left = device->worker_core_from_logical_core(left_core);
            CoreCoord physical_up = device->worker_core_from_logical_core(up_core);

            // Runtime args for reader (NOC0): physical coords of right and down neighbors
            SetRuntimeArgs(
                program,
                reader_kernel,
                core,
                std::vector<uint32_t>{physical_right.x, physical_right.y, physical_down.x, physical_down.y});

            // Runtime args for writer (NOC1): physical coords of left and up neighbors
            SetRuntimeArgs(
                program,
                writer_kernel,
                core,
                std::vector<uint32_t>{physical_left.x, physical_left.y, physical_up.x, physical_up.y});

            // Compute kernel doesn't need runtime args - uses compile-time L1 addresses
        }
    }

    return program;
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Runs the pre-fill program to populate L1 buffers, then runs the main program.
static bool run_single_device(const shared_ptr<distributed::MeshDevice>& mesh_device, MaxUtilConfig& cfg) {
    IDevice* device = mesh_device->impl().get_device(0);

    auto& cq = mesh_device->mesh_command_queue();
    auto target =
        distributed::MeshCoordinateRange(distributed::MeshCoordinate({0, 0}), distributed::MeshCoordinate({0, 0}));

    // Phase 1: Pre-fill L1 buffers from DRAM
    auto prefill_program = build_prefill_program(device, cfg);

    // Run pre-fill program
    auto prefill_workload = distributed::MeshWorkload();
    prefill_workload.add_program(target, std::move(prefill_program));
    distributed::EnqueueMeshWorkload(cq, prefill_workload, /*blocking=*/true);
    distributed::Finish(cq);

    log_info(LogTest, "Pre-fill phase complete");

    // Phase 2: Build and run main program
    auto mesh_workload = distributed::MeshWorkload();
    mesh_workload.add_program(target, build_program(device, cfg));

    distributed::EnqueueMeshWorkload(cq, mesh_workload, /*blocking=*/true);
    distributed::Finish(cq);

    log_info(
        LogTest,
        "MaxUtilWorkload [single-device] done: device={}, grid=[{},{}]->[{},{}], tiles={}, iters={}",
        device->id(),
        cfg.grid_start.x,
        cfg.grid_start.y,
        cfg.grid_end.x,
        cfg.grid_end.y,
        cfg.num_tiles,
        cfg.num_iterations);

    return true;
}

/// Runs the workload on every device in the mesh simultaneously.
static bool run_all_devices(
    const shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t num_tiles, uint32_t num_iterations) {
    auto& cq = mesh_device->mesh_command_queue();

    // Phase 1: Pre-fill on all devices
    auto prefill_workload = distributed::MeshWorkload();
    std::map<IDevice*, MaxUtilConfig> device_configs;

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        IDevice* device = mesh_device->get_device(coord[0], coord[1]);
        MaxUtilConfig cfg = full_grid_config(device, num_tiles, num_iterations);

        auto prefill_prog = build_prefill_program(device, cfg);
        device_configs[device] = cfg;

        auto target = distributed::MeshCoordinateRange(coord, coord);
        prefill_workload.add_program(target, std::move(prefill_prog));

        log_info(LogTest, "Pre-fill queuing: device={}", device->id());
    }

    distributed::EnqueueMeshWorkload(cq, prefill_workload, /*blocking=*/true);
    distributed::Finish(cq);
    log_info(LogTest, "Pre-fill phase complete on all devices");

    // Phase 2: Main program on all devices
    auto mesh_workload = distributed::MeshWorkload();

    for (const auto& coord : distributed::MeshCoordinateRange(mesh_device->shape())) {
        IDevice* device = mesh_device->get_device(coord[0], coord[1]);
        MaxUtilConfig cfg = device_configs[device];

        auto target = distributed::MeshCoordinateRange(coord, coord);
        mesh_workload.add_program(target, build_program(device, cfg));

        log_info(LogTest, "Main program queuing: device={}", device->id());
    }

    distributed::EnqueueMeshWorkload(cq, mesh_workload, /*blocking=*/true);
    distributed::Finish(cq);

    return true;
}

// ---------------------------------------------------------------------------
// Test entry points
// ---------------------------------------------------------------------------

/// Quick smoke run: 2x2 grid on device 0, one iteration.
void max_util_smoke(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    MaxUtilConfig cfg;
    cfg.grid_start = {0, 0};
    cfg.grid_end = {1, 1};
    cfg.num_tiles = 8;
    cfg.num_iterations = 1;
    EXPECT_TRUE(run_single_device(mesh_device, cfg));
}

/// Sustained stress run on device 0: all available cores, many iterations.
void max_util_stress(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    IDevice* device = mesh_device->impl().get_device(0);
    MaxUtilConfig cfg = full_grid_config(device, /*num_tiles=*/8, /*num_iterations=*/1000);
    EXPECT_TRUE(run_single_device(mesh_device, cfg));
}

/// All-devices stress run: every device in the mesh, all cores, many iterations.
void max_util_all_devices(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    EXPECT_TRUE(run_all_devices(mesh_device, /*num_tiles=*/8, /*num_iterations=*/1000));
}

}  // namespace unit_tests::didt::max_util_workload

// ---------------------------------------------------------------------------
// GTest fixtures
// ---------------------------------------------------------------------------

TEST_F(GenericMeshDeviceFixture, MaxUtilWorkload_Smoke) {
    unit_tests::didt::max_util_workload::max_util_smoke(get_mesh_device());
}

TEST_F(GenericMeshDeviceFixture, MaxUtilWorkload_Stress) {
    unit_tests::didt::max_util_workload::max_util_stress(get_mesh_device());
}

TEST_F(GenericMeshDeviceFixture, MaxUtilWorkload_AllDevices) {
    unit_tests::didt::max_util_workload::max_util_all_devices(get_mesh_device());
}

}  // namespace tt::tt_metal
