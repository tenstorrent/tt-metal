// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

#include "device_fixture.hpp"
#include "dm_common.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/host_api.hpp>
#include <tt-logger/tt-logger.hpp>
#include "impl/context/metal_context.hpp"

namespace tt::tt_metal {

using namespace tt::test_utils;

namespace unit_tests::dm::quasar_core_shift {

constexpr auto kSenderKernelPath = "tests/tt_metal/tt_metal/data_movement/quasar_core_shift/kernels/sender_bw.cpp";

// Offset from L1 base for the 64-bit cycle counter slot (kept clear of data buffer).
constexpr uint32_t kCyclesL1OffsetBytes = 256 * 1024;

enum class ShiftPosition : uint32_t {
    Baseline = 0,
    Left = 1,
    Right = 2,
    Up = 3,
    Down = 4,
};

const char* shift_position_name(ShiftPosition pos) {
    switch (pos) {
        case ShiftPosition::Baseline: return "baseline";
        case ShiftPosition::Left: return "left";
        case ShiftPosition::Right: return "right";
        case ShiftPosition::Up: return "up";
        case ShiftPosition::Down: return "down";
    }
    return "unknown";
}

bool should_skip_test() {
    const auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    if (arch != tt::ARCH::QUASAR) {
        return true;
    }
    return !MetalContext::instance().rtoptions().is_simulator_or_emulated();
}

// User (row, col) -> CoreCoord{x=col, y=row}
CoreCoord user_to_logical(uint32_t row, uint32_t col) { return CoreCoord{col, row}; }

struct ShiftReceiverCase {
    ShiftPosition position;
    CoreCoord receiver;
    int32_t dx;
    int32_t dy;
};

struct CoreShiftConfig {
    uint32_t test_id = 0;
    CoreCoord sender = {0, 0};
    CoreCoord receiver = {0, 0};
    uint32_t num_of_transactions = 256;
    uint32_t bytes_per_transaction = 4096;
};

struct RunResult {
    uint64_t cycles = 0;
    double bandwidth_bpc = 0.0;
    bool pass = false;
};

std::vector<ShiftReceiverCase> build_shift_receivers(CoreCoord baseline, uint32_t grid_x, uint32_t grid_y) {
    std::vector<ShiftReceiverCase> cases;

    auto try_add = [&](ShiftPosition pos, CoreCoord receiver, int32_t dx, int32_t dy) {
        if (receiver.x < grid_x && receiver.y < grid_y) {
            cases.push_back({pos, receiver, dx, dy});
        } else {
            log_warning(
                tt::LogTest,
                "Skipping {} receiver ({},{}) — out of grid ({}x{})",
                shift_position_name(pos),
                receiver.x,
                receiver.y,
                grid_x,
                grid_y);
        }
    };

    cases.push_back({ShiftPosition::Baseline, baseline, 0, 0});
    try_add(ShiftPosition::Left, CoreCoord{baseline.x - 1, baseline.y}, -1, 0);
    try_add(ShiftPosition::Right, CoreCoord{baseline.x + 1, baseline.y}, 1, 0);
    try_add(ShiftPosition::Up, CoreCoord{baseline.x, baseline.y + 1}, 0, 1);
    try_add(ShiftPosition::Down, CoreCoord{baseline.x, baseline.y - 1}, 0, -1);
    return cases;
}

static void run_sender_kernel(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    const CoreShiftConfig& cfg,
    uint32_t data_addr,
    uint32_t cycles_addr,
    uint32_t receiver_noc_x,
    uint32_t receiver_noc_y) {
    log_info(
        tt::LogTest,
        "Building program for sender=({},{}) receiver_logical=({},{}) receiver_noc=({},{}) txns={} size={}",
        cfg.sender.x,
        cfg.sender.y,
        cfg.receiver.x,
        cfg.receiver.y,
        receiver_noc_x,
        receiver_noc_y,
        cfg.num_of_transactions,
        cfg.bytes_per_transaction);

    Program program = CreateProgram();
    CoreRangeSet sender_cores({CoreRange(cfg.sender)});
    KernelHandle sender_kernel = experimental::quasar::CreateKernel(
        program,
        kSenderKernelPath,
        sender_cores,
        experimental::quasar::QuasarDataMovementConfig{
            .num_threads_per_cluster = 1,
        });

    SetRuntimeArgs(
        program,
        sender_kernel,
        sender_cores,
        {
            data_addr,
            cycles_addr,
            cfg.num_of_transactions,
            cfg.bytes_per_transaction,
            receiver_noc_x,
            receiver_noc_y,
        });

    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", cfg.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    log_info(
        tt::LogTest,
        "Launching sender_bw: sender=({},{}) receiver_logical=({},{}) receiver_noc=({},{}) txns={} size={}",
        cfg.sender.x,
        cfg.sender.y,
        cfg.receiver.x,
        cfg.receiver.y,
        receiver_noc_x,
        receiver_noc_y,
        cfg.num_of_transactions,
        cfg.bytes_per_transaction);

    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range(distributed::MeshCoordinate(0, 0));
    workload.add_program(device_range, std::move(program));
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    log_info(tt::LogTest, "sender_bw finished on host");
}

RunResult run_one(const std::shared_ptr<distributed::MeshDevice>& mesh_device, const CoreShiftConfig& cfg) {
    RunResult result;
    IDevice* device = mesh_device->get_devices()[0];

    L1AddressInfo l1_info = unit_tests::dm::get_l1_address_and_size(mesh_device, cfg.sender);
    const uint32_t data_addr = static_cast<uint32_t>(l1_info.base_address);
    const uint32_t cycles_addr = data_addr + kCyclesL1OffsetBytes;

    if (l1_info.size < kCyclesL1OffsetBytes + sizeof(uint64_t)) {
        log_error(tt::LogTest, "Insufficient L1 size for core shift benchmark");
        return result;
    }
    if (cfg.bytes_per_transaction > l1_info.size - kCyclesL1OffsetBytes) {
        log_error(tt::LogTest, "Transaction size {} exceeds available L1 below cycles slot", cfg.bytes_per_transaction);
        return result;
    }

    CoreCoord receiver_phys = mesh_device->worker_core_from_logical_core(cfg.receiver);

    constexpr size_t element_size_bytes = sizeof(bfloat16);
    const uint32_t num_elements = cfg.bytes_per_transaction / element_size_bytes;
    std::vector<uint32_t> golden = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        -100.0f, 100.0f, num_elements, static_cast<unsigned>(cfg.test_id + cfg.receiver.x * 100 + cfg.receiver.y));

    tt_metal::detail::WriteToDeviceL1(device, cfg.sender, data_addr, golden);
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    run_sender_kernel(
        mesh_device,
        cfg,
        data_addr,
        cycles_addr,
        static_cast<uint32_t>(receiver_phys.x),
        static_cast<uint32_t>(receiver_phys.y));

    std::vector<uint32_t> output;
    tt_metal::detail::ReadFromDeviceL1(device, cfg.receiver, data_addr, cfg.bytes_per_transaction, output);
    result.pass = (output == golden);

    std::vector<uint32_t> cycles_words(2, 0);
    tt_metal::detail::ReadFromDeviceL1(device, cfg.sender, cycles_addr, sizeof(uint64_t), cycles_words);
    result.cycles = static_cast<uint64_t>(cycles_words[0]) | (static_cast<uint64_t>(cycles_words[1]) << 32);

    if (result.cycles > 0) {
        const uint64_t total_bytes = static_cast<uint64_t>(cfg.num_of_transactions) * cfg.bytes_per_transaction;
        result.bandwidth_bpc = static_cast<double>(total_bytes) / static_cast<double>(result.cycles);
    }

    if (!result.pass) {
        log_error(
            tt::LogTest,
            "Correctness check failed: sender ({},{}) -> receiver ({},{}) size={}",
            cfg.sender.x,
            cfg.sender.y,
            cfg.receiver.x,
            cfg.receiver.y,
            cfg.bytes_per_transaction);
    }

    return result;
}

struct SweepRow {
    ShiftPosition position;
    int32_t dx;
    int32_t dy;
    CoreCoord receiver;
    uint32_t bytes_per_transaction;
    uint64_t cycles;
    double bandwidth_bpc;
    bool pass;
};

void write_csv(uint32_t test_id, CoreCoord sender, CoreCoord baseline, const std::vector<SweepRow>& rows) {
    const std::filesystem::path out_dir = "generated/quasar_core_shift";
    std::filesystem::create_directories(out_dir);
    const std::filesystem::path out_path = out_dir / ("test_" + std::to_string(test_id) + ".csv");

    std::ofstream csv(out_path);
    csv << "test_id,sender_x,sender_y,baseline_x,baseline_y,position,dx,dy,receiver_x,receiver_y,"
           "bytes_per_transaction,cycles,bandwidth_Bpc,pass\n";
    for (const auto& row : rows) {
        csv << test_id << ',' << sender.x << ',' << sender.y << ',' << baseline.x << ',' << baseline.y << ','
            << shift_position_name(row.position) << ',' << row.dx << ',' << row.dy << ',' << row.receiver.x << ','
            << row.receiver.y << ',' << row.bytes_per_transaction << ',' << row.cycles << ',' << row.bandwidth_bpc
            << ',' << (row.pass ? 1 : 0) << '\n';
    }
    log_info(tt::LogTest, "Wrote results to {}", out_path.string());
}

bool run_l1_smoke_one_to_one(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    CoreCoord sender,
    CoreCoord receiver,
    uint32_t num_bytes = 64) {
    CoreShiftConfig cfg{
        .test_id = 0,
        .sender = sender,
        .receiver = receiver,
        .num_of_transactions = 1,
        .bytes_per_transaction = num_bytes,
    };
    log_info(
        tt::LogTest,
        "L1 smoke: sender=({},{}) receiver=({},{}) bytes={}",
        sender.x,
        sender.y,
        receiver.x,
        receiver.y,
        num_bytes);
    return run_one(mesh_device, cfg).pass;
}

void run_shift_sweep(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t test_id,
    CoreCoord sender,
    CoreCoord receiver_baseline,
    uint32_t num_of_transactions = 256) {
    IDevice* device = mesh_device->get_devices()[0];
    const auto grid = device->compute_with_storage_grid_size();
    log_info(
        tt::LogTest,
        "Quasar core shift sweep test_id={} sender=({},{}) baseline=({},{}) grid={}x{}",
        test_id,
        sender.x,
        sender.y,
        receiver_baseline.x,
        receiver_baseline.y,
        grid.x,
        grid.y);

    ASSERT_LT(receiver_baseline.x, grid.x) << "Baseline receiver x out of grid";
    ASSERT_LT(receiver_baseline.y, grid.y) << "Baseline receiver y out of grid";

    const auto receivers = build_shift_receivers(receiver_baseline, grid.x, grid.y);

    // Keep the launch count low per process: the simulator/emulator accumulates
    // state across sequential program launches and becomes unstable after ~50.
    // A few representative sizes (small/medium/large) is enough to compare the
    // five shift scenarios. Add sizes here if running on stable hardware.
    const std::vector<uint32_t> transaction_sizes = {64, 1024, 8192};

    std::vector<SweepRow> all_rows;
    bool all_pass = true;

    log_info(
        tt::LogTest,
        "{:<10} {:>4} {:>4} {:>8} {:>12} {:>14} {:>6}",
        "position",
        "dx",
        "dy",
        "size_B",
        "cycles",
        "bandwidth_Bpc",
        "pass");

    for (const auto& rcv_case : receivers) {
        for (uint32_t tx_size : transaction_sizes) {
            log_info(
                tt::LogTest,
                "Starting run: position={} receiver=({},{}) size={}B",
                shift_position_name(rcv_case.position),
                rcv_case.receiver.x,
                rcv_case.receiver.y,
                tx_size);

            CoreShiftConfig cfg{
                .test_id = test_id,
                .sender = sender,
                .receiver = rcv_case.receiver,
                .num_of_transactions = num_of_transactions,
                .bytes_per_transaction = tx_size,
            };

            RunResult run = run_one(mesh_device, cfg);
            all_pass = all_pass && run.pass;

            SweepRow row{
                .position = rcv_case.position,
                .dx = rcv_case.dx,
                .dy = rcv_case.dy,
                .receiver = rcv_case.receiver,
                .bytes_per_transaction = tx_size,
                .cycles = run.cycles,
                .bandwidth_bpc = run.bandwidth_bpc,
                .pass = run.pass,
            };
            all_rows.push_back(row);

            log_info(
                tt::LogTest,
                "{:<10} {:>4} {:>4} {:>8} {:>12} {:>14.4f} {:>6}",
                shift_position_name(rcv_case.position),
                rcv_case.dx,
                rcv_case.dy,
                tx_size,
                run.cycles,
                run.bandwidth_bpc,
                run.pass ? "OK" : "FAIL");
        }
    }

    write_csv(test_id, sender, receiver_baseline, all_rows);
    EXPECT_TRUE(all_pass);
}

}  // namespace unit_tests::dm::quasar_core_shift

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarCoreShiftSmokeL1OneToOne) {
    if (unit_tests::dm::quasar_core_shift::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator or emulator";
    }
    EXPECT_TRUE(
        unit_tests::dm::quasar_core_shift::run_l1_smoke_one_to_one(devices_[0], CoreCoord{0, 0}, CoreCoord{1, 0}));
}

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarCoreShiftCorner0) {
    if (unit_tests::dm::quasar_core_shift::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator or emulator";
    }
    // Sender (0,0), baseline receiver (2,6) in user (row,col) notation -> CoreCoord{6,2}
    unit_tests::dm::quasar_core_shift::run_shift_sweep(
        devices_[0],
        /*test_id=*/920,
        CoreCoord{0, 0},
        CoreCoord{6, 2});
}

TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarCoreShiftCorner1) {
    if (unit_tests::dm::quasar_core_shift::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator or emulator";
    }
    // Sender (3,6) -> CoreCoord{6,3}, baseline receiver (1,1) -> CoreCoord{1,1}
    unit_tests::dm::quasar_core_shift::run_shift_sweep(
        devices_[0],
        /*test_id=*/921,
        CoreCoord{6, 3},
        CoreCoord{1, 1});
}

// Near-baseline variant of Corner0: same sender (0,0) but the baseline receiver
// sits only ~2 hops away at (1,1) instead of the far (6,2). Use this to compare
// against Corner0 and check whether NOC distance affects timing on this target.
TEST_F(QuasarMeshDeviceSingleCardFixture, QuasarCoreShiftNearBaseline) {
    if (unit_tests::dm::quasar_core_shift::should_skip_test()) {
        GTEST_SKIP() << "Test requires Quasar simulator or emulator";
    }
    // Sender (0,0), baseline receiver (1,1); L/R/U/D = (0,1)/(2,1)/(1,2)/(1,0)
    unit_tests::dm::quasar_core_shift::run_shift_sweep(
        devices_[0],
        /*test_id=*/922,
        CoreCoord{0, 0},
        CoreCoord{1, 1});
}

}  // namespace tt::tt_metal
