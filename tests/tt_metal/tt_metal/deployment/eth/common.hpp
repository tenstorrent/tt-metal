// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _ETH_COMMON_HPP
#define _ETH_COMMON_HPP

#include <chrono>

#include "tt_metal/tt_metal/deployment/deployment_common.hpp"
#include "tt_metal/api/tt-metalium/experimental/fabric/control_plane.hpp"

#include "tt_metal/test_utils/stimulus.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"

/* Performance debug helpers */
#define NOW() std::chrono::high_resolution_clock::now()
#define DELTA(s, start)                                                                       \
    do {                                                                                      \
        double delta_ms = std::chrono::duration<double, std::milli>(NOW() - (start)).count(); \
        log_info(tt::LogTest, "      {} done in {} ms", (s), delta_ms);                       \
        start = NOW();                                                                        \
    } while (0)

namespace tt::tt_metal {

struct LinkError {
    int send_device_id;
    int recv_device_id;
    const CoreCoord send_core;
    const CoreCoord recv_core;
    DataMovementProcessor processor;
};

[[maybe_unused]]
static inline void prepare_sender(
    tt::tt_metal::IDevice* const send_device,
    const CoreCoord& send_core,
    struct l1_allocator* send_allocator,
    uint32_t transfer_size,
    uint32_t transfer_count,
    uint32_t* send_delta_addr,
    std::vector<uint32_t>& inputs,
    DataMovementProcessor processor,
    uint32_t num_bytes_per_send,
    uint32_t progress_counter,
    uint32_t recv_l1_address,
    tt_metal::Program* send_program) {
    /* ==================== */
    *send_delta_addr = l1_alloc(send_allocator, sizeof(uint64_t));
    uint32_t send_l1_address = l1_alloc(send_allocator, transfer_size);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        send_device->id(), send_device->ethernet_core_from_logical_core(send_core), inputs, send_l1_address);

    auto send_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                num_bytes_per_send,
                transfer_size,
                transfer_count,
                *send_delta_addr,
                send_l1_address,
                recv_l1_address,
                progress_counter,
            },
    };
    eth_test_common::set_arch_specific_eth_config(send_eth_config);

    auto send_kernel = tt_metal::CreateKernel(
        *send_program,
        "tests/tt_metal/tt_metal/deployment/kernels/eth_simple_send_kernel.cpp",
        send_core,
        send_eth_config);

    tt_metal::SetRuntimeArgs(*send_program, send_kernel, send_core, {});
}

[[maybe_unused]]
static inline void prepare_receiver(
    tt::tt_metal::IDevice* const recv_device,
    const CoreCoord& recv_core,
    struct l1_allocator* recv_allocator,
    uint32_t transfer_size,
    uint32_t transfer_count,
    std::vector<uint32_t>& inputs,
    DataMovementProcessor processor,
    uint32_t progress_counter,
    uint32_t* recv_l1_address,
    tt_metal::Program* recv_program) {
    /* ==================== */
    std::vector<uint32_t> all_zeros(inputs.size(), 0);

    *recv_l1_address = l1_alloc(recv_allocator, transfer_size);
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        recv_device->id(), recv_device->ethernet_core_from_logical_core(recv_core), all_zeros, *recv_l1_address);

    auto recv_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                transfer_size,
                transfer_count,
                progress_counter,
            },
    };
    eth_test_common::set_arch_specific_eth_config(recv_eth_config);

    auto recv_kernel = tt_metal::CreateKernel(
        *recv_program,
        "tests/tt_metal/tt_metal/deployment/kernels/eth_simple_recv_kernel.cpp",
        recv_core,
        recv_eth_config);

    tt_metal::SetRuntimeArgs(*recv_program, recv_kernel, recv_core, {});
}

[[maybe_unused]]
static void prepare_bidir(
    tt::tt_metal::IDevice* const send_device,
    const CoreCoord& send_core,
    uint32_t transfer_size,
    uint32_t transfer_count,
    uint32_t send_delta_addr,
    std::span<uint32_t> inputs,
    DataMovementProcessor processor,
    uint32_t num_bytes_per_send,
    uint32_t iter_l1_address,
    uint32_t send_l1_address,
    uint32_t recv_l1_address,
    uint32_t channel0,
    uint32_t channel1,
    tt_metal::Program* send_program) {
    /* =================== */
    tt::tt_metal::MetalContext::instance().get_cluster().write_core(
        send_device->id(), send_device->ethernet_core_from_logical_core(send_core), inputs, send_l1_address);

    auto send_eth_config = tt_metal::EthernetConfig{
        .noc = tt_metal::NOC::NOC_0,
        .processor = processor,
        .compile_args =
            {
                channel0,
                channel1,
                iter_l1_address,
                num_bytes_per_send,
                transfer_size,
                transfer_count,
                send_delta_addr,
                send_l1_address,
                recv_l1_address,
            },
    };
    eth_test_common::set_arch_specific_eth_config(send_eth_config);

    auto send_kernel = tt_metal::CreateKernel(
        *send_program, "tests/tt_metal/tt_metal/deployment/kernels/eth_bidir_kernel.cpp", send_core, send_eth_config);

    tt_metal::SetRuntimeArgs(*send_program, send_kernel, send_core, {});
}

template <typename FIXTURE>
[[maybe_unused]]
static void wait_to_finish(
    FIXTURE* fixture,
    tt_metal::Program& send_program,
    tt_metal::Program& recv_program,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    distributed::MeshCoordinateRange& device_range) {
    /* ==================== */
    bool same_device = send_mesh_device == recv_mesh_device;

    distributed::MeshWorkload send_workload;
    distributed::MeshWorkload recv_workload_;
    distributed::MeshWorkload& recv_workload = same_device ? send_workload : recv_workload_;

    send_workload.add_program(device_range, std::move(send_program));
    if (!same_device) {
        recv_workload.add_program(device_range, std::move(recv_program));
    }

    fixture->RunProgram(send_mesh_device, send_workload, true);
    if (!same_device) {
        fixture->RunProgram(recv_mesh_device, recv_workload, true);
    }

    fixture->FinishCommands(send_mesh_device);
    if (!same_device) {
        fixture->FinishCommands(recv_mesh_device);
    }
}

[[maybe_unused]]
static void track_eth_progress_timeout(
    tt::tt_metal::IDevice* const send_device,
    tt::tt_metal::IDevice* const recv_device,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    uint32_t iter_l1_addr,
    uint32_t expected_count) {
    /* =================== */
    uint32_t prev_send = -1;
    uint32_t prev_recv = -1;

    for (;;) {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(10ms);

        uint32_t curr_send = read_eth_l1_u32(send_device, send_core, iter_l1_addr);
        uint32_t curr_recv = expected_count;
        if (recv_device) {
            curr_recv = read_eth_l1_u32(recv_device, recv_core, iter_l1_addr);
        }

        if ((curr_send == expected_count) && (curr_recv == expected_count)) {
            break;
        }

        // log_info(tt::LogTest, "Read {} {}, waiting until {}", curr_send, curr_recv, expected_count);

        if ((curr_send == prev_send) && (curr_send != expected_count)) {
            log_critical(tt::LogTest, "Timed out! You probably need to reset the device (tt-smi -r)");
            exit(1);
        }

        if ((curr_recv == prev_recv) && (curr_recv != expected_count)) {
            log_critical(tt::LogTest, "Timed out! You probably need to reset the device (tt-smi -r)");
            exit(1);
        }

        prev_send = curr_send;
        prev_recv = curr_recv;
    }
}

struct core_setup {
    std::shared_ptr<tt_metal::Program> program;
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    const CoreCoord core;
    std::string locinfo;
    uint32_t iter_l1_addr;
    uint32_t expected_count;
    uint32_t delta_time_addr;
    uint64_t total_transferred;
    double bw_threshold;
    uint32_t recv_l1_address;
    std::span<uint32_t> inp;
};

[[maybe_unused]]
static void track_eth_progress_timeout_cores(std::span<struct core_setup> cores) {
    std::vector<std::thread> threads;

    for (const auto& c : cores) {
        if (!c.iter_l1_addr) {
            continue;
        }
        threads.emplace_back([&] {
            auto* const device = c.mesh_device->get_devices()[0];
            track_eth_progress_timeout(device, nullptr, c.core, c.core, c.iter_l1_addr, c.expected_count);
        });
    }

    for (auto& t : threads) {
        t.join();
    }
}

template <typename FIXTURE>
[[maybe_unused]]
static void wait_to_finish_eth_timeout_cores(
    FIXTURE* fixture,
    std::span<struct core_setup> cores,
    std::map<std::shared_ptr<distributed::MeshDevice>, std::shared_ptr<tt_metal::Program>>& programs) {
    /* ==================== */
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    std::map<std::shared_ptr<distributed::MeshDevice>, std::shared_ptr<distributed::MeshWorkload>> devices;

    for (const auto& [dev, _] : programs) {
        devices[dev] = std::make_shared<distributed::MeshWorkload>();
    }

    for (const auto& [dev, workload] : devices) {
        detail::CompileProgram(dev->get_devices()[0], *programs[dev]);
        devices[dev]->add_program(device_range, std::move(*programs[dev]));
    }

    for (const auto& [dev, workload] : devices) {
        fixture->RunProgram(dev, *workload, true);
    }

    track_eth_progress_timeout_cores(cores);

    for (const auto& [dev, _] : devices) {
        fixture->FinishCommands(dev);
    }
}

template <typename FIXTURE>
[[maybe_unused]]
static void wait_to_finish_eth_timeout(
    FIXTURE* fixture,
    tt_metal::Program& send_program,
    tt_metal::Program& recv_program,
    const std::shared_ptr<distributed::MeshDevice>& send_mesh_device,
    const std::shared_ptr<distributed::MeshDevice>& recv_mesh_device,
    distributed::MeshCoordinateRange& device_range,
    const CoreCoord& send_core,
    const CoreCoord& recv_core,
    uint32_t iter_l1_addr,
    uint32_t expected_count) {
    /* ==================== */
    bool same_device = send_mesh_device == recv_mesh_device;

    distributed::MeshWorkload send_workload;
    distributed::MeshWorkload recv_workload_;
    distributed::MeshWorkload& recv_workload = same_device ? send_workload : recv_workload_;

    send_workload.add_program(device_range, std::move(send_program));
    if (!same_device) {
        recv_workload.add_program(device_range, std::move(recv_program));
    }

    fixture->RunProgram(send_mesh_device, send_workload, true);
    if (!same_device) {
        fixture->RunProgram(recv_mesh_device, recv_workload, true);
    }

    auto* const send_device = send_mesh_device->get_devices()[0];
    auto* const recv_device = recv_mesh_device->get_devices()[0];
    track_eth_progress_timeout(send_device, recv_device, send_core, recv_core, iter_l1_addr, expected_count);

    fixture->FinishCommands(send_mesh_device);
    if (!same_device) {
        fixture->FinishCommands(recv_mesh_device);
    }
}

[[maybe_unused]]
static bool data_check(
    tt::tt_metal::IDevice* const recv_device,
    const CoreCoord& recv_core,
    uint32_t recv_l1_address,
    std::span<uint32_t> inputs) {
    /* ==================== */
    auto readback_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        recv_device->id(),
        recv_device->ethernet_core_from_logical_core(recv_core),
        recv_l1_address,
        inputs.size() * sizeof(uint32_t));

    uint32_t first_error = -1;
    uint32_t last_error = 0;
    uint64_t total_errors = 0;

    for (int i = 0; i < inputs.size(); i++) {
        if (inputs[i] != readback_vec[i]) {
            uint32_t addr = i * 4;
            first_error = addr < first_error ? addr : first_error;
            last_error = addr > last_error ? addr : last_error;
            total_errors++;
        }
    }

    if (total_errors) {
        log_critical(
            tt::LogTest,
            "      [device: {}, core: {}] {} mismatched words"
            " starting at {:08x}, ending at {:08x}, out of {:08x}",
            recv_device->id(),
            recv_core,
            total_errors,
            first_error,
            last_error,
            inputs.size());
    }

    return !total_errors;
}

[[maybe_unused]]
static bool bandwidth_check(
    tt::tt_metal::IDevice* const send_device,
    const CoreCoord& send_core,
    uint32_t send_delta_addr,
    uint64_t total_transferred,
    double threshold) {
    /* ==================== */
    uint64_t delta = read_eth_l1_u64(send_device, send_core, send_delta_addr);
    double deltas = delta / 1.35e9; /* Assuming fixed max frequency */
    double bandwidth = 8 * total_transferred / 1e9 / deltas;
    log_info(tt::LogTest, "      Bandwidth {:.3f} Gbps, {:.3f} ms", bandwidth, deltas * 1000);

    bool pass = bandwidth >= threshold;
    if (!pass) {
        log_critical(tt::LogTest, "      Expected at least: {} Gbps, got {:.2f} Gbps", threshold, bandwidth);
    }

    return pass;
}

[[maybe_unused]]
static bool data_dram_check(
    tt::tt_metal::IDevice* const recv_device,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t dram_bank_id,
    std::span<uint32_t> inputs) {
    /* ==================== */
    uint64_t total_transferred = dram_end_addr - dram_start_addr;
    std::vector<uint32_t> outputs;

    detail::ReadFromDeviceDRAMChannel(recv_device, dram_bank_id, dram_start_addr, total_transferred, outputs);
    log_info(tt::LogTest, "      Read {} bytes from bank {}", outputs.size() * sizeof(uint32_t), dram_bank_id);
    TT_FATAL(inputs.size() == outputs.size(), "Input and output vector sizes must match");
    // inputs == std::span(outputs);
    // bool pass = !memcmp(&inputs[0], &outputs[0], inputs.size() * sizeof inputs[0]);

    uint64_t total_mismatches = 0;
    for (long i = 0; i < inputs.size(); i++) {
        if (inputs[i] != outputs[i]) {
            if (!total_mismatches) {
                log_critical(
                    tt::LogTest,
                    "      Input and output data don't match starting at: {:x}",
                    dram_start_addr + i * sizeof(uint32_t));
            }
            total_mismatches++;
            // log_critical(tt::LogTest, "      Input and output data don't match at {:08x}: {:08x} {:08x}", i,
            // inputs[i], outputs[i]);
        }
    }
    if (total_mismatches) {
        log_critical(tt::LogTest, "      Total mismatches: {} words", total_mismatches);
    }

    return !total_mismatches;
}

template <typename FIXTURE>
[[maybe_unused]]
static void tensix_zero_dram(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t dram_bank_id) {
    /* ==================== */
    TT_FATAL(dram_start_addr < dram_end_addr, "start addr must be less than end addr");
    tt_metal::Program zero_program = tt_metal::Program();

    auto* const device = mesh_device->get_devices()[0];
    CoreCoord core_grid = device->compute_with_storage_grid_size();
    uint32_t total_bytes = dram_end_addr - dram_start_addr;
    uint32_t core_count = core_grid.x * core_grid.y;
    uint64_t per_core_bytes = total_bytes / core_count;
    per_core_bytes = ((per_core_bytes + 15) >> 4) << 4;
    TT_FATAL((total_bytes % 16) == 0, "Total size must be divisible by 16");
    TT_FATAL((per_core_bytes % 16) == 0, "Per core size must be divisible by 16");

    uint32_t transfer_size = 160 * 1024;
    struct l1_allocator alloc = new_erisc_allocator();
    uint32_t buffer0 = l1_alloc(&alloc, transfer_size);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    DataMovementConfig config = {
        .compile_args =
            {
                buffer0,
                transfer_size,
            },
    };

    // log_info(tt::LogTest, "      start {:8x}", dram_start_addr);
    // log_info(tt::LogTest, "      end   {:8x}", dram_end_addr);
    uint32_t kernel_id = 0;
    for (uint32_t x = 0; x < core_grid.x; x++) {
        for (uint32_t y = 0; y < core_grid.y; y++) {
            CoreCoord core = CoreCoord(x, y);

            uint32_t id = kernel_id++;
            uint32_t start_addr = dram_start_addr + id * per_core_bytes;
            uint32_t end_addr =
                start_addr + per_core_bytes > dram_end_addr ? dram_end_addr : start_addr + per_core_bytes;

            start_addr = (start_addr >> 4) << 4;
            end_addr = ((end_addr + 15) >> 4) << 4;

            // log_info(tt::LogTest, "      kernel {}, {}", id, core);
            // log_info(tt::LogTest, "      start  {:8x}", start_addr);
            // log_info(tt::LogTest, "      end    {:8x}", end_addr);
            // log_info(tt::LogTest, "      size   {:8x}", end_addr - start_addr);
            auto kernel = tt_metal::CreateKernel(
                zero_program, "tests/tt_metal/tt_metal/deployment/kernels/zero_kernel.cpp", core, config);
            tt_metal::SetRuntimeArgs(
                zero_program,
                kernel,
                core,
                {
                    id,
                    dram_bank_id,
                    start_addr,
                    end_addr,
                });
        }
    }

    workload.add_program(device_range, std::move(zero_program));
    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    // log_info(tt::LogTest, "      done zeroing bank {}", dram_bank_id);
}

template <typename FIXTURE>
[[maybe_unused]]
static void tensix_counter_dram(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t dram_bank_id) {
    /* ==================== */
    TT_FATAL(dram_start_addr < dram_end_addr, "start addr must be less than end addr");
    tt_metal::Program zero_program = tt_metal::Program();

    auto* const device = mesh_device->get_devices()[0];
    CoreCoord core_grid = device->compute_with_storage_grid_size();
    uint32_t total_bytes = dram_end_addr - dram_start_addr;
    uint32_t core_count = core_grid.x * core_grid.y;
    uint64_t per_core_bytes = total_bytes / core_count;
    per_core_bytes = ((per_core_bytes + 15) >> 4) << 4;

    TT_FATAL((total_bytes % 16) == 0, "Total size must be divisible by 16");
    TT_FATAL((per_core_bytes % 16) == 0, "Per core size must be divisible by 16");

    uint32_t transfer_size = 160 * 1024;
    struct l1_allocator alloc = new_erisc_allocator();
    uint32_t buffer0 = l1_alloc(&alloc, transfer_size);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    DataMovementConfig config = {
        .compile_args =
            {
                buffer0,
                transfer_size,
            },
    };

    // log_info(tt::LogTest, "      start {:8x}", dram_start_addr);
    // log_info(tt::LogTest, "      end   {:8x}", dram_end_addr);
    uint32_t kernel_id = 0;
    for (uint32_t x = 0; x < core_grid.x; x++) {
        for (uint32_t y = 0; y < core_grid.y; y++) {
            CoreCoord core = CoreCoord(x, y);

            uint32_t id = kernel_id++;
            uint32_t start_addr = dram_start_addr + id * per_core_bytes / sizeof(uint32_t);
            uint32_t end_addr =
                start_addr + per_core_bytes > dram_end_addr ? dram_end_addr : start_addr + per_core_bytes;

            start_addr = (start_addr >> 4) << 4;
            end_addr = ((end_addr + 15) >> 4) << 4;

            // log_info(tt::LogTest, "      kernel {}, {}", id, core);
            // log_info(tt::LogTest, "      start  {:8x}", start_addr);
            // log_info(tt::LogTest, "      end    {:8x}", end_addr);
            // log_info(tt::LogTest, "      size   {:8x}", end_addr - start_addr);
            auto kernel = tt_metal::CreateKernel(
                zero_program, "tests/tt_metal/tt_metal/deployment/kernels/counter_kernel.cpp", core, config);
            tt_metal::SetRuntimeArgs(
                zero_program,
                kernel,
                core,
                {
                    id,
                    dram_bank_id,
                    start_addr,
                    end_addr,
                });
        }
    }

    workload.add_program(device_range, std::move(zero_program));
    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    // log_info(tt::LogTest, "      done zeroing bank {}", dram_bank_id);
}

template <typename FIXTURE>
[[maybe_unused]]
static bool tensix_compare_dram_banks(
    FIXTURE* fixture,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t dram_start_addr,
    uint32_t dram_end_addr,
    uint32_t dram_bank_id0,
    uint32_t dram_bank_id1) {
    /* ==================== */
    TT_FATAL(dram_start_addr < dram_end_addr, "start addr must be less than end addr");
    // log_critical(tt::LogTest, "      comparing {} and {}", dram_bank_id0, dram_bank_id1);

    tt_metal::Program cmp_program = tt_metal::Program();

    auto* const device = mesh_device->get_devices()[0];
    CoreCoord core_grid = device->compute_with_storage_grid_size();
    uint32_t total_bytes = dram_end_addr - dram_start_addr;
    uint32_t core_count = core_grid.x * core_grid.y;
    uint64_t per_core_bytes = total_bytes / core_count;
    per_core_bytes = ((per_core_bytes + 15) >> 4) << 4;
    TT_FATAL((total_bytes % 16) == 0, "Total size must be divisible by 16");
    TT_FATAL((per_core_bytes % 16) == 0, "Per core size must be divisible by 16");

    uint32_t transfer_size = 160 * 1024;
    struct l1_allocator alloc = new_tensix_allocator();
    uint32_t error_counter = l1_alloc(&alloc, sizeof(uint32_t));
    uint32_t first_error_addr = l1_alloc(&alloc, sizeof(uint32_t));
    uint32_t last_error_addr = l1_alloc(&alloc, sizeof(uint32_t));
    uint32_t buffer0 = l1_alloc(&alloc, transfer_size + 64);
    uint32_t buffer1 = l1_alloc(&alloc, transfer_size + 64);

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;

    // log_info(tt::LogTest, "      bank0 {}, bank1 {}", dram_bank_id0, dram_bank_id1);
    // log_info(tt::LogTest, "      start {:8x}", dram_start_addr);
    // log_info(tt::LogTest, "      end   {:8x}", dram_end_addr);
    DataMovementConfig config = {
        .compile_args =
            {
                buffer0,
                buffer1,
                transfer_size,
                error_counter,
                first_error_addr,
                last_error_addr,
            },
    };

    uint32_t kernel_id = 0;
    for (uint32_t x = 0; x < core_grid.x; x++) {
        for (uint32_t y = 0; y < core_grid.y; y++) {
            CoreCoord core = CoreCoord(x, y);

            uint32_t id = kernel_id++;
            uint32_t start_addr = dram_start_addr + id * per_core_bytes;
            uint32_t end_addr =
                start_addr + per_core_bytes > dram_end_addr ? dram_end_addr : start_addr + per_core_bytes;

            start_addr = (start_addr >> 4) << 4;
            end_addr = ((end_addr + 15) >> 4) << 4;

            // log_info(tt::LogTest, "      kernel {}, {}", id, core);
            // log_info(tt::LogTest, "      start  {:8x}", start_addr);
            // log_info(tt::LogTest, "      end    {:8x}", end_addr);
            // log_info(tt::LogTest, "      size   {:8x}", end_addr - start_addr);
            auto kernel = tt_metal::CreateKernel(
                cmp_program, "tests/tt_metal/tt_metal/deployment/kernels/cmp_kernel.cpp", core, config);
            tt_metal::SetRuntimeArgs(
                cmp_program,
                kernel,
                core,
                {
                    id,
                    dram_bank_id0,
                    dram_bank_id1,
                    start_addr,
                    end_addr,
                });
        }
    }

    workload.add_program(device_range, std::move(cmp_program));
    fixture->RunProgram(mesh_device, workload, true);
    fixture->FinishCommands(mesh_device);

    uint64_t total_errors = 0;
    uint32_t first_error = -1;
    uint32_t last_error = 0;

    for (uint32_t x = 0; x < core_grid.x; x++) {
        for (uint32_t y = 0; y < core_grid.y; y++) {
            CoreCoord core = CoreCoord(x, y);
            uint32_t errors = read_l1_u32(device, core, error_counter);
            total_errors += errors;

            if (errors) {
                uint32_t t = read_l1_u32(device, core, first_error_addr);
                if (t < first_error) {
                    first_error = t;
                }

                t = read_l1_u32(device, core, last_error_addr);
                if (t > last_error) {
                    last_error = t;
                }
            }
        }
    }

    if (total_errors) {
        log_critical(
            tt::LogTest,
            "      done comparing bank {} and {} with {} mismatched words"
            " starting at {:08x}, ending at {:08x}, out of {:08x}",
            dram_bank_id0,
            dram_bank_id1,
            total_errors,
            first_error,
            last_error,
            total_bytes);
    }
    return !total_errors;
}

[[maybe_unused]]
static bool test_check_cores(std::span<struct core_setup> cores) {
    bool pass = true;

    std::string prev = "";
    for (const auto& cs : cores) {
        if (prev != cs.locinfo) {
            log_info(tt::LogTest, "core_check: {}", cs.locinfo);
        }
        prev = cs.locinfo;
        auto* const dev = cs.mesh_device->get_devices()[0];
        pass &= bandwidth_check(dev, cs.core, cs.delta_time_addr, cs.total_transferred, cs.bw_threshold);
        pass &= data_check(dev, cs.core, cs.recv_l1_address, cs.inp);

        log_info(tt::LogTest, "    done");
    }

    return pass;
}

[[maybe_unused]]
static void print_summary(std::span<struct LinkError> errors) {
    if (!errors.size()) {
        return;
    }

    log_critical(tt::LogTest, "Failing links:");
    for (auto& e : errors) {
        log_critical(
            tt::LogTest,
            "\tSender device {}, receiver device {}, sender core {}, receiver core {}, processor {}",
            e.send_device_id,
            e.recv_device_id,
            e.send_core,
            e.recv_core,
            e.processor);
    }
    log_critical(tt::LogTest, "{} failing links in total", errors.size());
}

/* In Gbps */
[[maybe_unused]]
static double get_eth_bw() {
    // TODO More comprehensive hardware support
    switch (tt::tt_metal::GetClusterType()) {
        case tt::tt_metal::ClusterType::BLACKHOLE_GALAXY: return 200.0;
        default: return 400.0;
    }
}

[[maybe_unused]]
static void print_detected_devices() {
    log_info(tt::LogTest, "Detected devices:");
    for (auto& l : get_chip_physical_locations()) {
        log_info(tt::LogTest, "  {}", l);
    }
}

[[maybe_unused]]
static std::string get_connector(IDevice* sdev, CoreCoord score) {
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(sdev->id());
    tt::umd::ClusterDescriptor& cluster_desc = (*(cluster.get_cluster_desc()));

    auto subb = tt::tt_fabric::get_ubb_id(cluster_desc, sdev->id());

    EthernetChannel chan_id = soc_desc.logical_eth_core_to_chan_map.at(score);

    switch (subb.asic_id) {
        case 1:
            if ((chan_id >= 0 && chan_id <= 3) || (chan_id >= 10 && chan_id <= 11)) {
                return "QSFP";
            }
            if (chan_id >= 4 && chan_id <= 9) {
                return "TRACE";
            }
            break;

        case 2:
        case 3:
            if ((chan_id >= 0 && chan_id <= 1) || (chan_id >= 10 && chan_id <= 11)) {
                return "QSFP";
            }
            if (chan_id >= 2 && chan_id <= 9) {
                return "TRACE";
            }
            break;

        case 4:
            if ((chan_id >= 0 && chan_id <= 1) || (chan_id >= 10 && chan_id <= 11)) {
                return "QSFP";
            }
            if ((chan_id >= 2 && chan_id <= 3) || (chan_id >= 7 && chan_id <= 9)) {
                return "TRACE";
            }
            if (chan_id >= 4 && chan_id <= 6) {
                return "ExaMAX";
            }
            break;

        case 5:
            if ((chan_id >= 2 && chan_id <= 3) || (chan_id >= 10 && chan_id <= 11)) {
                return "QSFP";
            }
            if ((chan_id >= 0 && chan_id <= 1) || (chan_id >= 4 && chan_id <= 6)) {
                return "TRACE";
            }
            if (chan_id >= 7 && chan_id <= 9) {
                return "ExaMAX";
            }
            break;

        case 6:
        case 7:
            if (chan_id >= 10 && chan_id <= 11) {
                return "QSFP";
            }
            if (chan_id >= 0 && chan_id <= 6) {
                return "TRACE";
            }
            if (chan_id >= 7 && chan_id <= 9) {
                return "ExaMAX";
            }
            break;

        case 8:
            if (chan_id >= 10 && chan_id <= 11) {
                return "QSFP";
            }
            if (chan_id >= 0 && chan_id <= 3) {
                return "TRACE";
            }
            if (chan_id >= 4 && chan_id <= 9) {
                return "ExaMAX";
            }
            break;
    }

    return "unknown";
}

// Returns true if the ethernet core connects to another chip within this cluster (i.e. it is safe to
// call get_connected_ethernet_core on it). Cross-host cores (e.g. QSFP cables wired to another
// Galaxy) show up as active/linked-up but live in the remote-device connection map, so calling
// get_connected_ethernet_core on them fatals with "connects to a remote mmio device".
[[maybe_unused]]
static bool eth_core_connects_within_cluster(IDevice* device, const CoreCoord& logical_core) {
    const auto& cluster = MetalContext::instance().get_cluster();
    const auto& soc_desc = cluster.get_soc_desc(device->id());
    EthernetChannel eth_chan = soc_desc.logical_eth_core_to_chan_map.at(logical_core);
    const auto& within_cluster = cluster.get_ethernet_connections();
    auto it = within_cluster.find(device->id());
    return it != within_cluster.end() && it->second.contains(eth_chan);
}

[[maybe_unused]]
static std::string get_ubb(IDevice* device) {
    const auto& cluster = MetalContext::instance().get_cluster();
    umd::ClusterDescriptor* cluster_desc = cluster.get_cluster_desc();

    auto ubb = tt::tt_fabric::get_ubb_id(*cluster_desc, device->id());
    return fmt::format("ubb: {}, chip: {}", ubb.tray_id, ubb.asic_id);
}

[[maybe_unused]]
static std::string get_locinfo(
    IDevice* sdev, CoreCoord score, IDevice* rdev, CoreCoord rcore, DataMovementProcessor proc) {
    return fmt::format(
        "sdev: [{} ({}), {}], rdev: [{} ({}), {}]"
        ", score: [{}], rcore: [{}]"
        ", processor: [{}], link: [{}]",
        sdev->id(),
        pci_bdf_for_device_id(sdev->id()),
        get_ubb(sdev),
        rdev->id(),
        pci_bdf_for_device_id(rdev->id()),
        get_ubb(rdev),
        score,
        rcore,
        proc,
        get_connector(sdev, score));
}

static bool ensure_links(std::span<std::shared_ptr<distributed::MeshDevice>> devices) {
    bool pass = true;

    TEST_PARAM(uint32_t, expected_links, 0, "ETH_TEST_EXPECTED_LINKS");

    if (!expected_links) {
        return pass;
    }

    for (const auto& device : devices) {
        auto* const dev = device->get_devices()[0];
        int numlinks = dev->get_active_ethernet_cores().size();
        if (numlinks != expected_links) {
            pass = false;

            log_critical(
                tt::LogTest,
                "missing links: chip[{} ({}), {}]: expected {} links, got {}",
                dev->id(),
                pci_bdf_for_device_id(dev->id()),
                get_ubb(dev),
                expected_links,
                numlinks);
        }
    }

    return pass;
}

}  // namespace tt::tt_metal

#endif /* _ETH_COMMON_HPP */
