// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef _ETH_COMMON_HPP
#define _ETH_COMMON_HPP

#include "tt_metal/tt_metal/deployment/deployment_common.hpp"

#include "tt_metal/test_utils/stimulus.hpp"
#include "command_queue_fixture.hpp"
#include "tt_metal/tt_metal/eth/eth_test_common.hpp"

namespace tt::tt_metal {

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

static inline void prepare_receiver(
    tt::tt_metal::IDevice* const recv_device,
    const CoreCoord& recv_core,
    struct l1_allocator* recv_allocator,
    uint32_t transfer_size,
    uint32_t transfer_count,
    std::vector<uint32_t>& inputs,
    DataMovementProcessor processor,
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

template <typename FIXTURE>
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

static bool eth_data_check(
    tt::tt_metal::IDevice* const recv_device,
    const CoreCoord& recv_core,
    uint32_t recv_l1_address,
    std::vector<uint32_t>& inputs) {
    /* ==================== */
    auto readback_vec = tt::tt_metal::MetalContext::instance().get_cluster().read_core(
        recv_device->id(),
        recv_device->ethernet_core_from_logical_core(recv_core),
        recv_l1_address,
        inputs.size() * sizeof(uint32_t));

    bool pass = readback_vec == inputs;
    if (!pass) {
        for (int i = 0; i < inputs.size(); i++) {
            if (inputs[i] != readback_vec[i]) {
                log_critical(tt::LogTest, "      Mismatch at index: {}", i);
            }
        }
        log_critical(tt::LogTest, "      Mismatch at Core: {}", recv_core);
    }

    return pass;
}

[[maybe_unused]]
static bool eth_bandwidth_check(
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

}  // namespace tt::tt_metal

#endif /* _ETH_COMMON_HPP */
