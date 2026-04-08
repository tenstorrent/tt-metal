// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <memory>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

constexpr uint32_t WORKER_KEEP_RUNNING = 0;
constexpr uint32_t WORKER_TEARDOWN = 1;

namespace tt::tt_fabric::test_utils {

// Helper to allocate L1 addresses for worker kernel
struct WorkerMemoryLayout {
    uint32_t source_buffer_address;
    uint32_t teardown_signal_address;
    uint32_t packet_payload_size_bytes;
};

// Allocate L1 memory for worker kernel buffers
WorkerMemoryLayout allocate_worker_memory();

// Create program with traffic generator kernel
// FR-2: Launch worker kernel
std::shared_ptr<tt_metal::Program> create_traffic_generator_program(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    const FabricNodeId& dest_fabric_node,
    const WorkerMemoryLayout& mem_layout);

// FR-8: Signal teardown to worker kernel
void signal_worker_teardown(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    uint32_t teardown_signal_address);



} // namespace tt::tt_fabric::test_utils
