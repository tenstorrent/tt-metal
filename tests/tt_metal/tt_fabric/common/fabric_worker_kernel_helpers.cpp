// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "tests/tt_metal/tt_fabric/common/fabric_worker_kernel_helpers.hpp"
#include "tests/tt_metal/tt_fabric/common/fabric_fixture.hpp"
#include "tt_metal/host_api.hpp"

namespace tt::tt_fabric::test_utils {

// STUB IMPLEMENTATION - Replaced in CS-006
WorkerMemoryLayout allocate_worker_memory(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device) {
    WorkerMemoryLayout layout;
    // TODO: CS-006 will implement proper L1 memory allocation
    layout.source_buffer_address = 0x10000;  // Placeholder
    layout.teardown_signal_address = 0x11000;  // Placeholder
    layout.packet_payload_size_bytes = 256;
    return layout;
}

// STUB IMPLEMENTATION - Replaced in CS-006
std::shared_ptr<tt_metal::Program> create_traffic_generator_program(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    const FabricNodeId& dest_fabric_node,
    const WorkerMemoryLayout& mem_layout) {
    // TODO: CS-006 will implement full program creation
    // For now, return empty program
    return std::make_shared<tt_metal::Program>();
}

// STUB IMPLEMENTATION - Replaced in CS-006
FabricNodeId get_fabric_node_id(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device) {
    // TODO: CS-006 will implement proper node ID extraction
    FabricNodeId node_id;
    node_id.mesh_id = 0;
    node_id.logical_x = 0;
    node_id.logical_y = 0;
    return node_id;
}

// STUB IMPLEMENTATION - Replaced in CS-006
void signal_worker_teardown(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    uint32_t teardown_signal_address) {
    // TODO: CS-006 will implement proper teardown signaling
    log_info(LogTest, "Worker teardown signal sent to core ({}, {}) at address 0x{:x}",
        logical_core.x, logical_core.y, teardown_signal_address);
}

// STUB IMPLEMENTATION - Replaced in CS-006
void wait_for_worker_complete(
    BaseFabricFixture* fixture,
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    tt_metal::Program& program,
    std::chrono::milliseconds timeout) {
    // TODO: CS-006 will implement proper wait with timeout
    // For now, just log and return
    log_info(LogTest, "Waiting for worker to complete with timeout {} ms", timeout.count());
}

} // namespace tt::tt_fabric::test_utils
