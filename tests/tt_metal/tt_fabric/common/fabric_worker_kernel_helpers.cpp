// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "fabric_worker_kernel_helpers.hpp"
#include "fabric_fixture.hpp"
#include <stdexcept>

namespace tt::tt_fabric::test_utils {

WorkerMemoryLayout allocate_worker_memory(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device) {

    WorkerMemoryLayout layout;

    // Implementation stub - to be completed
    throw std::runtime_error("allocate_worker_memory() not implemented");
}

std::shared_ptr<tt_metal::Program> create_traffic_generator_program(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    const FabricNodeId& dest_fabric_node,
    const WorkerMemoryLayout& mem_layout) {

    // Implementation stub - to be completed
    throw std::runtime_error("create_traffic_generator_program() not implemented");
}

void signal_worker_teardown(
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    const CoreCoord& logical_core,
    uint32_t teardown_signal_address) {

    // Implementation stub - to be completed
    throw std::runtime_error("signal_worker_teardown() not implemented");
}

void wait_for_worker_complete(
    fabric_router_tests::BaseFabricFixture* fixture,
    const std::shared_ptr<tt_metal::distributed::MeshDevice>& device,
    tt_metal::Program& program,
    std::chrono::milliseconds timeout) {

    // Implementation stub - to be completed
    throw std::runtime_error("wait_for_worker_complete() not implemented");
}

} // namespace tt::tt_fabric::test_utils
