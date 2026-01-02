#pragma once

#include "common.hpp"

void execute_default_intra_mesh_routing_bench(
    const tt::tt_metal::FabricTestDescriptor& fabric_desc,
    const tt::tt_metal::GlobalSemaphore& cur_global_sema,
    uint32_t buffer_size,
    uint32_t page_size,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device,
    const tt::tt_fabric::FabricNodeId& src_fabric_node,
    const tt::tt_fabric::FabricNodeId& dst_fabric_node,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> src_buf,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dst_buf,
    const std::vector<uint32_t>& tx_send_data);

void execute_incremental_packet_size_bench(
    const tt::tt_metal::FabricTestDescriptor& fabric_desc,
    const tt::tt_metal::GlobalSemaphore& cur_global_sema,
    uint32_t page_size,
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device,
    const tt::tt_fabric::FabricNodeId& src_fabric_node,
    const tt::tt_fabric::FabricNodeId& dst_fabric_node,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> src_buf,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> dst_buf,
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> device_perf_buf);
