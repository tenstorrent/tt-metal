// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/fabric_types.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/fabric_edm_types.hpp>
#include <umd/device/types/cluster_descriptor_types.h>  // chip_id_t
#include <vector>
#include <umd/device/tt_core_coordinates.h>

namespace tt {
namespace tt_metal {
class Program;
}  // namespace tt_metal
}  // namespace tt

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshShape;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_fabric {
class FabricNodeId;
enum class RoutingDirection;
size_t get_tt_fabric_channel_buffer_size_bytes();
size_t get_tt_fabric_packet_header_size_bytes();
size_t get_tt_fabric_max_payload_size_bytes();

// Used to get the run-time args for estabilishing connection with the fabric router.
// The API appends the connection specific run-time args to the set of exisiting
// run-time args for the worker programs, which allows the workers to conveniently
// build connection management object(s) using the run-time args.
// It is advised to call the API once all the other run-time args for the prgram are
// determined/pushed to keep things clean and avoid any extra arg management.
//
// Inputs:
// src_chip_id: physical chip id/device id of the sender chip
// dst_chip_id: physical chip id/device id of the receiver chip
// link_idx: the link (0..n) to use b/w the src_chip_id and dst_chip_id. On WH for
//                instance we can have upto 4 active links b/w two chips
// worker_program: program handle
// worker_core: worker core logical coordinates
// worker_args: list of existing run-time args to which the connection args will be appended
// core_type: core type which the worker will be running on
//
// Constraints:
// 1. Currently the sender and reciever chip should be physically adjacent (for 1D)
// 2. Currently the sender and reciever chip should be on the same mesh (for 1D)
// 3. When connecting with 1D fabric routers, users are responsible for setting up the
// connection appropriately. The API will not perform any checks to ensure that the
// connection is indeed a 1D connection b/w all the workers.
void append_fabric_connection_rt_args(
    const FabricNodeId& src_fabric_node_id,
    const FabricNodeId& dst_fabric_node_id,
    uint32_t link_idx,
    tt::tt_metal::Program& worker_program,
    const CoreCoord& worker_core,
    std::vector<uint32_t>& worker_args,
    CoreType core_type = CoreType::WORKER);

// returns which links on a given src chip are available for forwarding the data to a dst chip
// these link indices can then be used to establish connection with the fabric routers
std::vector<uint32_t> get_forwarding_link_indices(
    const FabricNodeId& src_fabric_node_id, const FabricNodeId& dst_fabric_node_id);

FabricNodeId get_fabric_node_id_from_physical_chip_id(chip_id_t physical_chip_id);

std::vector<chan_id_t> get_active_fabric_eth_routing_planes_in_direction(
    FabricNodeId fabric_node_id, RoutingDirection routing_direction);

std::unordered_map<MeshId, tt::tt_metal::distributed::MeshShape> get_physical_mesh_shapes();

tt::tt_fabric::Topology get_fabric_topology();

/**
 * Call before CreateDevices to enable fabric, which uses the specified number of routing planes.
 * Currently, setting num_routing_planes dictates how many routing planes the fabric should be active on
 * for that init sequence. The number of routing planes fabric will be initialized on will be the max
 * of all the values specified by different clients. If a client wants to initialize fabric on all the
 * available routing planes, num_routing_planes can be left unspecifed.
 * NOTE: This does not 'reserve' routing planes for any clients, but is rather a global setting.
 *
 * Return value: void
 *
 * | Argument           | Description                         | Data type         | Valid range | Required |
 * |--------------------|-------------------------------------|-------------------|-------------|----------|
 * | fabric_config      | Fabric config to set                | FabricConfig      |             | Yes      |
 * | num_routing_planes | Number of routing planes for fabric | optional<uint8_t> |             | No       |
 */
void SetFabricConfig(
    FabricConfig fabric_config,
    FabricReliabilityMode reliability_mode = FabricReliabilityMode::STRICT_SYSTEM_HEALTH_SETUP_MODE,
    std::optional<uint8_t> num_routing_planes = std::nullopt);

FabricConfig GetFabricConfig();

namespace experimental {
size_t get_number_of_available_routing_planes(
    const tt::tt_metal::distributed::MeshDevice& mesh_device, size_t cluster_axis, size_t row_or_col);
}

}  // namespace tt::tt_fabric
