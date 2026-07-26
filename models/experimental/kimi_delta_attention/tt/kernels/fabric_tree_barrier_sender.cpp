// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// A single routed fabric atomic increment.  Together with the matching local
// waiter this is the building block for the SP8 prefix stage barrier.  Unlike
// a socket send it carries no tensor payload and does not allocate a fabric
// FIFO: the destination's fixed-address global semaphore is the message.

#ifdef API_TYPE_Mesh
#include "tt_metal/fabric/hw/inc/mesh/api.h"
using namespace tt::tt_fabric::mesh::experimental;
#else
#error "The KDA fabric tree barrier requires the Mesh fabric API"
#endif

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/routing_plane_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

void kernel_main() {
    // This is a GlobalSemaphore address.  Every member of the mesh owns the
    // same logical worker coordinate, so get_noc_addr addresses that worker
    // on the route destination.
    const uint32_t destination_semaphore_addr = get_arg_val<uint32_t>(0);
    size_t connection_arg_idx = 1;

    auto route_id = PacketHeaderPool::allocate_header_n(1);
    tt::tt_fabric::RoutingPlaneConnectionManager connections;
    open_connections(connections, 1, connection_arg_idx);

    fabric_unicast_noc_unicast_atomic_inc(
        connections,
        route_id,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{get_noc_addr(destination_semaphore_addr), 1, true});
    noc_async_writes_flushed();
    close_connections(connections);
    noc_async_write_barrier();
}
