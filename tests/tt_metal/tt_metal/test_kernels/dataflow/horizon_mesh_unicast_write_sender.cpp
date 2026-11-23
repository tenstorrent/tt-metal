// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Unified Mesh-API unicast/scatter write sender.
// Dispatches at runtime on noc_packet_type (0=unicast, 4=scatter) and with_state (0/1).

#include "api/dataflow/dataflow_api.h"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/mesh/api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"

static constexpr uint32_t HORIZON_NOC_UNICAST_SCATTER_WRITE = 4u;

void kernel_main() {
    const uint32_t src_l1_addr = get_arg(args::src_l1_addr);
    const uint32_t dst_noc_x = get_arg(args::dst_noc_x);
    const uint32_t dst_noc_y = get_arg(args::dst_noc_y);
    const uint32_t dst_dev_id = get_arg(args::dst_dev_id);
    const uint32_t dst_l1_addr_a = get_arg(args::dst_l1_addr_a);
    const uint32_t dst_l1_addr_b = get_arg(args::dst_l1_addr_b);
    const uint32_t chunk_size_a = get_arg(args::chunk_size_a);
    const uint32_t total_size = get_arg(args::total_size);
    const uint32_t noc_pkt_type = get_arg(args::noc_packet_type);
    const uint32_t with_state = get_arg(args::with_state);

    const uint64_t noc_addr_a =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_a);
    const uint64_t noc_addr_b =
        get_noc_addr(static_cast<uint8_t>(dst_noc_x), static_cast<uint8_t>(dst_noc_y), dst_l1_addr_b);

    using namespace tt::tt_fabric;
    using namespace tt::tt_fabric::mesh::experimental;

    alignas(16) uint8_t hdr_storage[sizeof(PACKET_HEADER_TYPE)]{};
    volatile PACKET_HEADER_TYPE* ph = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(hdr_storage);

    if (noc_pkt_type == HORIZON_NOC_UNICAST_SCATTER_WRITE) {
        NocUnicastScatterCommandHeader scatter_hdr({noc_addr_a, noc_addr_b}, {static_cast<uint16_t>(chunk_size_a)});
        if (with_state) {
            fabric_unicast_noc_scatter_write_set_state<UnicastScatterWriteUpdateMask::All>(
                ph, static_cast<uint8_t>(dst_dev_id), 0u, scatter_hdr, static_cast<uint16_t>(total_size));
            fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::None>(
                (tt_l1_ptr WorkerToFabricEdmSender*)nullptr, ph, src_l1_addr, scatter_hdr, 0u);
        } else {
            fabric_unicast_noc_scatter_write<WorkerToFabricEdmSender, false>(
                (tt_l1_ptr WorkerToFabricEdmSender*)nullptr,
                (volatile PACKET_HEADER_TYPE*)nullptr,
                static_cast<uint8_t>(dst_dev_id),
                0u,
                src_l1_addr,
                total_size,
                scatter_hdr);
        }
    } else {
        if (with_state) {
            fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::All>(
                ph,
                static_cast<uint8_t>(dst_dev_id),
                0u,
                NocUnicastCommandHeader{noc_addr_a},
                static_cast<uint16_t>(total_size));
            fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::None>(
                (tt_l1_ptr WorkerToFabricEdmSender*)nullptr, ph, src_l1_addr, NocUnicastCommandHeader{}, 0u);
        } else {
            fabric_unicast_noc_unicast_write<WorkerToFabricEdmSender, false>(
                (tt_l1_ptr WorkerToFabricEdmSender*)nullptr,
                (volatile PACKET_HEADER_TYPE*)nullptr,
                static_cast<uint8_t>(dst_dev_id),
                0u,
                src_l1_addr,
                total_size,
                NocUnicastCommandHeader{noc_addr_a});
        }
    }

    noc_async_write_barrier();
}
