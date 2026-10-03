// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// The fabric ping-pong kernel: each round, one end sends an atomic increment over the fabric (PP_TX) and waits for the
// reply, and the other waits (PP_RX on arrival) and replies; the ends swap roles every round.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "workload_layout.hpp"

void kernel_main() {
    using namespace tt::tt_fabric;
    size_t arg_index = 0;
    const uint32_t role = get_arg_val<uint32_t>(arg_index++);
    const uint32_t peer_x = get_arg_val<uint32_t>(arg_index++);
    const uint32_t peer_y = get_arg_val<uint32_t>(arg_index++);
    const uint32_t flag_addr = get_arg_val<uint32_t>(arg_index++);
    const uint32_t rounds = get_arg_val<uint32_t>(arg_index++);
    const uint32_t dst_chip = get_arg_val<uint32_t>(arg_index++);
    const uint32_t dst_mesh = get_arg_val<uint32_t>(arg_index++);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_index);
    conn.open();

    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    auto* hdr = PacketHeaderPool::allocate_header();
    fabric_set_unicast_route(hdr, static_cast<uint16_t>(dst_chip), static_cast<uint16_t>(dst_mesh));
    hdr->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(peer_x, peer_y, flag_addr, 0), 1});

    const auto send = [&] {
        conn.wait_for_empty_write_slot();
        DeviceZoneScopedN("PP_TX");
        conn.send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(hdr), sizeof(PACKET_HEADER_TYPE));
    };
    const auto arrived = [&](uint32_t round) {
        for (uint32_t polls = 0; flag[kRoundWord] < round; polls++) {
            invalidate_l1_cache();
            if (polls == kSpinLimit) {
                flag[kGaveUpRoundWord] = round;
                return false;
            }
        }
        return true;
    };
    for (uint32_t round = 1; round <= rounds; round++) {
        const bool sends_first = (round & 1u) == role;
        if (sends_first) {
            send();
        }
        if (!arrived(round)) {
            break;
        }
        {
            DeviceZoneScopedN("PP_RX");
        }
        if (!sends_first) {
            send();
        }
    }
    conn.close();
    noc_async_full_barrier();
}
