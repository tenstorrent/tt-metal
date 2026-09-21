// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// One end of a worker-to-worker ping-pong across a fabric link. Each round the initiator stamps PP_TX and sends a
// one-packet atomic increment of the peer's flag through the fabric; the peer spins on the flag, stamps PP_RX, stamps
// PP_TX and sends back; the initiator spins and stamps PP_RX. The initiator alternates by round, so both directions
// are measured on the same link. A spin gives up after kSpinLimit polls, leaving the round it was waiting for at
// flag_addr + 4 for the host.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t kSpinLimit = 1u << 26;

void kernel_main() {
    set_l1_data_cache<false>();
    using namespace tt::tt_fabric;
    size_t i = 0;
    const uint32_t role = get_arg_val<uint32_t>(i++);
    const uint32_t peer_x = get_arg_val<uint32_t>(i++);
    const uint32_t peer_y = get_arg_val<uint32_t>(i++);
    const uint32_t flag_addr = get_arg_val<uint32_t>(i++);
    const uint32_t rounds = get_arg_val<uint32_t>(i++);
    const uint32_t dst_chip = get_arg_val<uint32_t>(i++);
    const uint32_t dst_mesh = get_arg_val<uint32_t>(i++);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(i);
    conn.open();

    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(flag_addr);
    auto* hdr = PacketHeaderPool::allocate_header();
    if (!fabric_set_unicast_route(
            (HybridMeshPacketHeader*)hdr, static_cast<uint16_t>(dst_chip), static_cast<uint16_t>(dst_mesh))) {
        flag[1] = 0xDEAD0000u | dst_chip;  // no route: the host reads it as a round it gave up on
        conn.close();
        return;
    }
    hdr->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{safe_get_noc_addr(peer_x, peer_y, flag_addr, 0), 1});

    auto wait = [&](uint32_t r) {
        for (uint32_t polls = 0;; polls++) {
            invalidate_l1_cache();
            if (*flag >= r) {
                return true;
            }
            if (polls == kSpinLimit) {
                flag[1] = r;
                return false;
            }
        }
    };
    auto send = [&] { conn.send_payload_flush_non_blocking_from_address((uint32_t)hdr, sizeof(PACKET_HEADER_TYPE)); };
    for (uint32_t r = 1; r <= rounds; r++) {
        if ((r & 1u) == role) {
            conn.wait_for_empty_write_slot();
            {
                DeviceZoneScopedN("PP_TX");
                send();
            }
            if (!wait(r)) {
                break;
            }
            {
                DeviceZoneScopedN("PP_RX");
            }
        } else {
            if (!wait(r)) {
                break;
            }
            {
                DeviceZoneScopedN("PP_RX");
            }
            conn.wait_for_empty_write_slot();
            {
                DeviceZoneScopedN("PP_TX");
                send();
            }
        }
    }
    conn.close();
    noc_async_full_barrier();
}
