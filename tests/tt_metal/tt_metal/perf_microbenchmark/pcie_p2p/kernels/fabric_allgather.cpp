// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// TT-Fabric (Ethernet) all-gather step, the baseline for p2p_allgather.cpp. Same protocol:
// each iteration k every chip pushes its payload into slot[me] of every peer, then bumps the
// peer's flag[me] with a fabric atomic increment, then waits for flag[p] == k from every peer.
// 2D fabric: routes are set from destination device/mesh ids, one worker connection per
// distinct outgoing router.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

constexpr uint32_t MAX_PEERS = 7;
constexpr uint32_t MAX_CONNS = 4;

void kernel_main() {
    set_l1_data_cache<false>();
    using namespace tt::tt_fabric;
    size_t i = 0;
    const uint32_t me = get_arg_val<uint32_t>(i++);
    const uint32_t n = get_arg_val<uint32_t>(i++);
    const uint32_t FLAGS = get_arg_val<uint32_t>(i++);
    const uint32_t RESULTS = get_arg_val<uint32_t>(i++);
    const uint32_t SRC = get_arg_val<uint32_t>(i++);
    const uint32_t RECV = get_arg_val<uint32_t>(i++);
    const uint32_t stride = get_arg_val<uint32_t>(i++);
    const uint32_t size = get_arg_val<uint32_t>(i++);
    const uint32_t iters = get_arg_val<uint32_t>(i++);
    const uint32_t warmup = get_arg_val<uint32_t>(i++);
    const uint32_t timeout_lo = get_arg_val<uint32_t>(i++);
    const uint32_t timeout_hi = get_arg_val<uint32_t>(i++);
    const uint32_t max_payload = get_arg_val<uint32_t>(i++);
    const uint32_t num_conns = get_arg_val<uint32_t>(i++);
    // per peer: p, conn_idx, dst_dev_id, dst_mesh_id, dst_noc_x, dst_noc_y
    uint32_t peer_id[MAX_PEERS], peer_conn[MAX_PEERS], peer_dev[MAX_PEERS], peer_mesh[MAX_PEERS], peer_x[MAX_PEERS],
        peer_y[MAX_PEERS];
    const uint32_t num_peers = n - 1;
    for (uint32_t j = 0; j < num_peers; ++j) {
        peer_id[j] = get_arg_val<uint32_t>(i++);
        peer_conn[j] = get_arg_val<uint32_t>(i++);
        peer_dev[j] = get_arg_val<uint32_t>(i++);
        peer_mesh[j] = get_arg_val<uint32_t>(i++);
        peer_x[j] = get_arg_val<uint32_t>(i++);
        peer_y[j] = get_arg_val<uint32_t>(i++);
    }
    // connections, in conn_idx order
    auto c0 = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(i);
    auto c1 = (num_conns > 1) ? WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(i) : c0;
    auto c2 = (num_conns > 2) ? WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(i) : c0;
    auto c3 = (num_conns > 3) ? WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(i) : c0;
    WorkerToFabricEdmSender* conns[MAX_CONNS] = {&c0, &c1, &c2, &c3};
    for (uint32_t c = 0; c < num_conns; ++c) {
        conns[c]->open();
    }

    const uint64_t timeout = (static_cast<uint64_t>(timeout_hi) << 32) | timeout_lo;
    volatile tt_l1_ptr uint32_t* flags = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(FLAGS);
    volatile tt_l1_ptr uint32_t* src = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(SRC);
    volatile tt_l1_ptr uint32_t* recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RECV);
    volatile tt_l1_ptr uint32_t* res = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(RESULTS);
    const uint32_t last_w = (size / 4) - 1;
    const bool chunked = size > max_payload;

    // Headers: per peer, two payload headers (one per receive-slot parity) and one semaphore header.
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_pay[MAX_PEERS][2];
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr_sem[MAX_PEERS];
    for (uint32_t j = 0; j < num_peers; ++j) {
        for (uint32_t par = 0; par < 2; ++par) {
            hdr_pay[j][par] = PacketHeaderPool::allocate_header();
            fabric_set_unicast_route((HybridMeshPacketHeader*)hdr_pay[j][par], peer_dev[j], peer_mesh[j]);
            const uint64_t dst = safe_get_noc_addr(peer_x[j], peer_y[j], RECV + (me * 2 + par) * stride, 0);
            hdr_pay[j][par]->to_noc_unicast_write(NocUnicastCommandHeader{dst}, chunked ? max_payload : size);
        }
        hdr_sem[j] = PacketHeaderPool::allocate_header();
        fabric_set_unicast_route((HybridMeshPacketHeader*)hdr_sem[j], peer_dev[j], peer_mesh[j]);
        const uint64_t sem = safe_get_noc_addr(peer_x[j], peer_y[j], FLAGS + me * 64, 0);
        hdr_sem[j]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{sem, 1, true});
    }

    res[0] = 0;
    res[1] = 0;
    res[2] = 0;
    uint32_t violations = 0;
    uint64_t t_period_start = 0;
    for (uint32_t k = 1; k <= iters; ++k) {
        const uint64_t t0 = get_timestamp();
        if (k == warmup + 1) {
            t_period_start = t0;
        }
        const uint32_t par = k & 1;
        src[0] = k;
        src[last_w] = k;
        for (uint32_t j = 0; j < num_peers; ++j) {
            auto& conn = *conns[peer_conn[j]];
            if (!chunked) {
                conn.wait_for_empty_write_slot();
                conn.send_payload_without_header_non_blocking_from_address(SRC, size);
                conn.send_payload_flush_non_blocking_from_address(
                    (uint32_t)hdr_pay[j][par], sizeof(PACKET_HEADER_TYPE));
            } else {
                for (uint32_t off = 0; off < size; off += max_payload) {
                    const uint32_t chunk = (size - off < max_payload) ? (size - off) : max_payload;
                    const uint64_t dst =
                        safe_get_noc_addr(peer_x[j], peer_y[j], RECV + (me * 2 + par) * stride + off, 0);
                    hdr_pay[j][par]->to_noc_unicast_write(NocUnicastCommandHeader{dst}, chunk);
                    conn.wait_for_empty_write_slot();
                    conn.send_payload_without_header_non_blocking_from_address(SRC + off, chunk);
                    conn.send_payload_flush_blocking_from_address(
                        (uint32_t)hdr_pay[j][par], sizeof(PACKET_HEADER_TYPE));
                }
            }
        }
        for (uint32_t j = 0; j < num_peers; ++j) {
            auto& conn = *conns[peer_conn[j]];
            conn.wait_for_empty_write_slot();
            conn.send_payload_flush_non_blocking_from_address((uint32_t)hdr_sem[j], sizeof(PACKET_HEADER_TYPE));
        }
        noc_async_writes_flushed();
        for (uint32_t j = 0; j < num_peers; ++j) {
            const uint32_t p = peer_id[j];
            while (flags[p * 16] != k) {
                if (get_timestamp() - t0 > timeout) {
                    res[0] = 2;
                    res[1] = k - 1;
                    res[2] = violations;
                    res[3] = p;
                    for (uint32_t c = 0; c < num_conns; ++c) {
                        conns[c]->close();
                    }
                    noc_async_full_barrier();
                    return;
                }
            }
        }
        const uint64_t t1 = get_timestamp();
        for (uint32_t j = 0; j < num_peers; ++j) {
            const uint32_t base_w = ((peer_id[j] * 2 + par) * stride) / 4;
            violations += (recv[base_w] != k) + (recv[base_w + last_w] != k);
        }
        res[8 + (k - 1)] = static_cast<uint32_t>(t1 - t0);
    }
    const uint64_t t_end = get_timestamp();
    for (uint32_t c = 0; c < num_conns; ++c) {
        conns[c]->close();
    }
    noc_async_full_barrier();
    res[4] = static_cast<uint32_t>(t_end - t_period_start);
    res[5] = static_cast<uint32_t>((t_end - t_period_start) >> 32);
    res[0] = 1;
    res[1] = iters;
    res[2] = violations;
}
