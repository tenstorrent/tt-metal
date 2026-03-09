// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fabric reduce kernel for prefill MoE.
//
// Uses per-direction WorkerToFabricEdmSender connections (EAST/WEST) with
// explicit send_direction and NUM_HOPS parameters to support multi-hop routing
// on 1xN meshes (N >= 2).
//
// Compile-time defines:
//   EAST_CONNECTION: 1 if device has an EAST neighbor, 0 otherwise
//   WEST_CONNECTION: 1 if device has a WEST neighbor, 0 otherwise

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric::linear::experimental;

// Direction indices
constexpr uint32_t DIR_EAST = 0;
constexpr uint32_t DIR_WEST = 1;
constexpr uint32_t NUM_DIRS = 2;

// Direction validity from compile-time defines
constexpr bool HAS_EAST = (EAST_CONNECTION == 1);
constexpr bool HAS_WEST = (WEST_CONNECTION == 1);
constexpr std::array<bool, NUM_DIRS> directions = {HAS_EAST, HAS_WEST};

void kernel_main() {
    constexpr auto tensor_args = TensorAccessorArgs<0>();
    constexpr uint32_t SCALAR_BASE = TensorAccessorArgs<0>::NumArgsCT;
    constexpr uint32_t NUM_OUTPUT_TILES = get_compile_time_arg_val(SCALAR_BASE + 0);
    constexpr uint32_t NUM_HOPS = get_compile_time_arg_val(SCALAR_BASE + 1);

    uint32_t rt_idx = 0;
    const uint32_t output_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t recv_buf_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t remote_recv_buf_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t sem_combine_done_id = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t sem_fabric_recv_id = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t my_noc_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t my_noc_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t send_direction = get_arg_val<uint32_t>(rt_idx++);  // 0=EAST, 1=WEST

    constexpr uint32_t cb_send = 6;
    constexpr uint32_t cb_out = 7;
    constexpr uint32_t cb_recv = 8;

    const uint32_t page_bytes = get_local_cb_interface(cb_send).fifo_page_size;

    const auto out_accessor = TensorAccessor(tensor_args, output_addr, page_bytes);
    const auto recv_accessor = TensorAccessor(tensor_args, recv_buf_addr, page_bytes);
    const auto remote_recv_accessor = TensorAccessor(tensor_args, remote_recv_buf_addr, page_bytes);

    const uint32_t sem_combine_done_addr = get_semaphore(sem_combine_done_id);
    const uint32_t sem_fabric_recv_addr = get_semaphore(sem_fabric_recv_id);

    uint64_t remote_sem_noc_addr =
        safe_get_noc_addr(static_cast<uint8_t>(my_noc_x), static_cast<uint8_t>(my_noc_y), sem_fabric_recv_addr, 0);

    // Wait for local combine to finish
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_combine_done_addr);
    noc_semaphore_wait(sem_done, 1);

    // Build and open fabric connections (per-direction)
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, NUM_DIRS> fabric_connections;
    size_t fab_arg_idx = rt_idx;
    for (uint32_t i = 0; i < NUM_DIRS; i++) {
        if (directions[i]) {
            fabric_connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(fab_arg_idx);
            fabric_connections[i].open_start();
        }
    }
    for (uint32_t i = 0; i < NUM_DIRS; i++) {
        if (directions[i]) {
            fabric_connections[i].open_finish();
        }
    }

    auto* sender = &fabric_connections[send_direction];

    cb_reserve_back(cb_send, 1);
    const uint32_t send_l1 = get_write_ptr(cb_send);

    auto pkt_hdr_data = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem = PacketHeaderPool::allocate_header();

    // Phase 1: Send local output tiles to neighbor's recv_buf
    for (uint32_t tile = 0; tile < NUM_OUTPUT_TILES; ++tile) {
        noc_async_read_page(tile, out_accessor, send_l1);
        noc_async_read_barrier();

        uint64_t dest_noc_addr = tt::tt_fabric::linear::addrgen_detail::get_noc_address(remote_recv_accessor, tile);

        fabric_unicast_noc_unicast_write(
            sender,
            pkt_hdr_data,
            send_l1,
            page_bytes,
            tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr},
            static_cast<uint8_t>(NUM_HOPS));

        // Ensure NOC write to EDM completed before overwriting send_l1
        noc_async_write_barrier();
    }

    // Signal remote device
    fabric_unicast_noc_unicast_atomic_inc(
        sender,
        pkt_hdr_sem,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_sem_noc_addr, 1, true},
        static_cast<uint8_t>(NUM_HOPS));
    noc_async_write_barrier();

    // Phase 2: Wait for remote data
    volatile tt_l1_ptr uint32_t* sem_recv = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_fabric_recv_addr);
    noc_semaphore_wait(sem_recv, 1);

    // Phase 3: Accumulate recv_buf into output
    cb_reserve_back(cb_out, 1);
    const uint32_t out_l1 = get_write_ptr(cb_out);
    cb_reserve_back(cb_recv, 1);
    const uint32_t recv_l1 = get_write_ptr(cb_recv);

    for (uint32_t tile = 0; tile < NUM_OUTPUT_TILES; ++tile) {
        noc_async_read_page(tile, out_accessor, out_l1);
        noc_async_read_page(tile, recv_accessor, recv_l1);
        noc_async_read_barrier();

        uint16_t* op = reinterpret_cast<uint16_t*>(out_l1);
        uint16_t* rp = reinterpret_cast<uint16_t*>(recv_l1);
        for (uint32_t i = 0; i < 1024; ++i) {
            union {
                uint32_t u;
                float f;
            } a, b, c;
            a.u = static_cast<uint32_t>(op[i]) << 16;
            b.u = static_cast<uint32_t>(rp[i]) << 16;
            c.f = a.f + b.f;
            op[i] = static_cast<uint16_t>(c.u >> 16);
        }

        noc_async_write_page(tile, out_accessor, out_l1);
        noc_async_write_barrier();
    }

    // Phase 4: Close fabric connections
    for (uint32_t i = 0; i < NUM_DIRS; i++) {
        if (directions[i]) {
            fabric_connections[i].close();
        }
    }

    cb_push_back(cb_send, 1);
    cb_pop_front(cb_send, 1);
    cb_push_back(cb_out, 1);
    cb_pop_front(cb_out, 1);
    cb_push_back(cb_recv, 1);
    cb_pop_front(cb_recv, 1);
}
