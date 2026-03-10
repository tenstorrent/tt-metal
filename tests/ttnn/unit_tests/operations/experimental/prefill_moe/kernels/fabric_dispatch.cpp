// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fabric dispatch kernel for prefill MoE (v3 — multi-destination support).
//
// Receives routing metadata from host, exchanges tokens via TT-Fabric between
// devices on a 1xN mesh (N >= 2), then assembles per-expert pkt_buf regions
// in TILE_LAYOUT from ROW_MAJOR token pool (local hs_rm + received staging_buf).
//
// Supports sending to multiple destination devices (up to N-1) and receiving
// from multiple source devices. Each destination has its own direction, hop
// count, and staging_buf offset to avoid collisions.
//
// Uses per-direction WorkerToFabricEdmSender connections (EAST/WEST) with
// explicit send_direction and num_hops parameters for multi-hop routing.
//
// Runs on dispatch core (0, grid_y), dm1/RISCV_1.
//
// Data flow:
//   1. Open fabric connections for valid directions (EAST/WEST)
//   2. For each destination device:
//        a. Send remote-bound tokens from hs_rm -> remote staging_buf via fabric
//        b. Signal remote device (atomic_inc on SEM_FABRIC_RECV)
//   3. Wait for all remote senders (SEM_FABRIC_RECV >= recv_device_count)
//   4. Per-expert assembly: For each expert e:
//        a. Zero-fill tile buffer
//        b. For each token: read from hs_rm or staging_buf, scatter into tiles
//        c. Write assembled tiles to pkt_buf region for expert e
//   5. Signal SEM_PKT_READY on compute leader
//   6. Close fabric connections
//
// Semaphores:
//   SEM_PKT_READY (id=2): Incremented on compute leader after all pkt_bufs written.
//   SEM_FABRIC_RECV (id=4): Each remote sender increments by 1 when done.
//
// Compile-time defines:
//   EAST_CONNECTION: 1 if device has an EAST neighbor, 0 otherwise
//   WEST_CONNECTION: 1 if device has a WEST neighbor, 0 otherwise
//
// CT args:
//   [0..A] TensorAccessorArgs for pkt_buf (TILE_LAYOUT, page=2048B)
//   [A..B] TensorAccessorArgs for hs_rm/staging_buf (ROW_MAJOR, page=D_bytes)
//
// RT args (fixed — set by factory):
//   [0]  hs_rm_addr
//   [1]  staging_buf_addr
//   [2]  remote_staging_addr
//   [3]  pkt_buf_addr
//   [4]  leader_phys_x
//   [5]  leader_phys_y
//   [6]  my_phys_x
//   [7]  my_phys_y
//   [8]  D_bytes
//   [9]  D_tiles
//   [10] M_padded_tiles      - per-expert tile rows
//
// RT args (multi-dest dispatch metadata — from factory at [11..]):
//   [11] recv_device_count   - number of devices sending to us (0..N-1)
//   [12] send_dest_count     - number of destinations we send to (0..N-1)
//   For each dest d:
//     send_direction_d       - 0=EAST, 1=WEST
//     num_hops_d             - 1..N-1
//     remote_staging_offset_d - starting row in remote staging_buf
//     send_count_d           - tokens to send to this dest
//     send_indices_d[0..send_count_d-1]
//   num_experts
//   For each expert e:
//     M_e
//     token_sources[0..M_e-1]  - bit31: 0=hs_rm, 1=staging_buf; bits0-30: row
//
// RT args (fabric — appended by factory after all metadata):
//   fabric connection RT args (per valid direction: EAST then WEST)

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric::linear::experimental;

// Direction indices (matching factory's EAST=0, WEST=1 convention)
constexpr uint32_t DIR_EAST = 0;
constexpr uint32_t DIR_WEST = 1;
constexpr uint32_t NUM_DIRS = 2;

// Direction validity from compile-time defines
constexpr bool HAS_EAST = (EAST_CONNECTION == 1);
constexpr bool HAS_WEST = (WEST_CONNECTION == 1);
constexpr std::array<bool, NUM_DIRS> directions = {HAS_EAST, HAS_WEST};

// Scatter one ROW_MAJOR token row (in L1) into TILE_LAYOUT faces in L1.
inline void scatter_row_to_tiles(uint32_t src_l1, uint32_t tile_buf_l1, uint32_t row, uint32_t num_tiles) {
    uint32_t face_pair = row >> 4;
    uint32_t local_row = row & 0xF;
    uint32_t face0_off = (face_pair * 2) * 512 + local_row * 32;
    uint32_t face1_off = (face_pair * 2 + 1) * 512 + local_row * 32;

    for (uint32_t c = 0; c < num_tiles; ++c) {
        uint32_t tile_base = tile_buf_l1 + c * 2048;
        uint32_t src_off = c * 64;

        volatile uint32_t* dst0 = reinterpret_cast<volatile uint32_t*>(tile_base + face0_off);
        volatile uint32_t* s0 = reinterpret_cast<volatile uint32_t*>(src_l1 + src_off);
        for (uint32_t w = 0; w < 8; ++w) {
            dst0[w] = s0[w];
        }

        volatile uint32_t* dst1 = reinterpret_cast<volatile uint32_t*>(tile_base + face1_off);
        volatile uint32_t* s1 = reinterpret_cast<volatile uint32_t*>(src_l1 + src_off + 32);
        for (uint32_t w = 0; w < 8; ++w) {
            dst1[w] = s1[w];
        }
    }
}

void kernel_main() {
    // ---- CT args ----
    constexpr auto pkt_ct_args = TensorAccessorArgs<0>();
    constexpr auto rm_ct_args = TensorAccessorArgs<1>();

    // ---- RT args (fixed) ----
    uint32_t rt_idx = 0;
    const uint32_t hs_rm_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t staging_buf_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t remote_staging_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t pkt_buf_addr = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t leader_phys_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t leader_phys_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t my_phys_x = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t my_phys_y = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t D_bytes = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t D_tiles = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t M_padded_tiles = get_arg_val<uint32_t>(rt_idx++);

    // ---- Multi-dest dispatch metadata ----
    const uint32_t recv_device_count = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t send_dest_count = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t dest_meta_base = rt_idx;

    // Skip past all per-dest metadata to find expert metadata
    {
        uint32_t skip_cursor = dest_meta_base;
        for (uint32_t d = 0; d < send_dest_count; ++d) {
            skip_cursor += 3;  // direction, hops, offset
            uint32_t cnt = get_arg_val<uint32_t>(skip_cursor);
            skip_cursor += 1 + cnt;  // send_count + send_indices
        }
        rt_idx = skip_cursor;
    }

    const uint32_t num_experts = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t expert_meta_base = rt_idx;

    // Skip past all per-expert metadata to find fabric connection args
    {
        uint32_t skip_cursor = expert_meta_base;
        for (uint32_t e = 0; e < num_experts; ++e) {
            uint32_t M_e = get_arg_val<uint32_t>(skip_cursor);
            skip_cursor += 1 + M_e;
        }
        rt_idx = skip_cursor;
    }

    size_t fab_arg_idx = rt_idx;  // fabric connection args start here

    // ---- CB setup ----
    constexpr uint32_t cb_temp = 3;
    constexpr uint32_t cb_tiles = 6;
    constexpr uint32_t SEM_PKT_READY = 2;
    constexpr uint32_t SEM_FABRIC_RECV = 4;

    const uint32_t rm_page_bytes = D_bytes;
    const uint32_t tile_page_bytes = 2048;

    const auto hs_rm_accessor = TensorAccessor(rm_ct_args, hs_rm_addr, rm_page_bytes);
    const auto staging_accessor = TensorAccessor(rm_ct_args, staging_buf_addr, rm_page_bytes);
    const auto remote_stg_accessor = TensorAccessor(rm_ct_args, remote_staging_addr, rm_page_bytes);
    const auto pkt_accessor = TensorAccessor(pkt_ct_args, pkt_buf_addr, tile_page_bytes);

    cb_reserve_back(cb_temp, 1);
    const uint32_t temp_l1 = get_write_ptr(cb_temp);

    cb_reserve_back(cb_tiles, 1);
    const uint32_t tile_buf_l1 = get_write_ptr(cb_tiles);

    // Remote semaphore NOC address: SEM_FABRIC_RECV on remote device's dispatch core.
    uint64_t remote_sem_noc_addr = safe_get_noc_addr(
        static_cast<uint8_t>(my_phys_x), static_cast<uint8_t>(my_phys_y), get_semaphore(SEM_FABRIC_RECV), 0);

    // ---- Phase 1: Open fabric connections ----
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, NUM_DIRS> fabric_connections;
    size_t conn_idx = fab_arg_idx;
    for (uint32_t i = 0; i < NUM_DIRS; i++) {
        if (directions[i]) {
            fabric_connections[i] =
                tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(conn_idx);
            fabric_connections[i].open_start();
        }
    }
    for (uint32_t i = 0; i < NUM_DIRS; i++) {
        if (directions[i]) {
            fabric_connections[i].open_finish();
        }
    }

    auto pkt_hdr_data = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem = PacketHeaderPool::allocate_header();

    // ---- Phase 2: Send tokens to each destination device ----
    constexpr uint32_t MAX_FABRIC_CHUNK = 4096;

    uint32_t cursor = dest_meta_base;
    for (uint32_t d = 0; d < send_dest_count; ++d) {
        uint32_t dir = get_arg_val<uint32_t>(cursor++);
        uint32_t hops = get_arg_val<uint32_t>(cursor++);
        uint32_t staging_offset = get_arg_val<uint32_t>(cursor++);
        uint32_t cnt = get_arg_val<uint32_t>(cursor++);

        auto* sender = &fabric_connections[dir];

        for (uint32_t i = 0; i < cnt; ++i) {
            uint32_t t = get_arg_val<uint32_t>(cursor++);

            // Read token row t from hs_rm DRAM -> L1 temp
            noc_async_read_page(t, hs_rm_accessor, temp_l1);
            noc_async_read_barrier();

            // Remote staging_buf NOC address at row (staging_offset + i)
            uint64_t dest_noc_addr =
                tt::tt_fabric::linear::addrgen_detail::get_noc_address(remote_stg_accessor, staging_offset + i);

            // Send in chunks <= MAX_FABRIC_CHUNK bytes
            uint32_t bytes_sent = 0;
            while (bytes_sent < D_bytes) {
                uint32_t chunk = D_bytes - bytes_sent;
                if (chunk > MAX_FABRIC_CHUNK) {
                    chunk = MAX_FABRIC_CHUNK;
                }

                fabric_unicast_noc_unicast_write(
                    sender,
                    pkt_hdr_data,
                    temp_l1 + bytes_sent,
                    chunk,
                    tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr + bytes_sent},
                    static_cast<uint8_t>(hops));

                noc_async_write_barrier();
                bytes_sent += chunk;
            }
        }

        // Signal this remote device: all sends to it are complete
        if (cnt > 0) {
            fabric_unicast_noc_unicast_atomic_inc(
                sender,
                pkt_hdr_sem,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_sem_noc_addr, 1, true},
                static_cast<uint8_t>(hops));
            noc_async_write_barrier();
        }
    }

    // ---- Phase 3: Wait for all remote senders ----
    if (recv_device_count > 0) {
        volatile tt_l1_ptr uint32_t* recv_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_FABRIC_RECV));
        noc_semaphore_wait(recv_sem, recv_device_count);
        noc_semaphore_set(recv_sem, 0);
    }

    // ---- Phase 4: Per-expert pkt_buf assembly ----
    uint32_t expert_cursor = expert_meta_base;
    for (uint32_t e = 0; e < num_experts; ++e) {
        uint32_t M_e = get_arg_val<uint32_t>(expert_cursor);
        uint32_t src_base = expert_cursor + 1;

        uint32_t expert_page_offset = e * M_padded_tiles * D_tiles;

        for (uint32_t tr = 0; tr < M_padded_tiles; ++tr) {
            // Zero-fill tile buffer
            {
                volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(tile_buf_l1);
                uint32_t num_words = D_tiles * tile_page_bytes / 4;
                for (uint32_t z = 0; z < num_words; ++z) {
                    p[z] = 0;
                }
            }

            uint32_t row_lo = tr * 32;
            uint32_t row_hi = row_lo + 32;
            if (row_hi > M_e) {
                row_hi = M_e;
            }

            for (uint32_t i = row_lo; i < row_hi; ++i) {
                uint32_t src = get_arg_val<uint32_t>(src_base + i);
                uint32_t is_recv = (src >> 31) & 1;
                uint32_t src_idx = src & 0x7FFFFFFF;

                if (is_recv) {
                    noc_async_read_page(src_idx, staging_accessor, temp_l1);
                } else {
                    noc_async_read_page(src_idx, hs_rm_accessor, temp_l1);
                }
                noc_async_read_barrier();
                scatter_row_to_tiles(temp_l1, tile_buf_l1, i - row_lo, D_tiles);
            }

            // Write this tile row to pkt_buf DRAM
            for (uint32_t c = 0; c < D_tiles; ++c) {
                uint32_t page_idx = expert_page_offset + tr * D_tiles + c;
                noc_async_write_page(page_idx, pkt_accessor, tile_buf_l1 + c * tile_page_bytes);
            }
            noc_async_write_barrier();
        }

        expert_cursor += 1 + M_e;
    }

    // ---- Phase 5: Signal compute leader ----
    uint64_t leader_sem_addr = get_noc_addr(leader_phys_x, leader_phys_y, get_semaphore(SEM_PKT_READY));
    noc_semaphore_inc(leader_sem_addr, 1);

    // ---- Phase 6: Close fabric connections ----
    for (uint32_t i = 0; i < NUM_DIRS; i++) {
        if (directions[i]) {
            fabric_connections[i].close();
        }
    }

    // Release CB reservations
    cb_push_back(cb_temp, 1);
    cb_pop_front(cb_temp, 1);
    cb_push_back(cb_tiles, 1);
    cb_pop_front(cb_tiles, 1);
}
