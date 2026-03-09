// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fabric dispatch kernel for prefill MoE (v2 — per-expert pkt_buf assembly).
//
// Receives routing metadata from host, exchanges tokens via TT-Fabric between
// devices, then assembles per-expert pkt_buf regions in TILE_LAYOUT from the
// ROW_MAJOR token pool (local hs_rm + received staging_buf).
//
// Uses per-direction WorkerToFabricEdmSender connections (EAST/WEST) with
// explicit send_direction and num_hops parameters to support multi-hop routing
// on 1xN meshes (N >= 2).
//
// Runs on dispatch core (0, grid_y), dm1/RISCV_1.
//
// Data flow:
//   1. Open fabric connections for valid directions (EAST/WEST)
//   2. Send remote-bound tokens from local hs_rm -> remote staging_buf via fabric
//   3. Signal remote device (atomic_inc on SEM_FABRIC_RECV)
//   4. Wait for remote device's signal (SEM_FABRIC_RECV) if recv_count > 0
//   5. Per-expert assembly: For each expert e:
//        a. Zero-fill tile buffer
//        b. For each token in expert e's list: read from hs_rm or staging_buf,
//           scatter into tile buffer
//        c. Write assembled tiles to pkt_buf region for expert e
//   6. Signal SEM_PKT_READY on compute leader (once after all experts)
//   7. Close fabric connections
//
// Semaphores:
//   SEM_PKT_READY (id=2): Incremented on compute leader after all pkt_bufs written.
//   SEM_FABRIC_RECV (id=4): Remote dispatch signals when sends are complete.
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
//   [0]  hs_rm_addr          - hidden_states ROW_MAJOR DRAM address
//   [1]  staging_buf_addr    - local staging buffer ROW_MAJOR DRAM address
//   [2]  remote_staging_addr - remote staging buffer ROW_MAJOR DRAM address
//   [3]  pkt_buf_addr        - pkt_buf TILE_LAYOUT DRAM address (E * M_padded * D region)
//   [4]  leader_phys_x
//   [5]  leader_phys_y
//   [6]  my_phys_x           - this dispatch core's physical X (for remote sem addr)
//   [7]  my_phys_y           - this dispatch core's physical Y
//   [8]  D_bytes             - D * 2 (bytes per token row)
//   [9]  D_tiles             - D / 32 (tiles per row)
//   [10] M_padded_tiles      - per-expert tile rows (max_M_padded / 32)
//
// RT args (dispatch_metadata — from Python, inserted by factory at [11..]):
//   [11] recv_count          - tokens expected from remote device
//   [12] send_count          - tokens to send to remote device
//   [13..13+send_count-1]    - send_token_indices (hs_rm rows to send via fabric)
//   [13+send_count]          - num_experts
//   For each expert e:
//     [..] M_e               - actual token count for this expert
//     [..] token_sources[0..M_e-1] - bit 31: 0=local hs_rm, 1=staging_buf;
//                                    bits 0-30: row index in that source
//
// RT args (fabric — appended by factory after dispatch_metadata):
//   [..] send_direction      - 0=EAST, 1=WEST
//   [..] num_hops            - number of fabric hops to target device
//   [..] fabric connection RT args (per valid direction)

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
// All operations are L1-to-L1 memcpy (pure RISC-V, no NOC).
inline void scatter_row_to_tiles(
    uint32_t src_l1,       // L1 address of ROW_MAJOR token row (D*2 bytes)
    uint32_t tile_buf_l1,  // L1 address of tile buffer (D_tiles * 2048 bytes)
    uint32_t row,          // target row within tile row (0..31)
    uint32_t num_tiles     // D_tiles
) {
    uint32_t face_pair = row >> 4;   // row / 16
    uint32_t local_row = row & 0xF;  // row % 16
    uint32_t face0_off = (face_pair * 2) * 512 + local_row * 32;
    uint32_t face1_off = (face_pair * 2 + 1) * 512 + local_row * 32;

    for (uint32_t c = 0; c < num_tiles; ++c) {
        uint32_t tile_base = tile_buf_l1 + c * 2048;
        uint32_t src_off = c * 64;  // 32 BF16 = 64 bytes per tile column

        // Copy cols 0-15 (32 bytes) -> face 0 or 2
        volatile uint32_t* dst0 = reinterpret_cast<volatile uint32_t*>(tile_base + face0_off);
        volatile uint32_t* s0 = reinterpret_cast<volatile uint32_t*>(src_l1 + src_off);
        for (uint32_t w = 0; w < 8; ++w) {
            dst0[w] = s0[w];
        }

        // Copy cols 16-31 (32 bytes) -> face 1 or 3
        volatile uint32_t* dst1 = reinterpret_cast<volatile uint32_t*>(tile_base + face1_off);
        volatile uint32_t* s1 = reinterpret_cast<volatile uint32_t*>(src_l1 + src_off + 32);
        for (uint32_t w = 0; w < 8; ++w) {
            dst1[w] = s1[w];
        }
    }
}

void kernel_main() {
    // ---- CT args ----
    constexpr auto pkt_ct_args = TensorAccessorArgs<0>();  // pkt_buf (TILE_LAYOUT)
    constexpr auto rm_ct_args = TensorAccessorArgs<1>();   // hs_rm / staging_buf (ROW_MAJOR)

    // ---- RT args (fixed) ----
    uint32_t rt_idx = 0;
    const uint32_t hs_rm_addr = get_arg_val<uint32_t>(rt_idx++);           // [0]
    const uint32_t staging_buf_addr = get_arg_val<uint32_t>(rt_idx++);     // [1]
    const uint32_t remote_staging_addr = get_arg_val<uint32_t>(rt_idx++);  // [2]
    const uint32_t pkt_buf_addr = get_arg_val<uint32_t>(rt_idx++);         // [3]
    const uint32_t leader_phys_x = get_arg_val<uint32_t>(rt_idx++);        // [4]
    const uint32_t leader_phys_y = get_arg_val<uint32_t>(rt_idx++);        // [5]
    const uint32_t my_phys_x = get_arg_val<uint32_t>(rt_idx++);            // [6]
    const uint32_t my_phys_y = get_arg_val<uint32_t>(rt_idx++);            // [7]
    const uint32_t D_bytes = get_arg_val<uint32_t>(rt_idx++);              // [8]
    const uint32_t D_tiles = get_arg_val<uint32_t>(rt_idx++);              // [9]
    const uint32_t M_padded_tiles = get_arg_val<uint32_t>(rt_idx++);       // [10]

    // ---- dispatch_metadata (from Python, variable length) ----
    const uint32_t recv_count = get_arg_val<uint32_t>(rt_idx++);  // [11]
    const uint32_t send_count = get_arg_val<uint32_t>(rt_idx++);  // [12]
    const uint32_t send_indices_base = rt_idx;                    // [13]
    rt_idx += send_count;                                         // skip past send_indices

    const uint32_t num_experts = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t expert_meta_base = rt_idx;

    // Skip past all per-expert metadata to find send_direction/num_hops/fabric args
    {
        uint32_t skip_cursor = expert_meta_base;
        for (uint32_t e = 0; e < num_experts; ++e) {
            uint32_t M_e = get_arg_val<uint32_t>(skip_cursor);
            skip_cursor += 1 + M_e;  // skip M_e + token_sources
        }
        rt_idx = skip_cursor;
    }

    const uint32_t send_direction = get_arg_val<uint32_t>(rt_idx++);
    const uint32_t num_hops = get_arg_val<uint32_t>(rt_idx++);
    size_t fab_arg_idx = rt_idx;  // fabric connection args start here

    // ---- CB setup ----
    constexpr uint32_t cb_temp = 3;   // Temp buffer for one token row
    constexpr uint32_t cb_tiles = 6;  // Tile assembly buffer
    constexpr uint32_t SEM_PKT_READY = 2;
    constexpr uint32_t SEM_FABRIC_RECV = 4;

    // Page sizes
    const uint32_t rm_page_bytes = D_bytes;  // ROW_MAJOR page = one token row
    const uint32_t tile_page_bytes = 2048;   // TILE_LAYOUT page = one tile

    // Create TensorAccessors
    const auto hs_rm_accessor = TensorAccessor(rm_ct_args, hs_rm_addr, rm_page_bytes);
    const auto staging_accessor = TensorAccessor(rm_ct_args, staging_buf_addr, rm_page_bytes);
    const auto remote_stg_accessor = TensorAccessor(rm_ct_args, remote_staging_addr, rm_page_bytes);
    const auto pkt_accessor = TensorAccessor(pkt_ct_args, pkt_buf_addr, tile_page_bytes);

    // Get L1 buffer addresses via CB
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

    auto* sender = &fabric_connections[send_direction];

    auto pkt_hdr_data = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem = PacketHeaderPool::allocate_header();

    // ---- Phase 2: Send remote-bound tokens via fabric ----
    constexpr uint32_t MAX_FABRIC_CHUNK = 4096;

    for (uint32_t i = 0; i < send_count; ++i) {
        uint32_t t = get_arg_val<uint32_t>(send_indices_base + i);

        // Read token row t from hs_rm DRAM -> L1 temp
        noc_async_read_page(t, hs_rm_accessor, temp_l1);
        noc_async_read_barrier();

        // Compute remote staging_buf NOC address for page (row) i
        uint64_t dest_noc_addr = tt::tt_fabric::linear::addrgen_detail::get_noc_address(remote_stg_accessor, i);

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
                static_cast<uint8_t>(num_hops));

            noc_async_write_barrier();
            bytes_sent += chunk;
        }
    }

    // Signal remote device: all sends complete
    if (send_count > 0) {
        fabric_unicast_noc_unicast_atomic_inc(
            sender,
            pkt_hdr_sem,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_sem_noc_addr, 1, true},
            static_cast<uint8_t>(num_hops));
        noc_async_write_barrier();
    }

    // ---- Phase 3: Wait for remote device's sends (if expecting any) ----
    if (recv_count > 0) {
        volatile tt_l1_ptr uint32_t* recv_sem =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_FABRIC_RECV));
        noc_semaphore_wait(recv_sem, 1);
        noc_semaphore_set(recv_sem, 0);
    }

    // ---- Phase 4: Per-expert pkt_buf assembly ----
    uint32_t expert_cursor = expert_meta_base;
    for (uint32_t e = 0; e < num_experts; ++e) {
        uint32_t M_e = get_arg_val<uint32_t>(expert_cursor);
        uint32_t src_base = expert_cursor + 1;  // token_sources start here

        uint32_t expert_page_offset = e * M_padded_tiles * D_tiles;

        for (uint32_t tr = 0; tr < M_padded_tiles; ++tr) {
            // Zero-fill tile buffer (D_tiles * 2048 bytes)
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

            // Write this tile row to pkt_buf DRAM (2D page indexing with expert offset)
            for (uint32_t c = 0; c < D_tiles; ++c) {
                uint32_t page_idx = expert_page_offset + tr * D_tiles + c;
                noc_async_write_page(page_idx, pkt_accessor, tile_buf_l1 + c * tile_page_bytes);
            }
            noc_async_write_barrier();
        }

        expert_cursor += 1 + M_e;  // advance past M_e + token_sources
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
