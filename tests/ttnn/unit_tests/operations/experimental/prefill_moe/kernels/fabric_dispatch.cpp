// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// Fabric dispatch kernel for prefill MoE.
//
// Receives routing metadata from host, sends remote-bound tokens via TT-Fabric,
// receives tokens from the other device, and assembles a TILE_LAYOUT pkt_buf
// from ROW_MAJOR hidden_states sources.
//
// Runs on dispatch core (0, grid_y), dm1/RISCV_1.
//
// Data flow:
//   1. Open fabric connection
//   2. Send remote-bound tokens from local hs_rm -> remote staging_buf via fabric
//   3. Signal remote device (atomic_inc on SEM_FABRIC_RECV)
//   4. Wait for remote device's signal (SEM_FABRIC_RECV)
//   5. Assemble pkt_buf in TILE_LAYOUT from local + received tokens
//   6. Write assembled tiles to pkt_buf DRAM
//   7. Signal SEM_PKT_READY on compute leader
//   8. Close fabric connection
//
// Semaphores:
//   SEM_PKT_READY (id=2): Incremented on compute leader after pkt_buf is written.
//   SEM_FABRIC_RECV (id=4): Remote dispatch signals when sends are complete.
//
// CT args:
//   [0..A] TensorAccessorArgs for pkt_buf (TILE_LAYOUT, page=2048B)
//   [A..B] TensorAccessorArgs for hs_rm/staging_buf (ROW_MAJOR, page=D_bytes)
//
// RT args:
//   [0]  hs_rm_addr          - hidden_states ROW_MAJOR DRAM address
//   [1]  staging_buf_addr    - local staging buffer ROW_MAJOR DRAM address
//   [2]  remote_staging_addr - remote staging buffer ROW_MAJOR DRAM address
//   [3]  pkt_buf_addr        - pkt_buf TILE_LAYOUT DRAM address
//   [4]  leader_phys_x
//   [5]  leader_phys_y
//   [6]  my_phys_x           - this dispatch core's physical X (for remote sem addr)
//   [7]  my_phys_y           - this dispatch core's physical Y
//   [8]  D_bytes             - D * 2 (bytes per token row)
//   [9]  D_tiles             - D / 32 (tiles per row)
//   [10] local_count         - tokens from local HS for local expert
//   [11] recv_count          - tokens expected from remote device
//   [12] send_count          - tokens to send to remote device
//   [13..13+local_count-1]          - local_token_indices[]
//   [13+local_count..13+local_count+send_count-1] - send_token_indices[]
//   [...] FabricConnectionManager RT args

#include "api/dataflow/dataflow_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"

using namespace tt::tt_fabric::linear::experimental;

// Scatter one ROW_MAJOR token row (in L1) into TILE_LAYOUT faces in L1.
// All operations are L1-to-L1 memcpy (pure RISC-V, no NOC).
inline void scatter_row_to_tiles(
    uint32_t src_l1,       // L1 address of ROW_MAJOR token row (D*2 bytes)
    uint32_t tile_buf_l1,  // L1 address of tile buffer (D_tiles * 2048 bytes)
    uint32_t row,          // target row in pkt_buf (0..P-1)
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

    // ---- RT args ----
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
    const uint32_t local_count = get_arg_val<uint32_t>(rt_idx++);          // [10]
    const uint32_t recv_count = get_arg_val<uint32_t>(rt_idx++);           // [11]
    const uint32_t send_count = get_arg_val<uint32_t>(rt_idx++);           // [12]
    const uint32_t indices_base = rt_idx;                                  // [13]

    // Fabric connection args start after all indices
    size_t fab_arg_idx = indices_base + local_count + send_count;

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

    // Zero-fill the tile buffer in L1
    {
        volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(tile_buf_l1);
        uint32_t num_words = D_tiles * tile_page_bytes / 4;
        for (uint32_t i = 0; i < num_words; ++i) {
            p[i] = 0;
        }
    }

    // Remote semaphore NOC address: SEM_FABRIC_RECV on remote device's dispatch core.
    // Both devices have the same physical core mapping, so the semaphore L1 address
    // is the same. We use our own physical coords for the NOC address.
    uint64_t remote_sem_noc_addr = safe_get_noc_addr(
        static_cast<uint8_t>(my_phys_x), static_cast<uint8_t>(my_phys_y), get_semaphore(SEM_FABRIC_RECV), 0);

    // ---- Phase 1: Open fabric connection ----
    auto fabric_connection = FabricConnectionManager::build_from_args(fab_arg_idx);
    fabric_connection.open();

    auto* sender = fabric_connection.has_forward_connection() ? &fabric_connection.get_forward_connection()
                                                              : &fabric_connection.get_backward_connection();

    auto pkt_hdr_data = PacketHeaderPool::allocate_header();
    auto pkt_hdr_sem = PacketHeaderPool::allocate_header();

    // ---- Phase 2: Send remote-bound tokens via fabric ----
    for (uint32_t i = 0; i < send_count; ++i) {
        uint32_t t = get_arg_val<uint32_t>(indices_base + local_count + i);

        // Read token row t from hs_rm DRAM -> L1 temp
        noc_async_read_page(t, hs_rm_accessor, temp_l1);
        noc_async_read_barrier();

        // Compute remote staging_buf NOC address for page (row) i
        uint64_t dest_noc_addr = tt::tt_fabric::linear::addrgen_detail::get_noc_address(remote_stg_accessor, i);

        // Send via fabric to remote device
        fabric_unicast_noc_unicast_write(
            sender,
            pkt_hdr_data,
            temp_l1,
            D_bytes,
            tt::tt_fabric::NocUnicastCommandHeader{dest_noc_addr},
            static_cast<uint8_t>(1));  // 1 hop

        noc_async_write_barrier();  // Must barrier before reusing temp_l1
    }

    // Signal remote device: all sends complete
    fabric_unicast_noc_unicast_atomic_inc(
        sender,
        pkt_hdr_sem,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{remote_sem_noc_addr, 1, true},
        static_cast<uint8_t>(1));
    noc_async_write_barrier();

    // ---- Phase 3: Wait for remote device's sends ----
    volatile tt_l1_ptr uint32_t* recv_sem =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_FABRIC_RECV));
    noc_semaphore_wait(recv_sem, 1);
    noc_semaphore_set(recv_sem, 0);

    // ---- Phase 4: Assemble pkt_buf (ROW_MAJOR -> TILE_LAYOUT scatter in L1) ----
    uint32_t pkt_row = 0;

    // 4a: Local tokens from hs_rm
    for (uint32_t i = 0; i < local_count; ++i) {
        uint32_t t = get_arg_val<uint32_t>(indices_base + i);
        noc_async_read_page(t, hs_rm_accessor, temp_l1);
        noc_async_read_barrier();
        scatter_row_to_tiles(temp_l1, tile_buf_l1, pkt_row, D_tiles);
        pkt_row++;
    }

    // 4b: Received tokens from staging_buf
    for (uint32_t i = 0; i < recv_count; ++i) {
        noc_async_read_page(i, staging_accessor, temp_l1);
        noc_async_read_barrier();
        scatter_row_to_tiles(temp_l1, tile_buf_l1, pkt_row, D_tiles);
        pkt_row++;
    }

    // ---- Phase 5: Write assembled tiles to pkt_buf DRAM ----
    for (uint32_t c = 0; c < D_tiles; ++c) {
        noc_async_write_page(c, pkt_accessor, tile_buf_l1 + c * tile_page_bytes);
    }
    noc_async_write_barrier();

    // ---- Phase 6: Signal compute leader ----
    uint64_t leader_sem_addr = get_noc_addr(leader_phys_x, leader_phys_y, get_semaphore(SEM_PKT_READY));
    noc_semaphore_inc(leader_sem_addr, 1);

    // ---- Phase 7: Close fabric ----
    fabric_connection.close();

    // Release CB reservations
    cb_push_back(cb_temp, 1);
    cb_pop_front(cb_temp, 1);
    cb_push_back(cb_tiles, 1);
    cb_pop_front(cb_tiles, 1);
}
