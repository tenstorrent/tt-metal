// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// ExpertGatherSmall: one data-movement kernel on E_LOC cores (one local expert each) for a small prefill chunk whose
// hidden rows and dense routing matrix are replicated on every chip. Core e (global expert g = idx[e]):
// 1. reads the T/32 tiles of the routing column tile holding g and lists the tokens with a non-zero weight at
//    column g (count_e, up to T entries of (token, weight bits));
// 2. sends count_e to the leader (core 0), which lays out the tile-aligned expert regions (prefix sum of the counts
//    rounded up to 32, in local-expert order), writes `counts` / `region_offsets` (indexed by GLOBAL expert id, zero
//    for experts not on this chip) and sends every core the full offset table;
// 3. copies its tokens' ROW_MAJOR hidden rows into buffer[region_offset_e + i] (one read + one write per row);
// 4. writes its columns of the TILE weight matrix W [T, M]: every column tile of its region (and, for the last
//    core, the tail up to M) as full [32, 32] tiles: zeros except W[t_i, region_offset_e + i] = w_i. The combine is
//    then one matmul W @ expert_out plus the cross-chip reduce.
// Semaphores are reset after use so the program replays in a trace.
//
// compile-time args: [cb_routing, cb_scratch, cb_tile, E, E_LOC, T, H, M, rows_t(=T/32), cols_e_t(=E/32), m_t(=M/32),
//                     sem_id, leader_noc_x, leader_noc_y]
//                    + TensorAccessorArgs(hidden, routing, idx, buffer, counts, offsets, w)
// runtime args: [hidden_addr, routing_addr, idx_addr, buffer_addr, counts_addr, offsets_addr, w_addr, me, (noc_x,
// noc_y) x E_LOC]
#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t CB_ROUTING = get_compile_time_arg_val(0);
constexpr uint32_t CB_SCRATCH = get_compile_time_arg_val(1);
constexpr uint32_t CB_TILE = get_compile_time_arg_val(2);
constexpr uint32_t E = get_compile_time_arg_val(3);
constexpr uint32_t E_LOC = get_compile_time_arg_val(4);
constexpr uint32_t T = get_compile_time_arg_val(5);
constexpr uint32_t H = get_compile_time_arg_val(6);
constexpr uint32_t M = get_compile_time_arg_val(7);
constexpr uint32_t ROWS_T = get_compile_time_arg_val(8);
constexpr uint32_t COLS_E_T = get_compile_time_arg_val(9);
constexpr uint32_t M_T = get_compile_time_arg_val(10);
constexpr uint32_t SEM_ID = get_compile_time_arg_val(11);
constexpr uint32_t LEADER_X = get_compile_time_arg_val(12);
constexpr uint32_t LEADER_Y = get_compile_time_arg_val(13);
constexpr auto hidden_args = TensorAccessorArgs<14>();
constexpr auto routing_args = TensorAccessorArgs<hidden_args.next_compile_time_args_offset()>();
constexpr auto idx_args = TensorAccessorArgs<routing_args.next_compile_time_args_offset()>();
constexpr auto buffer_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
constexpr auto counts_args = TensorAccessorArgs<buffer_args.next_compile_time_args_offset()>();
constexpr auto offsets_args = TensorAccessorArgs<counts_args.next_compile_time_args_offset()>();
constexpr auto w_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
constexpr uint32_t TILE_BYTES = 2048;
constexpr uint32_t ROW_BYTES = H * 2;
constexpr uint32_t VEC_BYTES = E * 4;  // the counts / offsets pages
// scratch layout (bytes, every region 64-byte aligned -- Blackhole DRAM transfers need 64-byte aligned L1
// addresses, a 32-byte-aligned row staging read back shifted rows): idx table (E_LOC x 4) | my (token, wbits)
// list (T x 8) | count slots from the cores (E_LOC x 64, leader) | my count staged for the NoC (64) | offsets table
// (E_LOC + 1) x 4 padded to 512 | counts page (E x 4) | offsets page (E x 4) | one hidden row
constexpr uint32_t OFF_IDX = 0;
constexpr uint32_t OFF_LIST = 256;
constexpr uint32_t OFF_SLOTS = OFF_LIST + T * 8;
constexpr uint32_t OFF_MINE = OFF_SLOTS + E_LOC * 64;
constexpr uint32_t OFF_TABLE = OFF_MINE + 64;
constexpr uint32_t OFF_CNT = OFF_TABLE + 512;
static_assert(T % 8 == 0 && E % 32 == 0, "the scratch regions stay 64-byte aligned");
static_assert((OFF_CNT + 2 * VEC_BYTES) % 64 == 0, "the row staging must be 64-byte aligned");
constexpr uint32_t OFF_OFS = OFF_CNT + VEC_BYTES;
constexpr uint32_t OFF_ROW = OFF_OFS + VEC_BYTES;

FORCE_INLINE uint32_t tile_elem(uint32_t r, uint32_t c) {  // element offset of (row r, col c) inside a 32x32 tile
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

void kernel_main() {
    const uint32_t hidden_addr = get_arg_val<uint32_t>(0);
    const uint32_t routing_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t buffer_addr = get_arg_val<uint32_t>(3);
    const uint32_t counts_addr = get_arg_val<uint32_t>(4);
    const uint32_t offsets_addr = get_arg_val<uint32_t>(5);
    const uint32_t w_addr = get_arg_val<uint32_t>(6);
    const uint32_t me = get_arg_val<uint32_t>(7);
    constexpr uint32_t COORDS = 8;
    const uint32_t scratch = get_write_ptr(CB_SCRATCH);
    const uint32_t routing_l1 = get_write_ptr(CB_ROUTING);
    const uint32_t tile_l1 = get_write_ptr(CB_TILE);
    volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_ID));
    const uint32_t sem_addr = get_semaphore(SEM_ID);

    const auto s_hidden = TensorAccessor(hidden_args, hidden_addr, ROW_BYTES);
    const auto s_routing = TensorAccessor(routing_args, routing_addr, TILE_BYTES);
    const auto s_idx = TensorAccessor(idx_args, idx_addr, E_LOC * 4);
    const auto s_buffer = TensorAccessor(buffer_args, buffer_addr, ROW_BYTES);
    const auto s_counts = TensorAccessor(counts_args, counts_addr, VEC_BYTES);
    const auto s_offsets = TensorAccessor(offsets_args, offsets_addr, VEC_BYTES);
    const auto s_w = TensorAccessor(w_args, w_addr, TILE_BYTES);

    // 1. my expert and its routing column
    noc_async_read(s_idx.get_noc_addr(0), scratch + OFF_IDX, E_LOC * 4);
    noc_async_read_barrier();
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_IDX);
    const uint32_t g = idx[me];
    const uint32_t ct = g >> 5, cj = g & 31;
    for (uint32_t rt = 0; rt < ROWS_T; ++rt) {
        noc_async_read(s_routing.get_noc_addr(rt * COLS_E_T + ct), routing_l1 + rt * TILE_BYTES, TILE_BYTES);
    }
    noc_async_read_barrier();
    volatile tt_l1_ptr uint16_t* rl = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(routing_l1);
    volatile tt_l1_ptr uint32_t* list = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_LIST);
    uint32_t count = 0;
    for (uint32_t t = 0; t < T; ++t) {
        const uint16_t w = rl[(t >> 5) * 1024 + tile_elem(t & 31, cj)];
        if ((w & 0x7FFFu) != 0) {
            list[2 * count] = t;
            list[2 * count + 1] = w;
            ++count;
        }
    }

    // 2. counts -> leader; leader lays out the regions and broadcasts the offset table
    volatile tt_l1_ptr uint32_t* slots = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_SLOTS);
    volatile tt_l1_ptr uint32_t* table = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_TABLE);
    if (me != 0) {
        volatile tt_l1_ptr uint32_t* mine = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_MINE);
        mine[0] = count;  // staged for the NoC write (one 32-byte slot per core on the leader)
        noc_async_write(scratch + OFF_MINE, get_noc_addr(LEADER_X, LEADER_Y, scratch + OFF_SLOTS + me * 64), 64);
        noc_async_write_barrier();
        noc_semaphore_inc(get_noc_addr(LEADER_X, LEADER_Y, sem_addr), 1);
        noc_async_atomic_barrier();
    } else {
        slots[0] = count;  // the leader's own count
    }
    if (me == 0) {
        if constexpr (E_LOC > 1) {
            noc_semaphore_wait_min(sem, E_LOC - 1);
            noc_semaphore_set(sem, 0);
        }
        volatile tt_l1_ptr uint32_t* cnt = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_CNT);
        volatile tt_l1_ptr uint32_t* ofs = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch + OFF_OFS);
        for (uint32_t i = 0; i < E; ++i) {
            cnt[i] = 0;
            ofs[i] = 0;
        }
        uint32_t run = 0;
        for (uint32_t e = 0; e < E_LOC; ++e) {
            const uint32_t c = slots[e * 16];  // slot e = 64 bytes, its first word
            table[e] = run;
            cnt[idx[e]] = c;
            ofs[idx[e]] = run;
            run += (c + 31) & ~31u;
        }
        table[E_LOC] = run;  // the end of the last region (<= M by the drop-free contract)
        noc_async_write(scratch + OFF_CNT, s_counts.get_noc_addr(0), VEC_BYTES);
        noc_async_write(scratch + OFF_OFS, s_offsets.get_noc_addr(0), VEC_BYTES);
        noc_async_write_barrier();
        constexpr uint32_t TABLE_BYTES = ((E_LOC + 1) * 4 + 63) & ~63u;
        for (uint32_t e = 1; e < E_LOC; ++e) {
            const uint32_t nx = get_arg_val<uint32_t>(COORDS + 2 * e), ny = get_arg_val<uint32_t>(COORDS + 2 * e + 1);
            noc_async_write(scratch + OFF_TABLE, get_noc_addr(nx, ny, scratch + OFF_TABLE), TABLE_BYTES);
        }
        noc_async_write_barrier();
        for (uint32_t e = 1; e < E_LOC; ++e) {
            const uint32_t nx = get_arg_val<uint32_t>(COORDS + 2 * e), ny = get_arg_val<uint32_t>(COORDS + 2 * e + 1);
            noc_semaphore_inc(get_noc_addr(nx, ny, sem_addr), 1);
        }
        noc_async_atomic_barrier();
    } else {
        noc_semaphore_wait_min(sem, 1);
        noc_semaphore_set(sem, 0);
    }
    const uint32_t off = table[me];
    const uint32_t region_end = (me + 1 == E_LOC) ? M : table[me + 1];  // the last core owns the tail columns of W

    // 3. my tokens' rows -> buffer[off + i]
    for (uint32_t i = 0; i < count; ++i) {
        const uint32_t t = list[2 * i];
        noc_async_read(s_hidden.get_noc_addr(t), scratch + OFF_ROW, ROW_BYTES);
        noc_async_read_barrier();
        noc_async_write(scratch + OFF_ROW, s_buffer.get_noc_addr(off + i), ROW_BYTES);
        noc_async_write_barrier();
    }

    // 4. my column tiles of W: for every row tile rt and column tile mt in [off/32, region_end/32)
    volatile tt_l1_ptr uint16_t* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_l1);
    for (uint32_t mt = off >> 5; mt < (region_end >> 5); ++mt) {
        for (uint32_t rt = 0; rt < ROWS_T; ++rt) {
            for (uint32_t k = 0; k < 1024; ++k) {
                tile[k] = 0;
            }
            for (uint32_t i = 0; i < count; ++i) {
                const uint32_t m = off + i;
                const uint32_t t = list[2 * i];
                if ((m >> 5) == mt && (t >> 5) == rt) {
                    tile[tile_elem(t & 31, m & 31)] = uint16_t(list[2 * i + 1]);
                }
            }
            noc_async_write(tile_l1, s_w.get_noc_addr(rt * M_T + mt), TILE_BYTES);
            noc_async_write_barrier();
        }
    }
}
