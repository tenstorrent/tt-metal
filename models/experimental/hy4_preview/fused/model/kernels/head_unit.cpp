// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// LM head unit control (BRISC, NOC1) of the fused decode step (fused/model/ends_fused.py EndsHead; PLAN.md section
// 46), from fused/moe/kernels/router_writer.cpp: wait (bounded) for the hub's x (hand-over word SEM_X of the
// units' sync words), start the weight streamer (fused/moe/kernels/stream_reader.cpp, unchanged) and the compute
// kernel (head_compute.cpp), and per finished FP32 logits tile ([ROWS, 32]: 32 token ids for the 8 rows):
//   - keep each row's K best (sort key of the fp32 bits, token id) among this core's ids below the shard's size:
//     a strictly greater key wins and ids are visited in ascending order, so of equal logits the lower id stays
//     ahead;
//   - write the tile to tile column `col` of the hub's logits range, and to page `col` of the kept DRAM tensor when
//     the launch names one.
// Then the rows' lists (ROWS x K x (key, id), 64 K bytes) go to this unit's slot on the hub and one count to the
// hub's word. The count is sent whether or not the hub's release was seen: the hub does not wait for a core that
// gave up (its list then holds key 0 entries, below every real logit's key).
//
// The core's constants are its sync words (moe_common.hpp CFG_* for the streamer, and here): 12 the number of its
// columns (1 or 2), 13 its unit index, 14 its first tile column.
// compile-time args: [cb_x, cb_ctl, cb_out, x_tiles, hub_x, hub_y, K, shard ids, <the kept tensor's accessor args>]
// runtime args (the same on every unit core): [logits address on the hub, slots address on the hub, the hub's count
//   word, the kept tensor's address or 0]

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "ckernel.h"
#include "ckernel_defs.h"
#include "moe_dm.hpp"

namespace {

constexpr uint32_t CFG_COLS = 12;
constexpr uint32_t CFG_UNIT = 13;
constexpr uint32_t CFG_COL0 = 14;

struct Pair {
    uint32_t key;
    uint32_t id;
};

// Larger key <=> larger fp32 value (sign-adjusted bits).
FORCE_INLINE uint32_t sort_key(uint32_t fp32) {
    return fp32 ^ (static_cast<uint32_t>(static_cast<int32_t>(fp32) >> 31) | 0x80000000u);
}

template <uint32_t K>
FORCE_INLINE void insert(Pair* list, uint32_t key, uint32_t id) {
    if (key > list[K - 1].key) {
        uint32_t p = K - 1;
        while (p > 0 && key > list[p - 1].key) {
            list[p] = list[p - 1];
            --p;
        }
        list[p].key = key;
        list[p].id = id;
    }
}

}  // namespace

void kernel_main() {
    constexpr uint32_t cb_x = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ctl = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t x_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t hub_x = get_compile_time_arg_val(4);
    constexpr uint32_t hub_y = get_compile_time_arg_val(5);
    constexpr uint32_t K = get_compile_time_arg_val(6);
    constexpr uint32_t shard_ids = get_compile_time_arg_val(7);
    constexpr auto keep_args = TensorAccessorArgs<8>();
    constexpr uint32_t TILE_BYTES = hy4moe::ROW_TILE_BYTES_F32;
    constexpr uint32_t list_bytes = ROWS * K * sizeof(Pair);

    invalidate_l1_cache();
    const uint32_t cols = hy4moe::cfg(CFG_COLS);
    const uint32_t unit = hy4moe::cfg(CFG_UNIT);
    const uint32_t col0 = hy4moe::cfg(CFG_COL0);
    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t slots_addr = get_arg_val<uint32_t>(1);
    const uint32_t count_addr = get_arg_val<uint32_t>(2);
    const uint32_t keep_addr = get_arg_val<uint32_t>(3);

    // control page: [16 bytes the streamer's control words | the rows' lists], all on 16-byte words
    volatile tt_l1_ptr uint32_t* ctl = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_ctl));
    Pair* best = reinterpret_cast<Pair*>(get_write_ptr(cb_ctl) + 16);
    for (uint32_t i = 0; i < ROWS * K; ++i) {
        best[i].key = 0;
        best[i].id = col0 * 32;
    }

    volatile tt_l1_ptr uint32_t* start = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hy4moe::sem(hy4moe::SEM_X));
    const uint32_t released = hy4moe::wait_ge(start, 1) ? cols : 0;
    if (released) {
        start[0] = 0;  // seen: cleared for the next launch
    }
    ctl[0] = released;    // the streamer's item count
    ctl[1] = 0xFFFFFFFF;  // no plain slots
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(hy4moe::sem(hy4moe::SEM_LOCAL)), 1);
    ckernel::mailbox_write(ckernel::ThreadId::UnpackThreadId, released);
    ckernel::mailbox_write(ckernel::ThreadId::MathThreadId, released);
    ckernel::mailbox_write(ckernel::ThreadId::PackThreadId, released);
    if (released) {
        const auto keep = TensorAccessor(keep_args, keep_addr, TILE_BYTES);
        cb_push_back(cb_x, x_tiles);  // written by the hub's multicast
        for (uint32_t c = 0; c < released; ++c) {
            cb_wait_front(cb_out, 1);
            const uint32_t col = col0 + c;
            const uint32_t tile_l1 = get_read_ptr(cb_out);
            const uint32_t* tile = reinterpret_cast<const uint32_t*>(tile_l1);
            for (uint32_t r = 0; r < ROWS; ++r) {
                Pair* list = best + r * K;
                for (uint32_t face = 0; face < 2; ++face) {
                    const uint32_t* w = tile + (face * ROWS + r) * 16;
                    const uint32_t id0 = col * 32 + face * 16;
                    for (uint32_t j = 0; j < 16; ++j) {
                        if (id0 + j < shard_ids) {
                            insert<K>(list, sort_key(w[j]), id0 + j);
                        }
                    }
                }
            }
            if (hy4moe::write_buf_free()) {  // (bounded NOC waits)
                noc_async_write(tile_l1, get_noc_addr(hub_x, hub_y, logits_addr + col * TILE_BYTES), TILE_BYTES);
                if (keep_addr != 0 && hy4moe::write_buf_free()) {
                    noc_async_write(tile_l1, keep.get_noc_addr(col), TILE_BYTES);
                }
                hy4moe::writes_flushed();
            }
            cb_pop_front(cb_out, 1);
        }
    }
    if (hy4moe::write_buf_free()) {
        noc_async_write(
            get_write_ptr(cb_ctl) + 16, get_noc_addr(hub_x, hub_y, slots_addr + unit * list_bytes), list_bytes);
        hy4moe::writes_flushed();
    }
    // counted either way: the hub does not wait for a core that gave up
    if (hy4moe::atomic_buf_free()) {
        noc_semaphore_inc(get_noc_addr(hub_x, hub_y, count_addr), 1);
        hy4moe::atomics_flushed();
    }
}
