// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat spatial expert: pipelined gate/up weight forwarder (NCRISC, NOC1). Chunk c (one [KBLK x 2] block per receiver,
// the reader CB's slot c % CB_SLOTS) goes to landing slot c % SLOTS of each of the R receivers once all of them granted
// it, tagged with NoC transaction id 1 + c % DEPTH, so up to DEPTH chunks are in flight; when a chunk's id has no
// outstanding writes its receivers' data counters are bumped and the reader slot is freed, in order. (se_forward.cpp
// waited for every chunk's acknowledgements before the next one: the NoC estimator puts 4 writes per barrier at
// ~49 GB/s against ~71 GB/s at 16.)
//
// CT: 0 CB, 1 R, 2 TILE_BYTES, 3 SLOT_TILES (reader CB slot), 4 BLK_TILES (one receiver's block), 5 SLOTS (landing
//     ring), 6 CREDIT_SEM0, 7 DATA_SEM, 8 TOTAL_CHUNKS (SE_DYN: chunks per expert), 9 CB_SLOTS, 10 DEPTH (<= 15),
//     11 NUM_E (SE_DYN)
// RT: 0 landing base, 1..R receiver xy, then the se_dyn.hpp args (SE_DYN)
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#ifdef SE_DYN
#include "se_dyn.hpp"
#endif

void kernel_main() {
    constexpr uint32_t cb = get_compile_time_arg_val(0);
    constexpr uint32_t R = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t slot_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t blk_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t slots = get_compile_time_arg_val(5);
    constexpr uint32_t credit_sem0 = get_compile_time_arg_val(6);
    constexpr uint32_t data_sem = get_compile_time_arg_val(7);
#ifdef SE_DYN
    // CT 8 is then the chunks per expert (NUM_E = CT 11); the active experts' chunks are all that is sent (RT 1 + R..
    // are the se_dyn.hpp args, CB 7 this RISC's scratch)
    constexpr uint32_t per_e = get_compile_time_arg_val(8);
    constexpr uint32_t num_e = get_compile_time_arg_val(11);
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, 1 + R, get_write_ptr(tt::CBIndex::c_7) + 2048, 1);  // NCRISC: upper half of CB 7
    const uint32_t total = dyn.n_act * per_e;
#else
    constexpr uint32_t total = get_compile_time_arg_val(8);
#endif
    constexpr uint32_t cb_slots = get_compile_time_arg_val(9);
    constexpr uint32_t depth = get_compile_time_arg_val(10);
    static_assert(depth >= 1 && depth <= 15 && depth <= cb_slots);
    constexpr uint32_t slot_bytes = slot_tiles * tile_bytes;
    constexpr uint32_t blk_bytes = blk_tiles * tile_bytes;

    const uint32_t landing = get_arg_val<uint32_t>(0);
    uint64_t recv[R], recv_data[R];
    volatile tt_l1_ptr uint32_t* credit[R];
    for (uint32_t j = 0; j < R; ++j) {
        const uint32_t xy = get_arg_val<uint32_t>(1 + j);
        recv[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, 0);
        recv_data[j] = get_noc_addr(xy >> 16, xy & 0xFFFF, get_semaphore(data_sem));
        credit[j] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(credit_sem0 + j));
    }
    const uint32_t cb_base = get_read_ptr(cb);  // nothing popped yet: slot 0 of the reader CB
    uint32_t issued = 0, done = 0;
    while (done < total) {
        invalidate_l1_cache();
        if (issued < total && issued - done < depth &&
            cb_pages_available_at_front(cb, (issued - done + 1) * slot_tiles)) {
            bool granted = true;
            for (uint32_t j = 0; j < R; ++j) {
                granted = granted && *credit[j] >= issued + 1;
            }
            if (granted) {
                const uint32_t src = cb_base + (issued % cb_slots) * slot_bytes;
                const uint32_t dst = landing + (issued % slots) * blk_bytes;
                const uint32_t trid = 1 + issued % depth;
                for (uint32_t j = 0; j < R; ++j) {
                    noc_async_write_one_packet_with_trid(src + j * blk_bytes, recv[j] | dst, blk_bytes, trid);
                }
                ++issued;
            }
        }
        if (done < issued && ncrisc_noc_nonposted_write_with_transaction_id_flushed(noc_index, 1 + done % depth)) {
            for (uint32_t j = 0; j < R; ++j) {
                noc_semaphore_inc(recv_data[j], 1);
            }
            cb_pop_front(cb, slot_tiles);
            ++done;
        }
    }
    noc_async_write_set_trid(0);
    noc_async_write_barrier();
    noc_async_atomic_barrier();
}
