// SPDX-License-Identifier: Apache-2.0
// Regime-A in0 RING ALL-GATHER + column reduction (runs on the core's non-in1 RISC).
// Every core in a k-slice group is an injector: it reads only its OWN shard (W blocks) of in0[:,k-slice]
// from DRAM (parallel across the G cores), then the shards rotate cyclically around the ring so each core
// computes all G shards. cb0 holds the full k-slice (G slots of W blocks); slot s of core c ends up holding
// shard (c-s), which the in1 reader matches via its rotated read.
//   step 0: read own shard (slot 0) from DRAM, forward to next's slot 1, push slot 0.
//   step s (1..G-1): wait recv (prev forwarded into slot s), forward slot s -> next's slot s+1 (s<G-1),
//                    push slot s.  Every core forwards G-1 times cyclically; no head/tail.
// Then split-K column reduction (reuse compute REDUCE_K); top of the bank column writes DRAM.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t M_block = get_compile_time_arg_val(0);
    constexpr uint32_t K_block = get_compile_time_arg_val(1);  // kb
    constexpr uint32_t N_block = get_compile_time_arg_val(2);
    constexpr uint32_t K_num_blocks = get_compile_time_arg_val(3);  // G*W (full k-slice)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t Kt = get_compile_time_arg_val(5);
    constexpr uint32_t Nt = get_compile_time_arg_val(6);
    constexpr uint32_t W = get_compile_time_arg_val(7);  // blocks per shard
    constexpr uint32_t G = get_compile_time_arg_val(8);  // ring size
    constexpr uint32_t fwd_sem_id = get_compile_time_arg_val(9);
    constexpr uint32_t red_sem_id = get_compile_time_arg_val(10);
    constexpr uint32_t in0_on_reader = get_compile_time_arg_val(11);  // 1 = in1 RISC read slot 0 (skip our DRAM read)
    constexpr uint32_t in0ready_sem_id = get_compile_time_arg_val(12);
    constexpr uint32_t N_bpc = get_compile_time_arg_val(13);           // N-sub-blocks per core (reduction loops these)
    constexpr uint32_t redfree_sem_id = get_compile_time_arg_val(14);  // reverse credit for cb_reduce slot reuse
    constexpr uint32_t skip_in0 = get_compile_time_arg_val(15);  // ablation: skip the in0 DRAM read (isolate its cost)
    constexpr uint32_t in0_direct =
        get_compile_time_arg_val(16);  // 1 = each core reads its FULL [M-block,k-slice] directly (no ring/forward)
    constexpr uint32_t skip_fwd = get_compile_time_arg_val(
        17);  // ablation: skip the ring FORWARD (+recv waits); push garbage (isolate forward cost)
    constexpr uint32_t noreduce = get_compile_time_arg_val(
        18);  // ablation: no reduction chain (top writes its partial, non-top discards) -> isolate reduction comm
    constexpr uint32_t mshard = get_compile_time_arg_val(
        19);  // 1 = M-shard ring all-gather -> CONTIGUOUS [M,Kt_local] cb0 (read-once + deep-K)
    constexpr uint32_t in0share = get_compile_time_arg_val(
        20);  // 1 = N-slice shared in0: leader (nn==0) rings+forwards to siblings; receivers skip ring
    constexpr uint32_t share_valid_sem = get_compile_time_arg_val(21);  // leader -> receivers: in0 delivered
    constexpr uint32_t share_ready_sem = get_compile_time_arg_val(22);  // receivers -> leader: my cb0 is free
    constexpr uint32_t in0scatter = get_compile_time_arg_val(
        23);  // 1 = direct scatter all-gather (1 round of G-1 writes) instead of G-1 sequential ring rotations
    constexpr auto in0_args = TensorAccessorArgs<24>();
    constexpr auto out_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();

    uint32_t ai = 0;
    const uint32_t in0_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t out_addr = get_arg_val<uint32_t>(ai++);
    const uint32_t m0 = get_arg_val<uint32_t>(ai++);
    const uint32_t n0 = get_arg_val<uint32_t>(ai++);
    const uint32_t k_start = get_arg_val<uint32_t>(ai++);     // this slice's first K tile (global)
    const uint32_t ring_pos = get_arg_val<uint32_t>(ai++);    // 0..G-1
    const uint32_t fwd_next_x = get_arg_val<uint32_t>(ai++);  // ring next core (cyclic)
    const uint32_t fwd_next_y = get_arg_val<uint32_t>(ai++);
    const uint32_t red_next_x = get_arg_val<uint32_t>(ai++);  // reduction next core
    const uint32_t red_next_y = get_arg_val<uint32_t>(ai++);
    const uint32_t is_bottom = get_arg_val<uint32_t>(ai++);
    const uint32_t is_top = get_arg_val<uint32_t>(ai++);
    const uint32_t red_prev_x = get_arg_val<uint32_t>(ai++);  // reduction prev core (down; for reverse credit)
    const uint32_t red_prev_y = get_arg_val<uint32_t>(ai++);
    // in0-share (N-slice): role 0=none, 1=leader (rings+forwards), 2=receiver (recv only). Then:
    //   leader: [nsib, then nsib*(x,y)];  receiver: [leader_x, leader_y].
    uint32_t share_role = 0;
    if constexpr (in0share) {
        share_role = get_arg_val<uint32_t>(ai++);
    }

    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);
    constexpr uint32_t in0_cb = 0, out_cb = 2, cb_reduce = 7;
    constexpr uint32_t in0_blk = M_block * K_block;
    constexpr uint32_t in0_blk_bytes = in0_blk * tile_bytes;
    constexpr uint32_t shard_bytes = W * in0_blk_bytes;
    constexpr uint32_t out_blk = M_block * N_block;
    const uint32_t fwd_addr = get_semaphore(fwd_sem_id);
    volatile tt_l1_ptr uint32_t* fwd_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fwd_addr);
    const uint32_t red_addr = get_semaphore(red_sem_id);
    volatile tt_l1_ptr uint32_t* red_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red_addr);
    const uint32_t reduce_base = get_write_ptr(cb_reduce);  // capture BEFORE any cb_reduce use (drifts after recv)

    // phase 1 (M-SHARD RING): read-once in0 all-gather into a CONTIGUOUS [M_block, Kt_local] cb0 (deep-K).
    // Shard by M: core reads M-rows [ring_pos*Mw : +Mw] (Mw=M_block/G), all K, into cb0 at its M-offset (m-major
    // => contiguous). Shards rotate around the 8-bank ring (contiguous forwards, same M-offset on next). Every core
    // ends with the full [M_block, Kt_local]; compute uses kb=Kt_local, K_num_blocks=1 (deep). No redundant read.
    bool receiver = false;
    if constexpr (in0share) {
        receiver = (share_role == 2);
    }
    if (receiver) {
        // RECEIVER (N-slice nn>0): skip the ring; receive the k-slice from the leader SHARD-BY-SHARD (pipelined
        // with the leader's ring) so compute-feed isn't serialized behind the whole ring. Eliminates this core's
        // redundant DRAM read + ring participation.
        cb_reserve_back(in0_cb, K_num_blocks * in0_blk);
        const uint32_t rbase = get_write_ptr(in0_cb);
        const uint32_t lx = get_arg_val<uint32_t>(ai);  // leader coords appended after role
        const uint32_t ly = get_arg_val<uint32_t>(ai + 1);
        volatile tt_l1_ptr uint32_t* sv =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(share_valid_sem));
        for (uint32_t step = 0; step < G; ++step) {
            noc_semaphore_wait_min(sv, step + 1);  // leader's shard `step` is ready in its cb0
            uint32_t off = step * shard_bytes;     // PULL it from leader L1 (this idle writer RISC does the copy)
            noc_async_read(get_noc_addr(lx, ly, rbase + off), rbase + off, shard_bytes);
            noc_async_read_barrier();
            cb_push_back(in0_cb, W * in0_blk);
        }
    } else if constexpr (mshard) {
        constexpr uint32_t Mw = M_block / G;    // M-rows per shard
        constexpr uint32_t Kt_local = K_block;  // deep-K: kb == Kt_local (K_num_blocks==1)
        constexpr uint32_t shard_tiles = Mw * Kt_local;
        constexpr uint32_t shard_bytes_m = shard_tiles * tile_bytes;
        cb_reserve_back(in0_cb, M_block * Kt_local);
        const uint32_t base0 = get_write_ptr(in0_cb);
        for (uint32_t step = 0; step < G; ++step) {
            uint32_t g = (ring_pos + G - step) % G;     // shard (M-row-range) we hold this step
            uint32_t slot = base0 + g * shard_bytes_m;  // its contiguous M-offset in cb0
            if (step == 0) {
                uint32_t p = slot;
                for (uint32_t mr = 0; mr < Mw; ++mr) {
                    for (uint32_t k = 0; k < Kt_local; ++k) {
                        noc_async_read_page((m0 + ring_pos * Mw + mr) * Kt + (k_start + k), in0, p);
                        p += tile_bytes;
                    }
                }
                noc_async_read_barrier();
            } else {
                noc_semaphore_wait_min(fwd_ptr, step);  // prev forwarded shard g into our slot
            }
            if (step + 1 < G) {  // forward shard g to next's cb0[g] (same M-offset)
                uint64_t dst = get_noc_addr(fwd_next_x, fwd_next_y, slot);
                noc_async_write(slot, dst, shard_bytes_m);
                noc_semaphore_inc(get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr), 1);
            }
        }
        noc_async_write_barrier();
        cb_push_back(in0_cb, M_block * Kt_local);  // full contiguous [M_block, Kt_local] (deep-K)
    } else if constexpr (in0_direct) {
        cb_reserve_back(in0_cb, K_num_blocks * in0_blk);
        uint32_t p = get_write_ptr(in0_cb);
        for (uint32_t sb = 0; sb < K_num_blocks; ++sb) {
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t k = 0; k < K_block; ++k) {
                    if constexpr (!skip_in0) {
                        noc_async_read_page((m0 + m) * Kt + (k_start + sb * K_block + k), in0, p);
                    }
                    p += tile_bytes;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(in0_cb, K_num_blocks * in0_blk);
    } else {
        // phase 1: ring all-gather. cb0 = full k-slice (G slots of W blocks); reserve once, fill/forward/push.
        // in0-share LEADER: also forward each shard to the nn>0 siblings as it lands (pipelined with the ring).
        uint32_t leader_nsib = 0;
        if constexpr (in0share) {
            if (share_role == 1) {
                leader_nsib = get_arg_val<uint32_t>(ai);
            }
        }
        if constexpr (in0scatter) {
            // DIRECT SCATTER all-gather: read own shard, then write it to all G-1 peers' SAME slot concurrently
            // (1 round of G-1 async writes) instead of the ring's G-1 sequential rotations. Same bytes, ~1/(G-1)
            // rounds.
            cb_reserve_back(in0_cb, K_num_blocks * in0_blk);
            const uint32_t sbase = get_write_ptr(in0_cb);
            const uint32_t myslot = sbase + ring_pos * shard_bytes;
            {
                uint32_t p = myslot;
                for (uint32_t wb = 0; wb < W; ++wb) {
                    uint32_t sb = ring_pos * W + wb;
                    for (uint32_t m = 0; m < M_block; ++m) {
                        for (uint32_t k = 0; k < K_block; ++k) {
                            noc_async_read_page((m0 + m) * Kt + (k_start + sb * K_block + k), in0, p);
                            p += tile_bytes;
                        }
                    }
                }
            }
            noc_async_read_barrier();
            for (uint32_t s = 0; s < G - 1; ++s) {  // peer coords appended at ai (scatter is Ns==1 => no in0share args)
                uint32_t ox = get_arg_val<uint32_t>(ai + s * 2), oy = get_arg_val<uint32_t>(ai + 1 + s * 2);
                noc_async_write(myslot, get_noc_addr(ox, oy, myslot), shard_bytes);
            }
            noc_async_writes_flushed();
            for (uint32_t s = 0; s < G - 1; ++s) {
                uint32_t ox = get_arg_val<uint32_t>(ai + s * 2), oy = get_arg_val<uint32_t>(ai + 1 + s * 2);
                noc_semaphore_inc(get_noc_addr(ox, oy, fwd_addr), 1);
            }
            noc_semaphore_wait_min(fwd_ptr, G - 1);  // all peers scattered their shard to me
            noc_async_write_barrier();
            cb_push_back(in0_cb, K_num_blocks * in0_blk);
        } else {
            cb_reserve_back(in0_cb, K_num_blocks * in0_blk);
            const uint32_t base0 = get_write_ptr(in0_cb);
            for (uint32_t step = 0; step < G; ++step) {
                uint32_t slot = base0 + step * shard_bytes;
                if (step == 0) {
                    if constexpr (in0_on_reader) {
                        // the in1 RISC read our own shard into slot 0; wait for its signal (same-core shared L1)
                        volatile tt_l1_ptr uint32_t* rdy =
                            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(in0ready_sem_id));
                        noc_semaphore_wait(rdy, 1);
                    } else if constexpr (!skip_in0) {
                        // read own shard (shard index = ring_pos) into slot 0
                        uint32_t p = slot;
                        for (uint32_t wb = 0; wb < W; ++wb) {
                            uint32_t sb = ring_pos * W + wb;  // slice-block index of own shard
                            for (uint32_t m = 0; m < M_block; ++m) {
                                for (uint32_t k = 0; k < K_block; ++k) {
                                    noc_async_read_page((m0 + m) * Kt + (k_start + sb * K_block + k), in0, p);
                                    p += tile_bytes;
                                }
                            }
                        }
                        noc_async_read_barrier();
                    }
                } else if constexpr (!skip_fwd) {
                    noc_semaphore_wait_min(fwd_ptr, step);  // prev forwarded shard into our slot `step`
                }
                if (step + 1 < G && !skip_fwd) {  // forward this slot to next core's slot (step+1)
                    uint64_t dst = get_noc_addr(fwd_next_x, fwd_next_y, base0 + (step + 1) * shard_bytes);
                    noc_async_write(slot, dst, shard_bytes);
                    noc_semaphore_inc(get_noc_addr(fwd_next_x, fwd_next_y, fwd_addr), 1);
                }
                if constexpr (in0share) {
                    if (share_role ==
                        1) {  // LEADER: shard `step` is now in cb0; signal siblings so they PULL it (no data push)
                        noc_async_read_barrier();  // ensure this shard has landed in our L1 before siblings read it
                        for (uint32_t s = 0; s < leader_nsib; ++s) {
                            uint32_t sx = get_arg_val<uint32_t>(ai + 1 + s * 2);
                            uint32_t sy = get_arg_val<uint32_t>(ai + 2 + s * 2);
                            noc_semaphore_inc(get_noc_addr(sx, sy, get_semaphore(share_valid_sem)), 1);
                        }
                    }
                }
                cb_push_back(in0_cb, W * in0_blk);  // compute consumes this shard (W blocks)
            }
            noc_async_write_barrier();  // all ring + sibling forwards landed
        }  // end scatter-vs-ring
    }  // end ring (else in0_direct)

    // phase 2: split-K column reduction, looped over the N_bpc output blocks. cb_reduce holds 2 blocks;
    // red_recv (fwd) = prev sent block nb; redfree (reverse) = a core signals its prev that a cb_reduce slot
    // is free. Slot for block nb = reduce_base + (nb%2)*out_blk. compute (REDUCE_K) does copy/reduce_add.
    constexpr uint32_t out_blk_bytes = out_blk * tile_bytes;
    const uint32_t redfree_addr = get_semaphore(redfree_sem_id);
    volatile tt_l1_ptr uint32_t* redfree_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(redfree_addr);
    const uint64_t prev_redfree = get_noc_addr(red_prev_x, red_prev_y, redfree_addr);
    const uint64_t next_recv = get_noc_addr(red_next_x, red_next_y, red_addr);
    if constexpr (noreduce) {
        // ablation: NO reduction chain. Each core copied its partial to out_cb (is_bottom=1). Only the real top
        // writes DRAM (1 write/output, no redundant-write confound); non-top just discards. Isolates reduction comm.
        for (uint32_t nb = 0; nb < N_bpc; ++nb) {
            cb_wait_front(out_cb, out_blk);
            if (is_top) {
                uint32_t r = get_read_ptr(out_cb), n_off = n0 + nb * N_block;
                for (uint32_t m = 0; m < M_block; ++m) {
                    for (uint32_t n = 0; n < N_block; ++n) {
                        noc_async_write_page((m0 + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
                    }
                }
                noc_async_write_barrier();
            }
            cb_pop_front(out_cb, out_blk);
        }
        return;
    }
    for (uint32_t nb = 0; nb < N_bpc; ++nb) {
        if (!is_bottom) {
            cb_reserve_back(cb_reduce, out_blk);      // wait our compute freed slot (nb-2)
            noc_semaphore_inc(prev_redfree, 1);       // tell prev: our slot (nb%2) is free for block nb
            noc_semaphore_wait_min(red_ptr, nb + 1);  // prev forwarded block nb into it
            cb_push_back(cb_reduce, out_blk);         // compute reduce_add's it -> out_cb, pops cb_reduce
        }
        cb_wait_front(out_cb, out_blk);  // compute produced reduced block nb
        uint32_t r = get_read_ptr(out_cb);
        if (!is_top) {
            noc_semaphore_wait_min(redfree_ptr, nb + 1);  // next signalled its slot (nb%2) is free
            uint64_t dst = get_noc_addr(red_next_x, red_next_y, reduce_base + (nb % 2) * out_blk_bytes);
            noc_async_write(r, dst, out_blk_bytes);
            noc_async_write_barrier();
            noc_semaphore_inc(next_recv, 1);  // block nb delivered
        } else {
            uint32_t n_off = n0 + nb * N_block;
            for (uint32_t m = 0; m < M_block; ++m) {
                for (uint32_t n = 0; n < N_block; ++n) {
                    noc_async_write_page((m0 + m) * Nt + (n_off + n), out, r + (m * N_block + n) * tile_bytes);
                }
            }
            noc_async_write_barrier();
        }
        cb_pop_front(out_cb, out_blk);
    }
}
