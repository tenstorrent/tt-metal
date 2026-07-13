// SPDX-License-Identifier: Apache-2.0
// Regime-A ring all-gather in1 reader. Reads this core's bank N-band for its k-slice, but in ROTATED shard
// order (shard ring_pos, ring_pos-1, ... wrapping) so it matches the in0 blocks arriving via the ring
// (in0[k] must pair with in1[k]; the matmul sum is commutative so any order works as long as they match).
// Each shard = W contiguous in1 blocks (16 KB bursts, triple-TRID continues across shards). G = ring size.
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t K_block = get_compile_time_arg_val(0);  // kb
    constexpr uint32_t N_block = get_compile_time_arg_val(1);  // N_band
    constexpr uint32_t W = get_compile_time_arg_val(2);        // in1 blocks per shard
    constexpr uint32_t G = get_compile_time_arg_val(3);        // ring size (banks)
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t read_in0 = get_compile_time_arg_val(5);  // 1 = also read this core's in0 shard (same NoC as in1)
    constexpr uint32_t M_block = get_compile_time_arg_val(6);
    constexpr uint32_t Kt = get_compile_time_arg_val(7);
    constexpr uint32_t in0ready_sem_id = get_compile_time_arg_val(8);
    constexpr uint32_t in0_order = get_compile_time_arg_val(9);  // 0 before in1, 1 after in1, 2 interleaved
    constexpr uint32_t N_bpc = get_compile_time_arg_val(10);     // N-sub-blocks per core (1 = full N_band)
    constexpr uint32_t N_band = get_compile_time_arg_val(11);    // full bank width (stride for sub-band reads)
    constexpr uint32_t contig_nsb = get_compile_time_arg_val(
        12);  // 1 = read each sub-band contiguously (layout-optimal ceiling; valid only for constant inputs)
    constexpr uint32_t force_strided =
        get_compile_time_arg_val(13);  // 1 = strided sub-band read even at N_bpc==1 (nsring: full-K N-slice)
    constexpr uint32_t skip_in1 =
        get_compile_time_arg_val(14);  // ablation: feed compute without reading in1 (isolate compute/reduction)
    // M-split (milestone 2): the m==0 reader of each (b,k) group reads in1 from DRAM ONCE and forwards it to the
    // Sm-1 M-slaves (fanout Sm). role (runtime, differs per core): 0=slave (recv), 1=reader (read+fwd), 2=solo.
    constexpr uint32_t in1valid_sem = get_compile_time_arg_val(15);  // reader->slaves "block delivered"
    constexpr uint32_t in1ready_sem = get_compile_time_arg_val(16);  // slaves->reader "cb1 slot free"
    constexpr uint32_t in1mcast = get_compile_time_arg_val(17);      // 1 = forward via ONE mcast (else Sm-1 unicasts)
    constexpr uint32_t in0_direct =
        get_compile_time_arg_val(18);  // 1 = DEEP-K: no ring; in1 read contiguous [K_block,N_sub] per nb (kb=Kt_local)
    constexpr auto in0_args = TensorAccessorArgs<19>();

    const uint32_t in1_addr = get_arg_val<uint32_t>(0);
    const uint32_t bank_id = get_arg_val<uint32_t>(1);
    const uint32_t vc = get_arg_val<uint32_t>(2);
    const uint32_t slice_off = get_arg_val<uint32_t>(3);  // byte offset of the k-slice start within the bank
    const uint32_t ring_pos = get_arg_val<uint32_t>(4);   // this core's ring position
    const uint32_t in0_addr = get_arg_val<uint32_t>(5);
    const uint32_t m0 = get_arg_val<uint32_t>(6);
    const uint32_t k_start = get_arg_val<uint32_t>(7);
    const uint32_t n_base = get_arg_val<uint32_t>(8);  // nsring: this core's sub-band N-offset (tiles) within its bank
    const uint32_t mrole = get_arg_val<uint32_t>(9);   // 0=slave, 1=reader(read+fwd), 2=solo (no M-split)
    // M-split forward peers (arg 10 = count; then count*(x,y)). reader: the Sm-1 slaves; slave: the 1 reader.
    const uint32_t mpeers = get_arg_val<uint32_t>(10);

    constexpr uint32_t in0_cb = 0, in1_cb = 1;
    constexpr uint32_t in0_blk_tiles = M_block * K_block;
    const auto in0 = TensorAccessor(in0_args, in0_addr, tile_bytes);
    const uint32_t in0_base = get_write_ptr(in0_cb);  // cb0 base (slot 0), uniform layout
    volatile tt_l1_ptr uint32_t* rdy = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(in0ready_sem_id));
    // read in0 block wb (of this core's own shard) into cb0 slot0 region (no barrier here; caller barriers)
    auto read_in0_blk = [&](uint32_t wb) {
        uint32_t sb = ring_pos * W + wb;
        uint32_t p = in0_base + wb * in0_blk_tiles * tile_bytes;
        for (uint32_t m = 0; m < M_block; ++m) {
            for (uint32_t k = 0; k < K_block; ++k) {
                noc_async_read_page((m0 + m) * Kt + (k_start + sb * K_block + k), in0, p);
                p += tile_bytes;
            }
        }
    };

    // in0 order=before: read the whole in0 shard on the in1 RISC, signal, then stream in1.
    if constexpr (read_in0 && in0_order == 0) {
        for (uint32_t wb = 0; wb < W; ++wb) {
            read_in0_blk(wb);
        }
        noc_async_read_barrier();
        *rdy = 1;
    }

    constexpr uint32_t in1_blk = K_block * N_block;  // N_block = full N_band (N_bpc==1) or Nsb (N_bpc>1)
    constexpr uint32_t in1_blk_bytes = in1_blk * tile_bytes;

    // M-split SLAVE (mrole==0): don't read in1 from DRAM; receive it (forwarded by the m==0 reader) into cb1.
    if (mrole == 0) {
        volatile tt_l1_ptr uint32_t* valid =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(in1valid_sem));
        uint64_t reader_ready =
            get_noc_addr(get_arg_val<uint32_t>(11), get_arg_val<uint32_t>(12), get_semaphore(in1ready_sem));
        const uint32_t nblk = in0_direct ? N_bpc : (N_bpc * G * W);  // deep-K: 1 block/nb; ring: G*W blocks/nb
        for (uint32_t b = 0; b < nblk; ++b) {
            cb_reserve_back(in1_cb, in1_blk);
            noc_semaphore_inc(reader_ready, 1);    // our cb1 slot is free -> reader may forward block b
            noc_semaphore_wait_min(valid, b + 1);  // reader forwarded block b into it
            cb_push_back(in1_cb, in1_blk);
        }
        return;
    }
    // M-split READER (mrole==1): after reading each in1 block, forward it to the Sm-1 slaves (fanout). `mfwd`
    // waits all slaves freed the slot (in1ready), unicasts the block to each slave's cb1 (uniform L1), incs valid.
    const uint32_t in1valid_addr = get_semaphore(in1valid_sem);
    const uint32_t in1ready_addr = get_semaphore(in1ready_sem);
    volatile tt_l1_ptr uint32_t* in1ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1ready_addr);
    uint32_t mbc = 0;  // reader's forwarded-block counter
    auto mfwd = [&](uint32_t w1) {
        if (mrole != 1) {
            return;
        }
        noc_semaphore_wait_min(in1ready, (mbc + 1) * mpeers);  // every slave freed slot mbc
        if constexpr (in1mcast) {
            // ONE multicast to the Sm-1 slave strip (rect args 11..14; ndest = mpeers). Slave cb1 = same L1 offset.
            uint32_t x0 = get_arg_val<uint32_t>(11), y0 = get_arg_val<uint32_t>(12), x1 = get_arg_val<uint32_t>(13),
                     y1 = get_arg_val<uint32_t>(14);
            noc_async_write_multicast(w1, get_noc_multicast_addr(x0, y0, x1, y1, w1), in1_blk_bytes, mpeers);
            noc_async_writes_flushed();
            noc_semaphore_inc_multicast(get_noc_multicast_addr(x0, y0, x1, y1, in1valid_addr), 1, mpeers);
        } else {
            for (uint32_t s = 0; s < mpeers; ++s) {
                uint32_t sx = get_arg_val<uint32_t>(11 + s * 2), sy = get_arg_val<uint32_t>(12 + s * 2);
                noc_async_write(w1, get_noc_addr(sx, sy, w1), in1_blk_bytes);  // slave cb1 same L1 offset (uniform)
            }
            noc_async_writes_flushed();
            for (uint32_t s = 0; s < mpeers; ++s) {
                uint32_t sx = get_arg_val<uint32_t>(11 + s * 2), sy = get_arg_val<uint32_t>(12 + s * 2);
                noc_semaphore_inc(get_noc_addr(sx, sy, in1valid_addr), 1);
            }
        }
        ++mbc;
    };

    if constexpr (skip_in1) {  // ablation: hand compute empty in1 blocks (no DRAM read) to isolate compute+reduction
        const uint32_t nblk = in0_direct ? N_bpc : (N_bpc * G * W);
        for (uint32_t b = 0; b < nblk; ++b) {
            cb_reserve_back(in1_cb, in1_blk);
            cb_push_back(in1_cb, in1_blk);
        }
    } else if constexpr (in0_direct) {
        // DEEP-K (no ring): in1 = one contiguous [K_block(=Kt_local), N_block(=N_sub)] block per N-sub-band, natural
        // K order (strided: N_sub tiles per k-row, stride N_band). Matches the direct [M,Kt_local] in0
        // (K_num_blocks=1).
        constexpr uint32_t seg_bytes = N_block * tile_bytes;
        for (uint32_t nb = 0; nb < N_bpc; ++nb) {
            cb_reserve_back(in1_cb, in1_blk);
            uint32_t w1 = get_write_ptr(in1_cb);
            for (uint32_t kr = 0; kr < K_block; ++kr) {
                uint32_t off = ((k_start + kr) * N_band + n_base + nb * N_block) * tile_bytes;
                noc_async_read(get_noc_addr_from_bank_id<true>(bank_id, in1_addr + off), w1, seg_bytes);
                w1 += seg_bytes;
            }
            noc_async_read_barrier();
            mfwd(get_write_ptr(in1_cb));  // M-split reader forwards this deep block to the Sm-1 slaves
            cb_push_back(in1_cb, in1_blk);
        }
    } else if constexpr (N_bpc == 1 && !force_strided) {
        // contiguous full-N_band read: 16KB bursts, triple-TRID, rotated shard order
        constexpr uint32_t MAXBURST = 16384;
        constexpr uint32_t ppb = in1_blk_bytes / MAXBURST;
        const uint32_t shard_bytes = W * in1_blk_bytes;
        const uint32_t total_blocks = G * W;
        uint64_t src_base = get_noc_addr_from_bank_id<true>(bank_id, in1_addr);
        noc_async_read_one_packet_set_state<true>(src_base, MAXBURST, vc);
        uint32_t in0_done = 0, blk = 0;
        for (uint32_t step = 0; step < G; ++step) {
            uint32_t s = (ring_pos + G - step) % G;
            uint32_t shard_off = slice_off + s * shard_bytes;
            for (uint32_t wb = 0; wb < W; ++wb) {
                if constexpr (read_in0 && in0_order == 2) {
                    if (in0_done < W && blk == in0_done * G) {
                        read_in0_blk(in0_done++);
                    }
                }
                cb_reserve_back(in1_cb, in1_blk);
                uint32_t w1 = get_write_ptr(in1_cb);
                const uint32_t trid = blk % 3 + 1;
                noc_async_read_set_trid(trid);
                uint32_t off = shard_off + wb * in1_blk_bytes;
                for (uint32_t pp = 0; pp < ppb; ++pp) {
                    noc_async_read_one_packet_with_state_with_trid(src_base, off, w1, trid);
                    off += MAXBURST;
                    w1 += MAXBURST;
                }
                if (blk >= 2) {
                    noc_async_read_barrier_with_trid((blk - 2) % 3 + 1);
                    cb_push_back(in1_cb, in1_blk);
                }
                ++blk;
            }
        }
        for (uint32_t b = (total_blocks >= 2 ? total_blocks - 2 : 0); b < total_blocks; ++b) {
            noc_async_read_barrier_with_trid(b % 3 + 1);
            cb_push_back(in1_cb, in1_blk);
        }
    } else {
        // N-sub-block (large Mt): for each N-sub-band nb, read its [kb, Nsb] in1 blocks in rotated shard
        // order. Strided: Nsb tiles per K-row, stride N_band. Matches compute (n_block outer, k inner) + ring.
        constexpr uint32_t seg_bytes = N_block * tile_bytes;  // Nsb tiles per K-row
        const uint32_t kt_local = G * W * K_block;            // k-rows in this slice
        for (uint32_t nb = 0; nb < N_bpc; ++nb) {
            for (uint32_t step = 0; step < G; ++step) {
                // rotated shard order matches the ring in0 delivery; contig diagnostic reads natural order
                // (constant inputs => any pairing is correct) so a sub-band is one sequential DRAM run.
                uint32_t s = contig_nsb ? step : (ring_pos + G - step) % G;
                for (uint32_t wb = 0; wb < W; ++wb) {
                    uint32_t kblk = s * W + wb;
                    cb_reserve_back(in1_cb, in1_blk);
                    uint32_t w1 = get_write_ptr(in1_cb);
                    for (uint32_t kr = 0; kr < K_block; ++kr) {
                        // strided (real layout): Nsb tiles at k-row `ktile`, stride N_band.
                        // contig (diagnostic, constant-input only): read as if in1 were sub-band-major
                        // [N_bpc, Kt_local, Nsb] so each sub-band is one contiguous run -> measures the
                        // layout-optimal DRAM ceiling for large-Mt N-sub-blocking.
                        uint32_t ktile = k_start + kblk * K_block + kr;
                        uint32_t off = contig_nsb ? ((nb * kt_local + (kblk * K_block + kr)) * N_block) * tile_bytes
                                                  : (ktile * N_band + n_base + nb * N_block) *
                                                        tile_bytes;  // n_base = nsring sub-band offset
                        noc_async_read(get_noc_addr_from_bank_id<true>(bank_id, in1_addr + off), w1, seg_bytes);
                        w1 += seg_bytes;
                    }
                    noc_async_read_barrier();
                    mfwd(get_write_ptr(in1_cb));  // M-split reader: forward this block to the Sm-1 slaves
                    cb_push_back(in1_cb, in1_blk);
                }
            }
        }
    }

    // in0 order=after: read the shard after the in1 stream; interleave: finish any remaining + barrier & signal.
    if constexpr (read_in0 && in0_order == 1) {
        for (uint32_t wb = 0; wb < W; ++wb) {
            read_in0_blk(wb);
        }
        noc_async_read_barrier();
        *rdy = 1;
    } else if constexpr (read_in0 && in0_order == 2 && N_bpc == 1) {
        // interleaved in0 is only wired for the contiguous (N_bpc==1) path; in0_done lives in that branch.
        for (uint32_t wb = 0; wb < W; ++wb) {
            read_in0_blk(wb);
        }
        noc_async_read_barrier();
        *rdy = 1;
    }
}
