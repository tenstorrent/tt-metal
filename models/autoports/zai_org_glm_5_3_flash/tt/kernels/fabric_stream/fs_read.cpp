// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Stage 7a FabricStream, lane NCRISC (tt/fabric_stream.py; ported into the DSA shard tree with FS_WAIT: the producers'
// semaphore gate before the first read): streams this lane's share of the source rows (DRAM tiles of
// x from chunk src_base on) into the staging CB in fixed NB-page batches (NB divides the pages per chunk and the CB
// holds a whole number of batches: every push lands inside the CB), for the lane's BRISC (ag_send.cpp) to write and
// send. FS_SCATTER (tt/fabric_stream.py FabricStreamScatter): after the wait count, n_dst then the n_dst source chunk
// bases (block q of x for destination q): the lane streams its share of each destination's block in turn, in the order
// fs_send.cpp sends them. FS_KEYPACK (rev 68, below): each staged chunk's candidate records -> one key page.
// FS_SUBPC / FS_SUBD (GLM53_DSA_SHARD_RSNP, the scatter only; fs_send.cpp): a chunk's sub-chunks of FS_SUBPC pages
// stage only their first FS_SUBD (data) pages, in order -- the pad pages are never read nor sent.
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

constexpr uint32_t PPC = get_compile_time_arg_val(0);  // pages per chunk
constexpr uint32_t NB = get_compile_time_arg_val(1);   // pages per batch
constexpr uint32_t CB_STAGE = get_compile_time_arg_val(2);
constexpr uint32_t TB = 2048;
#if defined(FS_SUBD)  // GLM53_DSA_SHARD_RSNP: fs_send.cpp's SUB_N / DPC / sub_col (the same staged order)
#if !defined(FS_SCATTER) || defined(FS_KEYPACK)
#error "FS_SUBD: the scatter's plain chunks only"
#endif
constexpr uint32_t SUB_N = PPC / FS_SUBPC;
constexpr uint32_t DPC = (SUB_N * FS_SUBD + 1) / 2 * 2;
static_assert(PPC % FS_SUBPC == 0 && FS_SUBD > 0 && FS_SUBD < FS_SUBPC, "sub-chunks: data pages, then pad");
static_assert(DPC % NB == 0, "a batch never straddles a chunk");
inline uint32_t sub_col(uint32_t s) {
    const uint32_t k = s / FS_SUBD;
    return k < SUB_N ? k * FS_SUBPC + (s - k * FS_SUBD) : PPC - FS_SUBPC + FS_SUBD;
}
#else
static_assert(PPC % NB == 0, "a batch never straddles a chunk");
#endif
#if defined(FS_KEYPACK)
// rev 68, the KEY PACK (tt/fabric_stream.py keypack=; the 8x4 DSA shard's AG round 1, kernels/mla/dsa_shard.hpp AGK):
// a staged batch is ONE x chunk (NB = PPC pages: FS_KP_QPAGES q-piece pages, then the candidate bytes = a KP_REC_HDR-
// byte header whose word 0 is the record count, then KP_REC_BYTES-byte records whose word 0 is the key), rewritten in
// place before the push: page FS_KP_QPAGES = the n = min(count, KP_CAND_MAX) keys in record order as words 0 .. n - 1,
// zeros after (stale records of earlier launches never travel); fs_send.cpp ships pages [0, FS_KEYPACK) of the batch.
// The record format mirrors dsa_shard.hpp (tt/shard_plan.py asserts these constants against it).
constexpr uint32_t KP_REC_HDR = 16;
constexpr uint32_t KP_REC_BYTES = 16;
constexpr uint32_t KP_CAND_MAX = 512;
static_assert(NB == PPC, "keypack: one x chunk per staged batch");
static_assert(KP_CAND_MAX * 4 == TB, "keypack: one key page holds a full list");
static_assert(
    FS_KEYPACK == FS_KP_QPAGES + 1 && FS_KEYPACK % 2 == 0, "keypack: the y chunk = q pages + key page, pairs");
static_assert(FS_KP_QPAGES * TB + KP_REC_HDR + KP_CAND_MAX * KP_REC_BYTES <= PPC * TB, "keypack: records in the chunk");
#if defined(FS_SCATTER)
#error "FS_KEYPACK: the all-gather only"
#endif
#endif

void kernel_main() {
    constexpr auto x_args = TensorAccessorArgs<3>();
    const uint32_t x_addr = get_arg_val<uint32_t>(0);
    const uint32_t c_lo = get_arg_val<uint32_t>(1);  // this lane's first chunk (of the block)
    const uint32_t c_n = get_arg_val<uint32_t>(2);
    const uint32_t src_base = get_arg_val<uint32_t>(3);  // the block's first chunk (row tile) in x
    const auto x = TensorAccessor(x_args, x_addr);
#if defined(FS_WAIT)  // tt/fabric_stream.py wait=: the block is complete once FS_WAIT reached runtime arg 4
    const uint32_t wait_count = get_arg_val<uint32_t>(4);
    volatile tt_l1_ptr uint32_t* wsem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(FS_WAIT));
#if defined(FS_WAIT2)  // rev 67: two producer fields -- bits [0, 16) gate the first destinations, [16, 32) the rest
    while ((*wsem & 0xFFFFu) < wait_count) {
        invalidate_l1_cache();
    }
#else
    noc_semaphore_wait_min(wsem, wait_count);
#endif
#endif
#if defined(FS_SUBD)
    const uint32_t nbat = c_n * DPC / NB;
#else
    const uint32_t nbat = c_n * PPC / NB;
#endif
#if defined(FS_SCATTER)
#if defined(FS_WAIT)
    constexpr uint32_t SC0 = 5;
#else
    constexpr uint32_t SC0 = 4;
#endif
    const uint32_t n_dst = get_arg_val<uint32_t>(SC0);
#if defined(FS_WAIT2)  // tt/fabric_stream.py FabricStreamScatter first= / wait2= (rev 65, the shard's RS1 early): after
    // the bases, (k_split, wait2) -- destinations k >= k_split are streamed only once FS_WAIT's field [16, 32) reached
    // wait2 (rev 67: one count of both fields opened the first gate on the other gate's producers)
#if !defined(FS_WAIT)
#error "FS_WAIT2 needs FS_WAIT"
#endif
    const uint32_t k_split = get_arg_val<uint32_t>(SC0 + 1 + n_dst);
    const uint32_t wait2 = get_arg_val<uint32_t>(SC0 + 2 + n_dst);
#endif
    for (uint32_t k = 0; k < n_dst; k++) {
        const uint32_t base = get_arg_val<uint32_t>(SC0 + 1 + k);
#if defined(FS_WAIT2)
        if (k == k_split) {
            while ((*wsem >> 16) < wait2) {
                invalidate_l1_cache();
            }
        }
#endif
#else
    {
        const uint32_t base = src_base;
#endif
        for (uint32_t b = 0; b < nbat; b++) {
            cb_reserve_back(CB_STAGE, NB);
            const uint32_t wp = get_write_ptr(CB_STAGE);
#if defined(FS_SUBD)  // staged page idx of the lane: chunk idx / DPC, its page sub_col(idx % DPC)
            for (uint32_t i = 0; i < NB; i++) {
                const uint32_t idx = b * NB + i;
                noc_async_read_page((base + c_lo + idx / DPC) * PPC + sub_col(idx % DPC), x, wp + i * TB);
            }
#else
            for (uint32_t i = 0; i < NB; i++) {
                noc_async_read_page((base + c_lo) * PPC + b * NB + i, x, wp + i * TB);
            }
#endif
            noc_async_read_barrier();
#if defined(FS_KEYPACK)
            {
                invalidate_l1_cache();  // the landed pages, not stale cached words
                volatile tt_l1_ptr uint32_t* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(wp + FS_KP_QPAGES * TB);
                const uint32_t n = w[0] < KP_CAND_MAX ? w[0] : KP_CAND_MAX;
                for (uint32_t e = 0; e < n; e++) {  // in place: word e < record e's key word (KP_REC_HDR + e 16) / 4
                    w[e] = w[(KP_REC_HDR + e * KP_REC_BYTES) / 4];
                }
                for (uint32_t e = n; e < KP_CAND_MAX; e++) {
                    w[e] = 0;
                }
                invalidate_l1_cache();  // the stores land before fs_send.cpp's NoC reads them
            }
#endif
            cb_push_back(CB_STAGE, NB);
        }
    }
}
