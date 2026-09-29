// SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// sparse_sdpa_tiled reader (NCRISC, NOC 0).  Per tile of TQ query rows x H heads: the Q rows into cb_q_rm in
// (head, query) order, the tile's positions and block-id rows into reader scratch, then the three union passes
// (mark, compact with the seed prefix, membership), the control messages to compute (chunk count) and to the
// writer (U, chunk count, t0), and per chunk the K/V row gather of the union's blocks: the writer gathers rows
// [0, split) on its NoC, this kernel rows [split, valid) on NOC 0 into the same reserved cb_k_rm pages (the
// request / ack handshake of sparse_sdpa), then the padding rows [valid, k_chunk) are zero-filled.
//
// Producer contract on block_ids (uint32 [1, 1, S, IDS]): row j's first min(IDS, complete_j) entries are the
// selected complete blocks (complete_j = (p_j + 1) >> 2 for position p_j); an id at or past complete_j is
// skipped here (a masked slot, or a contract violation), so no future block can be gathered.  Every row gets a
// seed block in chunk 0 (its first valid id, or its own incomplete block) so the running max is finite from
// chunk 0 on; SEED = 0 disables the prefix for the negative test.
//
// Named compile-time args: TQ H S CB BT MAX_BLOCKS U_MAX IDS Q_ROW_BYTES K_ROW_BYTES RING_DEPTH SPLIT_NUM SPLIT_DEN
// SEED and the CB ids CB_Q_RM CB_K_RM CB_IDS CB_POS CB_BITMAP CB_SEEDMAP CB_RANK CB_UNION CB_MEMBER CB_CTRL
// CB_KREQ CB_KACK CB_TILECTL.  Positional compile-time args: TensorAccessorArgs of q, kv, block_ids, positions.
// Runtime args: 0 q, 1 kv, 2 block_ids, 3 positions addresses, 4 tile_start, 5 tile_count.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/noc_traits.h"
#include <ttnn/operations/pool/device/kernels/experimental_device_api.hpp>
#include "common.h"
#include "../../kernels/zones.h"

namespace {

// One 1 KB row per selected token: chunk row p is token p % BT of the block at union slot slot0 + p / BT.
// Depth-D transaction-id ring (sparse_sdpa_gather.hpp's trid_ring_gather over a block list).
template <uint32_t D, uint32_t BT, typename Accessor>
FORCE_INLINE void union_gather(
    Noc& noc,
    const Accessor& kv,
    uint32_t dst_l1,
    volatile tt_l1_ptr uint32_t* union_ptr,
    uint32_t slot0,
    uint32_t lo,
    uint32_t hi,
    uint32_t row_bytes) {
    const UnicastEndpoint local_l1;
    const uint32_t cnt = hi - lo;
    for (uint32_t i = 0; i < cnt; ++i) {
        const uint32_t p = lo + i;
        const uint32_t trid = (i % D) + 1;
        if (i >= D) {
            experimental::async_read_barrier_with_trid(noc, trid);
        }
        experimental::set_read_trid(noc, trid);
        const uint32_t page = union_ptr[slot0 + p / BT] * BT + p % BT;
        noc.async_read(kv, local_l1, row_bytes, {.page_id = page}, {.addr = dst_l1 + p * row_bytes});
    }
    const uint32_t to_drain = (cnt < D) ? cnt : D;
    for (uint32_t d = 0; d < to_drain; ++d) {
        experimental::async_read_barrier_with_trid(noc, ((cnt - to_drain + d) % D) + 1);
    }
    experimental::set_read_trid(noc, 0);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t TQ = get_named_compile_time_arg_val("TQ");
    constexpr uint32_t H = get_named_compile_time_arg_val("H");
    constexpr uint32_t S = get_named_compile_time_arg_val("S");
    constexpr uint32_t CB = get_named_compile_time_arg_val("CB");
    constexpr uint32_t BT = get_named_compile_time_arg_val("BT");
    constexpr uint32_t MAX_BLOCKS = get_named_compile_time_arg_val("MAX_BLOCKS");
    constexpr uint32_t U_MAX = get_named_compile_time_arg_val("U_MAX");
    constexpr uint32_t IDS = get_named_compile_time_arg_val("IDS");
    constexpr uint32_t Q_ROW_BYTES = get_named_compile_time_arg_val("Q_ROW_BYTES");
    constexpr uint32_t K_ROW_BYTES = get_named_compile_time_arg_val("K_ROW_BYTES");
    constexpr uint32_t RING_DEPTH = get_named_compile_time_arg_val("RING_DEPTH");
    constexpr uint32_t SPLIT_NUM = get_named_compile_time_arg_val("SPLIT_NUM");
    constexpr uint32_t SPLIT_DEN = get_named_compile_time_arg_val("SPLIT_DEN");
    constexpr bool SEED = get_named_compile_time_arg_val("SEED") != 0;
    constexpr uint32_t DEBUG_STAGE = get_named_compile_time_arg_val("DEBUG_STAGE");  // 5 / 6: study, no K/V reads
    constexpr uint32_t cb_q_rm = get_named_compile_time_arg_val("CB_Q_RM");
    constexpr uint32_t cb_k_rm = get_named_compile_time_arg_val("CB_K_RM");
    constexpr uint32_t cb_ids = get_named_compile_time_arg_val("CB_IDS");
    constexpr uint32_t cb_pos = get_named_compile_time_arg_val("CB_POS");
    constexpr uint32_t cb_bitmap = get_named_compile_time_arg_val("CB_BITMAP");
    constexpr uint32_t cb_seedmap = get_named_compile_time_arg_val("CB_SEEDMAP");
    constexpr uint32_t cb_rank = get_named_compile_time_arg_val("CB_RANK");
    constexpr uint32_t cb_union = get_named_compile_time_arg_val("CB_UNION");
    constexpr uint32_t cb_member = get_named_compile_time_arg_val("CB_MEMBER");
    constexpr uint32_t cb_ctrl = get_named_compile_time_arg_val("CB_CTRL");
    constexpr uint32_t cb_kreq = get_named_compile_time_arg_val("CB_KREQ");
    constexpr uint32_t cb_kack = get_named_compile_time_arg_val("CB_KACK");
    constexpr uint32_t cb_tilectl = get_named_compile_time_arg_val("CB_TILECTL");

    constexpr uint32_t R = TQ * H;
    constexpr uint32_t k_chunk = CB * BT;
    constexpr uint32_t words = (MAX_BLOCKS + 31) / 32;
    static_assert(TQ <= 32, "the membership word holds one bit per query of the tile");
    static_assert(CB >= TQ, "chunk 0 must hold every row's seed block");

    constexpr auto q_args = TensorAccessorArgs<0>();
    constexpr auto kv_args = TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
    constexpr auto ids_args = TensorAccessorArgs<kv_args.next_compile_time_args_offset()>();
    constexpr auto pos_args = TensorAccessorArgs<ids_args.next_compile_time_args_offset()>();

    const uint32_t q_addr = get_arg_val<uint32_t>(0);
    const uint32_t kv_addr = get_arg_val<uint32_t>(1);
    const uint32_t ids_addr = get_arg_val<uint32_t>(2);
    const uint32_t pos_addr = get_arg_val<uint32_t>(3);
    const uint32_t tile_start = get_arg_val<uint32_t>(4);
    const uint32_t tile_count = get_arg_val<uint32_t>(5);

    Noc noc;
    experimental::CB q_cb(cb_q_rm), k_cb(cb_k_rm), ids_cb(cb_ids), pos_cb(cb_pos);
    experimental::CB bitmap_cb(cb_bitmap), seedmap_cb(cb_seedmap), rank_cb(cb_rank), union_cb(cb_union);
    experimental::CB member_cb(cb_member), ctrl_cb(cb_ctrl), kreq_cb(cb_kreq), kack_cb(cb_kack), tilectl_cb(cb_tilectl);
    const auto q = TensorAccessor(q_args, q_addr);
    const auto kv = TensorAccessor(kv_args, kv_addr);
    const auto ids_acc = TensorAccessor(ids_args, ids_addr);
    const auto pos_acc = TensorAccessor(pos_args, pos_addr);

    // Reader scratch: reserved once, never pushed; the writer reads union / member / positions at the same
    // addresses (its copy of the CB interface starts at the same fifo base and never advances).
    ids_cb.reserve_back(1);
    pos_cb.reserve_back(1);
    bitmap_cb.reserve_back(1);
    seedmap_cb.reserve_back(1);
    rank_cb.reserve_back(1);
    union_cb.reserve_back(1);
    member_cb.reserve_back(1);
    volatile tt_l1_ptr uint32_t* ids = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ids_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* pos = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pos_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* bitmap = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bitmap_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* seedmap = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(seedmap_cb.get_write_ptr());
    volatile tt_l1_ptr uint16_t* rank = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(rank_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* uni = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(union_cb.get_write_ptr());
    volatile tt_l1_ptr uint32_t* member = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(member_cb.get_write_ptr());

    uint32_t complete_q[32];
    uint32_t n_q[32];

    for (uint32_t tile = tile_start; tile < tile_start + tile_count; ++tile) {
        FUSED_ZONE("fz_ss_r_tile");
        const uint32_t t0 = tile * TQ;

        // Q rows in (head, query) order; positions; block-id rows.
        q_cb.reserve_back(R);
        for (uint32_t h = 0; h < H; ++h) {
            for (uint32_t qi = 0; qi < TQ; ++qi) {
                noc.async_read(
                    q, q_cb, Q_ROW_BYTES, {.page_id = h * S + t0 + qi}, {.offset_bytes = (h * TQ + qi) * Q_ROW_BYTES});
            }
        }
        noc.async_read(
            pos_acc, pos_cb, TQ * sizeof(uint32_t), {.page_id = 0, .offset_bytes = t0 * sizeof(uint32_t)}, {});
        for (uint32_t qi = 0; qi < TQ; ++qi) {
            noc.async_read(
                ids_acc,
                ids_cb,
                IDS * sizeof(uint32_t),
                {.page_id = t0 + qi},
                {.offset_bytes = qi * IDS * sizeof(uint32_t)});
        }
        noc.async_read_barrier();
        q_cb.push_back(R);

        // Pass 1 (mark): the valid ids of every row, each row's seed, the incomplete diagonal blocks.
        for (uint32_t w = 0; w < words; ++w) {
            bitmap[w] = 0;
            seedmap[w] = 0;
        }
        for (uint32_t qi = 0; qi < TQ; ++qi) {
            const uint32_t p = pos[qi];
            // clamped to the cache: a position past it (outside the model's contract) must not index the bitmap or
            // the cache beyond their ends (review F2, 2026-09-25)
            const uint32_t complete = ((p + 1) >> 2) < MAX_BLOCKS ? ((p + 1) >> 2) : MAX_BLOCKS;
            const uint32_t tail = (p + 1) & 3;
            const uint32_t n = complete < IDS ? complete : IDS;
            complete_q[qi] = complete;
            n_q[qi] = n;
            volatile tt_l1_ptr uint32_t* row = ids + qi * IDS;
            uint32_t seed = sst::NO_BLOCK;
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t b = row[k];
                if (b < complete) {
                    bitmap[b >> 5] |= 1u << (b & 31);
                    if (seed == sst::NO_BLOCK) {
                        seed = b;
                    }
                }
            }
            if (tail > 0 && complete < MAX_BLOCKS) {
                // The own incomplete block: its first `tail` tokens are visible (band rule 1); inside the cache
                // whenever the position is (complete < T / BT when tail > 0 and p < T).
                bitmap[complete >> 5] |= 1u << (complete & 31);
            }
            if constexpr (SEED) {
                if (seed == sst::NO_BLOCK) {
                    // a row without a complete block (p <= 2) seeds with its diagonal block; a position past the
                    // cache has none inside it and seeds with the last block
                    seed = complete < MAX_BLOCKS ? complete : MAX_BLOCKS - 1;
                }
                bitmap[seed >> 5] |= 1u << (seed & 31);
                seedmap[seed >> 5] |= 1u << (seed & 31);
            }
        }

        // Pass 2 (compact): the distinct seeds first (ascending), then every other marked block ascending;
        // rank[w] = slots before word w's non-seed bits.
        uint32_t U = 0;
        if constexpr (SEED) {
            for (uint32_t w = 0; w < words; ++w) {
                uint32_t sw = seedmap[w];
                while (sw) {
                    const uint32_t bit = sst::ctz32(sw);
                    uni[U++] = 32 * w + bit;
                    sw &= sw - 1;
                }
            }
        }
        const uint32_t n_seeds = U;
        for (uint32_t w = 0; w < words; ++w) {
            rank[w] = static_cast<uint16_t>(U);
            uint32_t rest = bitmap[w] & ~seedmap[w];
            while (rest) {
                const uint32_t bit = sst::ctz32(rest);
                uni[U++] = 32 * w + bit;
                rest &= rest - 1;
            }
        }
        ASSERT(U <= U_MAX);
        const uint32_t n_chunks = (U + CB - 1) / CB;
        // Clear the membership words of every slot the chunks will read (the padding slots of the last chunk
        // included: they carry no member and the band gives them the floor).
        for (uint32_t s = 0; s < n_chunks * CB; ++s) {
            member[s] = 0;
        }

        // Pass 3 (membership): bit q of member[slot] = query q selected the block at that slot.
        for (uint32_t qi = 0; qi < TQ; ++qi) {
            const uint32_t complete = complete_q[qi];
            const uint32_t n = n_q[qi];
            volatile tt_l1_ptr uint32_t* row = ids + qi * IDS;
            const uint32_t qbit = 1u << qi;
            for (uint32_t k = 0; k < n; ++k) {
                const uint32_t b = row[k];
                if (b >= complete) {
                    continue;
                }
                const uint32_t w = b >> 5;
                const uint32_t bit = 1u << (b & 31);
                uint32_t slot;
                if (seedmap[w] & bit) {
                    slot = 0;
                    while (uni[slot] != b) {
                        ++slot;  // at most n_seeds - 1 steps
                    }
                } else {
                    slot = rank[w] + sst::popcount32(bitmap[w] & ~seedmap[w] & (bit - 1));
                }
                member[slot] |= qbit;
            }
        }

        // Control: chunk count to compute; U / chunks / t0 to the writer (it reads union, member and the
        // positions from this kernel's scratch once this message lands).
        ctrl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ctrl_cb.get_write_ptr());
            m[sst::ctrl::N_CHUNKS] = n_chunks;
        }
        ctrl_cb.push_back(1);
        tilectl_cb.reserve_back(1);
        {
            volatile tt_l1_ptr uint32_t* m = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tilectl_cb.get_write_ptr());
            m[sst::tilectl::U] = U;
            m[sst::tilectl::N_CHUNKS] = n_chunks;
            m[sst::tilectl::T0] = t0;
            m[sst::tilectl::N_SEEDS] = n_seeds;
        }
        tilectl_cb.push_back(1);

        // The union's K/V rows, one chunk of CB blocks at a time, split over both NoCs.
        for (uint32_t c = 0; c < n_chunks; ++c) {
            FUSED_ZONE("fz_ss_r_gather");
            const uint32_t slot0 = c * CB;
            const uint32_t n_slots = (U - slot0) < CB ? (U - slot0) : CB;
            const uint32_t valid = n_slots * BT;
            const uint32_t split = (valid * SPLIT_NUM) / SPLIT_DEN;
            k_cb.reserve_back(k_chunk);
            const uint32_t dst = k_cb.get_write_ptr();
            kreq_cb.reserve_back(1);
            {
                volatile tt_l1_ptr uint32_t* m =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(kreq_cb.get_write_ptr());
                m[sst::kreq::SLOT0] = slot0;
                m[sst::kreq::SPLIT] = split;
                m[sst::kreq::IS_LAST] = (c == n_chunks - 1) ? 1u : 0u;
                m[sst::kreq::DST_L1] = dst;
            }
            kreq_cb.push_back(1);
            if constexpr (DEBUG_STAGE != 5 && DEBUG_STAGE < 6) {
                union_gather<RING_DEPTH, BT>(noc, kv, dst, uni, slot0, split, valid, K_ROW_BYTES);
            }
            kack_cb.wait_front(1);
            kack_cb.pop_front(1);
            if (valid < k_chunk) {
                noc.async_write_zeros(k_cb, (k_chunk - valid) * K_ROW_BYTES, {.offset_bytes = valid * K_ROW_BYTES});
                noc.write_zeros_l1_barrier();
            }
            k_cb.push_back(k_chunk);
        }
    }
}
