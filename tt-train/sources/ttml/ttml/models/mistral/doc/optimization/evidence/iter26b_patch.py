"""Iteration 26b: 8-row K/V chunks in the SDPA forward (QK and exp stages in DEST-sized passes, the score CB
double-buffered so the exp writes a fresh slot, the mask stamped before the push)."""
import re
import sys

M = "tt-train/sources/ttml/metal/ops/sdpa_fw/device"
K = f"{M}/kernels/compute/sdpa_fw_compute_kernel.cpp"
U = f"{M}/kernels/compute/sdpa_compute_utils.hpp"
F = f"{M}/sdpa_fw_program_factory.cpp"


def sub(path, old, new, count=1, regex=False):
    s = open(path).read()
    if regex:
        s2, n = re.subn(old, new, s, flags=re.S)
    else:
        n = s.count(old)
        s2 = s.replace(old, new)
    if n != count:
        print(f"FAILED {path}: {old[:80]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s2)


# ---------------- kernel
sub(
    K,
    """constexpr uint32_t scaler_bf16_bits =
    get_compile_time_arg_val(10);  // host-side RNE BF16 of scaler_bits, for the WH fused scale+exp
""",
    """constexpr uint32_t scaler_bf16_bits =
    get_compile_time_arg_val(10);  // host-side RNE BF16 of scaler_bits, for the WH fused scale+exp
constexpr uint32_t dst_size = get_compile_time_arg_val(11);  // DEST tiles per pass (4 with fp32 accumulation)
// The QK^T and exp stages work on the chunk in passes of kPass tiles (iteration 26: chunks of up to 8 K/V rows).
constexpr uint32_t kPass = (Sk_chunk_t < dst_size) ? Sk_chunk_t : dst_size;
""",
)
sub(
    K,
    """        matmul_block_init(
            cb_query,
            cb_key,
            /* transpose */ 1,
            /* ct_dim */ Sk_chunk_t,
            /* rt_dim */ 1,
            /* kt_dim */ qWt);
        tile_regs_acquire();
        {
            // One matmul_block call does ONE K-step (1 Q tile × ct_dim K tiles → ct_dim
            // accumulated outputs). Iterate qWt times to contract the full feature dim.
            uint32_t q_idx = 0;
            uint32_t k_idx = 0;
            for (uint32_t feat = 0; feat < qWt; ++feat) {
                matmul_block(
                    cb_query,
                    cb_key,
                    /* in0 (Q) tile_idx */ q_idx,
                    /* in1 (K) tile_idx */ k_idx,
                    /* dst_idx */ 0,
                    /* transpose */ 1,
                    /* ct_dim */ Sk_chunk_t,
                    /* rt_dim */ 1,
                    /* kt_dim */ qWt);
                q_idx += 1U;
                k_idx += Sk_chunk_t;
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(cb_attention_weights, Sk_chunk_t);
        for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
            pack_tile(n, cb_attention_weights);
        }
        cb_push_back(cb_attention_weights, Sk_chunk_t);
        tile_regs_release();

        if (k_chunk == diag_chunk) {
            // Re-enter reserved state on the same L1 slots (single-buffered CB wraps back).
            cb_wait_front(cb_attention_weights, Sk_chunk_t);
            cb_pop_front(cb_attention_weights, Sk_chunk_t);
            cb_reserve_back(cb_attention_weights, Sk_chunk_t);

            // Unpacker → mask CB (BF16); packer stays on cb_attention_weights (FP32).
""",
    """        matmul_block_init(
            cb_query,
            cb_key,
            /* transpose */ 1,
            /* ct_dim */ kPass,
            /* rt_dim */ 1,
            /* kt_dim */ qWt);
        cb_reserve_back(cb_attention_weights, Sk_chunk_t);
        // The chunk's score tiles in passes of kPass columns (one DEST cycle each), packed in order into the
        // reserved chunk; one push after the passes (and the mask stamp below).
        for (uint32_t p = 0; p < Sk_chunk_t; p += kPass) {
            tile_regs_acquire();
            {
                // One matmul_block call does ONE K-step (1 Q tile × kPass K tiles → kPass accumulated
                // outputs). Iterate qWt times to contract the full feature dim; the K tiles of columns
                // p .. p + kPass - 1 at feature `feat` sit at feat * Sk_chunk_t + p (col-major chunk).
                uint32_t q_idx = 0;
                uint32_t k_idx = p;
                for (uint32_t feat = 0; feat < qWt; ++feat) {
                    matmul_block(
                        cb_query,
                        cb_key,
                        /* in0 (Q) tile_idx */ q_idx,
                        /* in1 (K) tile_idx */ k_idx,
                        /* dst_idx */ 0,
                        /* transpose */ 1,
                        /* ct_dim */ kPass,
                        /* rt_dim */ 1,
                        /* kt_dim */ qWt);
                    q_idx += 1U;
                    k_idx += Sk_chunk_t;
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t n = 0; n < kPass; ++n) {
                pack_tile(n, cb_attention_weights);
            }
            tile_regs_release();
        }

        if (k_chunk == diag_chunk) {
            // The chunk is still reserved (not pushed): the out-of-order packs below address its tiles.
            // Unpacker → mask CB (BF16); packer stays on cb_attention_weights (FP32).
""",
)
sub(
    K,
    """                pack_tile</* out_of_order */ true>(/* dst */ 0, cb_attention_weights, /* offset */ n);
                tile_regs_release();
            }
            pack_reconfig_l1_acc(0);

            cb_push_back(cb_attention_weights, Sk_chunk_t);
        }
""",
    """                pack_tile</* out_of_order */ true>(/* dst */ 0, cb_attention_weights, /* offset */ n);
                tile_regs_release();
            }
            pack_reconfig_l1_acc(0);
        }
        cb_push_back(cb_attention_weights, Sk_chunk_t);
""",
)
sub(
    K,
    "apply_exp_inplace_and_find_exp_sum<scaler_bits, scaler_bf16_bits, Sk_chunk_t>(",
    "apply_exp_inplace_and_find_exp_sum<scaler_bits, scaler_bf16_bits, Sk_chunk_t, kPass>(",
)
# ---------------- utils: exp stage in passes, output to a fresh slot of the (double-buffered) score CB
sub(
    U,
    """// DST budget: Sk_chunk_t registers are held live across sub_tiles_bcast_cols + exp_tile +
// (two) pack passes. With fp32_dest_acc_en=true the DST has 4 tiles, so Sk_chunk_t <= 4.
template <uint32_t scaler_fp32, uint32_t scaler_bf16, uint32_t Sk_chunk_t = 1U>
void apply_exp_inplace_and_find_exp_sum(uint32_t cb_attention_weights, uint32_t cb_cur_max, uint32_t cb_cur_exp_sum) {
    cb_wait_front(cb_attention_weights, Sk_chunk_t);
    cb_wait_front(cb_cur_max, onetile);

    reconfig_data_format(cb_attention_weights, cb_cur_max);
    sub_bcast_cols_init(cb_attention_weights, cb_cur_max);

    // Fused scale+exp: compute exp(scale * (score - max)) in a single SFPU pass.
    // sdpa_exp_tile_scaled dispatches to the arch-appropriate path (WH sfpi mul or BH LREG12 fold).
    tile_regs_acquire();
    for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
        sub_tiles_bcast_cols(
            cb_attention_weights, cb_cur_max, /* in0 tile_idx */ n, /* in1 tile_idx */ 0, /* dst_reg_idx */ n);
        sdpa_exp_tile_scaled<scaler_fp32, scaler_bf16>(n);
    }
    tile_regs_commit();

    // In-place overwrite of cb_attention_weights. The CB is sized exactly Sk_chunk_t (no
    // double-buffering), so pop+reserve aliases to the same physical L1 slots.
    cb_pop_front(cb_attention_weights, Sk_chunk_t);
    cb_reserve_back(cb_attention_weights, Sk_chunk_t);
    cb_reserve_back(cb_cur_exp_sum, onetile);

    tile_regs_wait();
    // Write the Sk_chunk_t exp tiles back to cb_attention_weights at offsets 0..Sk_chunk_t-1.
    pack_reconfig_data_format(cb_attention_weights);
    pack_reconfig_l1_acc(false);
    for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
        pack_tile(n, cb_attention_weights);
    }
    cb_push_back(cb_attention_weights, Sk_chunk_t);

    // Build the chunk's partial sum directly in cb_cur_exp_sum[0]:
    //   - first exp tile: overwrite slot 0 (L1-acc off);
    //   - subsequent tiles: L1-accumulate onto slot 0.
    pack_reconfig_data_format(cb_cur_exp_sum);
    pack_tile</* out_of_order */ true>(/* dst */ 0, cb_cur_exp_sum, /* offset */ 0);
    if constexpr (Sk_chunk_t > 1U) {
        pack_reconfig_l1_acc(true);
        for (uint32_t n = 1U; n < Sk_chunk_t; ++n) {
            pack_tile</* out_of_order */ true>(/* dst */ n, cb_cur_exp_sum, /* offset */ 0);
        }
        pack_reconfig_l1_acc(false);
    }
    cb_push_back(cb_cur_exp_sum, onetile);
    tile_regs_release();
}
""",
    """// DST budget: the chunk is processed in passes of Pass tiles (Pass <= the DEST tile count: 4 with fp32
// accumulation), each pass one tile_regs cycle. The score CB holds two chunks: the exp tiles go to the free
// slot and the raw scores are popped, so a chunk larger than DEST needs no in-place aliasing.
template <uint32_t scaler_fp32, uint32_t scaler_bf16, uint32_t Sk_chunk_t = 1U, uint32_t Pass = Sk_chunk_t>
void apply_exp_inplace_and_find_exp_sum(uint32_t cb_attention_weights, uint32_t cb_cur_max, uint32_t cb_cur_exp_sum) {
    static_assert(Sk_chunk_t % Pass == 0U, "the chunk is a whole number of passes");
    cb_wait_front(cb_attention_weights, Sk_chunk_t);
    cb_wait_front(cb_cur_max, onetile);
    cb_reserve_back(cb_attention_weights, Sk_chunk_t);  // the exp tiles' slot (the CB holds two chunks)
    cb_reserve_back(cb_cur_exp_sum, onetile);

    reconfig_data_format(cb_attention_weights, cb_cur_max);
    sub_bcast_cols_init(cb_attention_weights, cb_cur_max);
    pack_reconfig_data_format(cb_attention_weights);
    pack_reconfig_l1_acc(false);

    for (uint32_t p = 0; p < Sk_chunk_t; p += Pass) {
        // Fused scale+exp: compute exp(scale * (score - max)) in a single SFPU pass.
        // sdpa_exp_tile_scaled dispatches to the arch-appropriate path (WH sfpi mul or BH LREG12 fold).
        tile_regs_acquire();
        for (uint32_t n = 0; n < Pass; ++n) {
            sub_tiles_bcast_cols(
                cb_attention_weights, cb_cur_max, /* in0 tile_idx */ p + n, /* in1 tile_idx */ 0, /* dst_reg_idx */ n);
            sdpa_exp_tile_scaled<scaler_fp32, scaler_bf16>(n);
        }
        tile_regs_commit();
        tile_regs_wait();
        // The pass's exp tiles, in order, into the reserved chunk.
        for (uint32_t n = 0; n < Pass; ++n) {
            pack_tile(n, cb_attention_weights);
        }
        // The chunk's partial row sum directly in cb_cur_exp_sum[0]: the first exp tile overwrites slot 0
        // (L1-acc off), every later tile L1-accumulates onto it.
        for (uint32_t n = 0; n < Pass; ++n) {
            pack_reconfig_l1_acc(p + n > 0U);
            pack_tile</* out_of_order */ true>(/* dst */ n, cb_cur_exp_sum, /* offset */ 0);
        }
        pack_reconfig_l1_acc(false);
        tile_regs_release();
    }
    cb_pop_front(cb_attention_weights, Sk_chunk_t);  // the raw scores
    cb_push_back(cb_attention_weights, Sk_chunk_t);  // the exp tiles
    cb_push_back(cb_cur_exp_sum, onetile);
}
""",
)
# ---------------- factory
sub(F, "constexpr uint32_t kMaxSkChunkT = 4U;", "constexpr uint32_t kMaxSkChunkT = 8U;")
sub(
    F,
    """//   2. Must fit DST under fp32_dest_acc_en=true (DST has 4 tiles): the chunked softmax
//      holds Sk_chunk_t exp tiles live across one tile_regs cycle, so we cap at 4.
//   3. Larger chunks amortize softmax / correction overhead more, but consume L1
//      proportionally for cb_key, cb_value, and cb_attention_weights.
//
// We pick the largest power-of-2 in {1, 2, 4} that divides Ht. Bumping past 4 would require
// either fp32_dest_acc_en=false (loses precision) or splitting the exp pass into multiple
// DST cycles.
""",
    """//   2. The QK^T and exp stages process the chunk in DEST-sized passes (iteration 26), so the
//      chunk may exceed the DEST tile count; the PV matmul contracts the whole chunk in DEST.
//   3. Larger chunks amortize the softmax correction (and, in chain mode, the hop count) more,
//      but consume L1 proportionally for cb_key, cb_value, and cb_attention_weights.
//
// We pick the largest power-of-2 in {1, 2, 4, 8} that divides Ht.
""",
)
sub(
    F,
    """    // Holds the Sk_chunk_t QK^T score tiles for the current K chunk. Sized to one chunk
    // (not double-buffered) since softmax overwrites in-place, then the PV matmul consumes it.
""",
    """    // Holds the Sk_chunk_t QK^T score tiles of the current K chunk and, in the second slot, their exp
    // (iteration 26: the exp stage writes the free slot and pops the raw scores; the PV matmul consumes it).
""",
)
sub(
    F,
    "program, all_cores, kAttentionWeightsCbIndex, precise_data_format, float32_single_tile_size_bytes, Sk_chunk_t);",
    "program, all_cores, kAttentionWeightsCbIndex, precise_data_format, float32_single_tile_size_bytes, 2U * Sk_chunk_t);",
)
sub(
    F,
    r"(scaler_bf16,(\s*)// BF16 \(RNE\) scale for the WH fused scale\+exp path\n)(\s*)\};",
    r"\1\3    dst_size,\2// DEST tiles per pass (QK^T / exp passes of the chunk)\n\3};",
    count=3,
    regex=True,
)
print("patched")
