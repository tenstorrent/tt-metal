"""Iteration 30: the SDPA forward's diagonal chunk computes only the K tiles up to the diagonal (QK^T passes, mask
stamp, row max, exp / sum, PV contraction over n_cols tiles instead of the whole 8-row chunk). Bit-exact: the
skipped tiles were fully masked (-1e9 scores, exp = 0)."""
import sys

M = "tt-train/sources/ttml/metal/ops/sdpa_fw/device/kernels/compute"
K = f"{M}/sdpa_fw_compute_kernel.cpp"
U = f"{M}/sdpa_compute_utils.hpp"


def sub(path, old, new, count=1):
    s = open(path).read()
    n = s.count(old)
    if n != count:
        print(f"FAILED {path}: {old[:90]!r} found {n}, expected {count}")
        sys.exit(2)
    open(path, "w").write(s.replace(old, new))


# ---- kernel: n_cols per chunk
sub(
    K,
    """    for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
        cb_wait_front(cb_key, Sk_chunk_t * qWt);

        reconfig_data_format(cb_query, cb_key);
        pack_reconfig_data_format(cb_attention_weights);
""",
    """    for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
        cb_wait_front(cb_key, Sk_chunk_t * qWt);
#if defined(CAUSAL_MASK) || defined(BALANCED_PARALLELISM)
        // The diagonal chunk's K tiles past the diagonal are fully masked (exp = 0): only the tiles up to the
        // diagonal are computed (iteration 30); the CB pushes and pops stay whole chunks.
        const uint32_t n_cols = (k_chunk == diag_chunk) ? (diag_pos_in_chunk + 1U) : Sk_chunk_t;
#else
        const uint32_t n_cols = Sk_chunk_t;
#endif

        reconfig_data_format(cb_query, cb_key);
        pack_reconfig_data_format(cb_attention_weights);
""",
)
sub(
    K,
    """        for (uint32_t p = 0; p < Sk_chunk_t; p += kPass) {
            tile_regs_acquire();
            {""",
    """        for (uint32_t p = 0; p < n_cols; p += kPass) {
            const uint32_t pass_cols = (n_cols - p < kPass) ? (n_cols - p) : kPass;  // the last pass of the diagonal
            tile_regs_acquire();
            {""",
)
sub(
    K,
    """            for (uint32_t n = 0; n < kPass; ++n) {
                pack_tile(n, cb_attention_weights);
            }
            tile_regs_release();
        }

        if (k_chunk == diag_chunk) {""",
    """            for (uint32_t n = 0; n < pass_cols; ++n) {
                pack_tile(n, cb_attention_weights);
            }
            tile_regs_release();
        }

        if (k_chunk == diag_chunk) {""",
)
sub(
    K,
    """            for (uint32_t n = diag_pos_in_chunk; n < Sk_chunk_t; ++n) {
                // n == diag_pos_in_chunk → mask[0] (causal diagonal pattern).
                // n  > diag_pos_in_chunk → mask[1] (all -1e9, strictly past the diagonal).
                const uint32_t mask_tile_idx = (n == diag_pos_in_chunk) ? 0U : 1U;
                tile_regs_acquire();
                copy_tile(cb_attn_mask, mask_tile_idx, /* dst */ 0);
                tile_regs_commit();
                tile_regs_wait();
                // pack_tile<true> packs DST[0] to cb_attention_weights[n]; with l1_acc=1 the
                // packer reads the existing FP32 score, adds the FP32 mask in DST, writes FP32.
                pack_tile</* out_of_order */ true>(/* dst */ 0, cb_attention_weights, /* offset */ n);
                tile_regs_release();
            }
            pack_reconfig_l1_acc(0);""",
    """            {
                // Only the diagonal tile takes the causal pattern (mask[0]); the tiles past it are not computed.
                tile_regs_acquire();
                copy_tile(cb_attn_mask, /* mask_tile_idx */ 0, /* dst */ 0);
                tile_regs_commit();
                tile_regs_wait();
                // pack_tile<true> packs DST[0] to cb_attention_weights[diag]; with l1_acc=1 the
                // packer reads the existing FP32 score, adds the FP32 mask in DST, writes FP32.
                pack_tile</* out_of_order */ true>(/* dst */ 0, cb_attention_weights, /* offset */ diag_pos_in_chunk);
                tile_regs_release();
            }
            pack_reconfig_l1_acc(0);""",
)
sub(
    K,
    """            alias_cb_cur_max,
            alias_cb_prev_max,
            /* do_eltwise_max */ k_chunk > 0);""",
    """            alias_cb_cur_max,
            alias_cb_prev_max,
            /* do_eltwise_max */ k_chunk > 0,
            n_cols);""",
)
sub(
    K,
    "            cb_attention_weights, alias_cb_cur_max, alias_cb_cur_sum_exp);",
    "            cb_attention_weights, alias_cb_cur_max, alias_cb_cur_sum_exp, n_cols);",
)
sub(
    K,
    "        matmul_qk_by_v<Sk_chunk_t>(vWt, pv_block_size, cb_attention_weights, cb_value, alias_cb_cur_mm_out);",
    "        matmul_qk_by_v<Sk_chunk_t>(vWt, pv_block_size, cb_attention_weights, cb_value, alias_cb_cur_mm_out, n_cols);",
)
# ---- utils: runtime tile counts
sub(
    U,
    """    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    bool do_eltwise_max = false) {
    cb_wait_front(cb_attention_weights, Sk_chunk_t);
""",
    """    uint32_t cb_cur_max,
    uint32_t cb_prev_max,
    bool do_eltwise_max = false,
    uint32_t n_tiles = Sk_chunk_t) {  // tiles of the chunk that carry scores (the diagonal chunk's prefix)
    cb_wait_front(cb_attention_weights, Sk_chunk_t);
""",
)
sub(
    U,
    """    for (uint32_t n = 0; n < Sk_chunk_t; ++n) {
        reduce_tile<pool_type, reduce_dim>(""",
    """    for (uint32_t n = 0; n < n_tiles; ++n) {
        reduce_tile<pool_type, reduce_dim>(""",
)
sub(
    U,
    """void apply_exp_inplace_and_find_exp_sum(uint32_t cb_attention_weights, uint32_t cb_cur_max, uint32_t cb_cur_exp_sum) {
    static_assert(Sk_chunk_t % Pass == 0U, "the chunk is a whole number of passes");""",
    """void apply_exp_inplace_and_find_exp_sum(
    uint32_t cb_attention_weights, uint32_t cb_cur_max, uint32_t cb_cur_exp_sum, uint32_t n_tiles = Sk_chunk_t) {
    // n_tiles: the chunk's tiles that carry scores (the diagonal chunk's prefix, iteration 30); the CB traffic
    // stays whole chunks.
    static_assert(Sk_chunk_t % Pass == 0U, "the chunk is a whole number of passes");""",
)
sub(
    U,
    """    for (uint32_t p = 0; p < Sk_chunk_t; p += Pass) {
        // Fused scale+exp: compute exp(scale * (score - max)) in a single SFPU pass.
        // sdpa_exp_tile_scaled dispatches to the arch-appropriate path (WH sfpi mul or BH LREG12 fold).
        tile_regs_acquire();
        for (uint32_t n = 0; n < Pass; ++n) {""",
    """    for (uint32_t p = 0; p < n_tiles; p += Pass) {
        const uint32_t pass_tiles = (n_tiles - p < Pass) ? (n_tiles - p) : Pass;
        // Fused scale+exp: compute exp(scale * (score - max)) in a single SFPU pass.
        // sdpa_exp_tile_scaled dispatches to the arch-appropriate path (WH sfpi mul or BH LREG12 fold).
        tile_regs_acquire();
        for (uint32_t n = 0; n < pass_tiles; ++n) {""",
)
sub(
    U,
    """        // The pass's exp tiles, in order, into the reserved chunk.
        for (uint32_t n = 0; n < Pass; ++n) {
            pack_tile(n, cb_attention_weights);
        }
        // The chunk's partial row sum directly in cb_cur_exp_sum[0]: the first exp tile overwrites slot 0
        // (L1-acc off), every later tile L1-accumulates onto it.
        for (uint32_t n = 0; n < Pass; ++n) {""",
    """        // The pass's exp tiles, in order, into the reserved chunk.
        for (uint32_t n = 0; n < pass_tiles; ++n) {
            pack_tile(n, cb_attention_weights);
        }
        // The chunk's partial row sum directly in cb_cur_exp_sum[0]: the first exp tile overwrites slot 0
        // (L1-acc off), every later tile L1-accumulates onto it.
        for (uint32_t n = 0; n < pass_tiles; ++n) {""",
)
sub(
    U,
    """void matmul_qk_by_v(
    uint32_t Wt, uint32_t block_size, uint32_t cb_attention_weights, uint32_t cb_value, uint32_t cb_cur_mm_out) {
    cb_wait_front(cb_attention_weights, Sk_chunk_t);""",
    """void matmul_qk_by_v(
    uint32_t Wt,
    uint32_t block_size,
    uint32_t cb_attention_weights,
    uint32_t cb_value,
    uint32_t cb_cur_mm_out,
    uint32_t n_tiles = Sk_chunk_t) {  // K tiles of the chunk contracted (the diagonal chunk's prefix)
    cb_wait_front(cb_attention_weights, Sk_chunk_t);""",
)
sub(
    U,
    """        /* ct_dim */ block_size,
        /* rt_dim */ 1,
        /* kt_dim */ Sk_chunk_t);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {""",
    """        /* ct_dim */ block_size,
        /* rt_dim */ 1,
        /* kt_dim */ n_tiles);

    for (uint32_t tile_idx = 0; tile_idx < Wt; tile_idx += block_size) {""",
)
sub(
    U,
    """        for (uint32_t k_in_chunk = 0; k_in_chunk < Sk_chunk_t; ++k_in_chunk) {
            matmul_block(""",
    """        for (uint32_t k_in_chunk = 0; k_in_chunk < n_tiles; ++k_in_chunk) {
            matmul_block(""",
)
sub(
    U,
    """                /* ct_dim */ block_size,
                /* rt_dim */ 1,
                /* kt_dim */ Sk_chunk_t);
        }
        tile_regs_commit();""",
    """                /* ct_dim */ block_size,
                /* rt_dim */ 1,
                /* kt_dim */ n_tiles);
        }
        tile_regs_commit();""",
)
print("iteration 30 patch applied")
