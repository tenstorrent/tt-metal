# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated ordered prediction accumulator experiment; no device imports.

Apply prediction_sources to cumulative_exp_scalar's (reader, compute) pair.
No runtime source-file/AST reads, registration, default changes or writer changes.
Exact hardware equivalence and performance require the independent probe.
"""

BASELINE = "cumulative_exp_scalar"
VARIANT = "cumulative_prediction_registers"

PREDICTION_LOOP = """            for(uint32_t m=0;m<4;m++) {
                if constexpr(FUSE_DECAY_PREDICT) {
                    if(m) gdn_decay_predict<true,!HOIST_COL_INIT>(m,5,6,7,m);
                    else gdn_decay_predict<false,!HOIST_COL_INIT>(m,5,6,7,m);
                } else if constexpr(FUSE_ACCUMULATOR) {
                    if(m) {
                        // Exact separate FP32 product then add into accumulator.
                        // Skip the full temporary-tile store/reload/add pass.
                        gdn_col_mul_add_to<!HOIST_COL_INIT>(m,5,6,6);
                    } else { gdn_col_mul_to<!HOIST_COL_INIT>(m,5,6); }
                } else {
                    const uint32_t work=m?4:6;
                    gdn_col_mul_to(m,5,work);
                    if(m) { add_binary_tile_init(); add_binary_tile(6,4,6); }
                }
            }"""
PREDICTION_CALL = """            static_assert(FUSE_DECAY_PREDICT && FUSE_ACCUMULATOR && HOIST_COL_INIT,
                          "Ordered prediction specialization requires fused incumbent path");
            gdn_decay_predict_all(5,6,7);"""

# Concrete register indices are intentional: ACC1..4, K0, temporary5/6,
# scalar7. No dynamically indexed instruction operand or SFPU register array.
HELPER = r"""
// Ordered m0..3 prediction, retaining four SFPU accumulators per row band.
// The existing state tiles are DST0..3, K is DST5, result DST6, decay DST7.
// Stores of decayed state remain separate from its later K multiplication.
template<uint32_t ACC0,uint32_t ACC1,uint32_t OFF0,uint32_t OFF1>
inline __attribute__((always_inline)) void _gdn_prediction_pair_(uint32_t state_base,uint32_t left) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    TT_SFPLOAD(p_sfpu::LREG5,IM,ADDR_MOD_7,state_base+left+OFF0);
    TT_SFPLOAD(p_sfpu::LREG6,IM,ADDR_MOD_7,state_base+left+OFF1);
    TTI_SFPNOP;
    TTI_SFPMUL(p_sfpu::LREG5,p_sfpu::LREG7,p_sfpu::LCONST_0,p_sfpu::LREG5,0);
    TTI_SFPMUL(p_sfpu::LREG6,p_sfpu::LREG7,p_sfpu::LCONST_0,p_sfpu::LREG6,0);
    TTI_SFPNOP;
    TT_SFPSTORE(p_sfpu::LREG5,IM,ADDR_MOD_7,state_base+left+OFF0);
    TT_SFPSTORE(p_sfpu::LREG6,IM,ADDR_MOD_7,state_base+left+OFF1);
    TTI_SFPMUL(p_sfpu::LREG5,p_sfpu::LREG0,p_sfpu::LCONST_0,p_sfpu::LREG5,0);
    TTI_SFPMUL(p_sfpu::LREG6,p_sfpu::LREG0,p_sfpu::LCONST_0,p_sfpu::LREG6,0);
    TTI_SFPNOP;
    // Same native operand order: new_product*1 + previous_sum, separate add.
    TTI_SFPADD(p_sfpu::LREG5,p_sfpu::LCONST_1,ACC0,ACC0,0);
    TTI_SFPADD(p_sfpu::LREG6,p_sfpu::LCONST_1,ACC1,ACC1,0);
}

inline __attribute__((always_inline)) void _gdn_decay_predict_all_(uint32_t unused,uint32_t key,uint32_t output,uint32_t scalar) {
    constexpr InstrModLoadStore IM=InstrModLoadStore::DEFAULT;
    const uint32_t kb=key*DEST_TILE_SIZE_RAW, ob=output*DEST_TILE_SIZE_RAW;
    // Broadcast helper clobbers other LREGs only before accumulators are live.
    _broadcast_scalar_from_dest_(scalar*DEST_TILE_SIZE_RAW);
    TTI_SFPMOV(0,LREG_BCAST,p_sfpu::LREG7,0);
    for(uint32_t half=0;half<2;half++) for(uint32_t band=0;band<NUM_ROW_BANDS_PER_FACE_HALF;band++) {
        const uint32_t left=half*FACE2_BASE+band*ROW_BAND_STRIDE;
        const uint32_t right=left+FACE1_BASE;
        // m0 initializes directly from its product; never add to an initial zero.
        TT_SFPLOAD(p_sfpu::LREG0,IM,ADDR_MOD_7,kb+left);
        TT_SFPLOAD(p_sfpu::LREG1,IM,ADDR_MOD_7,left);
        TT_SFPLOAD(p_sfpu::LREG2,IM,ADDR_MOD_7,left+ODD_COLS_OFFSET);
        TT_SFPLOAD(p_sfpu::LREG3,IM,ADDR_MOD_7,right);
        TT_SFPLOAD(p_sfpu::LREG4,IM,ADDR_MOD_7,right+ODD_COLS_OFFSET);
        TTI_SFPMUL(p_sfpu::LREG1,p_sfpu::LREG7,p_sfpu::LCONST_0,p_sfpu::LREG1,0);
        TTI_SFPMUL(p_sfpu::LREG2,p_sfpu::LREG7,p_sfpu::LCONST_0,p_sfpu::LREG2,0);
        TTI_SFPMUL(p_sfpu::LREG3,p_sfpu::LREG7,p_sfpu::LCONST_0,p_sfpu::LREG3,0);
        TTI_SFPMUL(p_sfpu::LREG4,p_sfpu::LREG7,p_sfpu::LCONST_0,p_sfpu::LREG4,0);
        TT_SFPSTORE(p_sfpu::LREG1,IM,ADDR_MOD_7,left);
        TT_SFPSTORE(p_sfpu::LREG2,IM,ADDR_MOD_7,left+ODD_COLS_OFFSET);
        TT_SFPSTORE(p_sfpu::LREG3,IM,ADDR_MOD_7,right);
        TT_SFPSTORE(p_sfpu::LREG4,IM,ADDR_MOD_7,right+ODD_COLS_OFFSET);
        TTI_SFPMUL(p_sfpu::LREG1,p_sfpu::LREG0,p_sfpu::LCONST_0,p_sfpu::LREG1,0);
        TTI_SFPMUL(p_sfpu::LREG2,p_sfpu::LREG0,p_sfpu::LCONST_0,p_sfpu::LREG2,0);
        TTI_SFPMUL(p_sfpu::LREG3,p_sfpu::LREG0,p_sfpu::LCONST_0,p_sfpu::LREG3,0);
        TTI_SFPMUL(p_sfpu::LREG4,p_sfpu::LREG0,p_sfpu::LCONST_0,p_sfpu::LREG4,0);
        for(uint32_t m=1;m<4;m++) {
            const uint32_t slot=(m/2)*FACE1_BASE+(m%2)*ODD_COLS_OFFSET;
            TT_SFPLOAD(p_sfpu::LREG0,IM,ADDR_MOD_7,kb+slot+left);
            _gdn_prediction_pair_<p_sfpu::LREG1,p_sfpu::LREG2,0,ODD_COLS_OFFSET>(m*DEST_TILE_SIZE_RAW,left);
            _gdn_prediction_pair_<p_sfpu::LREG3,p_sfpu::LREG4,FACE1_BASE,FACE1_BASE+ODD_COLS_OFFSET>(m*DEST_TILE_SIZE_RAW,left);
        }
        TT_SFPSTORE(p_sfpu::LREG1,IM,ADDR_MOD_7,ob+left);
        TT_SFPSTORE(p_sfpu::LREG2,IM,ADDR_MOD_7,ob+left+ODD_COLS_OFFSET);
        TT_SFPSTORE(p_sfpu::LREG3,IM,ADDR_MOD_7,ob+right);
        TT_SFPSTORE(p_sfpu::LREG4,IM,ADDR_MOD_7,ob+right+ODD_COLS_OFFSET);
    }
}

"""
WRAPPER = r"""
ALWI void gdn_decay_predict_all(uint32_t key,uint32_t output,uint32_t scalar) {
    MATH((SFPU_BINARY_CALL_NO_TEMPLATE_ARGS(DST_SYNC_MODE,DST_ACCUM_MODE,
        _gdn_decay_predict_all_,0,key,output,VectorMode::None,scalar)));
}

"""
HELPER_ANCHOR = "// Match native calculate_typecast_fp32_to_fp16b's integer sequence exactly."
WRAPPER_ANCHOR = "template<bool ACCUMULATE>\nALWI void gdn_update_readout"


def _replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"Frozen prediction source drift: expected one anchor {before[:100]!r}")
    return source.replace(before, after, 1)


def prediction_sources(reader, compute):
    """Transform an already-selected source pair, preserving the reader verbatim.

    Baseline should be cumulative_exp_scalar from recurrent_variants.sources.
    Caller retains normal writer/CB/arguments/head ownership. Suitable for
    _local12_recurrent(source_overrides=...) without any global registration.
    """
    compute = _replace_once(compute, HELPER_ANCHOR, HELPER + HELPER_ANCHOR)
    compute = _replace_once(compute, WRAPPER_ANCHOR, WRAPPER + WRAPPER_ANCHOR)
    compute = _replace_once(compute, PREDICTION_LOOP, PREDICTION_CALL)
    return reader, compute
