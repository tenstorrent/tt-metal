// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// NOT COMPILED. Copy-paste source for graduating the measured phase-4 win into rms_norm.
// Measured by tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/rsqrt_lane_and_window as variant
// `chain_fused_cskip`: 29205 ns -> 8265 ns per core on the focus geometry (3.53x), col-0 PCC
// 0.9999968 vs the baseline's 0.9999967, all-ones absolute check identical.
//
// ---------------------------------------------------------------------------------------------
// PART 1 — the new chain element. Natural home: ttnn/cpp/ttnn/kernel_lib/eltwise_math.{hpp,inl}
//          (forward-declare in the .hpp, define in the .inl next to `Rsqrt`). It derives from the
//          same `UnaryOp` CRTP base every SFPU op there uses, so `eltwise_chain` keeps owning the CB
//          lifecycle, the dtype reconfig and the dst-sync window — only the SFPU body is new.
//
// WHY IT BYPASSES THE STOCK HELPERS (the raw-LLK justification the verifier needs):
//   * `rsqrt_tile` hardcodes `VectorMode::RC` and `ITERATIONS = 8`, and `add_unary_tile` likewise.
//     Neither the compute-API wrapper nor the `Rsqrt`/`AddUnary` chain elements expose a vector-mode,
//     iteration-count or DEST-address-stride knob. `cb_rms_sum` is a REDUCE_ROW result, so 24 of the
//     32 vector ops per pass compute lanes nobody reads (measured: 23 ns per 32-lane accurate-rsqrt
//     vector, so 8 vectors instead of 32 is the whole win).
//   * `AddUnary` and `Rsqrt` are separate elements, hence separate SFPU passes with separate
//     DEST-address setup + STALLWAIT and separate full walks. There is no "unary op with a pre-added
//     scalar" element to compose.
//   The BODY is the stock accurate rsqrt kernel verbatim — `_calculate_sqrt_body_<APPROX,
//   RECIPROCAL=true, FAST_APPROX=false>` (SQRT_23-bits) plus the `!fp32_dest_acc_en`
//   round-to-nearest store — i.e. exactly what `calculate_rsqrt<APPROX, 8, DST_ACCUM_MODE, false,
//   false>` runs. Same function, same precision, fewer lanes. The precision contract
//   (fp32_dest_acc_en / math_fidelity / math_approx_mode / dtypes) is untouched.
//
// SAFETY PRECONDITION (verified on device by
// tests/.../rsqrt_lane_and_window/test_phase4.py::test_bcast_col_reads_column_zero_only):
//   phase 5 consumes cb_rms_recip through `mul_tiles_bcast<BroadcastType::COL>` (srcB). Fed a tile
//   whose columns 1..31 hold poison (7.5e3 / 3e-4), that primitive reproduces column 0 across the
//   whole output with only bf16 rounding error (max rel-err 0.0078) — it reads column 0 ONLY. So
//   leaving columns 1..31 of cb_rms_recip unwritten cannot change the op's output. If a future
//   refinement ever gives cb_rms_recip a second consumer that reads other lanes, this element must
//   revert to a full-tile scope.
// ---------------------------------------------------------------------------------------------

#ifdef TRISC_MATH
namespace ckernel::sfpu {
// 4 even-parity vectors per face. The SFPU walks a face as [rg0-even, rg0-odd, rg1-even, ...] and
// column 0 lives only in the even-parity vectors, so we visit offsets 0,2,4,6. Net dst_reg advance
// is +8 == the stock ITERATIONS=8, so `VectorMode::C`'s face-0 -> face-2 stepping composes
// unchanged and column 0 is covered for all 32 rows.
template <bool APPROXIMATION_MODE, bool fp32_dest_acc_en>
inline void calculate_rsqrt_add_col0(uint32_t eps_bits) {
    const sfpi::vFloat eps = Converter::as_float(eps_bits);
    for (int d = 0; d < 4; d++) {
        sfpi::vFloat t = _calculate_sqrt_body_<APPROXIMATION_MODE, true, false>(sfpi::dst_reg[0] + eps);
        if constexpr (!fp32_dest_acc_en) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;  // skip the odd-parity vector (columns 1,3,..,15 — never column 0)
    }
}
}  // namespace ckernel::sfpu
#endif

namespace compute_kernel_lib {

/// `rsqrt(x + eps)` in ONE SFPU pass, scoped to the vector ops that hold COLUMN 0.
/// For a REDUCE_ROW statistic consumed through `BroadcastDim::Col` only. Columns 1..31 of the
/// output tile are left holding whatever the copy put there.
template <Dst Slot = Dst::D0>
struct RsqrtAddUnaryColZero : UnaryOp<RsqrtAddUnaryColZero<Slot>, Slot> {
    uint32_t eps_bits;
    constexpr explicit RsqrtAddUnaryColZero(uint32_t e) noexcept : eps_bits(e) {}
    static ALWI void init() { rsqrt_tile_init(); }  // programs the shared sqrt vConst*Prgm constants
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        const uint32_t idst = to_u32(Slot) + slot_offset;
        const uint32_t eps = eps_bits;
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            [eps]() { ckernel::sfpu::calculate_rsqrt_add_col0<APPROX, DST_ACCUM_MODE>(eps); },
            idst,
            ckernel::VectorMode::C)));
    }
};

}  // namespace compute_kernel_lib

// ---------------------------------------------------------------------------------------------
// PART 2 — the call-site change in ttnn/ttnn/operations/rms_norm/kernels/rms_norm_compute.cpp
//          (zone `cmp_rsqrt`). Two elements become one; EVERYTHING else is byte-identical, so the
//          CopyTile/PackTile policies, the reconfig folds and the (unblocked) window are unchanged.
//
//  BEFORE:
//      ckl::eltwise_chain(
//          ckl::EltwiseShape::tiles(ht),
//          ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
//          ckl::AddUnary<ckl::Dst::D0>{eps_bits},
//          ckl::Rsqrt<>{},
//          ckl::PackTile<cb_rms_recip, ckl::OutputLifecycle::Streaming>{});
//
//  AFTER:
//      ckl::eltwise_chain(
//          ckl::EltwiseShape::tiles(ht),
//          ckl::CopyTile<cb_rms_sum, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
//          ckl::RsqrtAddUnaryColZero<ckl::Dst::D0>{eps_bits},
//          ckl::PackTile<cb_rms_recip, ckl::OutputLifecycle::Streaming>{});
//
// NO host-side predicate is needed: the win is measured at every ht in {1,2,4,8,16} and at every
// CB format, and the mechanism (a REDUCE_ROW statistic consumed by a Col broadcast) is a structural
// invariant of the op, not a shape property.
//
// ---------------------------------------------------------------------------------------------
// PART 3 (optional, independent, numerically FREE) — narrow the two intermediate CBs.
//          rms_norm_program_descriptor.py cb_plan():
//              ("cb_rms_sum",   CB_RMS_SUM,   self.fp32_tile_bytes, H)   ->  self.tile_bytes
//              ("cb_rms_recip", CB_RMS_RECIP, self.fp32_tile_bytes, H)   ->  self.tile_bytes
//          (plus the matching CB dtypes wherever the descriptor declares them).
//          Measured: an extra 2.4% on phase 4 (8265 -> ~8060 ns extrapolating the raw pair's
//          264.6 -> 258.3 ns/tile at bf16 CBs) and col-0 PCC bit-identical to fp32 (0.9999967
//          baseline / 0.9999968 fused at ALL THREE of fp32/fp32, fp32/bf16, bf16/bf16) — because
//          fp32_dest_acc_en=False makes DEST bf16, so the fp32 container never held a bit the
//          hardware could keep. It also halves 2*H*4 KB of L1 per core, which is real headroom for
//          the L1-bound prefill geometries. NOTE: this is only free while fp32_dest_acc_en is
//          False; under fp32_dest_acc_en=True the fp32 CBs DO carry information and must stay fp32.
