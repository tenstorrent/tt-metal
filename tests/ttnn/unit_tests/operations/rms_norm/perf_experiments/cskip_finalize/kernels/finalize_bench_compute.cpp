// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH (perf experiment `cskip_finalize`) — rms_norm's ROOT FINALIZE chain
// and nothing else. NOT part of the op; never dispatched by rms_norm.py.
//
// WHAT IS ISOLATED
// The op's root finalize is
//     eltwise_chain(tiles(rows_t), CopyTile<in(cb_stat_sum)>, AddUnaryColValid{eps},
//                   RsqrtColValid{}, PackTile<out(cb_rstd_send)>)
// on a COLUMN-0-VALID stat tile (a `reduce<SUM, REDUCE_ROW>` result: 31 of 32 columns
// are structurally garbage, and the apply reads rstd as `OperandKind::Col`, i.e.
// column 0 alone). This kernel runs exactly that chain over `rows_t` resident stat
// tiles between one `cp_finalize` zone — no reduce, no combine, no multicast, no
// apply, one core. Everything else is held constant across variants, so the zone
// delta is attributable to the SFPU scope alone.
//
// THE VECTOR-LEVEL SCOPE (the idea under test)
// An SFPU vector op covers 4 rows x 8 stride-2 columns; a 16x16 FACE is 8 vector ops
// walked [rg0-even, rg0-odd, rg1-even, rg1-odd, ...] — column PARITY is the inner
// axis, row-group the outer. Column 0 lives in faces 0 and 2 (which `VectorMode::C`
// already selects, halving the tile) AND, within each of those faces, only in the
// EVEN-parity vectors. So a column-0 result needs 4 vectors per face = 8 per tile,
// not the 16 `VectorMode::C` runs. `ITERATIONS` cannot express it (it truncates the
// OUTER walk axis contiguously); a DEST address stride can (`dst_reg += 2`).
//
// RAW-LLK JUSTIFICATION (per variant; see the report for the measured numbers)
//   * `VectorMode` itself: `ckl::Rsqrt` / `ckl::AddUnary` cannot express any mode —
//     `rsqrt_tile` / `add_unary_tile` HARDCODE `VectorMode::RC`. Already the op's
//     established pattern (RsqrtColValid / AddUnaryColValid are `ckl::UnaryOp` chain
//     ELEMENTS, so eltwise_chain still owns the DEST window, CB lifecycle, init and
//     format reconfig).
//   * the even-parity stride: no wrapper anywhere exposes the SFPU's per-vector DEST
//     addressing, so the body must be hand-written sfpi and handed to
//     `_llk_math_eltwise_unary_sfpu_params_`. `_calculate_sqrt_body_` (the same body
//     the stock non-legacy `calculate_rsqrt` calls, at the same APPROX / DST_ACCUM /
//     ITERATION-per-vector precision) is reused verbatim — this changes WHICH vectors
//     run, never the arithmetic in one.
//   * the fusion: an `AddUnary` + `Rsqrt` pair is TWO distinct SFPU element types, so
//     `chain_sfpu_inits_uniform_v` is false and eltwise_chain re-emits BOTH SFPU inits
//     per tile inside the walk. One fused element makes the chain SFPU-uniform, so the
//     init is boot-hoisted. (Fusing also drops one whole DEST read/write per vector.)
//
// PRECISION CONTRACT — FIXED, IDENTICAL FOR EVERY VARIANT: bf16 in/out, HiFi2,
// fp32_dest_acc_en=False (so DST_ACCUM_MODE=false), math_approx_mode as the op sets
// it (APPROX), rsqrt ITERATIONS unchanged, FAST_APPROX=false, legacy_compat=false.
// Nothing here trades precision for speed. The one variant that CHANGES a value
// (`cskip_fused`, which keeps `x+eps` in an LREG at fp32 instead of round-tripping it
// through the 16-bit DEST) is strictly MORE accurate, and its bit-exact twin
// (`cskip_fused_bitexact`) is provided so the difference is a measured menu entry
// rather than a hidden change.

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_misc.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#endif

namespace ckl = compute_kernel_lib;
using ckernel::VectorMode;

namespace {
constexpr uint32_t cb_stat_sum = 0;    // N resident column-0-valid stat tiles (bf16)
constexpr uint32_t cb_rstd_send = 16;  // N result tiles (bf16)

// Scope selector shared by every element below.
constexpr uint32_t SCOPE_RC = 0;     // VectorMode::RC  — whole tile (pre-Refinement-5)
constexpr uint32_t SCOPE_C = 1;      // VectorMode::C   — faces 0+2 (what the op does today)
constexpr uint32_t SCOPE_CSKIP = 2;  // VectorMode::C + even-parity stride — column 0 only
}  // namespace

// ---------------------------------------------------------------------------
// Raw-sfpi bodies. Each is invoked through the SAME wrapper the stock ops use
// (`_llk_math_eltwise_unary_sfpu_params_`), so DEST addressing, the STALLWAIT and the
// face stepping are unchanged; only the per-face vector walk differs. NET dst_reg
// ADVANCE MUST BE 8 (== the stock ITERATIONS=8 == one face) or `VectorMode::C`'s
// face0 -> face2 stepping desynchronizes: the strided bodies do 4 x (+2).
// ---------------------------------------------------------------------------
#ifdef TRISC_MATH
namespace bench {

template <int STEP, int ITER>
sfpi_inline void rsqrt_body() {
    for (int i = 0; i < ITER; i++) {
        // The exact body of the stock non-legacy `calculate_rsqrt` (RECIPROCAL=true),
        // including its DST_ACCUM_MODE-gated bf16 round.
        sfpi::vFloat t =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STEP;
    }
}

template <int STEP, int ITER>
sfpi_inline void add_body(uint32_t param) {
    // The exact body of the stock `calculate_binop_with_scalar<..., ADD, ...>`.
    const sfpi::vFloat parameter = ckernel::sfpu::Converter::as_float(param);
    for (int i = 0; i < ITER; i++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = val + parameter;
        sfpi::dst_reg += STEP;
    }
}

// rsqrt(x + eps) in ONE pass over the scope.
//   ROUNDTRIP=true  — store the sum to DEST and read it back, so the sum is truncated
//                     to the DEST format exactly as the two-element chain does =>
//                     BIT-IDENTICAL result, at the cost of one extra store+load.
//   ROUNDTRIP=false — keep the sum in an LREG (fp32) => strictly more accurate than
//                     today, and one DEST round trip cheaper.
template <int STEP, int ITER, bool ROUNDTRIP>
sfpi_inline void fused_body(uint32_t param) {
    const sfpi::vFloat parameter = ckernel::sfpu::Converter::as_float(param);
    for (int i = 0; i < ITER; i++) {
        sfpi::vFloat s = sfpi::vFloat(sfpi::dst_reg[0]) + parameter;
        if constexpr (ROUNDTRIP) {
            sfpi::dst_reg[0] = s;
            s = sfpi::dst_reg[0];
        }
        sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(s);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += STEP;
    }
}

}  // namespace bench
#endif

// ---------------------------------------------------------------------------
// eltwise_chain ELEMENTS (the op's pattern: a missing block operation BUILT, so the
// helper keeps owning the DEST window / CB lifecycle / init / reconfig).
// ---------------------------------------------------------------------------
namespace {

template <uint32_t SCOPE, ckl::Dst Slot = ckl::Dst::D0>
struct RsqrtScoped : ckl::UnaryOp<RsqrtScoped<SCOPE, Slot>, Slot> {
    static ALWI void init() { ckernel::rsqrt_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        const uint32_t slot = ckl::to_u32(Slot) + slot_offset;
        if constexpr (SCOPE == SCOPE_CSKIP) {
            MATH((_llk_math_eltwise_unary_sfpu_params_((bench::rsqrt_body<2, 4>), slot, VectorMode::C)));
        } else {
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_rsqrt,
                (APPROX, 8 /* ITERATIONS */, DST_ACCUM_MODE, false /* FAST_APPROX */, false /* legacy */),
                slot,
                SCOPE == SCOPE_RC ? VectorMode::RC : VectorMode::C));
        }
    }
};

template <uint32_t SCOPE, ckl::Dst Slot = ckl::Dst::D0>
struct AddUnaryScoped : ckl::UnaryOp<AddUnaryScoped<SCOPE, Slot>, Slot> {
    uint32_t param;
    constexpr explicit AddUnaryScoped(uint32_t p) noexcept : param(p) {}
    static ALWI void init() { ckernel::binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        const uint32_t slot = ckl::to_u32(Slot) + slot_offset;
        if constexpr (SCOPE == SCOPE_CSKIP) {
            MATH((_llk_math_eltwise_unary_sfpu_params_((bench::add_body<2, 4>), slot, VectorMode::C, param)));
        } else {
            MATH(SFPU_UNARY_CALL(
                DST_SYNC_MODE,
                DST_ACCUM_MODE,
                calculate_binop_with_scalar,
                (APPROX, ckernel::ADD_UNARY, 8 /* ITERATIONS */),
                slot,
                SCOPE == SCOPE_RC ? VectorMode::RC : VectorMode::C,
                param));
        }
    }
};

// ONE element for `rsqrt(x + eps)`. Being a single SFPU type also makes the chain
// SFPU-init-uniform, which is what lets eltwise_chain boot-hoist the init.
template <uint32_t SCOPE, bool ROUNDTRIP, ckl::Dst Slot = ckl::Dst::D0>
struct AddRsqrtFused : ckl::UnaryOp<AddRsqrtFused<SCOPE, ROUNDTRIP, Slot>, Slot> {
    uint32_t param;
    constexpr explicit AddRsqrtFused(uint32_t p) noexcept : param(p) {}
    static ALWI void init() { ckernel::rsqrt_tile_init<false>(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        const uint32_t slot = ckl::to_u32(Slot) + slot_offset;
        if constexpr (SCOPE == SCOPE_CSKIP) {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                (bench::fused_body<2, 4, ROUNDTRIP>), slot, VectorMode::C, param)));
        } else {
            MATH((_llk_math_eltwise_unary_sfpu_params_(
                (bench::fused_body<1, 8, ROUNDTRIP>),
                slot,
                SCOPE == SCOPE_RC ? VectorMode::RC : VectorMode::C,
                param)));
        }
    }
};

}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(1);
    const uint32_t rows_t = get_arg_val<uint32_t>(0);

    {
        MaybeDeviceZoneScope("cp_hw_startup");
        compute_kernel_hw_startup(cb_stat_sum, cb_stat_sum, cb_rstd_send);
    }

    // The stat tiles are a resident L1 shard — mark them available once, exactly as
    // the op's combine leaves them in cb_stat_sum.
    cb_reserve_back(cb_stat_sum, rows_t);
    cb_push_back(cb_stat_sum, rows_t);

    {
        MaybeDeviceZoneScope("cp_finalize");
        if constexpr (VARIANT == 0) {
            // ablation floor: copy + pack only, no SFPU (== RMSN_ABLATE_FINALIZE 1).
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
        } else if constexpr (VARIANT == 1) {
            // pre-Refinement-5 shape: both ops over the whole tile.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                AddUnaryScoped<SCOPE_RC>{EPS_BITS},
                RsqrtScoped<SCOPE_RC>{},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
        } else if constexpr (VARIANT == 2) {
            // BASELINE — what rms_norm_compute.cpp does today.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                AddUnaryScoped<SCOPE_C>{EPS_BITS},
                RsqrtScoped<SCOPE_C>{},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
        } else if constexpr (VARIANT == 3) {
            // pure vector-level skip: same two elements, even-parity stride.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                AddUnaryScoped<SCOPE_CSKIP>{EPS_BITS},
                RsqrtScoped<SCOPE_CSKIP>{},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
        } else if constexpr (VARIANT == 4) {
            // fusion only (no parity skip), bit-exact.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                AddRsqrtFused<SCOPE_C, true>{EPS_BITS},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
        } else if constexpr (VARIANT == 5) {
            // THE CANDIDATE: fusion + parity skip, sum kept in an LREG (fp32).
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                AddRsqrtFused<SCOPE_CSKIP, false>{EPS_BITS},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
        } else if constexpr (VARIANT == 6) {
            // the candidate's bit-exact twin (sum round-tripped through DEST).
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                AddRsqrtFused<SCOPE_CSKIP, true>{EPS_BITS},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
        } else {
            // VARIANT == 7: fusion + parity skip over the WHOLE tile (RC), for the
            // vector-count ladder only.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<ckl::input(cb_stat_sum)>{},
                AddRsqrtFused<SCOPE_RC, false>{EPS_BITS},
                ckl::PackTile<ckl::output(cb_rstd_send)>{});
        }
    }
}
