// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// ISOLATED BENCH (perf experiment `batched_finalize`, Perf-2 idea I12) — rms_norm's
// ROOT FINALIZE chain and nothing else. NOT part of the op; never dispatched by
// rms_norm.py.
//
// WHAT IS ISOLATED
// The op's root finalize is today
//     eltwise_chain(EltwiseShape::tiles(rows_t),
//                   CopyTile<input(cb_stat_sum)>,          // DEFAULT policies:
//                   AddUnaryColValid{eps},                 //   WaitPolicy::PerTile
//                   RsqrtColValid{},                       //   PopPolicy::PerTile
//                   PackTile<output(cb_rstd_send)>)        //   Reserve/Push PerTile
// over `rows_t` column-0-valid stat tiles. Those PER-TILE (Streaming) CB policies make
// `chain_supports_block_v` FALSE, so eltwise_chain CLAMPS block_size to 1
// (eltwise_chain.inl:3054) and the per-tile body becomes:
//     tile_regs_acquire -> [copy, SFPU init, add] [SFPU init, rsqrt] -> commit/wait
//     -> pack -> release
// i.e. one DEST-sync window AND two SFPU inits PER STAT TILE. (The add and the rsqrt
// are two distinct SFPU element types, so `chain_sfpu_inits_uniform_v` is false and
// neither init is boot-hoisted — eltwise_chain.inl:2751-2757 re-emits them inside the
// walk.) On the single-tile-row decode geometry rows_t == 1 and none of that matters;
// on BLOCK_SHARDED (1,1,8192,1024) rows_t == 32 and this is the SERIALIZED ROOT cost
// that 56 other cores wait on.
//
// THE IDEA UNDER TEST (I12): BATCH the finalize across tile-rows — one DEST-sync
// window and one SFPU init per BLOCK of B stat tiles instead of per tile. The lever
// already exists in the helper: `EltwiseShape::tiles(rows_t, B)`. It is unreachable
// today ONLY because of the input/output CB policies, so the candidates swap those to
// a block-capable family:
//   * PerChunk / PerChunk  — same streaming semantics as today, at B-tile granularity
//     (the writer still sees rstd tiles pushed in B-sized groups, not all at the end).
//   * Upfront / AtEnd      — the whole rows_t window at once; strictly fewer CB ops,
//     but the first rstd tile is not published until the last one is finalized.
// Plus the round-1 (`cskip_finalize`) FUSION of `+eps` and `rsqrt` into ONE SFPU
// element, which additionally makes the chain SFPU-init-UNIFORM so the single
// remaining init is boot-hoisted out of the walk entirely.
//
// RAW-LLK / HELPER-BYPASS JUSTIFICATION — carried over verbatim from the op, NOT
// introduced here. The `*ColValid` elements hand a hand-written `sfpi` body to
// `_llk_math_eltwise_unary_sfpu_params_` (the same wrapper `SFPU_UNARY_CALL` uses)
// because the helper-side entry points `ckernel::rsqrt_tile` / `add_unary_tile`
// HARDCODE `VectorMode::RC`, and because NO wrapper anywhere exposes the SFPU's
// per-vector DEST address stride — which is the only handle on the even/odd COLUMN
// PARITY walk inside a face, the axis on which a column-0-valid result needs just 4 of
// 8 vectors. `_calculate_sqrt_body_` / the scalar-add body are reused verbatim, so this
// changes WHICH vectors run, never the arithmetic in one. The BATCHING itself
// (this experiment's actual idea) needs NO raw LLK: it is `EltwiseShape::tiles(n, B)`
// plus block-capable CB policies, all public helper API.
//
// PRECISION CONTRACT — FIXED AND IDENTICAL FOR EVERY VARIANT: bf16 in/out, HiFi2,
// fp32_dest_acc_en=False, math_approx_mode as the op sets it, rsqrt ITERATIONS
// unchanged, FAST_APPROX=false, legacy_compat=false. Batching cannot change a value:
// every variant computes the same per-tile arithmetic, only the DEST window and init
// placement move. The one variant that DOES change a value (`*_fused`, which keeps
// `x+eps` in an LREG at fp32 instead of round-tripping it through the 16-bit DEST) is
// strictly MORE accurate, and its bit-exact twin is provided so the difference is a
// measured menu entry rather than a hidden trade.

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
constexpr uint32_t cb_stat_sum = 0;    // rows_t resident column-0-valid stat tiles
constexpr uint32_t cb_rstd_send = 16;  // rows_t result tiles
}  // namespace

// ---------------------------------------------------------------------------
// The op's column-0-valid SFPU bodies, verbatim (see the header's justification).
// NET dst_reg ADVANCE MUST STAY +8 (== the stock ITERATIONS=8 == one face) or
// VectorMode::C's face0 -> face2 stepping desynchronizes: these do 4 x (+2).
// ---------------------------------------------------------------------------
#ifdef TRISC_MATH
namespace bench {

sfpi_inline void rsqrt_body() {
    for (int i = 0; i < 4; i++) {
        sfpi::vFloat t =
            ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(sfpi::dst_reg[0]);
        if constexpr (!DST_ACCUM_MODE) {
            t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
        }
        sfpi::dst_reg[0] = t;
        sfpi::dst_reg += 2;
    }
}

sfpi_inline void add_body(uint32_t param) {
    const sfpi::vFloat parameter = ckernel::sfpu::Converter::as_float(param);
    for (int i = 0; i < 4; i++) {
        sfpi::vFloat val = sfpi::dst_reg[0];
        sfpi::dst_reg[0] = val + parameter;
        sfpi::dst_reg += 2;
    }
}

// rsqrt(x + eps) in ONE pass over the even-parity scope.
//   ROUNDTRIP=true  — store the sum to DEST and read it back, so it is truncated to the
//                     DEST format exactly as the two-element chain does => BIT-IDENTICAL.
//   ROUNDTRIP=false — keep the sum in an LREG (fp32): strictly closer to the fp64
//                     reference, one DEST round trip cheaper.
template <bool ROUNDTRIP>
sfpi_inline void fused_body(uint32_t param) {
    const sfpi::vFloat parameter = ckernel::sfpu::Converter::as_float(param);
    for (int i = 0; i < 4; i++) {
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
        sfpi::dst_reg += 2;
    }
}

}  // namespace bench
#endif

// ---------------------------------------------------------------------------
// eltwise_chain ELEMENTS (the op's pattern: a missing block operation BUILT, so the
// helper keeps owning the DEST window / CB lifecycle / init / reconfig — including,
// crucially for this experiment, the BLOCK walk over `block_size` DEST lanes).
// ---------------------------------------------------------------------------
namespace {

template <ckl::Dst Slot = ckl::Dst::D0>
struct RsqrtColValid : ckl::UnaryOp<RsqrtColValid<Slot>, Slot> {
    static ALWI void init() { ckernel::rsqrt_tile_init<false>(); }
    static ALWI void exec_impl(uint32_t slot_offset) {
        [[maybe_unused]] const uint32_t slot = ckl::to_u32(Slot) + slot_offset;
        MATH((_llk_math_eltwise_unary_sfpu_params_((bench::rsqrt_body), slot, VectorMode::C)));
    }
};

template <ckl::Dst Slot = ckl::Dst::D0>
struct AddUnaryColValid : ckl::UnaryOp<AddUnaryColValid<Slot>, Slot> {
    uint32_t param;
    constexpr explicit AddUnaryColValid(uint32_t p) noexcept : param(p) {}
    static ALWI void init() { ckernel::binop_with_scalar_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        [[maybe_unused]] const uint32_t slot = ckl::to_u32(Slot) + slot_offset;
        MATH((_llk_math_eltwise_unary_sfpu_params_((bench::add_body), slot, VectorMode::C, param)));
    }
};

template <bool ROUNDTRIP, ckl::Dst Slot = ckl::Dst::D0>
struct AddRsqrtFused : ckl::UnaryOp<AddRsqrtFused<ROUNDTRIP, Slot>, Slot> {
    uint32_t param;
    constexpr explicit AddRsqrtFused(uint32_t p) noexcept : param(p) {}
    static ALWI void init() { ckernel::rsqrt_tile_init<false>(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        [[maybe_unused]] const uint32_t slot = ckl::to_u32(Slot) + slot_offset;
        MATH((_llk_math_eltwise_unary_sfpu_params_((bench::fused_body<ROUNDTRIP>), slot, VectorMode::C, param)));
    }
};

// ---- CB policy families -------------------------------------------------
// STREAMING == what the op does today (per-tile everything) => block clamped to 1.
constexpr auto in_stream = ckl::input(cb_stat_sum);
constexpr auto out_stream = ckl::output(cb_rstd_send);
// PER-CHUNK == the same streaming semantics at B-tile granularity, block-capable.
constexpr auto in_chunk =
    ckl::input(cb_stat_sum, ckl::WaitPolicy::PerChunk, ckl::PopPolicy::PerChunk, ckl::OperandKind::Block);
constexpr auto out_chunk = ckl::output(cb_rstd_send, ckl::ReservePolicy::PerChunk, ckl::PushPolicy::PerChunk);
// UPFRONT == the whole rows_t window in one go; fewest CB ops, latest publish.
constexpr auto in_upfront =
    ckl::input(cb_stat_sum, ckl::WaitPolicy::Upfront, ckl::PopPolicy::AtEnd, ckl::OperandKind::Block);
constexpr auto out_upfront = ckl::output(cb_rstd_send, ckl::ReservePolicy::Upfront, ckl::PushPolicy::AtEnd);

}  // namespace

void kernel_main() {
    constexpr uint32_t VARIANT = get_compile_time_arg_val(0);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(1);
    const uint32_t rows_t = get_arg_val<uint32_t>(0);
    const uint32_t blk = get_arg_val<uint32_t>(1);

    {
        MaybeDeviceZoneScope("cp_hw_startup");
        compute_kernel_hw_startup(cb_stat_sum, cb_stat_sum, cb_rstd_send);
    }

    // The stat tiles are a resident L1 shard — marked available once, exactly as the
    // op's combine leaves them in cb_stat_sum.
    cb_reserve_back(cb_stat_sum, rows_t);
    cb_push_back(cb_stat_sum, rows_t);

    {
        MaybeDeviceZoneScope("cp_finalize");
        if constexpr (VARIANT == 0) {
            // BASELINE — exactly what rms_norm_compute.cpp does today (block clamps to 1).
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<in_stream>{},
                AddUnaryColValid<>{EPS_BITS},
                RsqrtColValid<>{},
                ckl::PackTile<out_stream>{});
        } else if constexpr (VARIANT == 1) {
            // policy change ONLY (run this with blk = 1 to separate it from batching).
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t, blk),
                ckl::CopyTile<in_chunk>{},
                AddUnaryColValid<>{EPS_BITS},
                RsqrtColValid<>{},
                ckl::PackTile<out_chunk>{});
        } else if constexpr (VARIANT == 2) {
            // THE CANDIDATE: batched DEST window, same two elements.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t, blk),
                ckl::CopyTile<in_upfront>{},
                AddUnaryColValid<>{EPS_BITS},
                RsqrtColValid<>{},
                ckl::PackTile<out_upfront>{});
        } else if constexpr (VARIANT == 3) {
            // batched (per-chunk publish) + fusion (SFPU-init-uniform => init hoisted).
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t, blk),
                ckl::CopyTile<in_chunk>{},
                AddRsqrtFused<false>{EPS_BITS},
                ckl::PackTile<out_chunk>{});
        } else if constexpr (VARIANT == 4) {
            // batched (upfront) + fusion.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t, blk),
                ckl::CopyTile<in_upfront>{},
                AddRsqrtFused<false>{EPS_BITS},
                ckl::PackTile<out_upfront>{});
        } else if constexpr (VARIANT == 5) {
            // batched + fusion, bit-exact twin (sum round-tripped through DEST).
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t, blk),
                ckl::CopyTile<in_upfront>{},
                AddRsqrtFused<true>{EPS_BITS},
                ckl::PackTile<out_upfront>{});
        } else {
            // VARIANT == 6: fusion WITHOUT batching (round 1's `cskip_fused`), the
            // control that separates the fusion's win from the batching's.
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(rows_t),
                ckl::CopyTile<in_stream>{},
                AddRsqrtFused<false>{EPS_BITS},
                ckl::PackTile<out_stream>{});
        }
    }
}
