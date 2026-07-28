# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off for rms_norm PHASE 4's two round-1 follow-ups (Perf 2):

  (a) narrow `cb_rms_sum` / `cb_rms_recip` from Float32 to Bfloat16 (free at
      `fp32_dest_acc_en=False`, load-bearing at `True`).
  (b) pack the `ht` per-tile-row statistics of a row-block into ONE tile (tile-row h's
      32 row-sums into column h), run ONE SFPU rsqrt pass over the packed tile instead of
      `ht` separate cskip passes, then extract each column back to a column-0-only tile
      for phase 5's `mul_tiles_bcast<BroadcastType::COL>` consumer (verified by looking at
      `tt_metal/hw/inc/api/compute/bcast.h`: `mul_tiles_bcast<COL>` takes a CB + a TILE INDEX,
      never a column offset -- there is no knob to make it read column h, so the extract is
      structurally required, not a convenience).

WHAT IS ISOLATED
----------------
Same isolation contract as round 1's `rsqrt_lane_and_window`: `ht` resident tiles in
`cb_rms_sum`, `ht` tiles out in `cb_rms_recip`, both backed directly on L1-sharded tensors
(`cb_descriptor_from_sharded_tensor`) -- zero NoC traffic, whole device kernel duration is
phase-4 compute. The row-block loop (`n_groups` groups of `ht` tiles) mirrors the op.

THE PACK / EXTRACT MECHANISM (raw compute API, adapted from the `gather_payload_shrink`
sibling's device-verified column-pack -- see the kernel-head comment in
`kernels_source()` for the full raw-LLK justification and the (face,row,col) -> (j,k)
address derivation this bench relies on).

MODES
-----
  baseline    : today's op, byte-for-byte (`ht` separate per-tile eltwise_chain windows,
                `RsqrtAddUnaryColZero`, VectorMode::C + even-parity stride).
  pack_here   : build the packed statistic tile LOCALLY (raw FPU reduce with a one-hot
                [output-column h, input-column 0] scaler bank, "PICK0"), fuse the rsqrt SFPU
                pass into the SAME dst-sync window (no separate copy-in), pack the packed
                tile to a compute-internal CB, then extract each column back to a
                column-0-only tile via a second one-hot scaler bank ("COLSEL").
  pack_given  : same as pack_here but SKIPS the pack step -- the packed tile is assumed to
                already be resident, as if it arrived as the cross-core gather payload from
                the `gather_payload_shrink` sibling's column-packed multicast. Only a
                copy-in + fused rsqrt + extract is paid.

SCOPE (which SFPU vector-mode footprint covers the packed tile's `ht` valid columns)
  c      : columns land CONTIGUOUSLY at 0..ht-1 (both parities), rsqrt uses VectorMode::C
           (16 vectors/pass, faces {0,2}). Valid for any ht in [1, 16].
  cskip  : columns land at EVEN positions 0,2,4,...,2*(ht-1) (odd columns never used),
           rsqrt uses VectorMode::C + even-parity stride (8 vectors/pass, same as the
           shipped single-tile fast path). Requires ht <= 8.

STAT_BF16 (sub-lever (a), orthogonal axis): narrows `cb_in` / `cb_packed_in` /
`cb_packed_stat` / `cb_out` from Float32 to Bfloat16. The one-hot scaler banks (PICK0 /
COLSEL) always stay Float32 -- they are tiny (mostly zero) and carry no user data, so the
precision contract does not apply to them.

PRECISION CONTRACT (fixed, never a lever): every variant runs under the SAME
`ComputeConfigDescriptor` (HiFi2 / `fp32_dest_acc_en` as requested / `math_approx_mode=False`
/ `dst_full_sync_en=False`). The rsqrt body is the IDENTICAL stock accurate kernel
(`_calculate_sqrt_body_<APPROX, RECIPROCAL=true, FAST_APPROX=false>`) round-1 already
shipped, reused verbatim -- this bench changes SFPU LANE COUNT and PACK LAYOUT only, never
the function or its precision.
"""

from __future__ import annotations

import torch

import ttnn

TILE = 32

# ---- CB indices --------------------------------------------------------------------
CB_IN = 0  # ht separate tiles (op: cb_rms_sum). MODE baseline / pack_here.
CB_PACKED_IN = 4  # 1 tile per group, pre-packed "arrived via mcast". MODE pack_given.
CB_PICK0 = 8  # ht resident scaler tiles for the PACK step. MODE pack_here only.
CB_COLSEL = 9  # ht resident scaler tiles for the EXTRACT step. pack_here / pack_given.
CB_PACKED_STAT = 12  # compute-internal: the packed tile, post-rsqrt.
CB_OUT = 16  # ht separate tiles out (op: cb_rms_recip). All modes.

ZONE_NAME = "PH4PACK"

_MODE_BASELINE, _MODE_PACK_HERE, _MODE_PACK_GIVEN = 0, 1, 2
_SCOPE_C, _SCOPE_CSKIP = 0, 1

# name -> (mode, scope, stat_bf16)
VARIANTS = {
    "baseline": (_MODE_BASELINE, _SCOPE_C, 0),
    "baseline_bf16": (_MODE_BASELINE, _SCOPE_C, 1),
    "pack_here_c": (_MODE_PACK_HERE, _SCOPE_C, 0),
    "pack_here_c_bf16": (_MODE_PACK_HERE, _SCOPE_C, 1),
    "pack_here_cskip": (_MODE_PACK_HERE, _SCOPE_CSKIP, 0),  # ht <= 8 only
    "pack_given_c": (_MODE_PACK_GIVEN, _SCOPE_C, 0),
    "pack_given_c_bf16": (_MODE_PACK_GIVEN, _SCOPE_C, 1),
    "pack_given_cskip": (_MODE_PACK_GIVEN, _SCOPE_CSKIP, 0),  # ht <= 8 only
}
BASELINE = "baseline"
CSKIP_VARIANTS = ("pack_here_cskip", "pack_given_cskip")

LABEL = {
    "baseline": "today's op: ht separate cskip passes, fp32 CBs",
    "baseline_bf16": "today's op, bf16 CBs (sub-lever a alone)",
    "pack_here_c": "pack LOCALLY (PICK0 reduce, fused rsqrt) + extract (COLSEL), scope=C",
    "pack_here_c_bf16": "pack_here_c + bf16 CBs (a+b composed)",
    "pack_here_cskip": "pack LOCALLY at EVEN columns + cskip rsqrt (8 vec) + extract",
    "pack_given_c": "packed tile ASSUMED resident (sibling composition) + rsqrt(C) + extract",
    "pack_given_c_bf16": "pack_given_c + bf16 CBs",
    "pack_given_cskip": "packed tile pre-placed at EVEN cols + cskip rsqrt + extract",
}


def core_range_set(grid_x, grid_y):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))])


def sharded_memory_config(tiles_per_core, grid_x, grid_y):
    return ttnn.create_sharded_memory_config(
        shape=(tiles_per_core * TILE, TILE),
        core_grid=core_range_set(grid_x, grid_y),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def tensor_height(tiles_per_core, grid_x, grid_y):
    return tiles_per_core * TILE * grid_x * grid_y


def compute_config(fp32_dest_acc_en=False):
    """The focus case's user precision contract. IDENTICAL for every variant -- never a lever."""
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        fp32_dest_acc_en=fp32_dest_acc_en,
        math_approx_mode=False,
        dst_full_sync_en=False,
    )


def _uniform_runtime_args(grid_x, grid_y, args):
    rt = ttnn.RuntimeArgs()
    for x in range(grid_x):
        for y in range(grid_y):
            rt[x][y] = list(args)
    return rt


# =============================================================================
# Scaler bank construction (PICK0 / COLSEL) -- built as ORDINARY (row, col) torch
# tensors and handed to `ttnn.from_torch(..., layout=TILE_LAYOUT)`, which performs the
# logical-(row,col) -> physical-(face, row-in-face, col-in-face) tilization for us. This
# is the SAME physical byte layout the raw kernels' `put(face,row,col,v)` helpers target
# (standard 4-face tile decomposition: face = 2*(row>=16) + (col>=16); row-in-face =
# row%16; col-in-face = col%16) -- so a plain row-major torch build is exactly as
# correct as, and far less error-prone than, a hand-rolled raw-L1 fill.
#
# DERIVATION (see the kernel-head comment for the MVMUL model dest[i,j] = sum_k W[j,k] *
# data[i,k], scaler W addressed by its own (row=j, col=k)):
#   PICK0[h]:  W[h,0] = W[h+16,0] = 1, else 0   -> dest[i,h] = data[i,0]
#              ("take column 0 of the source tile, place it at destination column h")
#   COLSEL[h]: W[0,h] = W[16,h] = 1, else 0     -> dest[i,0] = data[i,h]
#              ("take column h of the packed tile, place it at destination column 0")
# CSKIP variants target column 2*h instead of h (h in [0,7]), so the packed values land
# on EVEN columns only and the rsqrt pass can use the narrower even-parity-stride scope.
# =============================================================================
def _one_hot_bank(ht, col_of_h, out_col_of_h=None):
    """[ht*32, 32] torch tensor: tile h has W[j, col_of_h(h)] = 1 at j in {row, row+16}.

    `out_col_of_h` selects which OUTPUT/COLUMN role h plays; when None (PICK0 case) the
    varying axis is the output row-pair (h, h+16) and the fixed column is `col_of_h(h)`
    (== 0, the source's valid column). When given (COLSEL case) the roles swap: the
    output row-pair is fixed at (0, 16) and the varying axis is the column `col_of_h(h)`.
    """
    bank = torch.zeros(ht * TILE, TILE, dtype=torch.float32)
    for h in range(ht):
        tile = torch.zeros(TILE, TILE, dtype=torch.float32)
        c = col_of_h(h)
        if out_col_of_h is None:
            # COLSEL: output row-pair FIXED at (0, 16) regardless of h; only the input
            # column `c` varies. (Bug fixed: this branch used to key the row on the loop
            # variable `h` instead of 0, which happened to be a no-op for h==0 -- the row-0
            # extract worked -- but placed every h>=1's result at DEST column h instead of
            # 0, which the extract's own packer edge mask then zeroed away entirely.)
            tile[0, c] = 1.0
            tile[16, c] = 1.0
        else:
            r = out_col_of_h(h)
            tile[r, c] = 1.0
            tile[r + 16, c] = 1.0
        bank[h * TILE : (h + 1) * TILE, :] = tile
    return bank


def build_pick0_bank(ht, scope):
    """PICK0[h]: select source column 0, place at destination column target(h)."""
    target = (lambda h: h) if scope == _SCOPE_C else (lambda h: 2 * h)
    # varying axis = OUTPUT row-pair (target(h), target(h)+16); fixed column = 0 (source).
    return _one_hot_bank(ht, col_of_h=lambda h: 0, out_col_of_h=target)


def build_colsel_bank(ht, scope):
    """COLSEL[h]: select packed column target(h), place at destination column 0."""
    target = (lambda h: h) if scope == _SCOPE_C else (lambda h: 2 * h)
    # varying axis = INPUT column target(h); fixed output row-pair = (0, 16).
    return _one_hot_bank(ht, col_of_h=target, out_col_of_h=None)


def pack_columns(col0_values, ht, n_groups, scope):
    """Host-side reference pack: col0_values [n_groups*ht*32] (flat, row-major within a
    core) -> [n_groups*32, 32] packed tiles (one packed tile per group), used to build the
    `pack_given` input tensor directly from the SAME per-tile column-0 values `cb_in`
    carries, so pack_given's correctness does not depend on `pack_here`'s own reduce.
    """
    target = (lambda h: h) if scope == _SCOPE_C else (lambda h: 2 * h)
    data = col0_values.reshape(n_groups, ht, TILE)  # [g, h, row]
    packed = torch.zeros(n_groups, TILE, TILE, dtype=col0_values.dtype)
    for h in range(ht):
        packed[:, :, target(h)] = data[:, h, :]
    return packed.reshape(n_groups * TILE, TILE)


# =============================================================================
# Compute kernel
# =============================================================================
def kernel_source():
    return r"""
// Isolated rms_norm PHASE-4 bench (Perf 2, phase4_packed_rsqrt_and_bf16_stats).
//
// MODES (CT arg 0): 0 = baseline (today's op, byte-for-byte), 1 = pack_here (build the
// packed tile locally, fuse rsqrt into the same window, extract), 2 = pack_given (packed
// tile already resident -- e.g. arrived from the gather_payload_shrink sibling's mcast --
// only copy-in + fused rsqrt + extract is paid).
// SCOPE (CT arg 1): 0 = C (columns 0..ht-1 contiguous, VectorMode::C, 16 vec/pass, any
// ht<=16), 1 = CSKIP (columns 0,2,..,2*(ht-1), VectorMode::C + parity stride, 8 vec/pass,
// ht<=8).
// STAT_BF16 (CT arg 2): narrows cb_in/cb_packed_in/cb_packed_stat/cb_out to Bfloat16
// (sub-lever a). The PICK0/COLSEL scaler banks always stay Float32.
//
// RAW-LLK JUSTIFICATION for the pack/extract mechanism (adapted from the
// gather_payload_shrink sibling's device-verified column-pack; that idea's own
// probe_mechanism.py established the (scaler position) -> (dest position) map on real
// silicon). BH's REDUCE_ROW SUM is an MVMUL with the scaler tile as the WEIGHT matrix
// (SrcA) and the data as the moving operand (SrcB):
//     dest[i, j] = sum_k W[j, k] * data[i, k]
// where the scaler's own tile-relative (face, row-in-face, col-in-face) address IS (j, k)
// via the standard 4-face decomposition (row-in-face + 16*bottom_face = j, col-in-face +
// 16*right_face = k) -- i.e. W is just an ordinary 32x32 tile addressed by its own (row,
// col). No kernel_lib reduce helper can express a non-canonical W (ckl::reduce /
// reduce_mean take exactly ONE canonical "row 0 of every face" scaler), so this is raw
// compute-API by necessity:
//   PICK0[h]:  W[h,0] = W[h+16,0] = 1, else 0
//              -> dest[i,h] = data[i,0]: "take column 0 of source tile h, place it at
//                 destination column h (or 2h under CSKIP)". `ht` reduce_tile calls (one
//                 source tile per h, all writing idst=0) accumulate into ONE dest tile
//                 because each call's nonzero output column is disjoint from every other
//                 call's -- non-interference verified by the sibling's probe (tile 9).
//   COLSEL[h]: W[0,h] = W[16,h] = 1, else 0
//              -> dest[i,0] = data[i,h]: the inverse selector.
// PICK0 needs `reduce_uninit()` between `tile_regs_commit` and the pack -- reduce_init's
// default packer edge mask force-zeros every output column but 0, which would erase
// PICK0's whole point (a non-zero column h). COLSEL wants column-0-only output, which the
// mask already gives for free, so it is left ON through the whole per-h extract loop and
// cleared ONCE afterward (mirrors the sibling's step-2 exactly).
#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_math.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise_scalar.hpp"

#ifdef TRISC_MATH
#include "ckernel_sfpu_sqrt.h"
#include "sfpu/ckernel_sfpu_converter.h"
#endif

namespace ckl = compute_kernel_lib;
using ckernel::VectorMode;

constexpr uint32_t cb_in = 0;
constexpr uint32_t cb_packed_in = 4;
constexpr uint32_t cb_pick0 = 8;
constexpr uint32_t cb_colsel = 9;
constexpr uint32_t cb_packed_stat = 12;
constexpr uint32_t cb_out = 16;

constexpr uint32_t MODE_BASELINE = 0, MODE_PACK_HERE = 1, MODE_PACK_GIVEN = 2;
constexpr uint32_t SCOPE_C = 0, SCOPE_CSKIP = 1;

// ---------------------------------------------------------------------------
// The shipped fast-path element (round 1, RsqrtAddUnaryColZero), reused verbatim as the
// `baseline` mode so the comparison is apples-to-apples in ONE binary.
// ---------------------------------------------------------------------------
#ifdef TRISC_MATH
sfpi_inline sfpi::vFloat rsqrt_body(sfpi::vFloat x, uint32_t eps_bits) {
    const sfpi::vFloat eps = ckernel::sfpu::Converter::as_float(eps_bits);
    sfpi::vFloat t = ckernel::sfpu::_calculate_sqrt_body_<APPROX, true /*RECIPROCAL*/, false /*FAST_APPROX*/>(
        x + eps);
    if constexpr (!DST_ACCUM_MODE) {
        t = sfpi::convert<sfpi::vFloat16b>(t, sfpi::RoundMode::Nearest);
    }
    return t;
}

// Fused rsqrt(x+eps) over NVEC vector-ops at DEST-address stride STRIDE, starting at the
// current dst_reg cursor (caller positions it via VectorMode + the SFPU loop harness).
template <int NVEC, int STRIDE>
sfpi_inline void rsqrt_body_strided(uint32_t eps_bits) {
    for (int d = 0; d < NVEC; d++) {
        sfpi::dst_reg[0] = rsqrt_body(sfpi::dst_reg[0], eps_bits);
        sfpi::dst_reg += STRIDE;
    }
}
#endif

namespace compute_kernel_lib {
template <Dst Slot = Dst::D0>
struct RsqrtAddUnaryColZero : UnaryOp<RsqrtAddUnaryColZero<Slot>, Slot> {
    uint32_t eps_bits;
    constexpr explicit RsqrtAddUnaryColZero(uint32_t e) noexcept : eps_bits(e) {}
    static ALWI void init() { rsqrt_tile_init(); }
    ALWI void exec(uint32_t /*i*/, uint32_t slot_offset) const {
        const uint32_t idst = to_u32(Slot) + slot_offset;
        const uint32_t eps = eps_bits;
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            [eps]() { rsqrt_body_strided<4, 2>(eps); }, idst, VectorMode::C)));
    }
};
}  // namespace compute_kernel_lib

// One SFPU pass over DEST slot `idst`, scope selected at compile time: C covers the whole
// low-column half (16 vectors -- columns 0..15, both parities); CSKIP covers only the
// even-parity vectors (8 vectors -- columns 0,2,..,14).
template <uint32_t scope>
ALWI void packed_rsqrt_sfpu(uint32_t idst, uint32_t eps_bits) {
#ifdef TRISC_MATH
    if constexpr (scope == SCOPE_CSKIP) {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            [eps_bits]() { rsqrt_body_strided<4, 2>(eps_bits); }, idst, VectorMode::C)));
    } else {
        MATH((_llk_math_eltwise_unary_sfpu_params_(
            [eps_bits]() { rsqrt_body_strided<8, 1>(eps_bits); }, idst, VectorMode::C)));
    }
#else
    (void)idst;
    (void)eps_bits;
#endif
}

void kernel_main() {
    constexpr uint32_t MODE = get_compile_time_arg_val(0);
    constexpr uint32_t SCOPE = get_compile_time_arg_val(1);
    constexpr uint32_t STAT_BF16 = get_compile_time_arg_val(2);
    (void)STAT_BF16;  // CB *formats* carry the dtype; nothing here branches on it.
    // LIVENESS PROVEN (not re-checked on every run): a deliberate
    // `static_assert(!(MODE == MODE_PACK_GIVEN && SCOPE == SCOPE_CSKIP), ...)` inserted here
    // failed the build on trisc0 (chlkc_unpack.cpp), trisc1 (chlkc_math.cpp) AND trisc2
    // (chlkc_pack.cpp) when running the `pack_given_cskip` variant -- confirming the fast
    // path is genuinely compiled for that CT-arg combination, not dead code behind a stale
    // JIT cache.

    const uint32_t ht = get_arg_val<uint32_t>(0);
    const uint32_t n_groups = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits = get_arg_val<uint32_t>(2);

    constexpr uint32_t DEST_BLOCK = ckl::DEST_AUTO_LIMIT;

    // cb_in is UNDECLARED in MODE_PACK_GIVEN's program (it only attaches cb_packed_in), so
    // the startup call must reference CBs present in EVERY mode's descriptor -- cb_colsel
    // and cb_out both are (see `_tensors_for_variant`). Referencing an undeclared CB here
    // corrupted operand tracking for the OTHER real CBs (measured: pack_given_c read back
    // PCC 0.889 instead of failing loudly -- a silent format-state corruption, not a crash).
    compute_kernel_hw_startup(cb_colsel, cb_colsel, cb_out);

    // The CBs backed on resident L1 shards (`cb_descriptor_from_sharded_tensor`) are
    // physically already there but the CB's own read/write-pointer bookkeeping still needs
    // an explicit publish before any `cb_wait_front` on it can succeed -- mirrors round 1's
    // "publish once" note on `cb_in` verbatim, extended to every zero-copy CB this kernel
    // reads (`cb_in`/`cb_packed_in` per MODE, `cb_pick0`/`cb_colsel` unconditionally on the
    // packed paths).
    if constexpr (MODE == MODE_BASELINE || MODE == MODE_PACK_HERE) {
        cb_reserve_back(cb_in, ht * n_groups);
        cb_push_back(cb_in, ht * n_groups);
    }
    if constexpr (MODE == MODE_PACK_GIVEN) {
        cb_reserve_back(cb_packed_in, n_groups);
        cb_push_back(cb_packed_in, n_groups);
    }
    if constexpr (MODE != MODE_BASELINE) {
        // PICK0 / COLSEL are true CONSTANTS: published once, waited once here, held
        // resident for the whole kernel, never popped (R8 pattern -- cb_gamma).
        if constexpr (MODE == MODE_PACK_HERE) {
            cb_reserve_back(cb_pick0, ht);
            cb_push_back(cb_pick0, ht);
            cb_wait_front(cb_pick0, ht);
        }
        cb_reserve_back(cb_colsel, ht);
        cb_push_back(cb_colsel, ht);
        cb_wait_front(cb_colsel, ht);
    }

    for (uint32_t g = 0; g < n_groups; ++g) {
        DeviceZoneScopedN("PH4PACK");
        if constexpr (MODE == MODE_BASELINE) {
            // ---- EXACTLY the op's current phase 4 (post round-1 graduation) ----
            ckl::eltwise_chain(
                ckl::EltwiseShape::tiles(ht),
                ckl::CopyTile<cb_in, ckl::Dst::D0, ckl::InputLifecycle::Streaming>{},
                ckl::RsqrtAddUnaryColZero<ckl::Dst::D0>{eps_bits},
                ckl::PackTile<cb_out, ckl::OutputLifecycle::Streaming>{});
        } else {
            // ================= STEP 1: get the RSQRT'd packed tile ==================
            if constexpr (MODE == MODE_PACK_HERE) {
                // ---- PACK (raw FPU reduce, PICK0) fused with RSQRT in ONE window ----
                cb_wait_front(cb_in, ht);
                cb_reserve_back(cb_packed_stat, 1);
                reconfig_data_format(cb_pick0, cb_in);  // scaler->SrcA, data->SrcB (REDUCE_ROW)
                pack_reconfig_data_format(cb_packed_stat);
                reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                    cb_in, cb_pick0, cb_packed_stat);
                tile_regs_acquire();
                for (uint32_t h = 0; h < ht; ++h) {
                    reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                        cb_in, cb_pick0, /*itile=*/h, /*itile_scaler=*/h, /*idst=*/0);
                }
                // Fused: rsqrt(x+eps) runs on the SAME DEST slot, in the SAME window --
                // no separate copy-in / pack-out round trip for the raw packed sum.
                packed_rsqrt_sfpu<SCOPE>(0, eps_bits);
                tile_regs_commit();
                reduce_uninit();  // MUST precede the pack -- see the kernel-head note.
                tile_regs_wait();
                pack_tile(0, cb_packed_stat);
                tile_regs_release();
                cb_push_back(cb_packed_stat, 1);
                cb_pop_front(cb_in, ht);
            } else {
                // ---- PACK_GIVEN: packed tile already resident, just copy-in + rsqrt ----
                cb_wait_front(cb_packed_in, 1);
                cb_reserve_back(cb_packed_stat, 1);
                // `copy_tile_to_dst_init_short` documents "this does NOT reconfigure the
                // unpacker data types" -- UNPACK SrcA is still whatever hw_startup last set
                // (cb_colsel, Float32). Without this call, a bf16 cb_packed_in is read back
                // through an fp32-configured unpacker and silently misinterpreted (measured:
                // PCC ~0.59, not a crash -- a believable-looking but WRONG value).
                reconfig_data_format_srca(cb_packed_in);
                pack_reconfig_data_format(cb_packed_stat);  // cb_packed_in may be bf16 or fp32
                tile_regs_acquire();
                copy_tile_to_dst_init_short(cb_packed_in);
                copy_tile(cb_packed_in, 0, 0);
                packed_rsqrt_sfpu<SCOPE>(0, eps_bits);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_packed_stat);
                tile_regs_release();
                cb_push_back(cb_packed_stat, 1);
                cb_pop_front(cb_packed_in, 1);
            }

            // ================= STEP 2: extract ht columns -> ht col-0 tiles =========
            cb_wait_front(cb_packed_stat, 1);
            cb_reserve_back(cb_out, ht);
            reconfig_data_format(cb_colsel, cb_packed_stat);
            pack_reconfig_data_format(cb_out);
            reduce_init<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                cb_packed_stat, cb_colsel, cb_out);
            for (uint32_t base = 0; base < ht; base += DEST_BLOCK) {
                const uint32_t n = (ht - base < DEST_BLOCK) ? (ht - base) : DEST_BLOCK;
                tile_regs_acquire();
                for (uint32_t j = 0; j < n; ++j) {
                    reduce_tile<ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
                        cb_packed_stat, cb_colsel, /*itile=*/0, /*itile_scaler=*/base + j, /*idst=*/j);
                }
                tile_regs_commit();
                // Packer edge mask stays ON: it already zeros everything but column 0,
                // exactly the output shape we want here (mirrors the sibling's step 2).
                tile_regs_wait();
                for (uint32_t j = 0; j < n; ++j) {
                    pack_tile(j, cb_out, base + j);
                }
                tile_regs_release();
            }
            reduce_uninit();
            cb_push_back(cb_out, ht);
            cb_pop_front(cb_packed_stat, 1);
        }
    }
}
"""


# =============================================================================
# Host side — program descriptor
# =============================================================================
def create_program_descriptor(
    tensors,
    *,
    variant,
    ht,
    n_groups,
    eps_bits,
    grid_x,
    grid_y,
    fp32_dest_acc_en=False,
):
    """`tensors` = dict of the tensors this variant actually needs:
    {'in': ..., 'out': ..., 'packed_in': ..., 'pick0': ..., 'colsel': ...} (only the keys
    the selected variant/mode uses need be present).
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {variant!r}")
    mode, scope, stat_bf16 = VARIANTS[variant]
    cores = core_range_set(grid_x, grid_y)
    compute = ttnn.KernelDescriptor(
        kernel_source=kernel_source(),
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[mode, scope, stat_bf16],
        runtime_args=_uniform_runtime_args(grid_x, grid_y, [ht, n_groups, eps_bits]),
        config=compute_config(fp32_dest_acc_en),
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_IN, tensors["in"])] if "in" in tensors else []
    if "packed_in" in tensors:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_PACKED_IN, tensors["packed_in"]))
    if "pick0" in tensors:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_PICK0, tensors["pick0"]))
    if "colsel" in tensors:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_COLSEL, tensors["colsel"]))
    cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, tensors["out"]))
    # cb_packed_stat is compute-internal: no tensor backs it, just a plain program CB.
    packed_stat_dtype = ttnn.bfloat16 if stat_bf16 else ttnn.float32
    packed_stat_bytes = ttnn.tile_size(packed_stat_dtype)
    cbs.append(
        ttnn.CBDescriptor(
            total_size=packed_stat_bytes * 2,
            core_ranges=cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_PACKED_STAT, data_format=packed_stat_dtype, page_size=packed_stat_bytes
                )
            ],
        )
    )
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def run_op(tensors, *, variant, ht, n_groups, eps_bits, grid_x, grid_y, fp32_dest_acc_en=False):
    descriptor = create_program_descriptor(
        tensors,
        variant=variant,
        ht=ht,
        n_groups=n_groups,
        eps_bits=eps_bits,
        grid_x=grid_x,
        grid_y=grid_y,
        fp32_dest_acc_en=fp32_dest_acc_en,
    )
    io_tensors = [t for t in tensors.values()]
    return ttnn.generic_op(io_tensors, descriptor)


def dtype_of(bf16: bool):
    return ttnn.bfloat16 if bf16 else ttnn.float32
