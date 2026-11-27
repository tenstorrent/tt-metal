// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// PMK KDA item cores — compute (TRISC): per (head, chunk) item
//   P4: q | k | v = silu(causal conv_4(u)) on the chunk's 32 rows (the three shifted copies are shift-matrix matmuls
//       against this chunk's and the previous chunk's u tiles: exact, fp32 DST), taps as row-0 broadcasts (SFPU);
//       g = lower_bound * sigmoid((f + dt_bias) * rate) (fp32, the OptimizedKDA._decay arithmetic); beta = sigmoid(b);
//       k, v, g, beta rows past the valid end multiplied by the row mask (identity steps, as the op's zeroed rows);
//   P5: the chunk terms of prepare_chunk_recurrence -- the compute body of
//       ttnn/cpp/ttnn/operations/experimental/kda/prepare_chunk_recurrence/device/kernels/compute/
//       prepare_chunk_recurrence.cpp ported helper by helper from the DataflowBuffer API to circular buffers
//       (reserve / push / wait / pop on CB ids, tile indices relative to the CB read pointer, outputs packed at fixed
//       offsets of ONE merged output CB). The L2-norm reduction uses the classic reduce_tile API with a full ones tile
//       (the op's compute_kernel_lib::reduce); the arithmetic (anchored decay factors, nested block inverse) is the
//       op's.
// Gates from kernels/kda_pre/compute.cpp (decay, sigmoid, row broadcasts).
// Compile-time (PH_CT0 +): Kt, VTF, NCH, CPH, SCALE_BITS (1/sqrt(d)), EPS_BITS (1e-6), LB_BITS (lower_bound).
// Runtime (ph_rt0 +): slot.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/cb_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/reduce.h"
#include "api/compute/sfpu_binary_bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mk/phase_args.hpp"
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/kda_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/compute/matmul_subblock.hpp"
#if defined(PMK_KDA_RNE) || \
    defined(PMK_KDA_RNE_PACK)  // (GLM53_PMK_KDA_RNE) tf32 RNE of every packed fp32 tile (14's prepare)
#include "models/autoports/zai_org_glm_5_3_flash/tt/kernels/pmk/rne_tf32.hpp"
#endif

using namespace pk;

// W2 a15prof3 (GLM53_PMK_ITEM_ZONES=1 in a zoned PMK composite: PMK_ITEM_ZONES, PMK_ZS = the composite's zone salt;
// the standalone KdaPhase passes PMK_ITEM_ZONES_SA): device-profiler zones around the item steps on the math thread
// (each step in its own block scope)
#if defined(PMK_ITEM_ZONES_SA) && !defined(MK_PHASED)
#include "api/debug/kernel_profiler.hpp"
#define PMK_ITEM_ZONES 1
#define PMK_ZS ""
#endif
#if defined(PMK_ITEM_ZONES) && defined(TRISC_MATH) && defined(PMK_ZS)
#define PMK_IZ(n) DeviceZoneScopedN("PMK_I_" n PMK_ZS)
#else
#define PMK_IZ(n)
#endif

namespace {

constexpr uint32_t max_dst_tiles =
    ckernel::get_dest_max_tiles<DST_SYNC_MODE, DST_ACCUM_MODE, ckernel::DstTileShape::Tile32x32>();

// a CB-backed operand: tiles at `off` + i of CB `cb`; outputs: managed (reserve / push here) or a fixed window of a
// CB someone else reserved (the merged OUT CB: packed at off + i, pushed by the caller)
struct Buf {
    uint32_t cb;
    uint32_t off;
    bool managed;
};
constexpr Buf B(uint32_t cb, uint32_t off = 0) { return Buf{cb, off, true}; }
constexpr Buf OUTW(uint32_t off) { return Buf{CB_OUT, off, false}; }

// the packer-gasket mode: arm after every pack reconfig to an fp32 CB (no-op otherwise)
FORCE_INLINE void rne_pack_arm() {
#if defined(PMK_KDA_RNE_PACK)
    kda_rne_pack_arm();
#endif
}
// the merged OUT CB may be bf16 (the chunk-term knob): the packer follows its format for the window and returns to fp32
FORCE_INLINE void o_begin(const Buf& o, uint32_t n) {
    if (o.managed) {
        cb_reserve_back(o.cb, n);
    } else {
        pack_reconfig_data_format(CB_OUT);
#if defined(PMK_KDA_RNE_OUT)  // fp32 chunk terms: the scan's FPU reads them
        rne_pack_arm();
#endif
    }
}
FORCE_INLINE void o_end(const Buf& o, uint32_t n) {
    if (o.managed) {
        cb_push_back(o.cb, n);
    } else {
        pack_reconfig_data_format(CB_W0);
        rne_pack_arm();
    }
}

// the tf32 RNE pre-round of DST tiles [0, n) before their pack (no-op without PMK_KDA_RNE), as 14's prepare: the SFPU
// init before an FPU op's own init; after an SFPU op the tile call follows directly
FORCE_INLINE void rne_init() {
#if defined(PMK_KDA_RNE)
    kda_rne_tf32_init();
#endif
}
FORCE_INLINE void rne_tiles(uint32_t n) {
#if defined(PMK_KDA_RNE)
    for (uint32_t t = 0; t < n; ++t) {
        kda_rne_tf32_tile(t);
    }
#else
    (void)n;
#endif
}

enum class Op { Add, Sub, Mul };

// out[Mt,Nt] = A[Mt,Kt_] @ (Tr ? B[Nt,Kt_]^T : B[Kt_,Nt]) (matmul_blocks of the op)
template <uint32_t Mt, uint32_t Kt_, uint32_t Nt, bool Tr>
FORCE_INLINE void matmul_blocks(const Buf& a, const Buf& b, const Buf& o) {
    constexpr uint32_t sc = kda::MatmulSubblock<Mt, Nt>::columns;
    constexpr uint32_t sr = kda::MatmulSubblock<Mt, Nt>::rows;
    o_begin(o, Mt * Nt);
    reconfig_data_format<SrcOrder::Reverse>(a.cb, b.cb);
    rne_init();
    matmul_block_init(a.cb, b.cb, Tr, sc, sr, Kt_);
    for (uint32_t mi = 0; mi < Mt; mi += sr) {
        for (uint32_t ni = 0; ni < Nt; ni += sc) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt_; ki++) {
                const uint32_t bi = Tr ? ni * Kt_ + ki : ki * Nt + ni;
                matmul_block(a.cb, b.cb, a.off + mi * Kt_ + ki, b.off + bi, 0, Tr, sc, sr, Kt_);
            }
            rne_tiles(sr * sc);
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t r = 0; r < sr; r++) {
                for (uint32_t cc = 0; cc < sc; cc++) {
                    pack_tile(r * sc + cc, o.cb, o.off + (mi + r) * Nt + ni + cc);
                }
            }
            tile_regs_release();
        }
    }
    o_end(o, Mt * Nt);
}

template <Op op>
FORCE_INLINE void elementwise_binary(const Buf& a, const Buf& b, const Buf& o, uint32_t n) {
    o_begin(o, n);
    reconfig_data_format(a.cb, b.cb);
    rne_init();
    if constexpr (op == Op::Add) {
        add_init(a.cb, b.cb);
    } else if constexpr (op == Op::Sub) {
        sub_init(a.cb, b.cb);
    } else {
        mul_init(a.cb, b.cb);
    }
    for (uint32_t s = 0; s < n; s += max_dst_tiles) {
        const uint32_t bt = n - s < max_dst_tiles ? n - s : max_dst_tiles;
        tile_regs_acquire();
        if constexpr (op == Op::Add) {
            add_block(a.cb, b.cb, a.off + s, b.off + s, 0, bt);
        } else if constexpr (op == Op::Sub) {
            sub_block(a.cb, b.cb, a.off + s, b.off + s, 0, bt);
        } else {
            mul_block(a.cb, b.cb, a.off + s, b.off + s, 0, bt);
        }
        rne_tiles(bt);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < bt; ++t) {
            pack_tile(t, o.cb, o.off + s + t);
        }
        tile_regs_release();
    }
    o_end(o, n);
}

FORCE_INLINE void multiply_selected_tile(const Buf& a, uint32_t ai, const Buf& b, uint32_t bi, const Buf& o) {
    o_begin(o, 1);
    reconfig_data_format(a.cb, b.cb);
    rne_init();
    mul_init(a.cb, b.cb);
    tile_regs_acquire();
    mul_block(a.cb, b.cb, a.off + ai, b.off + bi, 0, 1);
    rne_tiles(1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o.cb, o.off);
    tile_regs_release();
    o_end(o, 1);
}

// unary SFPU over n tiles: F applied on DST (copy in, op, pack out)
template <bool Rne = true, typename F>
FORCE_INLINE void unary_tiles(const Buf& in, const Buf& o, uint32_t n, F f) {
    o_begin(o, n);
    reconfig_data_format_srca(in.cb);
    copy_init(in.cb);
    for (uint32_t s = 0; s < n; s += max_dst_tiles) {
        const uint32_t bt = n - s < max_dst_tiles ? n - s : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t t = 0; t < bt; ++t) {
            copy_tile(in.cb, in.off + s + t, t);
        }
        f(bt);
        if constexpr (Rne) {
            rne_tiles(bt);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < bt; ++t) {
            pack_tile(t, o.cb, o.off + s + t);
        }
        tile_regs_release();
    }
    o_end(o, n);
}
FORCE_INLINE void square_tiles(const Buf& in, const Buf& o, uint32_t n) {
    unary_tiles(in, o, n, [](uint32_t bt) {
        square_tile_init();
        for (uint32_t t = 0; t < bt; ++t) {
            square_tile(t);
        }
    });
}
FORCE_INLINE void exponential_tiles(const Buf& in, const Buf& o, uint32_t n) {
    unary_tiles(in, o, n, [](uint32_t bt) {
        exp_tile_init();
        for (uint32_t t = 0; t < bt; ++t) {
            exp_tile(t);
        }
    });
}
FORCE_INLINE void multiply_by_half(const Buf& in, const Buf& o, uint32_t n) {
    constexpr uint32_t half = __builtin_bit_cast(uint32_t, 0.5F);
    unary_tiles<false>(in, o, n, [](uint32_t bt) {  // x * 0.5 of a rounded tile is exact: no pre-round (as 14)
        binop_with_scalar_tile_init();
        for (uint32_t t = 0; t < bt; ++t) {
            mul_unary_tile(t, half);
        }
    });
}
FORCE_INLINE void negated_exponential_tiles(const Buf& in, const Buf& o, uint32_t n) {
    unary_tiles(in, o, n, [](uint32_t bt) {
        negative_tile_init();
        for (uint32_t t = 0; t < bt; ++t) {
            negative_tile(t);
        }
        exp_tile_init();
        for (uint32_t t = 0; t < bt; ++t) {
            exp_tile(t);
        }
    });
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1] (column 0 broadcast across N)
FORCE_INLINE void multiply_by_column(const Buf& a, const Buf& col, const Buf& o, uint32_t Mt, uint32_t Nt) {
    const uint32_t n = Mt * Nt;
    o_begin(o, n);
    reconfig_data_format(a.cb, col.cb);
    rne_init();
    mul_bcast_cols_init(a.cb, col.cb);
    for (uint32_t s = 0; s < n; s += max_dst_tiles) {
        const uint32_t bt = n - s < max_dst_tiles ? n - s : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t t = 0; t < bt; ++t) {
            const uint32_t i = s + t;
            mul_tiles_bcast_cols(a.cb, col.cb, a.off + i, col.off + i / Nt, t);
        }
        rne_tiles(bt);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < bt; ++t) {
            pack_tile(t, o.cb, o.off + s + t);
        }
        tile_regs_release();
    }
    o_end(o, n);
}

// a tiled row [1, n] -> a tiled column [n, 1]
FORCE_INLINE void transpose_tile_row_to_column(const Buf& in, const Buf& o, uint32_t n) {
    o_begin(o, n);
    reconfig_data_format_srca(in.cb);
    rne_init();
    transpose_init(in.cb);
    for (uint32_t s = 0; s < n; s += max_dst_tiles) {
        const uint32_t bt = n - s < max_dst_tiles ? n - s : max_dst_tiles;
        tile_regs_acquire();
        for (uint32_t t = 0; t < bt; ++t) {
            transpose_tile(in.cb, in.off + s + t, t);
        }
        rne_tiles(bt);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t t = 0; t < bt; ++t) {
            pack_tile(t, o.cb, o.off + s + t);
        }
        tile_regs_release();
    }
    o_end(o, n);
}

// row sums of `sq` (Kt tiles, one tile row) -> rsqrt(sum + eps) [* scale] as a column tile in `o`
template <bool Scale>
FORCE_INLINE void reduce_rows_to_inverse_norms(uint32_t Kt, const Buf& sq, const Buf& o, uint32_t eps, uint32_t scale) {
    o_begin(o, 1);
    reconfig_data_format(sq.cb, CB_CONST);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(sq.cb, CB_CONST, o.cb);
    tile_regs_acquire();
    for (uint32_t t = 0; t < Kt; ++t) {
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(sq.cb, CB_CONST, sq.off + t, K_ONES, 0);
    }
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if constexpr (Scale) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale);
    }
    rne_tiles(1);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o.cb, o.off);
    tile_regs_release();
    reduce_uninit();
    o_end(o, 1);
}

template <uint32_t Kt, bool Scale>
FORCE_INLINE void normalize_l2_rows(const Buf& input, const Buf& normalized, uint32_t eps, uint32_t scale) {
    square_tiles(input, B(CB_W3), Kt);
    cb_wait_front(CB_W3, Kt);
    reduce_rows_to_inverse_norms<Scale>(Kt, B(CB_W3), B(CB_TW0), eps, scale);
    cb_wait_front(CB_TW0, 1);
    multiply_by_column(input, B(CB_TW0), normalized, 1, Kt);
    cb_wait_front(normalized.cb, Kt);
    cb_pop_front(CB_TW0, 1);
    cb_pop_front(CB_W3, Kt);
    cb_pop_front(input.cb, Kt);
}

// (I - N)^-1 for the strictly lower N by three nested block levels (invert_block_nested of the op)
FORCE_INLINE void invert_block_nested(const Buf& nsl, const Buf& inverse) {
    const Buf eye = B(CB_CONST, K_EYE), masks = B(CB_CONST, K_MASK0);
    const Buf matrix = B(CB_TW0), total = B(CB_TW1), product = B(CB_TW2);
    multiply_selected_tile(nsl, 0, masks, 0, matrix);  // N1
    cb_wait_front(CB_TW0, 1);
    elementwise_binary<Op::Add>(eye, matrix, total, 1);
    cb_wait_front(CB_TW1, 1);
    for (uint32_t step = 0; step < 2; ++step) {
        matmul_blocks<1, 1, 1, false>(matrix, total, product);
        cb_wait_front(CB_TW2, 1);
        elementwise_binary<Op::Add>(eye, product, total, 1);
        cb_wait_front(CB_TW1, 2);
        cb_pop_front(CB_TW1, 1);
        cb_pop_front(CB_TW2, 1);
    }
    cb_pop_front(CB_TW0, 1);  // total holds A

    multiply_selected_tile(nsl, 0, masks, 1, matrix);  // L2
    cb_wait_front(CB_TW0, 1);
    matmul_blocks<1, 1, 1, false>(total, matrix, product);  // M2 = A L2
    cb_wait_front(CB_TW2, 1);
    cb_pop_front(CB_TW0, 1);
    matmul_blocks<1, 1, 1, false>(product, total, matrix);  // M2 A
    cb_wait_front(CB_TW0, 1);
    cb_pop_front(CB_TW2, 1);
    elementwise_binary<Op::Add>(total, matrix, total, 1);  // B = A + M2 A
    cb_wait_front(CB_TW1, 2);
    cb_pop_front(CB_TW1, 1);
    cb_pop_front(CB_TW0, 1);

    multiply_selected_tile(nsl, 0, masks, 2, product);  // L3
    cb_wait_front(CB_TW2, 1);
    cb_pop_front(nsl.cb, 1);                                // -strict_lower(Akk) dead: its CB reused for I + M3
    matmul_blocks<1, 1, 1, false>(total, product, matrix);  // M3 = B L3
    cb_wait_front(CB_TW0, 1);
    cb_pop_front(CB_TW2, 1);
    matmul_blocks<1, 1, 1, false>(matrix, matrix, product);  // M3^2
    cb_wait_front(CB_TW2, 1);
    elementwise_binary<Op::Add>(eye, matrix, nsl, 1);
    cb_wait_front(nsl.cb, 1);
    cb_pop_front(CB_TW0, 1);
    elementwise_binary<Op::Add>(eye, product, matrix, 1);
    cb_wait_front(CB_TW0, 1);
    cb_pop_front(CB_TW2, 1);
    matmul_blocks<1, 1, 1, false>(nsl, matrix, product);  // (I + M3)(I + M3^2)
    cb_wait_front(CB_TW2, 1);
    cb_pop_front(nsl.cb, 1);
    cb_pop_front(CB_TW0, 1);
    matmul_blocks<1, 1, 1, false>(product, total, inverse);
    cb_pop_front(CB_TW2, 1);
    cb_pop_front(CB_TW1, 1);
}

#if defined(PMK_KDA_KDEC_DEST)
// (GLM53_PMK_KDA_KDEC=dest; Stage 14 kprec3's op fix fc999c8e7f9, root cause of record 2026-10-26 03:55:13Z)
// ek = exp(G_last - G) with the exponent formed EXACTLY in DEST: G_last (ones @ g) and G (tril @ g) as two fp32 matmul
// accumulations (0 / 1 masks and the packed g: exact operands), one SFPU subtract, the precise exp, one DEST pass. The
// anchored factors exp(anchor - G) * exp(G_last / 2) reach the SFPU exp through SrcA (tf32: spacing 1/16 at the ~ +-72
// exponents of fast channels, |G_last| ~ 145 per chunk): k_dec_t came out biased LOW (0.984 median above |G_last| 80).
template <uint32_t Kt>
FORCE_INLINE void exact_key_decay(const Buf& g, const Buf& prefix, const Buf& sum_bcast, const Buf& o) {
    o_begin(o, Kt);
    for (uint32_t t = 0; t < Kt; ++t) {
        tile_regs_acquire();
        reconfig_data_format<SrcOrder::Reverse>(sum_bcast.cb, g.cb);
        matmul_block_init(sum_bcast.cb, g.cb, false, 1, 1, 1);
        matmul_block(sum_bcast.cb, g.cb, sum_bcast.off, g.off + t, 0, false, 1, 1, 1);  // DST 0 = G_last in every row
        matmul_block(prefix.cb, g.cb, prefix.off, g.off + t, 1, false, 1, 1, 1);        // DST 1 = G
        sub_binary_tile_init();
        sub_binary_tile(0, 1, 0);  // DST 0 = G_last - G (<= 0), fp32
        exp_tile_init();
        exp_tile(0);   // the precise exp of an exact fp32 argument
        rne_tiles(1);  // ek: an FPU operand (k * ek)
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o.cb, o.off + t);
        tile_regs_release();
    }
    o_end(o, Kt);
}
#endif

// dst[d] <- tile i of cb (any format: SrcA reconfigured per load, as kda_pre)
FORCE_INLINE void load(uint32_t cb, uint32_t i, uint32_t d) {
    reconfig_data_format_srca(cb);
    copy_init(cb);
    copy_tile(cb, i, d);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(PH_CT0 + 0);
    constexpr uint32_t VTF = get_compile_time_arg_val(PH_CT0 + 1);
    constexpr uint32_t NCH = get_compile_time_arg_val(PH_CT0 + 2);
    constexpr uint32_t CPH = get_compile_time_arg_val(PH_CT0 + 3);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(PH_CT0 + 4);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(PH_CT0 + 5);
    constexpr uint32_t LB_BITS = get_compile_time_arg_val(PH_CT0 + 6);
    constexpr uint32_t CT = 2 * Kt + VTF;
    constexpr uint32_t UPI = 2 * CT;  // CB U pages per item: previous | this (PMK_ITEM_CONV_FPU: S1..S3 in CB_SH)
    constexpr uint32_t NP = out_pages(Kt, VTF);
    static_assert(Kt == VTF && Kt <= 4, "KDA head dim 128: 4 key / value tiles");
    const uint32_t slot = get_arg_val<uint32_t>(ph_rt0 + 0);

    compute_kernel_hw_startup<SrcOrder::Reverse>(CB_CONST, CB_U, CB_CONV);
    cb_wait_front(CB_CONST, K_PAGES);
    cb_wait_front(CB_TAPS, 4 * CT);
    cb_wait_front(CB_DTR, 2 * Kt);
    const Buf tril = B(CB_CONST, K_TRIL), ones = B(CB_CONST, K_ONES);

    for (uint32_t c = slot; c < NCH; c += CPH) {
        cb_wait_front(CB_BETA, 1);
        const uint32_t live = read_tile_value(CB_BETA, 0, CTL_COL);
        const uint32_t partial = read_tile_value(CB_BETA, 0, CTL_COL + 1);
        if (live == 0) {
            cb_pop_front(CB_BETA, 1);
            continue;
        }
        cb_wait_front(CB_MASK, 1);
        cb_wait_front(CB_U, UPI);
#ifdef PMK_ITEM_CONV_FPU
        cb_wait_front(CB_SH, 3 * CT);
#endif
        cb_wait_front(CB_F, Kt);

        // ---- P4 conv: column tile t of q | k | v; DST 0 = acc, 1..3 = shifted copies, 4 = tap / mask row
        {
            PMK_IZ("conv");
            cb_reserve_back(CB_CONV, CT);
            pack_reconfig_data_format(CB_CONV);
#ifdef PMK_ITEM_CONV_FPU
            // DST 0 = this row x tap 3, DST s = S_s x tap 3 - s (FPU row-broadcast products: bf16 x bf16 at HiFi4 =
            // exact fp32), then acc += DST 1, 2, 3 on the SFPU in the shift-matmul conv's order -> bitwise its result
            bool fpu_ready = false;
            for (uint32_t t = 0; t < CT; ++t) {
                if (!fpu_ready) {
                    reconfig_data_format(CB_U, CB_TAPS);
                    mul_bcast_rows_init(CB_U, CB_TAPS);
                    fpu_ready = true;
                }
                tile_regs_acquire();
                mul_tiles_bcast_rows(CB_U, CB_TAPS, CT + t, 3 * CT + t, 0);
                for (uint32_t s = 1; s <= 3; ++s) {  // (CB_SH: the same bf16 format as CB_U, one unpack config)
                    mul_tiles_bcast_rows(CB_SH, CB_TAPS, (s - 1) * CT + t, (3 - s) * CT + t, s);
                }
                add_binary_tile_init();
                for (uint32_t s = 1; s <= 3; ++s) {
                    add_binary_tile(0, s, 0);
                }
                silu_tile_init();
                silu_tile(0);
                if (partial != 0 && t >= Kt) {  // k and v rows past the valid end: zero (q only reaches padded outputs)
                    load(CB_MASK, 0, 4);
                    sfpu_mul_bcast_col_init();
                    sfpu_mul_bcast_col(0, 4);
                    fpu_ready = false;  // the mask load re-configured the unpacker for a copy
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_CONV, t);
                tile_regs_release();
            }
#else
            for (uint32_t t = 0; t < CT; ++t) {
                tile_regs_acquire();
                load(CB_U, CT + t, 0);
                for (uint32_t s = 1; s <= 3; ++s) {
                    reconfig_data_format<SrcOrder::Reverse>(CB_CONST, CB_U);
                    matmul_block_init(CB_CONST, CB_U, false, 1, 1, 1);
                    matmul_block(CB_CONST, CB_U, K_SDN1 + s - 1, CT + t, s, false, 1, 1, 1);
                    matmul_block(CB_CONST, CB_U, K_SUP1 + s - 1, t, s, false, 1, 1, 1);
                }
                load(CB_TAPS, 3 * CT + t, 4);  // tap 3 multiplies the row itself
                sfpu_mul_bcast_row_init();
                sfpu_mul_bcast_row(0, 4);
                for (uint32_t s = 1; s <= 3; ++s) {  // tap 3 - s multiplies row t - s
                    load(CB_TAPS, (3 - s) * CT + t, 4);
                    sfpu_mul_bcast_row_init();
                    sfpu_mul_bcast_row(s, 4);
                    add_binary_tile_init();
                    add_binary_tile(0, s, 0);
                }
                silu_tile_init();
                silu_tile(0);
                if (partial != 0 && t >= Kt) {  // k and v rows past the valid end: zero (q only reaches padded outputs)
                    load(CB_MASK, 0, 4);
                    sfpu_mul_bcast_col_init();
                    sfpu_mul_bcast_col(0, 4);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_CONV, t);
                tile_regs_release();
            }
#endif
            cb_push_back(CB_CONV, CT);
            cb_pop_front(CB_U, UPI);
#ifdef PMK_ITEM_CONV_FPU
            cb_pop_front(CB_SH, 3 * CT);
#endif
        }

        // ---- gate g = lb * sigmoid((f + dt_bias) * rate), masked; beta = sigmoid(b), masked (into CB_TW1: a ring used
        // with 1-tile blocks only -- a 1-tile block on the 4-tile W0..W3 rings straddles their wrap, r4's garbage
        // terms)
        {
            PMK_IZ("gate");
            cb_reserve_back(CB_G, Kt);
            pack_reconfig_data_format(CB_G);
            rne_pack_arm();  // g .. the item's last fp32 pack (the conv's bf16 reconfig cleared it)
#ifdef PMK_ITEM_GATE_BATCH
            // the Kt gate tiles in ONE DST pass (DST 0..Kt-1; the dt_bias / rate rows in DST Kt, the mask in Kt + 1):
            // the same per-element op order as the per-tile loop, one init per op kind and one copy init per source CB
            static_assert(Kt + 2 <= max_dst_tiles, "gate batch: Kt + 2 DST tiles");
            tile_regs_acquire();
            reconfig_data_format_srca(CB_F);
            copy_init(CB_F);
            for (uint32_t t = 0; t < Kt; ++t) {
                copy_tile(CB_F, t, t);
            }
            reconfig_data_format_srca(CB_DTR);
            copy_init(CB_DTR);
            sfpu_add_bcast_row_init();
            for (uint32_t t = 0; t < Kt; ++t) {
                copy_tile(CB_DTR, t, Kt);
                sfpu_add_bcast_row(t, Kt);
            }
            sfpu_mul_bcast_row_init();
            for (uint32_t t = 0; t < Kt; ++t) {
                copy_tile(CB_DTR, Kt + t, Kt);
                sfpu_mul_bcast_row(t, Kt);
            }
            sigmoid_tile_init();
            for (uint32_t t = 0; t < Kt; ++t) {
                sigmoid_tile(t);
            }
            binop_with_scalar_tile_init();
            for (uint32_t t = 0; t < Kt; ++t) {
                mul_unary_tile(t, LB_BITS);
            }
            if (partial != 0) {
                load(CB_MASK, 0, Kt + 1);
                sfpu_mul_bcast_col_init();
                for (uint32_t t = 0; t < Kt; ++t) {
                    sfpu_mul_bcast_col(t, Kt + 1);
                }
            }
            rne_tiles(Kt);  // g: an FPU operand (tril @ g, ones @ g)
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < Kt; ++t) {
                pack_tile(t, CB_G, t);
            }
            tile_regs_release();
#else
            for (uint32_t t = 0; t < Kt; ++t) {
                tile_regs_acquire();
                load(CB_F, t, 0);
                load(CB_DTR, t, 1);
                sfpu_add_bcast_row_init();
                sfpu_add_bcast_row(0, 1);
                load(CB_DTR, Kt + t, 1);
                sfpu_mul_bcast_row_init();
                sfpu_mul_bcast_row(0, 1);
                sigmoid_tile_init();
                sigmoid_tile(0);
                binop_with_scalar_tile_init();
                mul_unary_tile(0, LB_BITS);
                if (partial != 0) {
                    load(CB_MASK, 0, 1);
                    sfpu_mul_bcast_col_init();
                    sfpu_mul_bcast_col(0, 1);
                }
                rne_tiles(1);  // g: an FPU operand (tril @ g, ones @ g)
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_G, t);
                tile_regs_release();
            }
#endif
            cb_push_back(CB_G, Kt);
            cb_pop_front(CB_F, Kt);
            cb_reserve_back(CB_TW1, 1);
            {
                tile_regs_acquire();
                load(CB_BETA, 0, 0);
                sigmoid_tile_init();
                sigmoid_tile(0);
                if (partial != 0) {
                    load(CB_MASK, 0, 1);
                    sfpu_mul_bcast_col_init();
                    sfpu_mul_bcast_col(0, 1);
                }
                rne_tiles(1);  // beta: an FPU operand (v * beta, k * beta; 14 rounds it by its rne_tf32 op)
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, CB_TW1, 0);
                tile_regs_release();
            }
            cb_push_back(CB_TW1, 1);
            cb_pop_front(CB_BETA, 1);
            cb_pop_front(CB_MASK, 1);
        }

        // ---- P5 prepare_chunk_recurrence (the op's compute body; q k v = CB_CONV tiles 0.., popped in order)
        cb_wait_front(CB_CONV, CT);
        cb_wait_front(CB_G, Kt);
        cb_wait_front(CB_TW1, 1);
        cb_reserve_back(CB_OUT, NP);
        {
            PMK_IZ("l2");
            normalize_l2_rows<Kt, true>(B(CB_CONV), B(CB_NQ), EPS_BITS, SCALE_BITS);
            normalize_l2_rows<Kt, false>(B(CB_CONV), B(CB_NK), EPS_BITS, SCALE_BITS);
            // v_beta = v * beta
            multiply_by_column(B(CB_CONV), B(CB_TW1), OUTW(off_vbeta(Kt, VTF)), 1, VTF);
            cb_pop_front(CB_CONV, VTF);
        }
        // gate factors: G = cumsum(g) anchored at G_last / 2
        {
            PMK_IZ("gfac");
            const Buf g = B(CB_G), centered_decay = B(CB_W0), decay = B(CB_SD), g_last = B(CB_W3), anchor_g = B(CB_W2);
            const Buf cid = B(CB_CID), anchor_decay = B(CB_ANCH);
            matmul_blocks<1, 1, Kt, false>(tril, g, centered_decay);
            cb_wait_front(CB_W0, Kt);
            exponential_tiles(centered_decay, decay, Kt);  // exp(G): the scan-facing q_decay / kd factor
            cb_wait_front(CB_SD, Kt);
            matmul_blocks<1, 1, Kt, false>(ones, g, g_last);  // replicated G_last
            cb_wait_front(CB_W3, Kt);
#if !defined(PMK_KDA_KDEC_DEST)  // (KDEC=dest: g is held to the final block)
            cb_pop_front(CB_G, Kt);
#endif
            multiply_by_half(g_last, anchor_g, Kt);
            cb_wait_front(CB_W2, Kt);
            {
                const Buf centered_g = anchor_decay;
                elementwise_binary<Op::Sub>(centered_decay, anchor_g, centered_g, Kt);
                cb_wait_front(CB_ANCH, Kt);
                cb_pop_front(CB_W0, Kt);
                exponential_tiles(centered_g, centered_decay, Kt);
                cb_wait_front(CB_W0, Kt);
                negated_exponential_tiles(centered_g, cid, Kt);
                cb_wait_front(CB_CID, Kt);
                cb_pop_front(CB_ANCH, Kt);
            }
#if !defined(PMK_KDA_KDEC_DEST)                             // exp(G_last / 2) feeds only the anchored k_dec_t
            exponential_tiles(anchor_g, anchor_decay, Kt);  // exp(G_last / 2)
            cb_wait_front(CB_ANCH, Kt);
#else
            (void)anchor_decay;
#endif
            cb_pop_front(CB_W2, Kt);
        }
        // scan-facing q_decay / kd (exact factors) and the anchored pairwise inputs
        {
            PMK_IZ("pin");
            {
                const Buf nq = B(CB_NQ), nk = B(CB_NK), beta = B(CB_TW1), decay = B(CB_SD), centered_decay = B(CB_W0);
                const Buf k_beta_pairwise = B(CB_W1), q_pairwise = B(CB_W2);
                {
                    const Buf beta_k = q_pairwise;
                    multiply_by_column(nk, beta, beta_k, 1, Kt);
                    cb_wait_front(CB_W2, Kt);
                }
                cb_pop_front(CB_TW1, 1);
                elementwise_binary<Op::Mul>(nq, decay, OUTW(off_qdec(Kt, VTF)), Kt);
                {
                    const Buf beta_k = q_pairwise;
                    elementwise_binary<Op::Mul>(beta_k, decay, OUTW(off_kd(Kt, VTF)), Kt);
                    elementwise_binary<Op::Mul>(beta_k, centered_decay, k_beta_pairwise, Kt);
                    cb_wait_front(CB_W1, Kt);  // beta k exp(G - anchor)
                    cb_pop_front(CB_W2, Kt);
                }
                elementwise_binary<Op::Mul>(nq, centered_decay, q_pairwise, Kt);
                cb_wait_front(CB_W2, Kt);  // q exp(G - anchor)
                cb_pop_front(CB_NQ, Kt);
                cb_pop_front(CB_W0, Kt);
                cb_pop_front(CB_SD, Kt);
            }
            // final decay rows exp(G_last) (W3 -> W0), k_pairwise = k exp(anchor - G) (-> W3)
            exponential_tiles(B(CB_W3), B(CB_W0), Kt);
            cb_wait_front(CB_W0, Kt);
            cb_pop_front(CB_W3, Kt);
            elementwise_binary<Op::Mul>(B(CB_NK), B(CB_CID), B(CB_W3), Kt);
            cb_wait_front(CB_W3, Kt);
#if !defined(PMK_KDA_KDEC_DEST)  // (KDEC=dest: k is held to the final block)
            cb_pop_front(CB_NK, Kt);
#endif
            cb_pop_front(CB_CID, Kt);
        }
        // pairwise matrices: Akk (raw), intra = tril(Aqk)
        {
            const Buf k_beta_pairwise = B(CB_W1), q_pairwise = B(CB_W2), k_pairwise = B(CB_W3), akk = B(CB_AKK);
            const Buf aqk = B(CB_TW0);
            PMK_IZ("pair");
            matmul_blocks<1, Kt, 1, true>(k_beta_pairwise, k_pairwise, akk);
            cb_wait_front(CB_AKK, 1);
            cb_pop_front(CB_W1, Kt);
            matmul_blocks<1, Kt, 1, true>(q_pairwise, k_pairwise, aqk);
            cb_wait_front(CB_TW0, 1);
            cb_pop_front(CB_W2, Kt);
            elementwise_binary<Op::Mul>(aqk, tril, OUTW(off_intra(Kt, VTF)), 1);
            cb_pop_front(CB_TW0, 1);
        }
        // T_inv = (I + strictly_lower(Akk))^-1
        {
            const Buf akk = B(CB_AKK), lower_akk = B(CB_TW0), diagonal_akk = B(CB_TW1), eye = B(CB_CONST, K_EYE);
            PMK_IZ("inv");
            elementwise_binary<Op::Mul>(akk, tril, lower_akk, 1);
            cb_wait_front(CB_TW0, 1);
            cb_pop_front(CB_AKK, 1);
            elementwise_binary<Op::Mul>(lower_akk, eye, diagonal_akk, 1);
            cb_wait_front(CB_TW1, 1);
            elementwise_binary<Op::Sub>(diagonal_akk, lower_akk, akk, 1);
            cb_wait_front(CB_AKK, 1);  // -strictly_lower(A)
            cb_pop_front(CB_TW1, 1);
            cb_pop_front(CB_TW0, 1);
            invert_block_nested(akk, OUTW(off_tinv(Kt, VTF)));
        }
        // final_decay [K, 1] = exp(G_last)^T; k_dec_t = (k_pairwise * exp(G_last / 2))^T
        // (KDEC=dest: k_dec_t = (k * exp(G_last - G))^T from the exact exponent, formed HERE after the pairwise
        // matrices with g and k held -- W1 / W2 are free; the same block placed before the gate factors corrupted every
        // later term of the item on device, d0 2026-10-26 04:10Z)
        {
            const Buf final_decay_rows = B(CB_W0), k_pairwise = B(CB_W3), anchor_decay = B(CB_ANCH), k_dec = B(CB_W0);
            PMK_IZ("tr");
#if defined(PMK_KDA_KDEC_DEST)
            {
                const Buf ek = B(CB_W1), k_dec_exact = B(CB_W2);
                exact_key_decay<Kt>(B(CB_G), tril, ones, ek);
                cb_wait_front(CB_W1, Kt);
                elementwise_binary<Op::Mul>(B(CB_NK), ek, k_dec_exact, Kt);
                cb_wait_front(CB_W2, Kt);
                cb_pop_front(CB_W1, Kt);
                cb_pop_front(CB_G, Kt);
                cb_pop_front(CB_NK, Kt);
            }
#endif
            transpose_tile_row_to_column(final_decay_rows, OUTW(off_fd(Kt, VTF)), Kt);
            cb_pop_front(CB_W0, Kt);
#if defined(PMK_KDA_KDEC_DEST)
            (void)anchor_decay;
            (void)k_dec;
            (void)k_pairwise;
            transpose_tile_row_to_column(B(CB_W2), OUTW(off_kdt(Kt, VTF)), Kt);
            cb_pop_front(CB_W2, Kt);
            cb_pop_front(CB_W3, Kt);
#else
            elementwise_binary<Op::Mul>(k_pairwise, anchor_decay, k_dec, Kt);
            cb_wait_front(CB_W0, Kt);
            cb_pop_front(CB_W3, Kt);
            cb_pop_front(CB_ANCH, Kt);
            transpose_tile_row_to_column(k_dec, OUTW(off_kdt(Kt, VTF)), Kt);
            cb_pop_front(CB_W0, Kt);
#endif
        }
        cb_push_back(CB_OUT, NP);
    }
#if defined(PMK_KDA_RNE_PACK)
    kda_rne_pack_disarm();  // the composite's next phase packs unrounded
#endif
}
