#if defined(TT_WH_PERIODIC_CAST_NORMALIZATION) && !defined(TT_WH_PERIODIC_SIGNED_NORMALIZATION)
#error "WH CAST normalization requires the complete signed-normalization owner"
#endif
// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#ifdef TRISC_MATH

// Exact BF16 materialization of the Q192 radix-23 reduction.
// The ordinary arm uses seven mantissa-bit additions per radix23 limb.
// The certified optional arm forms the same product with integer-exact
// radix15 FMAs, then reuses normal-FP32 digits from per-row DST scratch.
#if !defined(ARCH_WORMHOLE) || !defined(USE_BF16)
#error "WH Q192 reduction requires BF16 Wormhole ingress"
#endif
#include <cstdint>
#include <limits>
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#if !defined(PERIODIC_Q192_INDEX_REUSE_DISABLE) && !defined(PERIODIC_Q192_STATE_REUSE_DISABLE) && \
    !defined(PERIODIC_Q192_PRODUCT_LOAD_DISABLE) && !defined(PERIODIC_Q192_UNROLL_DISABLE)
#define TT_WH_PERIODIC_INDEX_REUSE 1
#endif
#if defined(TT_WH_PERIODIC_PRODUCT_CACHE) && !defined(PERIODIC_Q192_PRODUCT_CACHE_DISABLE) &&     \
    !defined(PERIODIC_Q192_INDEX_REUSE_DISABLE) && !defined(PERIODIC_Q192_STATE_REUSE_DISABLE) && \
    !defined(PERIODIC_Q192_PRODUCT_LOAD_DISABLE) && !defined(PERIODIC_Q192_UNROLL_DISABLE)
#define TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE 1
#endif
#if defined(TT_WH_PERIODIC_WINDOW37) && !defined(PERIODIC_Q192_PRODUCT_CACHE_DISABLE) &&           \
    !defined(PERIODIC_Q192_WINDOW_DISABLE) && !defined(PERIODIC_Q192_INDEX_REUSE_DISABLE) &&       \
    !defined(PERIODIC_Q192_STATE_REUSE_DISABLE) && !defined(PERIODIC_Q192_PRODUCT_LOAD_DISABLE) && \
    !defined(PERIODIC_Q192_UNROLL_DISABLE)
#define TT_WH_PERIODIC_WINDOW37_ACTIVE 1
#endif
#if defined(TT_WH_PERIODIC_REGISTER_OFFSET) || defined(TT_WH_PERIODIC_DEFAULT_ENTRY)
#if !defined(TT_WH_PERIODIC_OUTPUT_WINDOW26) || !defined(TT_WH_PERIODIC_REGISTER_DIGITS) || \
    !defined(TT_WH_PERIODIC_WINDOW_POOL)
#error "periodic transport requires its certified pooled 26-bit register owner"
#endif
#endif
#ifdef TT_WH_PERIODIC_FLOAT_HIGH_LIMB
#if !defined(TT_WH_PERIODIC_WINDOW37_ACTIVE) || !defined(TT_WH_PERIODIC_OUTPUT_WINDOW26) || \
    !defined(TT_WH_PERIODIC_WINDOW_POOL) || !defined(TT_WH_PERIODIC_CARRY_PROJECTION) ||    \
    !defined(TT_WH_PERIODIC_REGISTER_DIGITS) || !defined(TT_WH_PERIODIC_DEFAULT_ENTRY)
#error "floating high limb requires the pooled three-digit register owner"
#endif
#endif
#ifdef TT_WH_PERIODIC_PREDICATED_LOADS
#if !defined(TT_WH_PERIODIC_FLOAT_HIGH_LIMB) || !defined(TT_WH_PERIODIC_REGISTER_OFFSET)
#error "predicated pool loads require the complete floating high-limb owner"
#endif
#endif
#ifdef TT_WH_PERIODIC_SIGNED_NORMALIZATION
#if !defined(TT_WH_PERIODIC_PREDICATED_LOADS) || !defined(TT_WH_PERIODIC_OUTPUT_WINDOW26)
#error "signed normalization requires the complete 26-bit predicated window owner"
#endif
#endif
namespace sfpi {
sfpi_inline vUInt tt_wh_product(int dst_index, uint32_t k) {
    vUInt result = 0u;
#if !defined(PERIODIC_Q192_PRODUCT_LOAD_DISABLE) && !defined(PERIODIC_Q192_STATE_REUSE_DISABLE)
    vUInt bits = as<vUInt>(vFloat(dst_reg[dst_index]));
#endif
    for (unsigned b = 0; b < 7; ++b) {
#if defined(PERIODIC_Q192_PRODUCT_LOAD_DISABLE) || defined(PERIODIC_Q192_STATE_REUSE_DISABLE)
        vUInt bits = as<vUInt>(vFloat(dst_reg[dst_index]));
#endif
        v_if((bits & (1u << (16 + b))) != 0u) { result = result + (k << b); }
        v_endif;
    }
    return result;
}
sfpi_inline vUInt tt_wh_radix_index(vUInt distance) {
    vUInt out = 0u;
    for (unsigned i = 1; i <= 9; ++i) {
        v_if(distance >= 23u * i) { out = i; }
        v_endif;
    }
    return out;
}
static constexpr uint32_t tt_q192_limbs[] = {
    0x439041u, 0x2b3278u, 0x036d8au, 0x29a6eeu, 0x757d1fu, 0x253f84u, 0x139105u, 0x7cc1b7u, 0x0000a2u, 0x000000u};

#ifdef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
static_assert(64 >= 32 + 32 && 64 + 10 <= 108, "periodic scratch must not overlap data/gradient rows");
// Per-row exact product digits, encoded as normal F32 words in DST64..73.
// All slots are overwritten before any reduction or tangent-parity consumer.
// Radix15 integer FMA: significand*constant+carry is below 2^23.
// Adding 2^23 makes its entire exact integer result the FP32 mantissa.
static constexpr uint32_t tt_q192_limbs15[] = {
    4161u, 30855u, 22116u, 23316u, 23565u, 9883u, 29821u, 5035u, 10748u, 2090u, 14649u, 19483u, 2607u, 0u};
sfpi_inline void tt_q192_prefill_digits(int dst_index) {
    vUInt raw = as<vUInt>(vFloat(dst_reg[dst_index]));
    vFloat significand = as<vFloat>((raw & 0x007f0000u) | 0x43000000u);
    vUInt carry = 0u;
    vUInt packed = 0u;
#pragma GCC unroll 14
    for (int i = 0; i < 14; ++i) {
        vFloat k = float(tt_q192_limbs15[i]);
        vFloat bias = as<vFloat>(carry | 0x4b000000u);
        vFloat product = __builtin_rvtt_sfpmad(significand.get(), k.get(), bias.get(), SFPMAD_MOD1_OFFSET_NONE);
        vUInt integer = as<vUInt>(product) & 0x7fffffu;
        carry = integer >> 15;
        vUInt digit = integer & 0x7fffu;
        int offset = (16 + 15 * i) % 23;
        packed = packed | (digit << offset);
        if (offset + 15 >= 23) {
            dst_reg[64 + (16 + 15 * i) / 23].mode<DataLayout::F32>() = as<vFloat>((packed & 0x7fffffu) | 0x4b000000u);
            packed = digit >> (23 - offset);
        }
    }
    dst_reg[73].mode<DataLayout::F32>() = as<vFloat>(packed | 0x4b000000u);
}
#endif

#ifdef TT_WH_PERIODIC_WINDOW37_ACTIVE
static_assert(64 + 6 <= 96 && 96 + 2 <= 108, "periodic window scratch ranges must be disjoint");
// The two preceding radix15 limbs certify the exact incoming carry for
// every BF16 significand. The output certificate permits three product
// digits; the original owner retains four digits and its 37-bit fraction.
#if defined(TT_WH_PERIODIC_OUTPUT_WINDOW26) && defined(TT_WH_PERIODIC_REGISTER_DIGITS)
static constexpr int tt_q192_window_bits = 26;
static constexpr int tt_q192_window_digits = 3;
#else
static constexpr int tt_q192_window_bits = 37;
static constexpr int tt_q192_window_digits = 4;
#endif
static constexpr int tt_q192_window_first = (71 - tt_q192_window_bits) / 15;
static constexpr int tt_q192_window_last = (200 - tt_q192_window_bits) / 15;
#ifdef TT_WH_PERIODIC_WINDOW_POOL
#ifdef TT_WH_PERIODIC_FLOAT_HIGH_LIMB
static_assert(
    64 + 20 <= 84 && 84 + 9 <= 96 && 96 + 2 <= 108,
    "pooled constants, nine floating high limbs, and state must be disjoint");
#else
static_assert(
    64 + 20 <= 84 && 84 + 4 <= 96 && 96 + 2 <= 108, "periodic constants/products/state must own disjoint rows");
#endif
sfpi_inline void tt_q192_prefill_window_constants() {
    static constexpr uint32_t limbs[] = {
        4161u, 30855u, 22116u, 23316u, 23565u, 9883u, 29821u, 5035u, 10748u, 2090u, 14649u, 19483u, 2607u, 0u, 0u};
#pragma GCC unroll 9
    for (int i = 0; i < 9; ++i) {
        dst_reg[64 + i].mode<DataLayout::F32>() =
            float(limbs[i + tt_q192_window_first - 2] | (limbs[i + tt_q192_window_first - 1] << 15)) * 0x1p-30f;
    }
#ifdef TT_WH_PERIODIC_FLOAT_HIGH_LIMB
#pragma GCC unroll 9
    for (int i = 0; i < 9; ++i) {
        dst_reg[84 + i].mode<DataLayout::F32>() = float(limbs[tt_q192_window_first + i + 2]);
    }
#endif
#pragma GCC unroll 11
    for (int i = 0; i < 11; ++i) {
        dst_reg[73 + i].mode<DataLayout::F32>() = as<vFloat>(
            vUInt((limbs[i + tt_q192_window_first] | (limbs[i + tt_q192_window_first + 1] << 15)) | 0x40000000u));
    }
}
#endif
sfpi_inline void tt_q192_window37(int dst_index) {
    static constexpr uint32_t limbs[] = {
        4161u, 30855u, 22116u, 23316u, 23565u, 9883u, 29821u, 5035u, 10748u, 2090u, 14649u, 19483u, 2607u, 0u, 0u};
#ifdef TT_WH_PERIODIC_CARRY_PROJECTION
#ifdef TT_WH_PERIODIC_DEFAULT_ENTRY
    vFloat carry_coefficient = dst_reg[64].mode<DataLayout::F32>();
#else
    vFloat carry_coefficient = 0.0f;
#endif
#else
    vUInt chunk0 = 0u;
#endif
#ifdef TT_WH_PERIODIC_DEFAULT_ENTRY
    vUInt chunk1 = as<vUInt>(vFloat(dst_reg[73].mode<DataLayout::F32>()));
#ifdef TT_WH_PERIODIC_FLOAT_HIGH_LIMB
    vFloat high_limb = dst_reg[84].mode<DataLayout::F32>();
#else
    vUInt chunk2 = as<vUInt>(vFloat(dst_reg[75].mode<DataLayout::F32>()));
#endif
    constexpr int first_selected = tt_q192_window_first + 1;
#else
    vUInt chunk1 = 0u;
    vUInt chunk2 = 0u;
    constexpr int first_selected = tt_q192_window_first;
#endif
#ifdef TT_WH_PERIODIC_REGISTER_DIGITS
#ifndef TT_WH_PERIODIC_WINDOW_POOL
#error "register digits require the certified pooled projected window"
#endif
    vUInt register_lo = 0u, register_hi = 0u;
#ifdef TT_WH_PERIODIC_REGISTER_OFFSET
    vUInt register_offset = 0u;
#endif
#endif
#ifdef TT_WH_PERIODIC_PREDICATED_LOADS
    // Keep the intrinsic live value as a raw vector. Rewrapping it inside
    // v_if adds a redundant merge move to each predicated FP32 load.
    // The standalone SFPU init clears LaneConfig index capture/read blocking;
    // selection enters enabled and each v_endif restores the enabled state.
    auto raw_carry = carry_coefficient.get();
    auto raw_chunk = chunk1.get();
    auto raw_high = high_limb.get();
#endif
    {
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
        vInt exponent = exexp(vFloat(dst_reg[dst_index]), ExponentMode::Biased);
#else
        vUInt raw = as<vUInt>(vFloat(dst_reg[dst_index]));
        vInt exponent = as<vInt>((raw >> 23) & 255u);
#endif
        v_if(exponent < 126) { exponent = 126; }
        v_endif;
        vInt shift = 326 - tt_q192_window_bits - exponent;
        // floor(n/15) = ((n+1)*17)>>8 for 0 <= n <= 254;
        // Both admitted window widths stay within that exact division range.
        vUInt factor = as<vUInt>(shift + 1);
        vInt index = as<vInt>((factor + (factor << 4)) >> 8);
#pragma GCC unroll 9
        for (int i = first_selected; i <= tt_q192_window_last; ++i) {
            v_if(index == i) {
#ifdef TT_WH_PERIODIC_WINDOW_POOL
#ifdef TT_WH_PERIODIC_PREDICATED_LOADS
                raw_carry = __builtin_rvtt_sfpload_lv(
                    ckernel::instrn_buffer,
                    raw_carry,
                    (64 + i - tt_q192_window_first) * SFP_DESTREG_STRIDE,
                    0,
                    0,
                    SFPLOAD_MOD0_FMT_FP32,
                    SFPLOAD_ADDR_MODE_NOINC);
                raw_chunk = __builtin_rvtt_sfpload_lv(
                    ckernel::instrn_buffer,
                    raw_chunk,
                    (73 + i - tt_q192_window_first) * SFP_DESTREG_STRIDE,
                    0,
                    0,
                    SFPLOAD_MOD0_FMT_FP32,
                    SFPLOAD_ADDR_MODE_NOINC);
                raw_high = __builtin_rvtt_sfpload_lv(
                    ckernel::instrn_buffer,
                    raw_high,
                    (84 + i - tt_q192_window_first) * SFP_DESTREG_STRIDE,
                    0,
                    0,
                    SFPLOAD_MOD0_FMT_FP32,
                    SFPLOAD_ADDR_MODE_NOINC);
#else
                carry_coefficient = dst_reg[64 + i - tt_q192_window_first].mode<DataLayout::F32>();
                chunk1 = as<vUInt>(vFloat(dst_reg[73 + i - tt_q192_window_first].mode<DataLayout::F32>()));
#ifdef TT_WH_PERIODIC_FLOAT_HIGH_LIMB
                high_limb = dst_reg[84 + i - tt_q192_window_first].mode<DataLayout::F32>();
#else
                chunk2 = as<vUInt>(vFloat(dst_reg[73 + i - tt_q192_window_first + 2].mode<DataLayout::F32>()));
#endif
#endif
#else
#ifdef TT_WH_PERIODIC_CARRY_PROJECTION
                carry_coefficient = float(limbs[i - 2] | (limbs[i - 1] << 15)) * 0x1p-30f;
#else
                chunk0 = limbs[i - 2] | (limbs[i - 1] << 15);
#endif
                chunk1 = limbs[i] | (limbs[i + 1] << 15);
                chunk2 = limbs[i + 2] | (limbs[i + 3] << 15);
#endif
            }
            v_endif;
        }
        vInt offset = shift - ((index << 4) - index);
#ifdef TT_WH_PERIODIC_REGISTER_OFFSET
        register_offset = as<vUInt>(offset);
#elif defined(TT_WH_PERIODIC_PRIMITIVE_BITS)
        dst_reg[96].mode<DataLayout::F32>() = setexp(as<vFloat>(offset), 150);
#else
        dst_reg[96].mode<DataLayout::F32>() = as<vFloat>(as<vUInt>(offset) | 0x4b000000u);
#endif
    }
#ifdef TT_WH_PERIODIC_PREDICATED_LOADS
    carry_coefficient = vFloat(raw_carry);
    chunk1 = as<vUInt>(vFloat(raw_chunk));
    high_limb = vFloat(raw_high);
#endif
#ifdef TT_WH_PERIODIC_WINDOW_POOL
    chunk1 = chunk1 ^ 0x40000000u;
#ifndef TT_WH_PERIODIC_FLOAT_HIGH_LIMB
    chunk2 = chunk2 ^ 0x40000000u;
#endif
#endif
    {
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
        vFloat m = setexp(setsgn(vFloat(dst_reg[dst_index]), 0), 134);
#else
        vUInt raw = as<vUInt>(vFloat(dst_reg[dst_index]));
        vFloat m = as<vFloat>((raw & 0x007f0000u) | 0x43000000u);
#endif
#ifdef TT_WH_PERIODIC_CARRY_PROJECTION
        vFloat carry_bias = 0x1.fffffep+22f;
        vFloat carry_value =
            __builtin_rvtt_sfpmad(m.get(), carry_coefficient.get(), carry_bias.get(), SFPMAD_MOD1_OFFSET_NONE);
        vUInt carry = as<vUInt>(exman(carry_value));
#pragma GCC unroll 4
        for (int i = 0; i < tt_q192_window_digits; ++i) {
#ifdef TT_WH_PERIODIC_FLOAT_HIGH_LIMB
            vUInt part = chunk1;
#else
            vUInt part = (i < 2 ? chunk1 : chunk2);
#endif
#else
        vUInt carry = 0u;
#pragma GCC unroll 6
        for (int i = 0; i < 6; ++i) {
            vUInt part = (i < 2 ? chunk0 : (i < 4 ? chunk1 : chunk2));
#endif
#ifdef TT_WH_PERIODIC_FLOAT_HIGH_LIMB
            vFloat k = high_limb;
            if (i < 2) {
                if (i % 2) {
                    part = part >> 15;
                }
                part = part & 0x7fffu;
                k = convert<vFloat>(as<vSMag>(part), RoundMode::Nearest);
            }
#else
            if (i % 2) {
                part = part >> 15;
            }
            part = part & 0x7fffu;
            vFloat k = convert<vFloat>(as<vSMag>(part), RoundMode::Nearest);
#endif
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
            vFloat bias = setexp(as<vFloat>(carry), 150);
#else
            vFloat bias = as<vFloat>(carry | 0x4b000000u);
#endif
            vFloat product = __builtin_rvtt_sfpmad(m.get(), k.get(), bias.get(), SFPMAD_MOD1_OFFSET_NONE);
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
            vUInt integer = as<vUInt>(exman(product));
#else
            vUInt integer = as<vUInt>(product) & 0x7fffffu;
#endif
            carry = integer >> 15;
#ifdef TT_WH_PERIODIC_CARRY_PROJECTION
            {
#ifdef TT_WH_PERIODIC_WINDOW_POOL
#ifdef TT_WH_PERIODIC_REGISTER_DIGITS
                if (i == 0) {
                    register_lo = integer & 0x7fffu;
                }
                if (i == 1) {
                    register_lo = register_lo | ((integer & 0x7fffu) << 15);
                }
                if (i == 2) {
                    register_hi = integer & 0x7fffu;
                }
                if (i == 3) {
                    register_hi = register_hi | ((integer & 0x7fffu) << 15);
                }
#else
                dst_reg[84 + i].mode<DataLayout::F32>() = setexp(as<vFloat>(integer & 0x7fffu), 150);
#endif
#else
                dst_reg[64 + i].mode<DataLayout::F32>() = setexp(as<vFloat>(integer & 0x7fffu), 150);
#endif
#else
            if (i >= 2) {
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
                dst_reg[64 + i - 2].mode<DataLayout::F32>() = setexp(as<vFloat>(integer & 0x7fffu), 150);
#else
                dst_reg[64 + i - 2].mode<DataLayout::F32>() = as<vFloat>((integer & 0x7fffu) | 0x4b000000u);
#endif
#endif
            }
        }
    }
#ifdef TT_WH_PERIODIC_REGISTER_OFFSET
    vUInt offset = register_offset;
#elif defined(TT_WH_PERIODIC_PRIMITIVE_BITS)
    vUInt offset = as<vUInt>(exman(vFloat(dst_reg[96].mode<DataLayout::F32>())));
#else
vUInt offset = as<vUInt>(vFloat(dst_reg[96].mode<DataLayout::F32>())) & 15u;
#endif
#ifdef TT_WH_PERIODIC_REGISTER_DIGITS
    vUInt lo = register_lo, hi = register_hi;
#else
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
#ifdef TT_WH_PERIODIC_WINDOW_POOL
    vUInt lo = as<vUInt>(exman(vFloat(dst_reg[84].mode<DataLayout::F32>())));
#else
    vUInt lo = as<vUInt>(exman(vFloat(dst_reg[64].mode<DataLayout::F32>())));
#endif
#else
    vUInt lo = as<vUInt>(vFloat(dst_reg[64].mode<DataLayout::F32>())) & 0x7fffu;
#endif
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
#ifdef TT_WH_PERIODIC_WINDOW_POOL
    vUInt d1 = as<vUInt>(exman(vFloat(dst_reg[85].mode<DataLayout::F32>())));
#else
    vUInt d1 = as<vUInt>(exman(vFloat(dst_reg[65].mode<DataLayout::F32>())));
#endif
#else
    vUInt d1 = as<vUInt>(vFloat(dst_reg[65].mode<DataLayout::F32>())) & 0x7fffu;
#endif
    lo = lo | (d1 << 15);
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
#ifdef TT_WH_PERIODIC_WINDOW_POOL
    vUInt hi = as<vUInt>(exman(vFloat(dst_reg[86].mode<DataLayout::F32>())));
#else
    vUInt hi = as<vUInt>(exman(vFloat(dst_reg[66].mode<DataLayout::F32>())));
#endif
#else
    vUInt hi = as<vUInt>(vFloat(dst_reg[66].mode<DataLayout::F32>())) & 0x7fffu;
#endif
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
#ifdef TT_WH_PERIODIC_WINDOW_POOL
    vUInt d3 = as<vUInt>(exman(vFloat(dst_reg[87].mode<DataLayout::F32>())));
#else
    vUInt d3 = as<vUInt>(exman(vFloat(dst_reg[67].mode<DataLayout::F32>())));
#endif
#else
    vUInt d3 = as<vUInt>(vFloat(dst_reg[67].mode<DataLayout::F32>())) & 0x7fffu;
#endif
    hi = hi | (d3 << 15);
#endif
#if defined(TT_WH_PERIODIC_OUTPUT_WINDOW26) && defined(TT_WH_PERIODIC_REGISTER_DIGITS)
    vUInt window = ((lo >> offset) | (hi << (30u - offset))) & 0x0fffffffu;
    vUInt round = (window >> 25) & 1u;
    vUInt state = (((window >> 26) + round) & 3u) | (round << 2);
    dst_reg[96].mode<DataLayout::F32>() = setexp(as<vFloat>(state), 150);
    lo = window & 0x03ffffffu;
    v_if(round != 0u) { lo = lo ^ 0x03ffffffu; }
    v_endif;
#ifdef TT_WH_PERIODIC_CAST_NORMALIZATION
    vUInt truncated = lo & ~(lo >> 24);
    vUInt fp = as<vUInt>(convert<vFloat>(as<vSMag>(truncated), RoundMode::Nearest)) - 0x0d000000u;
    v_if(lo == 0u) { fp = 0x32000000u; }
    v_endif;
#else
    vInt bit = 31 - as<vInt>(lz(lo));
    vUInt mantissa = lo << as<vUInt>(23 - bit);
#ifndef TT_WH_PERIODIC_SIGNED_NORMALIZATION
    v_if(bit > 23) { mantissa = lo >> as<vUInt>(bit - 23); }
    v_endif;
#endif
    vUInt fp = (as<vUInt>(bit + 101) << 23) | (mantissa & 0x7fffffu);
#endif
#else
    lo = ((lo >> offset) | (hi << (30u - offset))) & 0x3fffffffu;
    hi = hi >> offset;
    vUInt round = (hi >> 6) & 1u;
    vUInt state = (((hi >> 7) + round) & 3u) | (round << 2);
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
    dst_reg[96].mode<DataLayout::F32>() = setexp(as<vFloat>(state), 150);
#else
    dst_reg[96].mode<DataLayout::F32>() = as<vFloat>(state | 0x4b000000u);
#endif
    hi = hi & 127u;
    v_if(round != 0u) {
        lo = lo ^ 0x3fffffffu;
        hi = hi ^ 127u;
    }
    v_endif;
    vInt bit = 31 - as<vInt>(lz(lo));
    v_if(hi != 0u) { bit = 61 - as<vInt>(lz(hi)); }
    v_endif;
    vUInt mantissa = (hi << as<vUInt>(53 - bit)) | (lo >> as<vUInt>(bit - 23));
    vUInt fp = (as<vUInt>(bit + 90) << 23) | (mantissa & 0x7fffffu);
#endif
    dst_reg[97].mode<DataLayout::F32>() = as<vFloat>(fp);
}
#endif

#if defined(TT_WH_PERIODIC_PHASE_SIGN) && !defined(TT_WH_PERIODIC_SIGNED_NORMALIZATION)
#error "periodic phase sign requires the complete signed-normalized window owner"
#endif

sfpi_inline vUInt tt_q192_top(int dst_index) {
    vUInt input_bits = as<vUInt>(vFloat(dst_reg[dst_index]));
    vInt work_exponent = as<vInt>((input_bits >> 23) & 0xffu);
    v_if(work_exponent < 126) { work_exponent = 126; }
    v_endif;
    vUInt distance = as<vUInt>(342 - work_exponent);
    vInt limb_index = as<vInt>(tt_wh_radix_index(distance));
#ifndef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
    vUInt addend = 0u;
#endif
    vUInt packed_top = as<vUInt>(limb_index) << 28;
#ifdef PERIODIC_Q192_UNROLL_DISABLE
#pragma GCC unroll 1
#else
#pragma GCC unroll 10
#endif
    for (int i = 0; i < 10; ++i) {
#ifdef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
        vUInt sum = as<vUInt>(vFloat(dst_reg[64 + i].mode<DataLayout::F32>())) & 0x7fffffu;
#else
        vUInt product = tt_wh_product(dst_index, tt_q192_limbs[i]);
        vUInt sum;
        sum = ((product & 127u) << 16);
        sum = vUInt(__builtin_rvtt_sfpiadd_v_lv(sum.get(), sum.get(), addend.get(), SFPIADD_MOD1_CC_NONE));
        addend = tt_q192_limbs[i] + (sum >> 23);
        sum = sum & 0x7fffffu;
#endif
        vUInt index = packed_top >> 28;
        v_if(index == unsigned(i) + 1u) { packed_top = packed_top | ((sum & 0x00400000u) << 2); }
        v_endif;
        v_if(index == unsigned(i)) { packed_top = packed_top | sum; }
        v_endif;
        v_if(index + 1u == unsigned(i)) { packed_top = packed_top | ((sum & 1u) << 23); }
        v_endif;
#ifndef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
        product = product >> 7;
        addend = vUInt(__builtin_rvtt_sfpiadd_v_lv(addend.get(), product.get(), addend.get(), SFPIADD_MOD1_CC_NONE));
#endif
    }
    return packed_top;
}

sfpi_inline vUInt tt_q192_state_post(vUInt packed_top, int dst_index) {
#ifdef TT_WH_PERIODIC_INDEX_REUSE
    vInt limb_index = as<vInt>(packed_top >> 28);
#endif
    packed_top = packed_top & 0x01ffffffu;
    vUInt input_bits = as<vUInt>(vFloat(dst_reg[dst_index]));
    vInt work_exponent = as<vInt>((input_bits >> 23) & 0xffu);
    v_if(work_exponent < 126) { work_exponent = 126; }
    v_endif;
#ifndef TT_WH_PERIODIC_INDEX_REUSE
    vInt limb_index = as<vInt>(tt_wh_radix_index(as<vUInt>(342 - work_exponent)));
#endif
    vInt limb_bits = shft(limb_index, 4, ShiftMode::Logical) + shft(limb_index, 2, ShiftMode::Logical);
    limb_bits = limb_bits + shft(limb_index, 1, ShiftMode::Logical) + limb_index;
    vInt r = 342 - work_exponent - limb_bits;
    vUInt top = packed_top & 0x00ffffffu;
    vUInt quotient = (top >> as<vUInt>(r)) & 3u;
    vUInt round_up = (((top << 1) | (packed_top >> 24)) >> as<vUInt>(r)) & 1u;
    vUInt state = (input_bits & 0xff800000u) | (as<vUInt>(limb_index) << 5);
    state = state | ((quotient + round_up) & 3u) | (round_up << 2);
    v_if((input_bits & 0x7fffffu) == 0u) { state = state | 8u; }
    v_endif;
    v_if(limb_index != 3) { state = state & ~8u; }
    v_endif;
    return state;
}

sfpi_inline vUInt tt_q192_state(int dst_index) {
#ifdef TT_WH_PERIODIC_WINDOW37_ACTIVE
#ifdef TT_WH_PERIODIC_PRIMITIVE_BITS
    vUInt low = as<vUInt>(exman(vFloat(dst_reg[96].mode<DataLayout::F32>())));
#else
    vUInt low = as<vUInt>(vFloat(dst_reg[96].mode<DataLayout::F32>())) & 7u;
#endif
    return (as<vUInt>(vFloat(dst_reg[dst_index])) & 0xff800000u) | low;
#else
    return tt_q192_state_post(tt_q192_top(dst_index), dst_index);
#endif
}

#ifdef PERIODIC_Q192_STATE_REUSE_DISABLE
sfpi_inline vUInt tt_q192_control(int dst_index) {
    vUInt state = tt_q192_state(dst_index);
#else
sfpi_inline vUInt tt_q192_control(vUInt state, int dst_index) {
#endif
    vUInt input_bits = as<vUInt>(vFloat(dst_reg[dst_index]));
    vUInt m = input_bits & 0x7fffffu;
    vUInt limb_index = (state >> 5) & 15u;
    vInt exponent = as<vInt>((input_bits >> 23) & 0xffu);
    v_if(exponent < 126) { exponent = 126; }
    v_endif;
    vInt limb_bits = as<vInt>(limb_index);
    limb_bits = shft(limb_bits, 4, ShiftMode::Logical) + shft(limb_bits, 2, ShiftMode::Logical) +
                shft(limb_bits, 1, ShiftMode::Logical) + limb_bits;
    vUInt remainder_mask = shft(vUInt(1u), 342 - exponent - limb_bits) - 1u;
    vUInt control = (limb_index << 25) | ((state & 4u) << 21) | remainder_mask;
    v_if(m == 0u) {
        v_if(limb_index == 3u) { control = control | 0x01000000u; }
        v_endif;
    }
    v_endif;
    return control;
}

sfpi_inline vUInt tt_q192_find_target(vUInt control, int dst_index) {
#ifndef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
    vUInt addend = 0u;
#endif
#ifdef PERIODIC_Q192_UNROLL_DISABLE
#pragma GCC unroll 1
#else
#pragma GCC unroll 10
#endif
    for (int i = 0; i < 10; ++i) {
#ifdef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
        vUInt sum = as<vUInt>(vFloat(dst_reg[64 + i].mode<DataLayout::F32>())) & 0x7fffffu;
#else
        vUInt product = tt_wh_product(dst_index, tt_q192_limbs[i]);
        vUInt sum;
        sum = ((product & 127u) << 16);
        sum = vUInt(__builtin_rvtt_sfpiadd_v_lv(sum.get(), sum.get(), addend.get(), SFPIADD_MOD1_CC_NONE));
        addend = tt_q192_limbs[i] + (sum >> 23);
        sum = sum & 0x7fffffu;
#endif
        v_if((control & 0x00800000u) != 0u) {
            sum = ~sum;
            sum = sum & 0x7fffffu;
        }
        v_endif;
        v_if((control & 0x01800000u) == 0x01800000u) { sum = sum + 1u; }
        v_endif;
        control = control & ~0x01000000u;
        v_if((sum & 0x00800000u) != 0u) { control = control | 0x01000000u; }
        v_endif;
        sum = sum & 0x7fffffu;
#ifndef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
        product = product >> 7;
        addend = vUInt(__builtin_rvtt_sfpiadd_v_lv(addend.get(), product.get(), addend.get(), SFPIADD_MOD1_CC_NONE));
#endif
        vUInt limb_index = (control >> 25) & 15u;
        v_if(limb_index == unsigned(i)) { sum = sum & (control & 0x7fffffu); }
        v_endif;
        // Digits increase in significance with i.  Prefer limb_index, then
        // limb_index-1, then limb_index-2; later matches overwrite earlier.
        v_if((limb_index == unsigned(i) + 2u) && (sum != 0u)) { control = (control & 0x9fffffffu) | 0xc0000000u; }
        v_endif;
        v_if((limb_index == unsigned(i) + 1u) && (sum != 0u)) { control = (control & 0x9fffffffu) | 0xa0000000u; }
        v_endif;
        v_if((limb_index == unsigned(i)) && (sum != 0u)) { control = (control & 0x9fffffffu) | 0x80000000u; }
        v_endif;
    }
    return control;
}

template <unsigned OFFSET, bool PRESERVE>
sfpi_inline vUInt tt_q192_dynamic_digit(vUInt control, int dst_index) {
#ifndef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
    vUInt addend = 0u;
#endif
#ifdef PERIODIC_Q192_UNROLL_DISABLE
#pragma GCC unroll 1
#else
#pragma GCC unroll 10
#endif
    for (int i = 0; i < 10; ++i) {
#ifdef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
        vUInt sum = as<vUInt>(vFloat(dst_reg[64 + i].mode<DataLayout::F32>())) & 0x7fffffu;
#else
        vUInt product = tt_wh_product(dst_index, tt_q192_limbs[i]);
        vUInt sum;
        sum = ((product & 127u) << 16);
        sum = vUInt(__builtin_rvtt_sfpiadd_v_lv(sum.get(), sum.get(), addend.get(), SFPIADD_MOD1_CC_NONE));
        addend = tt_q192_limbs[i] + (sum >> 23);
        sum = sum & 0x7fffffu;
#endif
        v_if((control & 0x00800000u) != 0u) {
            sum = ~sum;
            sum = sum & 0x7fffffu;
        }
        v_endif;
        v_if((control & 0x01800000u) == 0x01800000u) { sum = sum + 1u; }
        v_endif;
        control = control & ~0x01000000u;
        v_if((sum & 0x00800000u) != 0u) { control = control | 0x01000000u; }
        v_endif;
        sum = sum & 0x7fffffu;
#ifndef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
        product = product >> 7;
        addend = vUInt(__builtin_rvtt_sfpiadd_v_lv(addend.get(), product.get(), addend.get(), SFPIADD_MOD1_CC_NONE));
#endif
        vUInt limb_index = (control >> 25) & 15u;
        v_if(limb_index == unsigned(i)) { sum = sum & (control & 0x7fffffu); }
        v_endif;
        vUInt wanted = limb_index - ((control >> 29) & 3u) - OFFSET;
        v_if(wanted == unsigned(i)) {
            if constexpr (PRESERVE) {
                control = (control & 0xff800000u) | sum;
            } else {
                control = sum;
            }
        }
        v_endif;
    }
    return control;
}

sfpi_inline vUInt tt_q192_pack_mantissa(vUInt high, vUInt low) {
    vUInt valid = 1u;
    v_if(high == 0u) {
        high = 1u;
        valid = 0u;
    }
    v_endif;
    vInt bit_index = 31 - as<vInt>(lz(high));
    vInt take = 23 - bit_index;
    vUInt mantissa = shft(high, take) | shft(low, -bit_index);
    return (mantissa & 0x00ffffffu) | (as<vUInt>(bit_index) << 24) | (valid << 29);
}

sfpi_inline vFloat tt_q192_scale(vUInt packed, int dst_index) {
    vUInt metadata = packed >> 24;
    packed = packed & 0x7fffffu;
    vInt exponent = as<vInt>((as<vUInt>(vFloat(dst_reg[dst_index])) >> 23) & 0xffu);
    v_if(exponent < 126) { exponent = 126; }
    v_endif;
    vInt limb_index = as<vInt>(tt_wh_radix_index(as<vUInt>(342 - exponent)));
    limb_index = limb_index - as<vInt>((metadata >> 6) & 3u);
    limb_index = shft(limb_index, 4, ShiftMode::Logical) + shft(limb_index, 2, ShiftMode::Logical) +
                 shft(limb_index, 1, ShiftMode::Logical) + limb_index;
    // Product digit i has weight 2**(23*i), and the exact Q192 quotient
    // denominator contributes 2**(342-exponent).  Add the FP32 bias here:
    // 23*i + bit_index + exponent - 342 + 127.
    exponent = limb_index + as<vInt>(metadata & 31u) + exponent - 215;
    metadata = as<vUInt>(exponent) | shft(((metadata >> 5) & 1u) ^ 1u, 8);
    metadata = shft(metadata, 23);
    packed = vUInt(__builtin_rvtt_sfpiadd_v_lv(packed.get(), packed.get(), metadata.get(), SFPIADD_MOD1_CC_NONE));
    return as<vFloat>(packed);
}

sfpi_inline vFloat tt_q192_scale_with_state(vUInt packed, vUInt state, int dst_index) {
    vUInt metadata = packed >> 24;
    packed = packed & 0x7fffffu;
    vInt exponent = as<vInt>((state >> 23) & 0xffu);
    v_if(exponent < 126) { exponent = 126; }
    v_endif;
    vInt limb_index = as<vInt>((state >> 5) & 15u);
    limb_index = limb_index - as<vInt>((metadata >> 6) & 3u);
    limb_index = shft(limb_index, 4, ShiftMode::Logical) + shft(limb_index, 2, ShiftMode::Logical) +
                 shft(limb_index, 1, ShiftMode::Logical) + limb_index;
    // Product digit i has weight 2**(23*i), and the exact Q192 quotient
    // denominator contributes 2**(342-exponent).  Add the FP32 bias here:
    // 23*i + bit_index + exponent - 342 + 127.
    exponent = limb_index + as<vInt>(metadata & 31u) + exponent - 215;
    metadata = as<vUInt>(exponent) | shft(((metadata >> 5) & 1u) ^ 1u, 8);
    metadata = shft(metadata, 23);
    packed = vUInt(__builtin_rvtt_sfpiadd_v_lv(packed.get(), packed.get(), metadata.get(), SFPIADD_MOD1_CC_NONE));
    return as<vFloat>(packed);
}

#ifdef PERIODIC_Q192_STATE_REUSE_DISABLE
sfpi_inline vUInt tt_q192_candidate(int dst_index) {
    vUInt high_pack =
        tt_q192_dynamic_digit<0, true>(tt_q192_find_target(tt_q192_control(dst_index), dst_index), dst_index);
#else
sfpi_inline vUInt tt_q192_candidate(vUInt state, int dst_index) {
    vUInt high_pack =
        tt_q192_dynamic_digit<0, true>(tt_q192_find_target(tt_q192_control(state, dst_index), dst_index), dst_index);
#endif
    vUInt low_pack = tt_q192_dynamic_digit<1, false>(high_pack, dst_index);
    vUInt packed = tt_q192_pack_mantissa(high_pack & 0x7fffffu, low_pack & 0x7fffffu);
    vUInt target_bits = (high_pack >> 29) << 30;
    packed = vUInt(__builtin_rvtt_sfpor_lv(packed.get(), packed.get(), target_bits.get()));
    return packed;
}
#if defined(TT_WH_PERIODIC_ASCENDING_GATHER) && !defined(TT_WH_PERIODIC_FLOAT_EXPANSION) && \
    !defined(TT_WH_PERIODIC_FLOAT_EXPANSION_PARITY)
#error "ascending gather requires its complete analytic expansion owner"
#endif
#if defined(TT_WH_PERIODIC_FLOAT_EXPANSION) || defined(TT_WH_PERIODIC_FLOAT_EXPANSION_PARITY)
#if defined(FUSE_GRAD_MUL) || !defined(EMBEDDED_LUT) || !defined(TT_WH_PERIODIC_CAST_NORMALIZATION) || \
    (defined(TT_WH_PERIODIC_FLOAT_EXPANSION) && !defined(TT_WH_PERIODIC_PHASE_SIGN))
#error "analytic expansion requires exclusive unary bank"
#endif
template <bool PARITY = false>
sfpi_inline void tt_q192_expansion_prefill() {
    dst_reg[32].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3b22f980u : 0x3aa2f980u));
    dst_reg[33].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x305b9380u : 0x2fdb9380u));
    dst_reg[34].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x26882a40u : 0x26082a40u));
    dst_reg[35].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3fbe60c0u : 0x3f3e60c0u));
    dst_reg[36].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x365c9c80u : 0x35dc9c80u));
    dst_reg[37].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x2c02a540u : 0x2b82a540u));
    dst_reg[38].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3f5b9380u : 0x3edb9380u));
    dst_reg[39].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x35882a40u : 0x35082a40u));
    dst_reg[40].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x2c1fc260u : 0x2b9fc260u));
    dst_reg[41].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3fe44140u : 0x3f644140u));
    dst_reg[42].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3614fe00u : 0x3594fe00u));
    dst_reg[43].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x2c9d5f40u : 0x2c1d5f40u));
    dst_reg[44].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3fa94fe0u : 0x3f294fe0u));
    dst_reg[45].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x341d5f40u : 0x339d5f40u));
    dst_reg[46].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x29fa9a60u : 0x297a9a60u));
    dst_reg[47].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3ff09d40u : 0x3ff84ea0u));
    dst_reg[48].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x367a3ea0u : 0x35fa3ea0u));
    dst_reg[49].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x2bd37700u : 0x2b537700u));
    dst_reg[50].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3fafa3e0u : 0x3f2fa3e0u));
    dst_reg[51].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x35a69ba0u : 0x35269ba0u));
    dst_reg[52].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x2c40db60u : 0x2bc0db60u));
    dst_reg[53].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3ff534c0u : 0x3ffa9a60u));
    dst_reg[54].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x366e06c0u : 0x35ee06c0u));
    dst_reg[55].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x2cd8a560u : 0x2c58a560u));
    dst_reg[56].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x3f5dc0c0u : 0x3eddc0c0u));
    dst_reg[57].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x35db14a0u : 0x355b14a0u));
    dst_reg[58].mode<DataLayout::F32>() = __builtin_bit_cast(float, uint32_t(PARITY ? 0x2bcc9e20u : 0x2b4c9e20u));
}
template <int PHASE, bool PARITY = false>
sfpi_inline vFloat tt_q192_expansion(int dst_index) {
    vFloat input = dst_reg[dst_index];
    vInt exponent = exexp(input, ExponentMode::Biased);
    v_if(exponent < 126) { exponent = 126; }
    v_endif;
    vUInt factor = as<vUInt>(exponent - 125);
    vInt bucket = as<vInt>((factor + (factor << 4)) >> 8);
    vFloat high = dst_reg[32].mode<DataLayout::F32>();
    vFloat low = dst_reg[33].mode<DataLayout::F32>();
    vFloat tail = dst_reg[34].mode<DataLayout::F32>();
    auto rh = high.get();
    auto rl = low.get();
    auto rt = tail.get();
#ifndef TT_WH_PERIODIC_ASCENDING_GATHER
    v_if(bucket == 1) {
#else
    v_if(bucket >= 1) {
#endif
        rh = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rh,
            (35) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rl = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rl,
            (36) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rt = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rt,
            (37) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
    }
    v_endif;
#ifndef TT_WH_PERIODIC_ASCENDING_GATHER
    v_if(bucket == 2) {
#else
    v_if(bucket >= 2) {
#endif
        rh = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rh,
            (38) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rl = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rl,
            (39) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rt = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rt,
            (40) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
    }
    v_endif;
#ifndef TT_WH_PERIODIC_ASCENDING_GATHER
    v_if(bucket == 3) {
#else
    v_if(bucket >= 3) {
#endif
        rh = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rh,
            (41) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rl = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rl,
            (42) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rt = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rt,
            (43) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
    }
    v_endif;
#ifndef TT_WH_PERIODIC_ASCENDING_GATHER
    v_if(bucket == 4) {
#else
    v_if(bucket >= 4) {
#endif
        rh = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rh,
            (44) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rl = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rl,
            (45) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rt = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rt,
            (46) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
    }
    v_endif;
#ifndef TT_WH_PERIODIC_ASCENDING_GATHER
    v_if(bucket == 5) {
#else
    v_if(bucket >= 5) {
#endif
        rh = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rh,
            (47) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rl = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rl,
            (48) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rt = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rt,
            (49) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
    }
    v_endif;
#ifndef TT_WH_PERIODIC_ASCENDING_GATHER
    v_if(bucket == 6) {
#else
    v_if(bucket >= 6) {
#endif
        rh = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rh,
            (50) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rl = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rl,
            (51) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rt = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rt,
            (52) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
    }
    v_endif;
#ifndef TT_WH_PERIODIC_ASCENDING_GATHER
    v_if(bucket == 7) {
#else
    v_if(bucket >= 7) {
#endif
        rh = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rh,
            (53) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rl = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rl,
            (54) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rt = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rt,
            (55) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
    }
    v_endif;
#ifndef TT_WH_PERIODIC_ASCENDING_GATHER
    v_if(bucket == 8) {
#else
    v_if(bucket >= 8) {
#endif
        rh = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rh,
            (56) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rl = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rl,
            (57) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
        rt = __builtin_rvtt_sfpload_lv(
            ckernel::instrn_buffer,
            rt,
            (58) * SFP_DESTREG_STRIDE,
            0,
            0,
            SFPLOAD_MOD0_FMT_FP32,
            SFPLOAD_ADDR_MODE_NOINC);
    }
    v_endif;
    high = vFloat(rh);
    low = vFloat(rl);
    tail = vFloat(rt);
    vInt offset = exponent - 126 - ((bucket << 4) - bucket);
    vFloat m = setexp(setsgn(input, 0), offset + 134);
    vFloat product = m * high;
    product = m * low + product;
    product = m * tail + product;
    if constexpr (PHASE == 1) {
        product = product + 0.5f;
    }
    vFloat biased = product + 8388608.0f;
    vUInt q = as<vUInt>(biased);
    vFloat qf = biased - 8388608.0f;
    vFloat residue;
    if constexpr (PARITY) {
        residue = -qf;
    } else {
        residue = float(PHASE) * 0.5f - qf;
    }
    residue = m * high + residue;
    residue = m * low + residue;
    residue = m * tail + residue;
    vFloat coordinate;
    if constexpr (PARITY) {
        static_assert(PHASE == 0);
        vFloat magnitude = setsgn(residue, 0);
        v_if(magnitude > 0.5f) {
            residue = -copysgn(1.0f - magnitude, residue);
            q = q ^ 1u;
        }
        v_endif;
        dst_reg[96].mode<DataLayout::F32>() = setexp(as<vFloat>(q & 1u), 150);
        coordinate = residue * 1.5707963705062866f;
        coordinate = as<vFloat>(as<vUInt>(coordinate) ^ (as<vUInt>(input) & 0x80000000u));
        exponent = exexp(input, ExponentMode::Biased);
        v_if(exponent <= 125) { coordinate = input; }
        v_endif;
    } else {
        vFloat magnitude = setsgn(residue, 0);
        v_if(magnitude > 0.5f) { magnitude = 1.0f - magnitude; }
        v_endif;
        residue = copysgn(magnitude, residue);
        coordinate = residue * 3.1415927410125732f;
        vUInt sign = q << 31;
        if constexpr (PHASE == 0) {
            sign = sign ^ (as<vUInt>(input) & 0x80000000u);
        }
        coordinate = as<vFloat>(as<vUInt>(coordinate) ^ sign);
        exponent = exexp(input, ExponentMode::Biased);
        v_if(exponent <= 125) {
            if constexpr (PHASE == 0) {
                coordinate = input;
            } else {
                coordinate = 1.5707963705062866f - setsgn(input, 0);
            }
        }
        v_endif;
        v_if(exponent == 255) { coordinate = std::numeric_limits<float>::quiet_NaN(); }
        v_endif;
    }
    return coordinate;
}
#endif
template <int PHASE_QUADRANTS>
sfpi_inline vFloat tt_periodic_reduce_q192(int dst_index) {
#ifdef TT_WH_PERIODIC_FLOAT_EXPANSION
    static_assert(PHASE_QUADRANTS == 0 || PHASE_QUADRANTS == 1);
    return tt_q192_expansion<PHASE_QUADRANTS>(dst_index);
#else

#ifdef TT_WH_PERIODIC_WINDOW37_ACTIVE
    tt_q192_window37(dst_index);
    vUInt state = tt_q192_state(dst_index);
    vFloat residual = dst_reg[97].mode<DataLayout::F32>();
#else

#ifdef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
    tt_q192_prefill_digits(dst_index);
#endif
#ifdef PERIODIC_Q192_STATE_REUSE_DISABLE
    vFloat residual = tt_q192_scale(tt_q192_candidate(dst_index), dst_index);
    vUInt state = tt_q192_state(dst_index);
#else
    vUInt state = tt_q192_state(dst_index);
#ifdef TT_WH_PERIODIC_INDEX_REUSE
    vFloat residual = tt_q192_scale_with_state(tt_q192_candidate(state, dst_index), state, dst_index);
#else
    vFloat residual = tt_q192_scale(tt_q192_candidate(state, dst_index), dst_index);
#endif
#endif
#endif
    residual = residual * 1.5707963705062866f;
#ifdef TT_WH_PERIODIC_PHASE_SIGN
    vUInt folded = (state ^ unsigned(PHASE_QUADRANTS)) & 1u;
    vUInt negative = state >> 1;
    if constexpr ((PHASE_QUADRANTS & 1) == 0) {
        negative = negative ^ (state >> 31) ^ ((~state) & (state >> 2));
    } else {
        negative = negative ^ state ^ (state & (state >> 2));
    }
    if constexpr ((PHASE_QUADRANTS & 2) != 0) {
        negative = negative ^ 1u;
    }
    vFloat coordinate = residual;
    v_if(folded != 0u) { coordinate = 1.5707963705062866f - coordinate; }
    v_endif;
    coordinate = as<vFloat>(as<vUInt>(coordinate) | (negative << 31));
#else
    vUInt quotient = state & 3u;
    vUInt input_sign = state >> 31;
    vUInt residual_bits = as<vUInt>(residual) | ((state & 4u) << 29);
    residual_bits = residual_bits ^ (input_sign << 31);
    quotient = (quotient ^ (0u - input_sign)) + input_sign;
    quotient = (quotient + PHASE_QUADRANTS) & 3u;
    vFloat coordinate = as<vFloat>(residual_bits);
    v_if((quotient & 1u) != 0u) { coordinate = 1.5707963705062866f - setsgn(coordinate, 0); }
    v_endif;
    v_if((quotient & 2u) != 0u) { coordinate = -coordinate; }
    v_endif;
#endif
    vInt exponent = as<vInt>((state >> 23) & 0xffu);
    v_if(exponent <= 125) {
        vFloat input = vFloat(dst_reg[dst_index]);
        if constexpr (PHASE_QUADRANTS == 0) {
            coordinate = input;
        }
        if constexpr (PHASE_QUADRANTS == 1) {
            coordinate = 1.5707963705062866f - setsgn(input, 0);
        }
        if constexpr (PHASE_QUADRANTS == 2) {
            coordinate = -input;
        }
        if constexpr (PHASE_QUADRANTS == 3) {
            coordinate = -(1.5707963705062866f - setsgn(input, 0));
        }
    }
    v_endif;
    vUInt original_bits = as<vUInt>(vFloat(dst_reg[dst_index]));
    v_if((original_bits & 0x7f800000u) == 0x7f800000u) { coordinate = std::numeric_limits<float>::quiet_NaN(); }
    v_endif;
    return coordinate;
#endif
}
// Principal pi/2 residual for the existing rational-plus-reciprocal graph.
sfpi_inline vFloat tt_periodic_reduce_q192_tan(int dst_index) {
#ifdef TT_WH_PERIODIC_FLOAT_EXPANSION_PARITY
    return tt_q192_expansion<0, true>(dst_index);
#else

#ifdef TT_WH_PERIODIC_WINDOW37_ACTIVE
    tt_q192_window37(dst_index);
    vUInt state = tt_q192_state(dst_index);
    vFloat residual = dst_reg[97].mode<DataLayout::F32>();
#else

#ifdef TT_WH_PERIODIC_PRODUCT_CACHE_ACTIVE
    tt_q192_prefill_digits(dst_index);
#endif
#ifdef PERIODIC_Q192_STATE_REUSE_DISABLE
    vFloat residual = tt_q192_scale(tt_q192_candidate(dst_index), dst_index);
    vUInt state = tt_q192_state(dst_index);
#else
    vUInt state = tt_q192_state(dst_index);
#ifdef TT_WH_PERIODIC_INDEX_REUSE
    vFloat residual = tt_q192_scale_with_state(tt_q192_candidate(state, dst_index), state, dst_index);
#else
    vFloat residual = tt_q192_scale(tt_q192_candidate(state, dst_index), dst_index);
#endif
#endif
#endif
    residual = residual * 1.5707963705062866f;
    vUInt bits = as<vUInt>(residual) | ((state & 4u) << 29);
    bits = bits ^ (state & 0x80000000u);
    residual = as<vFloat>(bits);
    v_if(((state >> 23) & 0xffu) <= 125u) { residual = dst_reg[dst_index]; }
    v_endif;
    return residual;
#endif
}

}  // namespace sfpi

#endif  // TRISC_MATH
