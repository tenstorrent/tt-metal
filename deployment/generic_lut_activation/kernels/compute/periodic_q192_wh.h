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
// every BF16 significand. Four following limbs cover the 37-bit fraction,
// two quadrant bits, and the at-most-fourteen-bit extraction offset.
sfpi_inline void tt_q192_window37(int dst_index) {
    static constexpr uint32_t limbs[] = {
        4161u, 30855u, 22116u, 23316u, 23565u, 9883u, 29821u, 5035u, 10748u, 2090u, 14649u, 19483u, 2607u, 0u};
    vUInt chunk0 = 0u;
    vUInt chunk1 = 0u;
    vUInt chunk2 = 0u;
    {
        vUInt raw = as<vUInt>(vFloat(dst_reg[dst_index]));
        vInt exponent = as<vInt>((raw >> 23) & 255u);
        v_if(exponent < 126) { exponent = 126; }
        v_endif;
        vInt shift = 289 - exponent;
        // floor(n/15) = ((n+1)*17)>>8 for 0 <= n <= 254;
        // the complete clamped BF16 exponent range gives 34 <= n <= 163.
        vUInt factor = as<vUInt>(shift + 1);
        vInt index = as<vInt>((factor + (factor << 4)) >> 8);
#pragma GCC unroll 9
        for (int i = 2; i <= 10; ++i) {
            v_if(index == i) {
                chunk0 = limbs[i - 2] | (limbs[i - 1] << 15);
                chunk1 = limbs[i] | (limbs[i + 1] << 15);
                chunk2 = limbs[i + 2] | (limbs[i + 3] << 15);
            }
            v_endif;
        }
        vInt offset = shift - ((index << 4) - index);
        dst_reg[96].mode<DataLayout::F32>() = as<vFloat>(as<vUInt>(offset) | 0x4b000000u);
    }
    {
        vUInt raw = as<vUInt>(vFloat(dst_reg[dst_index]));
        vFloat m = as<vFloat>((raw & 0x007f0000u) | 0x43000000u);
        vUInt carry = 0u;
#pragma GCC unroll 6
        for (int i = 0; i < 6; ++i) {
            vUInt part = (i < 2 ? chunk0 : (i < 4 ? chunk1 : chunk2));
            if (i % 2) {
                part = part >> 15;
            }
            part = part & 0x7fffu;
            vFloat k = convert<vFloat>(as<vSMag>(part), RoundMode::Nearest);
            vFloat bias = as<vFloat>(carry | 0x4b000000u);
            vFloat product = __builtin_rvtt_sfpmad(m.get(), k.get(), bias.get(), SFPMAD_MOD1_OFFSET_NONE);
            vUInt integer = as<vUInt>(product) & 0x7fffffu;
            carry = integer >> 15;
            if (i >= 2) {
                dst_reg[64 + i - 2].mode<DataLayout::F32>() = as<vFloat>((integer & 0x7fffu) | 0x4b000000u);
            }
        }
    }
    vUInt offset = as<vUInt>(vFloat(dst_reg[96].mode<DataLayout::F32>())) & 15u;
    vUInt lo = as<vUInt>(vFloat(dst_reg[64].mode<DataLayout::F32>())) & 0x7fffu;
    vUInt d1 = as<vUInt>(vFloat(dst_reg[65].mode<DataLayout::F32>())) & 0x7fffu;
    lo = lo | (d1 << 15);
    vUInt hi = as<vUInt>(vFloat(dst_reg[66].mode<DataLayout::F32>())) & 0x7fffu;
    vUInt d3 = as<vUInt>(vFloat(dst_reg[67].mode<DataLayout::F32>())) & 0x7fffu;
    hi = hi | (d3 << 15);
    lo = ((lo >> offset) | (hi << (30u - offset))) & 0x3fffffffu;
    hi = hi >> offset;
    vUInt round = (hi >> 6) & 1u;
    vUInt state = (((hi >> 7) + round) & 3u) | (round << 2);
    dst_reg[96].mode<DataLayout::F32>() = as<vFloat>(state | 0x4b000000u);
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
    dst_reg[97].mode<DataLayout::F32>() = as<vFloat>(fp);
}
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
    vUInt low = as<vUInt>(vFloat(dst_reg[96].mode<DataLayout::F32>())) & 7u;
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
template <int PHASE_QUADRANTS>
sfpi_inline vFloat tt_periodic_reduce_q192(int dst_index) {
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
}
// Principal pi/2 residual for the existing rational-plus-reciprocal graph.
sfpi_inline vFloat tt_periodic_reduce_q192_tan(int dst_index) {
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
}

}  // namespace sfpi

#endif  // TRISC_MATH
