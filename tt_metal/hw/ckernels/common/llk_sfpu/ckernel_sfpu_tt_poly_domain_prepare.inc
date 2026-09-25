// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
inline vFloat prepare_raw_domain_input(vFloat x_raw) {
#if defined(TT_SELECTED_WH_SIGNED_ABS_RECIPROCAL)
    vFloat coordinate = setsgn(x_raw, 0);
    v_if(coordinate > TT_SELECTED_RECIP_CORE_BOUND) {
        coordinate = ckernel::sfpu::sfpu_reciprocal_iter<2>(coordinate);
    }
    v_endif;
    return coordinate;
#elif defined(TT_SELECTED_CORE_AGGREGATE_STIRLING)
    return selected_aggregate_stirling_prepare(x_raw);
#elif defined(TT_SELECTED_CORE_BRIDGE_EXPONENT_REPLAY_TAIL_1)
    return selected_core_bridge_prepare(x_raw);
#elif defined(TT_SELECTED_CORE_SQUARE_AFFINE_EXP2_MILLS_TAIL_1)
    vFloat coordinate = x_raw * x_raw;
    vFloat lower = 0.0f;
    vFloat upper = TT_SELECTED_CORE_UPPER * TT_SELECTED_CORE_UPPER;
    ordered_min_max(lower, coordinate);
    ordered_min_max(coordinate, upper);
    return coordinate;
#elif defined(TT_SELECTED_CORE_SCALED_SQUARE_EXP2_TAIL_1)
    vFloat coordinate = (x_raw * x_raw) * TT_SELECTED_CORE_COORDINATE_SCALE + TT_SELECTED_CORE_COORDINATE_BIAS;
    vFloat lower = -1.0f;
    vFloat upper = 1.0f;
    ordered_min_max(lower, coordinate);
    ordered_min_max(coordinate, upper);
    return coordinate;
#elif defined(TT_SELECTED_CORE_TOTAL_FORM)
    vFloat coordinate = x_raw;
    v_if(coordinate < TT_SELECTED_CORE_LOWER) { coordinate = TT_SELECTED_CORE_LOWER; }
    v_elseif(coordinate > TT_SELECTED_CORE_UPPER) { coordinate = TT_SELECTED_CORE_UPPER; }
    v_endif;
    return coordinate;
#elif defined(TT_WH_EXPONENT_ALU_LOG2_TERMINALS)
    vFloat prepared = sfpi::max(x_raw, std::numeric_limits<float>::min());
    vInt exponent = exexp(x_raw, ExponentMode::Biased);
    v_if(exponent == 255) { prepared = 1.0f; }
    v_endif;
    return prepared;
#elif defined(TT_DOMAIN_ACTION_LOWER_CLAMP_RESULT_CLAMP)
    // A one-sided constant terminal can be folded into an ingress maximum and
    // an egress maximum.  Only the opposite raw action remains below.
#if defined(ARCH_WORMHOLE)
    return dense_lower_clamp<true, __builtin_bit_cast(uint32_t, TT_DOMAIN_ACTION_LOWER_CLAMP_BOUND)>(x_raw);
#else
    return dense_lower_clamp<false, __builtin_bit_cast(uint32_t, TT_DOMAIN_ACTION_LOWER_CLAMP_BOUND)>(x_raw);
#endif
#elif defined(TT_DOMAIN_ACTION_TERMINAL_INGRESS_ONLY)
    // Both typed terminals are already the selected core's values at its
    // callable endpoints, as proved over the complete physical BF16 lattice.
    vFloat prepared = sfpi::max(x_raw, TT_DOMAIN_ACTION_CALLABLE_LOWER_BOUND);
    return sfpi::min(prepared, TT_DOMAIN_ACTION_CALLABLE_UPPER_BOUND);
#elif defined(TT_DOMAIN_ACTION_RAW_TERMINAL_ENVELOPE)
    // The compiler emits this only for one lower/upper pair whose untouched
    // interval is contained in the selected CSV's callable domain.  Evaluation
    // is therefore always in-domain; finalization below replays the terminal
    // actions from the original raw coordinate.
#if defined(TT_DOMAIN_ACTION_WH_ORDERED_INGRESS)
    vFloat prepared = x_raw;
    v_if(prepared < TT_DOMAIN_ACTION_RAW_LOWER_BOUND) { prepared = TT_DOMAIN_ACTION_RAW_LOWER_BOUND; }
    v_elseif(prepared > TT_DOMAIN_ACTION_RAW_UPPER_BOUND) { prepared = TT_DOMAIN_ACTION_RAW_UPPER_BOUND; }
    v_endif;
    return prepared;
#else
    vFloat prepared = sfpi::max(x_raw, TT_DOMAIN_ACTION_RAW_LOWER_BOUND);
    return sfpi::min(prepared, TT_DOMAIN_ACTION_RAW_UPPER_BOUND);
#endif
#else
    return x_raw;
#endif
}
