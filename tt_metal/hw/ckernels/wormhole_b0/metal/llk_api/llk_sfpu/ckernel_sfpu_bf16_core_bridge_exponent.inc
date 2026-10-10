// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
// Namespace-relative canonical terminal definitions.
static inline vFloat selected_core_total_exponent_replay(vFloat x, vFloat core) {
    vInt biased = exexp(x, ExponentMode::Biased);
    vFloat exponent = convert<vFloat>(as<vSMag>(biased - TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT), RoundMode::Nearest);
    vFloat result = exponent * TT_SELECTED_CORE_LOG_TWO + core;
    v_if(biased == 255) { result *= x; }
    v_endif;
    return result;
}

static inline vFloat selected_core_exponent_replay_prepare(vFloat x, vInt& exponent_delta) {
    exponent_delta = 0;
    vFloat coordinate = x;
    v_if(coordinate > TT_SELECTED_CORE_UPPER) {
        exponent_delta = exexp(coordinate, ExponentMode::Biased) - TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT;
        coordinate = setexp(coordinate, TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT);
    }
    v_endif;
    vFloat lower = TT_SELECTED_CORE_LOWER;
    ordered_min_max(lower, coordinate);
    return coordinate;
}

static inline void selected_core_exponent_replay_finalize(vFloat x, vInt exponent_delta, vFloat& result) {
    vFloat exponent = convert<vFloat>(as<vSMag>(exponent_delta), RoundMode::Nearest);
    result = exponent * TT_SELECTED_CORE_LOG_TWO + result;
    v_if(x < TT_SELECTED_CORE_INVALID_BELOW) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x < TT_SELECTED_CORE_BRIDGE_UPPER) {
        result = (x - TT_SELECTED_CORE_ENDPOINT) * TT_SELECTED_CORE_BRIDGE_SLOPE;
    }
    v_endif;
    constexpr int special_delta = 255 - TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT;
    v_if(exponent_delta == special_delta) { result *= x; }
    v_endif;
}

static inline vFloat selected_core_bridge_prepare(vFloat x_raw) {
    vFloat coordinate = x_raw;
    v_if(coordinate > TT_SELECTED_CORE_UPPER) {
        coordinate = setexp(coordinate, TT_SELECTED_CORE_REPLAY_BIASED_EXPONENT);
    }
    v_endif;
    vFloat lower = TT_SELECTED_CORE_LOWER;
    ordered_min_max(lower, coordinate);
    return coordinate;
}

static inline void selected_core_bridge_finalize(vFloat x, vFloat& result) {
    v_if(x < TT_SELECTED_CORE_INVALID_BELOW) { result = std::numeric_limits<float>::quiet_NaN(); }
    v_elseif(x < TT_SELECTED_CORE_BRIDGE_UPPER) {
        result = (x - TT_SELECTED_CORE_ENDPOINT) * TT_SELECTED_CORE_BRIDGE_SLOPE;
    }
    v_elseif(x > TT_SELECTED_CORE_UPPER) { result = selected_core_total_exponent_replay(x, result); }
    v_endif;
}
