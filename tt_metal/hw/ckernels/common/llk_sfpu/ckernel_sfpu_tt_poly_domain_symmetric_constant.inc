// This macro is emitted only for the complete typed action shape certified by
// detect_symmetric_constant_raw_tails. Keep the original records and hashes in
// generated source: these assertions make the cheaper lowering subordinate to
// that unchanged program rather than a replacement contract.
inline void apply_raw_pre_domain_actions(vFloat x_raw, vFloat& result) {
    static_assert(TT_DOMAIN_ACTION_COUNT == 2, "symmetric tails require exactly two actions");
    constexpr auto first = TT_DOMAIN_ACTION_DATA[0];
    constexpr auto second = TT_DOMAIN_ACTION_DATA[1];
    static_assert(first.action_kind == 0 && second.action_kind == 0, "symmetric tails require constant actions");
    static_assert(first.inclusive == second.inclusive, "symmetric tails require matching inclusivity");
    static_assert(
        first.inclusive == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_INCLUSIVE,
        "symmetric tail certificate does not match action records");
    static_assert(
        (first.direction == 0 && second.direction == 1 && first.bound == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
         second.bound == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND) ||
            (first.direction == 1 && second.direction == 0 && first.bound == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND &&
             second.bound == -TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND),
        "symmetric tail certificate does not match action bounds");
    static_assert(
        first.value == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE && second.value == TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE,
        "symmetric tail certificate does not match action values");

    vFloat absolute_raw = setsgn(x_raw, 0);
    if constexpr (TT_DOMAIN_ACTION_SYMMETRIC_TAIL_INCLUSIVE != 0) {
        v_if(absolute_raw >= TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND) { result = TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE; }
        v_endif;
    } else {
        v_if(absolute_raw > TT_DOMAIN_ACTION_SYMMETRIC_TAIL_BOUND) { result = TT_DOMAIN_ACTION_SYMMETRIC_TAIL_VALUE; }
        v_endif;
    }
}
