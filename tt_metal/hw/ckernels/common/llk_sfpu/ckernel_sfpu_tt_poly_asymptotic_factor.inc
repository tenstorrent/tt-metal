// The CSV codegen declares whether the asymptotic correction is active on the
// left tail, right tail, or whole domain.  Keep that decision compile-time so a
// right-tail factor does not materialize the old 1e38 upper-bound comparison
// (and, more importantly, does not run on the non-tail segments).
#if (                                                                                    \
    defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || \
    defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X) ||           \
    defined(ASYMPTOTIC_FACTOR_QUADRATIC)) &&                                             \
    ((defined(ASYMPTOTIC_REGION_LEFT) + defined(ASYMPTOTIC_REGION_RIGHT) + defined(ASYMPTOTIC_REGION_ALL)) != 1)
#error "asymptotic factor requires exactly one LEFT/RIGHT/ALL region"
#endif
#if defined(ASYMPTOTIC_REGION_LEFT)
#define TT_ASYMPTOTIC_IF(x) v_if((x) < ASYMPTOTIC_BOUND)
#define TT_ASYMPTOTIC_END_IF v_endif
#elif defined(ASYMPTOTIC_REGION_RIGHT)
#define TT_ASYMPTOTIC_IF(x) v_if((x) >= ASYMPTOTIC_BOUND)
#define TT_ASYMPTOTIC_END_IF v_endif
#elif defined(ASYMPTOTIC_REGION_ALL)
#define TT_ASYMPTOTIC_IF(x) if constexpr (true)
#define TT_ASYMPTOTIC_END_IF
#endif

#if defined(ASYMPTOTIC_FACTOR_EXP_QUADRATIC) || defined(ASYMPTOTIC_FACTOR_EXP_LINEAR) || \
    defined(ASYMPTOTIC_FACTOR_X_EXP_LINEAR) || defined(ASYMPTOTIC_FACTOR_X) || defined(ASYMPTOTIC_FACTOR_QUADRATIC)
inline vFloat apply_asymptotic_scale(vFloat value) {
    value = value * ASYMPTOTIC_SCALE;
#ifdef ASYMPTOTIC_NEGATE_OUTPUT
    // Do not spell this as multiplication by -1 or unary minus: SFPI GCC
    // 15.1's rvtt_combine pass ICEs in try_combine_negated_result for the
    // right-tail logsigmoid chain.  XOR is exact IEEE sign negation and is an
    // established SFPI idiom (also used by the trig/cbrt reconstruction paths).
    return as<vFloat>(as<vInt>(value) ^ (vInt)0x80000000);
#else
    return value;
#endif
}
#endif
