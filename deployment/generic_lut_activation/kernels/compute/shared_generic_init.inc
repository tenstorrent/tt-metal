// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Shared initialization: include after init_sfpu, before the tile loop.
// Caller retains LUT context and the original initialization lifetime.

#if defined(USE_BF16) && defined(TT_TARGET_BUILTIN_PERIODIC_SINE)
    ckernel::sin_tile_init();
#elif defined(USE_BF16) && defined(TT_TARGET_BUILTIN_PERIODIC_COSINE)
    ckernel::cos_tile_init();
#elif defined(USE_BF16) && defined(TT_TARGET_BUILTIN_PERIODIC_TANGENT)
    ckernel::tan_tile_init();
#endif

#if defined(EVAL_METHOD_HALF_ANGLE_RATIO_1) && \
    defined(TT_HALF_ANGLE_ROOT_PRGM_PRELOAD) && defined(TRISC_MATH)
    static_assert(TT_HALF_ANGLE_ROOT_PRGM_REGISTER_COUNT == 3u);
    static_assert(TT_HALF_ANGLE_ROOT_PRGM_LOADS_REMOVED_PER_ELEMENT == 3u);
    static_assert(TT_HALF_ANGLE_ROOT_PRGM_MAGIC_BITS == HALF_ANGLE_SQRT_MAGIC);
    static_assert(
        TT_HALF_ANGLE_ROOT_PRGM_NEWTON_BIAS_BITS ==
        __builtin_bit_cast(uint32_t, HALF_ANGLE_SQRT_C1));
    sfpi::vConstIntPrgm0 = TT_HALF_ANGLE_ROOT_PRGM_MAGIC_BITS;
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(
        float, TT_HALF_ANGLE_ROOT_PRGM_NEGATIVE_HALF_BITS);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(
        float, TT_HALF_ANGLE_ROOT_PRGM_NEWTON_BIAS_BITS);
#endif

#if defined(TT_FACTORED_CW_REPLAY) && defined(TRISC_MATH)
    static_assert(POLY_DEGREE == sfpi::TT_FACTORED_CW_REPLAY_DEGREE &&
                      NUM_SEGMENTS == sfpi::TT_FACTORED_CW_REPLAY_SEGMENTS &&
                      LUT_SIZE == sfpi::TT_FACTORED_CW_REPLAY_LUT_SIZE,
                  "verified replay descriptor/LUT mismatch");
    static_assert(
        (sfpi::TT_FACTORED_CW_REPLAY_LOWER_BITS >> 16) ==
                sfpi::TT_FACTORED_CW_REPLAY_LOWER_BF16 &&
            (sfpi::TT_FACTORED_CW_REPLAY_UPPER_BITS >> 16) ==
                sfpi::TT_FACTORED_CW_REPLAY_UPPER_BF16 &&
            __builtin_bit_cast(
                uint32_t,
                LUT_DATA[sfpi::TT_FACTORED_CW_REPLAY_C0_INDEX]) ==
                sfpi::TT_FACTORED_CW_REPLAY_LEAF_C0_BITS &&
            __builtin_bit_cast(
                uint32_t,
                LUT_DATA[sfpi::TT_FACTORED_CW_REPLAY_C1_INDEX]) ==
                sfpi::TT_FACTORED_CW_REPLAY_LEAF_C1_BITS &&
            sfpi::TT_FACTORED_CW_REPLAY_LEAF_C1_BITS ==
                sfpi::TT_FACTORED_CW_REPLAY_EXPONENT_HALF_BITS,
        "verified replay constants do not match the selected resource");
    sfpi::vConstFloatPrgm0 =
        __builtin_bit_cast(float, sfpi::TT_FACTORED_CW_REPLAY_INVERSE_SCALE_BITS);
    sfpi::vConstFloatPrgm1 =
        __builtin_bit_cast(float, sfpi::TT_FACTORED_CW_REPLAY_RESIDUAL_SCALE_BITS);
    sfpi::vConstFloatPrgm2 =
        LUT_DATA[sfpi::TT_FACTORED_CW_REPLAY_C1_INDEX];
    ckernel::addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ckernel::ADDR_MOD_6);
#endif

#if defined(USE_BF16) && defined(CLAMPED_AFFINE_BF16_NATIVE_RELU)
    // The native tile API owns Wormhole's BF16 face/register traversal.
    ckernel::relu_tile_init();
#endif

#if defined(USE_BF16) && \
    defined(TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_EXACT_STORE) && \
    !defined(TT_TARGET_BH_BF16_THRESHOLD_IDENTITY_TTI)
    // The certificate binds the exact public threshold-identity primitive;
    // its wrapper owns destination start/face/done traversal.
    ckernel::hardshrink_tile_init();
#endif
#if defined(USE_BF16) && \
    defined(TT_TARGET_BH_BF16_THRESHOLD_SOFTSHIFT_NATIVE)
    // The descriptor binds the selected threshold-softshift algebra to the
    // exact public target primitive; its wrapper owns destination traversal.
    ckernel::softshrink_tile_init();
#endif

#if DIRECT_ALGEBRAIC_TTI_CANDIDATE && defined(TRISC_MATH)
#if defined(TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_PRGM0_PRELOAD) && \
    !defined(SLOPE_PRGM_PRELOAD_DISABLE)
    static_assert(
        TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_PRGM0_BITS ==
            __builtin_bit_cast(uint32_t, (float)(SLOPE_MAX_SLOPE_LO)),
        "typed slope program constant must preserve exact coefficient bits");
    sfpi::vConstFloatPrgm0 = SLOPE_MAX_SLOPE_LO;
#if TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BOUNDARY_COUNT == 1u
    static_assert(
        TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_PRGM1_BITS ==
            TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_TTI_BOUNDARY_RAW_U16,
        "typed slope boundary program constant must preserve physical word");
    sfpi::vConstIntPrgm1 =
        TT_TARGET_BH_BF16_UNIT_HIGH_SLOPE_PRGM1_BITS;
#endif
#endif
    if constexpr (sfpi::kDirectAlgebraicTtiReplay) {
        // Every direct-algebraic body is two-element interleaved.  Its final B
        // store advances the hardware destination counter by four rows; the
        // paired replay epilogue clears that counter after each tile.
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 4},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#endif

#if GATED_SINGLE_ROW_JOINT_TTI_CANDIDATE && defined(TRISC_MATH)
    if constexpr (sfpi::kGatedSingleRowJointTtiReplay) {
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#endif

#if defined(EVAL_METHOD_ZONE_FACTORED_AFFINE_EVEN_DECAY_2) && \
    !defined(TT_ZONE_PRGM_PRELOAD_DISABLE) && defined(TRISC_MATH)
    // Zone-factored constant-pool preload (mirrors the HW_PRELOAD pattern
    // below; structural — driven only by the declared eval kind).  Program the
    // three constants every row touches ONCE: the biased-exponent multiplier
    // and the two upper pre-scaled exponent-leaf coefficients.  The reader is
    // zone_factored_affine_even_decay_2_eval; the value expressions here are
    // exactly the constexpr expressions its escape arm materializes in-body,
    // so lane arithmetic is bit-identical with or without the preload.
    // Nothing else under this eval kind writes or reads Prgm0/1/2 (no
    // sfpu_reciprocal, no trig, no replay body — see the PRGM CONSTANT
    // OWNERSHIP RULE below).
    sfpi::vConstFloatPrgm0 = sfpi::kZoneNegativeHalfLog2E;
    sfpi::vConstFloatPrgm1 =
        sfpi::TT_ZONE_EXP_COEFFS[1] * sfpi::zone_exp_fraction_scale(1);
    sfpi::vConstFloatPrgm2 =
        sfpi::TT_ZONE_EXP_COEFFS[2] * sfpi::zone_exp_fraction_scale(2);
#endif

#if defined(HW_PRELOAD) && defined(TRISC_MATH)
    // GENERIC constant-pool preload: program the 3 hottest loop-invariant
    // constants for this exponent-ALU kind into the programmable const registers
    // ONCE (they persist across every replayed body and every tile). The
    // per-kind ranking matches the *_hw_eval_preloaded readers above.
#if defined(EXPONENT_ALU_EXP2)
#if !defined(EXP_HW_COMPOSE_SIGMOID) && !defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) && \
    !defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) && \
    !defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) && \
    !defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) && \
    !defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)
    // Under sigmoid compose, prgm0 is reserved for sfpu_reciprocal (set to 2.0 by
    // sfpu_reciprocal_init); MULT is hoisted into an LREG instead (see hw_reduce).
    sfpi::vConstFloatPrgm0 = EXP_HW_MULT;  // touched every element
#endif
#if (defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
     defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL)) && defined(ARCH_WORMHOLE)
    // WH compose: sfpu_reciprocal owns Prgm1/2 as well (its Sollya seed
    // constants are read at every sfpu_reciprocal_iter call) — program NOTHING
    // here. The top-2 Horner coeffs are foldable constexpr locals inside
    // exp_hw_eval_preloaded instead. Anything written here would be clobbered
    // by sfpu_reciprocal_init below AND would have corrupted the reciprocal if
    // it survived. See the PRGM CONSTANT OWNERSHIP RULE below.
#elif defined(EXP_HW_FUSED)
    // FUSED: prgm1/prgm2 carry the pre-scaled coeffs c[k]*2^-23k so the
    // per-element Horner runs over the UNNORMALIZED exman fraction (no
    // `* 0x1p-23f`). Scale is a compile-time constant per coeff index.
    sfpi::vConstFloatPrgm1 = EXP_HW_COEFFS[EXP_HW_DEGREE] * sfpi::exp_hw_fused_scale(EXP_HW_DEGREE);
    sfpi::vConstFloatPrgm2 =
        (EXP_HW_DEGREE >= 1) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 1] * sfpi::exp_hw_fused_scale(EXP_HW_DEGREE - 1) : 0.0f;
#else
    sfpi::vConstFloatPrgm1 = EXP_HW_COEFFS[EXP_HW_DEGREE];
    sfpi::vConstFloatPrgm2 = (EXP_HW_DEGREE >= 1) ? EXP_HW_COEFFS[EXP_HW_DEGREE - 1] : 0.0f;
#endif
#if EXP_HW_BOUNDED_TWO_SIDED_TTI_CANDIDATE
    if constexpr (sfpi::kBoundedTwoSidedTtiReplay) {
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#endif
#if EXP_HW_TTI_CANDIDATE
    if constexpr (sfpi::kExpHwTtiReplayBare) {
        // TTI replay fast path: SFPSTORE walks dst via ADDR_MOD_6 dest += 4
        // (TTNN _init_exponential_tti_bf16_ does exactly this; 2-element
        // interleaved body). Programmed once; nothing else in this kernel
        // touches ADDR_MOD_6.
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 4},
        }
            .set(ckernel::ADDR_MOD_6);
    } else if constexpr (sfpi::kExpHwTtiReplayPe) {
        // PE-corrected (non-bare) variant is a SINGLE-element body: the
        // ADDR_MOD_6 dst walk is dest += 2 per body (mutually exclusive with
        // the bare gate above by construction).
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#endif
#if EXP_HW_COMPOSE_TTI_CANDIDATE
    if constexpr (sfpi::kSigmoidTtiReplay) {
        // Sigmoid/silu compose TTI replay: SINGLE element per body, so the
        // ADDR_MOD_6 dst walk is dest += 2 (the plain-exp gate above uses +4
        // for its 2-element interleave; the two gates are mutually exclusive
        // by construction — compose is excluded from kExpHwTtiReplay).
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
        if constexpr (sfpi::kSymmetricSigmoidProductTtiReplayD2Bare) {
            // Reuse the common raw-ingress RNE/store LOADMACRO carrier.  The
            // macro's load into dead L2 is overwritten by the next replay's
            // absolute-value head; RNE reads literal L0 into staging and the
            // fixed store advances ADDR_MOD_6 at issue+3.
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                0,
                ckernel::p_sfpu::LREG0,
                13,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
            constexpr uint32_t kSymmetricProductLmSequence = 0x13850000u;
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_UPPER,
                (kSymmetricProductLmSequence >> 16));
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_LOWER,
                (kSymmetricProductLmSequence & 0xFFFFu));
            TTI_SFPCONFIG(0, 4, 0);
            TTI_SFPCONFIG(0x110, 8, 1);
            TTI_SFPNOP;
        }
    }
#endif
#if ABS_RESIDUAL_TTI_CANDIDATE
    if constexpr (sfpi::kAbsResidualTtiReplay) {
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#endif
#if EXP_HW_HYPERBOLIC_TTI_CANDIDATE
    if constexpr (sfpi::kHyperbolicEvenTtiReplay) {
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#endif
#elif defined(EXPONENT_ALU_LOG2)
    sfpi::vConstFloatPrgm0 = LOG_HW_SCALE;
    sfpi::vConstFloatPrgm1 = LOG_HW_COEFFS[LOG_HW_DEGREE];
    sfpi::vConstFloatPrgm2 = (LOG_HW_DEGREE >= 1) ? LOG_HW_COEFFS[LOG_HW_DEGREE - 1] : 0.0f;
#if LOG_HW_TTI_CANDIDATE
    if constexpr (sfpi::kLogHwTtiReplay) {
        // Single-element replay body: SFPSTORE walks dst via ADDR_MOD_6
        // dest += 2 (one vector per body). Programmed once; nothing else in
        // this kernel touches ADDR_MOD_6. (Under the LM tail fold the
        // SFPLOADMACRO itself carries ADDR_MOD_6 — same +2/element walk.)
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#if LOG_LM_CANDIDATE
    if constexpr (sfpi::kLogHwLmReplay) {
        // SFPLOADMACRO one-time programming (after the prgm-constant loads
        // above; backdoor captures do not execute and do not write L12-L14).
        //
        // T_MUL -> programmable instruction 0 (backdoor capture: dest field
        // 12). Fired in the MAD slot with SRCB_OVERRIDE=1: computes
        // L12(=LOG_HW_SCALE) * VD + 0.0 -> VD. Mirrors the plain body's
        // SFPMUL(L3, L12, LCONST_0, L3) with commuted (bit-identical) a*b.
        TTI_SFPMUL(
            ckernel::p_sfpu::LREG12, ckernel::p_sfpu::LREG0, ckernel::p_sfpu::LCONST_0, 12, 0);
        // T_RND -> programmable instruction 1 (dest field 13). Fired in the
        // Round slot with default srcC-override: fp32 -> bf16 RNE of VD,
        // result -> staging flop (ROUND_USE_STAGING avoids the shared
        // Simple/Round LREG write port while the next iteration's SETEXP
        // writes through the Simple port on the same cycle).
        TTI_SFP_STOCH_RND(
            sfpi::SFPSTOCHRND_RND_EVEN,
            0,
            ckernel::p_sfpu::LREG0,
            ckernel::p_sfpu::LREG0,
            13,
            sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
        // Macro sequence register 0 (front door through LREG0 — the word has
        // a non-zero upper half, so the SFPCONFIG imm16 form cannot carry it).
        // Byte layout [store][round][mad][simple]:
        //   scale != 1: store 0x73 (fixed STORE, delay 6, src=staging),
        //               round 0x65 (T_RND, delay 4, dst=staging),
        //               mad   0x94 (T_MUL, delay 2, srcB-override), simple 0.
        //   scale == 1: store 0x63 (delay 4, src=staging),
        //               round 0x55 (T_RND, delay 2, dst=staging), mad 0.
        constexpr uint32_t kLogLmSeq =
            ((float)(LOG_HW_SCALE) != 1.0f) ? 0x73659400u : 0x63550000u;
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_UPPER, (kLogLmSeq >> 16));
        TTI_SFPLOADI(ckernel::p_sfpu::LREG0, sfpi::SFPLOADI_MOD0_LOWER, (kLogLmSeq & 0xFFFFu));
        TTI_SFPCONFIG(0, 4, 0);  // Macro Sequence Register 0
        // LoadMacroConfig.Misc: DEFAULT_STORE_INSMOD=0 (macro store InstrMod
        // 0, identical to the plain body's SFPSTORE mod0), no
        // STORE_INHERITS_INSMOD, UNIT_DEPENDENCY_ENABLE=0xF — all four units
        // in WaitForElapsedInstructions mode so macro delays count issued
        // SFPU instructions, never wall-clock (bubble-immune; the in-tree
        // exp_21f LOADMACRO init uses the same 0xF00).
        TTI_SFPCONFIG(0xF00, 8, 1);
        TTI_SFPNOP;
    }
#endif
#endif
#elif defined(EXPONENT_ALU_LOG1P)
    // Juffa log1p: prgm0 = the k*ln2 recombine multiplier (touched every
    // element in the final SFPMAD), prgm1/prgm2 = top-2 Horner coeffs.
    sfpi::vConstFloatPrgm0 = (float)(LOG1P_LN2 * 0x1p-23f);
    sfpi::vConstFloatPrgm1 = LOG1P_HW_COEFFS[LOG1P_HW_DEGREE];
    sfpi::vConstFloatPrgm2 = (LOG1P_HW_DEGREE >= 1) ? LOG1P_HW_COEFFS[LOG1P_HW_DEGREE - 1] : 0.0f;
#if LOG1P_TTI_CANDIDATE
    if constexpr (sfpi::kLog1pTtiReplay) {
        // TTI replay fast path: dst walk via ADDR_MOD_6 dest += 2 (single
        // element per body; see the exp2 branch above).
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#endif
#elif defined(EXPONENT_ALU_POW)
    // Only plain sqrt (ROOT_N==2, no reciprocal) uses the preload fast path;
    // cbrt/rsqrt fall back to the general evaluator (see pow_hw_eval_preloaded),
    // which reads no prgm registers, so we must not clobber them there.
#if (POW_HW_ROOT_N == 2) && !defined(POW_HW_RECIPROCAL)
    sfpi::vConstFloatPrgm0 = POW_HW_SCALE_C1;  // sqrt(2)
    sfpi::vConstFloatPrgm1 = POW_HW_COEFFS[POW_HW_DEGREE];
    sfpi::vConstFloatPrgm2 = (POW_HW_DEGREE >= 1) ? POW_HW_COEFFS[POW_HW_DEGREE - 1] : 0.0f;
#endif
#endif
#endif

#if (defined(EVAL_METHOD_NEWTON_ROOT) || \
     defined(EVAL_METHOD_ZONE_FACTORED_EXP_ROOT_CORRECTION_2)) && \
    defined(TRISC_MATH)
    // Newton-Raphson magic-seed root: preload the seed magic + Newton coeffs into
    // the programmable const registers ONCE so the recorded per-element body
    // reloads none of them (mirrors native sqrt_init).
    //   sqrt : prgm0=magic, prgm1=C1, prgm2=C2 (SQRT_23-bit double-Newton).
    //   rsqrt: prgm0=inverse-sqrt magic, prgm1=C1 (=1.5 step constant).
    //   cbrt : native-style magic-root uses prgm0/1/2 as correction coeffs.
#if (NEWTON_ROOT_N == 3) && defined(NEWTON_ROOT_ALGORITHM_CBRT_MAGIC)
#if defined(NEWTON_ROOT_BACKEND_NATIVE_SFPU_TILE)
#if defined(NATIVE_ROOT_GRAPH_INIT_DISABLE)
    // Validation escape: retain the stock callback initializer so CVER can
    // prove the graph-owned decomposition against the same arithmetic body.
    ckernel::cbrt_tile_init();
#else
    // The lowered graph carries the exact constants consumed by tt-metal's
    // native cbrt body.  Preserve the public bare cbrt common init (SFPU
    // config, address modifiers, and one SETRWC), then materialize that graph
    // state directly.  cbrt_tile_init's callback would issue a second,
    // immediately redundant SETRWC before loading these same three words.
    SFPU_UNARY_INIT(cbrt);
    sfpi::vConstFloatPrgm0 = NEWTON_ROOT_C0;
    sfpi::vConstFloatPrgm1 = NEWTON_ROOT_C1;
    sfpi::vConstFloatPrgm2 = NEWTON_ROOT_C2;
#endif
#else
    sfpi::vConstFloatPrgm0 = NEWTON_ROOT_C0;
    sfpi::vConstFloatPrgm1 = NEWTON_ROOT_C1;
    sfpi::vConstFloatPrgm2 = NEWTON_ROOT_C2;
#endif

#elif defined(NEWTON_ROOT_BACKEND_NATIVE_SFPU_TILE) && \
    (NEWTON_ROOT_N == 2) && defined(NEWTON_ROOT_RECIPROCAL)
    // The native reciprocal-root initializer owns the fixed-root constants
    // and counter state for this metadata-declared backend.
    ckernel::rsqrt_tile_init();
#elif (NEWTON_ROOT_N != 3)
    sfpi::vConstIntPrgm0 = NEWTON_ROOT_MAGIC;
    sfpi::vConstFloatPrgm1 = NEWTON_ROOT_C1;  // sqrt: C1; rsqrt: 1.5 step constant
    sfpi::vConstFloatPrgm2 = NEWTON_ROOT_C2;  // sqrt: C2; rsqrt: unused
#endif
#if NEWTON_TTI_CANDIDATE || NEWTON_RSQRT_TTI_CANDIDATE || NEWTON_CBRT_EXACT_DESCRIPTOR
    if constexpr (
        sfpi::kNewtonSqrtTtiReplay || sfpi::kNewtonRsqrtTtiReplay ||
        NEWTON_CBRT_EXACT_DESCRIPTOR) {
        // TTI replay fast path: SFPSTORE walks dst via ADDR_MOD_6 dest += 2
        // (single element per body). Programmed once; nothing else in this
        // kernel touches ADDR_MOD_6.
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#endif
#endif

#if defined(EVAL_METHOD_ROOT_LOG1P_HANDOFF_1) && defined(TRISC_MATH)
    // Both records are single-row bodies. As in the ordinary SFPU loops, their
    // final stores advance D-RWC by two physical rows; fp32_dest_acc_en changes
    // capacity, not this architectural traversal. The replay epilogue resets
    // D-RWC between phases and after the final row. Tile 0 therefore occupies
    // physical rows 0..63, and the FP32 handoff uses certified tile-1 offset
    // 64 (rows 64..127) without aliasing unconsumed input rows.
    ckernel::addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = sfpi::TT_ROOT_LOG1P_DST_ROW_INCREMENT},
    }
        .set(ckernel::ADDR_MOD_6);
#endif

#if defined(EVAL_METHOD_TRIG_RESIDUAL) && defined(TRISC_MATH)
    // Four-part Cody-Waite tail constants and 1/PI multiplier for the standalone
    // trig residual evaluator.
#if defined(TRIG_RESIDUAL_PHASE_SINE_PI_ODD)
    sfpi::vConstFloatPrgm0 = -0x1.51p-21f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-33f;
#else
    sfpi::vConstFloatPrgm0 = -0x1.51p-22f;
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-34f;
#endif
    sfpi::vConstFloatPrgm2 = 0.31830987334251404f;
#if TRIG_TTI_CANDIDATE
    if constexpr (sfpi::kTrigTtiReplay) {
        // TTI replay fast path: SFPSTORE walks dst via ADDR_MOD_6 dest += 2
        // (single element per body). Programmed once; nothing else in this
        // kernel touches ADDR_MOD_6 (the sfpi fallback loop is constexpr'd
        // out under this gate).
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
    }
#endif
#endif

#if defined(EVAL_METHOD_TAN_STANDALONE) && defined(TRISC_MATH)
#if !defined(ARCH_WORMHOLE)
    // BH: Prgm0 is reciprocal-owned (sfpu_reciprocal_init writes 2.0 below and
    // the quadrant reciprocal reads it at every call) — the 3rd Cody-Waite
    // tail term that previously lived there is an LREG pin at the dst-loop
    // site instead. Prgm1/2 are free on BH: 4th tail term + 2/pi stay here.
    sfpi::vConstFloatPrgm1 = -0x1.0b4612p-34f;
    sfpi::vConstFloatPrgm2 = 0.63661974668502808f;
#endif
    // WH: sfpu_reciprocal owns ALL of Prgm0/1/2 — program nothing here; the
    // eval reads arch-gated locals instead (see tan_standalone_eval).
#endif

    // =========================================================================
    // PRGM CONSTANT OWNERSHIP RULE — READ THIS BEFORE ADDING ANY
    // vConstFloatPrgm WRITE (above) OR READ (in an eval body).
    //
    // sfpu_reciprocal_iter reads the programmable constants at EVERY call:
    //   * WORMHOLE  : Prgm0/1/2 = its Sollya quadratic-seed constants
    //                 (0.3232..., 1.4545..., 2.1212...). The reciprocal owns
    //                 ALL THREE prgm registers, period.
    //   * BLACKHOLE : Prgm0 = 2.0 (Newton step). The reciprocal owns Prgm0
    //                 only; Prgm1/2 stay free for op constants.
    // sfpu_reciprocal_init programs exactly those registers, so on any config
    // that links a reciprocal it MUST run AFTER every op-specific prgm write
    // above, and NOTHING may write a reciprocal-owned register afterwards.
    // Reordering can never reconcile a conflict: the op eval and the
    // reciprocal both read their registers at eval time, inside the same
    // per-element body — a shared register means one of them computes garbage
    // (shipped example: tan_standalone's Cody-Waite tail term in Prgm0 read
    // 2.0 on BH; exp-compose Horner top coeffs in Prgm1/2 read Sollya seeds
    // on WH). The only fix is DEMOTION to an LREG pin or a foldable local.
    //
    // Ownership as enforced in this file:
    //   * EXP_HW_COMPOSE_SIGMOID(_PRODUCT): MULT demoted from Prgm0 to a
    //     hoisted local (all arches); on WH the top-2 Horner coeffs are also
    //     demoted from Prgm1/2 to foldable locals (exp_hw_eval_preloaded).
    //   * EVAL_METHOD_TAN_STANDALONE: 3rd Cody-Waite tail term demoted from
    //     Prgm0 to an LREG pin (all arches); on WH the 4th term + 2/pi are
    //     demoted from Prgm1/2 to locals too.
    //   * POW_HW_RECIPROCAL: programs no prgm registers; the init below is
    //     REQUIRED (previously missing — the reciprocal ran on whatever
    //     state dispatch left in the prgm regs).
    //   * TT_SELECTED_CORE_NEEDS_RECIPROCAL: certified selected schedule emits
    //     sfpu_reciprocal_iter<2>; it owns Prgm0 on BH and therefore shares
    //     this same last-writer init, never a legacy activation macro.
    //   * RANGE_REDUCTION_TAN: reads no prgm registers — init only.
    // Configs with no reciprocal (sin/cos TRIG_RESIDUAL, plain exp/log/pow,
    // NEWTON_ROOT, poly cascades) may use all three prgm registers freely.
    // =========================================================================
#if (defined(RANGE_REDUCTION_TAN) || defined(EVAL_METHOD_TAN_STANDALONE) ||        \
     defined(EXP_HW_COMPOSE_SIGMOID) || defined(EXP_HW_COMPOSE_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_SYMMETRIC_SIGMOID_PRODUCT) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_EVEN) || \
     defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) || \
     defined(EXP_HW_COMPOSE_BOUNDED_TWO_SIDED_RATIONAL) || \
     defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_EVEN) || \
     defined(REDUCE_EXP_COMPOSE_HYPERBOLIC_ODD) || \
     defined(REDUCE_EXP_COMPOSE_SIGMOID) || \
     defined(REDUCE_EXP_COMPOSE_SIGMOID_PRODUCT) || \
     defined(POW_HW_RECIPROCAL) || \
     defined(TT_ABS_RESIDUAL_AFFINE_1_NEEDS_RECIPROCAL) || \
     defined(TT_SQUARE_DECAY_NEEDS_RECIPROCAL) || \
     defined(EVAL_METHOD_ROOT_NATIVE_LOG_RECIPROCAL_1) || \
     defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1) || \
     defined(EVAL_METHOD_EXPONENT_BUCKET_LOG_DERIVATIVE_1) || \
     (defined(TT_INLINE_PROGRAM_NEEDS_RECIPROCAL) || \
      defined(TT_SELECTED_CORE_TOTAL_NEEDS_RECIPROCAL) || \
      defined(TT_SELECTED_CORE_NEEDS_RECIPROCAL))) && \
    defined(TRISC_MATH)
    ckernel::sfpu::sfpu_reciprocal_init<false>();
#endif

#if defined(EVAL_METHOD_SHARED_INVERSE_SQUARE_RECURRENCE_P0_P2_1) && \
    defined(ARCH_BLACKHOLE) && defined(USE_BF16) && defined(TRISC_MATH)
    // Prgm0 was just programmed to 2.0 by reciprocal init.  The anonymous
    // P2 regularizer owns the two remaining programmable constants.
    sfpi::vConstFloatPrgm1 = SHARED_INVERSE_SQUARE_P2[1];
    sfpi::vConstFloatPrgm2 = SHARED_INVERSE_SQUARE_P2[0];
    ckernel::addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ckernel::ADDR_MOD_6);
#endif

#if defined(TT_TARGET_SELECTED_TTI_CLASS_REPAIR) && defined(TRISC_MATH)
    ckernel::addr_mod_t{
        .srca = {.incr = 0},
        .srcb = {.incr = 0},
        .dest = {.incr = 2},
    }
        .set(ckernel::ADDR_MOD_6);
#endif

#if EXP_HW_HYPERBOLIC_TTI_CANDIDATE && \
    defined(EXP_HW_COMPOSE_HYPERBOLIC_ODD_FACTOR) && defined(TRISC_MATH)
    if constexpr (sfpi::kHyperbolicOddFactorTtiReplay) {
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 2},
        }
            .set(ckernel::ADDR_MOD_6);
        sfpi::exp_hw_hyperbolic_odd_tti_lm_init();
    }
#endif

#if defined(TT_SELECTED_TTI_LOG_SQUARE_FACTORIZED_ODD) && defined(POLY_TTI_DISABLE) && defined(TRISC_MATH)
    // The non-replay baseline consumes the same immutable log companion.
    sfpi::vConstFloatPrgm0 = 0.69314718055994530942f * 0x1p-23f;
    sfpi::vConstFloatPrgm1 = __builtin_bit_cast(float, sfpi::TT_SELECTED_LOG_SQUARE_LOG_COEFF_BITS[2]);
    sfpi::vConstFloatPrgm2 = __builtin_bit_cast(float, sfpi::TT_SELECTED_LOG_SQUARE_LOG_COEFF_BITS[1]);
#endif
#if POLY_TTI_CANDIDATE && defined(TRISC_MATH)
    if constexpr (sfpi::kPolyTtiReplay) {
        // Poly-cascade replay preamble: coefficients into only the
        // structurally allocated programmable const registers once (the poly
        // path otherwise never touches them), and the ADDR_MOD_6 dst walk for
        // the body's final
        // SFPSTORE: +4 rows (2 vectors) for the 2-element plain/parity
        // interleaves, +2 rows for the single-element signed-abs body.
        // kPolyTtiReplay proves this write set is disjoint from every row
        // owned by the typed overlay's reciprocal primitive. In particular,
        // no write below can clobber sfpu_reciprocal_init's last-writer state.
#if POLY_TTI_SHAPE_LOG_SQUARE
        sfpi::vConstFloatPrgm0 = 0.69314718055994530942f * 0x1p-23f;
        sfpi::vConstFloatPrgm1 = __builtin_bit_cast(
            float, sfpi::TT_SELECTED_LOG_SQUARE_LOG_COEFF_BITS[2]);
        sfpi::vConstFloatPrgm2 = __builtin_bit_cast(
            float, sfpi::TT_SELECTED_LOG_SQUARE_LOG_COEFF_BITS[1]);
#elif POLY_TTI_SHAPE_PARITY_EVEN
        // Parity-even stride-2 top coefficients: c[TOP], c[TOP-2], c[TOP-4].
        sfpi::vConstFloatPrgm1 = sfpi::poly_tti_c(sfpi::kPolyTtiParityTop);
        sfpi::vConstFloatPrgm2 = sfpi::poly_tti_c(sfpi::kPolyTtiParityTop - 2u);
        sfpi::vConstFloatPrgm0 = sfpi::poly_tti_c(sfpi::kPolyTtiParityTop - 4u);
#else
        sfpi::vConstFloatPrgm1 = sfpi::poly_tti_c(POLY_DEGREE);
        sfpi::vConstFloatPrgm2 = sfpi::poly_tti_c(POLY_DEGREE - 1);
        if constexpr (sfpi::kPolyTtiPrgmWriteMask & sfpi::kPolyTtiPrgm0Mask) {
            sfpi::vConstFloatPrgm0 = sfpi::poly_tti_c(POLY_DEGREE - 2);
        }
#endif
        ckernel::addr_mod_t{
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = (sfpi::kPolyTtiPlain2x || sfpi::kPolyTtiParityEven2x) ? 4 : 2},
        }
            .set(ckernel::ADDR_MOD_6);
#if defined(TT_SELECTED_TTI_SIGNED_ABS_RECIPROCAL) || \
    defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)
        if constexpr (
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)
            sfpi::kPolyTtiSignedAbsAffineReplay
#else
            sfpi::kPolyTtiSignedAbsRecipReplay
#endif
        ) {
            // Capture the unchanged signed-abs reconstruction tail into
            // programmable templates.  LOADMACRO supplies the current raw
            // input as VD=L3: template 7 restores that sign onto L2, template 5
            // performs the ordinary fp32->bf16 RNE into the staging flop,
            // and the fixed store drains staging through ADDR_MOD_6.
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
            // The 32-slot raw-ingress body restores and repairs sign itself.
            // Template 5 rounds literal L0 into LOADMACRO's VD=L2; fixed
            // store then drains VD at issue+3 (the proven rational-tail form).
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                0,
                ckernel::p_sfpu::LREG0,
                13,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);
#else
            TTI_SFPSETSGN(
                0,
                ckernel::p_sfpu::LREG2,
                15,
                0);  // backdoor capture -> simple template 7
            TTI_SFP_STOCH_RND(
                sfpi::SFPSTOCHRND_RND_EVEN,
                0,
                ckernel::p_sfpu::LREG0,
                ckernel::p_sfpu::LREG0,
                13,
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16B);  // capture -> round template 5
#endif
            constexpr uint32_t kSignedAbsRecipLmSeq =
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_OVERLAY)
                TT_TYPED_TTI_ABS_AFFINE_LM_SEQUENCE_BITS;
#else
                TT_SELECTED_TTI_RECIP_LM_SEQUENCE_BITS;
#endif
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_UPPER,
                (kSignedAbsRecipLmSeq >> 16));
            TTI_SFPLOADI(
                ckernel::p_sfpu::LREG0,
                sfpi::SFPLOADI_MOD0_LOWER,
                (kSignedAbsRecipLmSeq & 0xFFFFu));
            TTI_SFPCONFIG(0, 4, 0);  // Macro Sequence Register 0
            // All macro units count issued SFPU instructions, matching the
            // in-tree exp/log LOADMACRO convention and making stalls inert.
#if defined(TT_TYPED_TTI_SIGNED_ABS_AFFINE_RAW_INGRESS_CLOSURE)
            TTI_SFPCONFIG(0x110, 8, 1);
#else
            TTI_SFPCONFIG(0xF00, 8, 1);
#endif
            TTI_SFPNOP;
        }
#endif
    }
#endif
