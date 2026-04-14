# Quasar Kernel Generation Prompts

Run any prompt below from the `codegen/` directory:

```bash
cd codegen
claude -p "Generate abs for Quasar" --dangerously-skip-permissions --effort max --verbose --model opus
```

**Batch execution**: Use the batch script for automated runs:
```bash
./scripts/batch_generate.sh --wave 1               # run Wave 1 sequentially
./scripts/batch_generate.sh --wave 1 --parallel     # run Wave 1 in parallel
./scripts/batch_generate.sh --wave 1 -j 4           # max 4 concurrent
./scripts/batch_generate.sh --kernel abs             # single kernel
./scripts/batch_generate.sh --from 5                 # resume from #5
./scripts/batch_generate.sh --wave 1 --dry-run       # preview
```

**After generating**: Add `#include` lines to `tt_llk_quasar/common/inc/ckernel_sfpu.h` for each new kernel.

---

## Wave 1: Testable Simple SFPU (4)

These have golden generators. Functional test possible after wiring into Quasar test.
**All 4 can run in parallel.**

```
Generate abs for Quasar
```

```
Generate negative for Quasar
```

```
Generate fill for Quasar
```

```
Generate threshold for Quasar
```

---

## Wave 2: Testable Medium SFPU (5)

Also have golden generators. Functional test possible.
**All 5 can run in parallel.**

```
Generate elu for Quasar
```

```
Generate exp2 for Quasar
```

```
Generate log for Quasar
```

```
Generate trigonometry for Quasar
```

```
Generate activations for Quasar
```

---

## Wave 3: Remaining Simple SFPU (6)

Compile-only validation — no golden generators exist for these.
**All 6 can run in parallel.**

```
Generate sign for Quasar
```

```
Generate hardtanh for Quasar
```

```
Generate clamp for Quasar
```

```
Generate dropout for Quasar
```

```
Generate is_fp16_zero for Quasar
```

```
Generate isinf_isnan for Quasar
```

---

## Wave 4: Remaining Medium SFPU (9)

Compile-only validation.
**All 9 can run in parallel.**

```
Generate cdf for Quasar
```

```
Generate tanh_derivative for Quasar
```

```
Generate rsqrt_compat for Quasar
```

```
Generate rounding_ops for Quasar
```

```
Generate polyval for Quasar
```

```
Generate load_config for Quasar
```

```
Generate cast_fp32_to_fp16a for Quasar
```

```
Generate converter for Quasar
```

```
Generate typecast for Quasar
```

---

## Wave 5: Complex SFPU with Test Potential (4)

These have cross-arch tests or dedicated test files.
**All 4 can run in parallel.**

```
Generate comp for Quasar
```

```
Generate topk for Quasar
```

```
Generate quant for Quasar
```

```
Generate binary for Quasar
```

---

## Wave 6: Remaining Complex SFPU (8)

Compile-only validation.
**All 8 can run in parallel.**

```
Generate binary_bitwise for Quasar
```

```
Generate add_int for Quasar
```

```
Generate sub_int for Quasar
```

```
Generate mul_int for Quasar
```

```
Generate shift for Quasar
```

```
Generate where for Quasar
```

```
Generate cumsum for Quasar
```

```
Generate ema for Quasar
```

---

## Wave 7: Specialized SFPU (6)

Compile-only validation. Unusual data flow or multi-tile patterns.
**All 6 can run in parallel.**

```
Generate welfords for Quasar
```

```
Generate reduce for Quasar
```

```
Generate reduce_custom for Quasar
```

```
Generate max_pool_indices for Quasar
```

```
Generate add_top_row for Quasar
```

```
Generate reshuffle_rows for Quasar
```

---

## Wave 8: LLK Submodule — Core (10)

Generate AFTER all SFPU kernels (Waves 1-7) — the math wrappers `#include` them.

**Dependency constraints within Wave 8:**
- `#45` depends on `#44` (binary params depends on binary)
- `#47` depends on `#46` (ternary params depends on ternary)
- `#49` depends on `#48` (welfords params depends on welfords)
- `#43`, `#50`, `#51`, `#52` are independent

**Safe parallel groups within Wave 8:**
- Group A (parallel): `#43`, `#44`, `#46`, `#48`, `#50`, `#51`, `#52`
- Group B (after A): `#45`, `#47`, `#49`

```
Generate math_eltwise_unary_sfpu_params for Quasar
```

```
Generate math_eltwise_binary_sfpu for Quasar
```

```
Generate math_eltwise_binary_sfpu_params for Quasar
```

```
Generate math_eltwise_ternary_sfpu for Quasar
```

```
Generate math_eltwise_ternary_sfpu_params for Quasar
```

```
Generate math_welfords_sfpu for Quasar
```

```
Generate math_welfords_sfpu_params for Quasar
```

```
Generate math_transpose_dest for Quasar
```

```
Generate pack_rows for Quasar
```

```
Generate unpack_untilize for Quasar
```

---

## Wave 9: LLK Submodule — Experimental (13)

Low priority. These are in `llk_lib/experimental/` on Blackhole and may not be needed for initial Quasar support.

```
Generate math_eltwise_binary_custom for Quasar
```

```
Generate math_eltwise_unary_datacopy_custom for Quasar
```

```
Generate math_matmul_custom_no_mop for Quasar
```

```
Generate math_mul_reduce_scalar for Quasar
```

```
Generate math_reduce_custom for Quasar
```

```
Generate math_reduce_runtime_custom for Quasar
```

```
Generate pack_custom for Quasar
```

```
Generate unpack_A_custom for Quasar
```

```
Generate unpack_AB_matmul_custom for Quasar
```

```
Generate unpack_AB_reduce_custom for Quasar
```

```
Generate unpack_AB_reduce_custom_runtime for Quasar
```

```
Generate unpack_AB_sub_bcast_col_custom for Quasar
```

```
Generate unpack_mul_reduce_scalar for Quasar
```
