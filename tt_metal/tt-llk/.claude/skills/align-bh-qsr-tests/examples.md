# Eltwise binary SFPU — BH vs QSR alignment

Canonical walkthrough. Other ops follow the same levers; do not copy these mathop lists blindly.

## Starting mismatches

BH perf families pair by name (`float`, `int`, `div`, `atan2`, `bcast`, …). QSR began as **one** lumped `test_perf_eltwise_binary_sfpu_quasar`, so compare reported unmatched.

On the paired float/int families, BH swept `formats` + `dest_acc` + `mathop` + `approx_mode`. QSR packed `formats_dest_acc` (`QuasarSfpuVariant`) and tuples `binary_op_mathop_approx` / `binary_op_mathop_clamp`. Compare keys by axis name, so `formats.input` never met `formats_dest_acc.formats.input`.

## Fixes (in order)

1. **Split QSR perf** into six wrappers that pair with functional: `int`, `float`, `bf16_rne`, `max_min_float`, `max_min_int32`, `quant`. Leave unmatched QSR-only families unmatched.

2. **Float unbundle** — keep the three `generate_quasar_sfpu_format_variants(SfpuElwadd, …)` representatives; explode to named axes with a dest_acc constraint. Functional `approx_mode` is Yes/No for atan2 and DIV, else No. Perf copies that sweep and pins DIV back to No, leaving atan2 as the only Yes/No pair. Resolve `resolve_quasar_sfpu_variant(SfpuElwadd, formats, dest_acc)` and `binary_op = mathop.cpp_enum_value` inside the test. Keep QSR-only `implied_math_format`; the comparer ignores it. Do not add `bcast_dim`. `tile_indices` stays functional-only.

3. **Int unbundle** — `formats` + `dest_acc=Yes` + `mathop` using `SfpuElwLt/Gt/Le/Ge`. The `Sfpu*Int` comparison members are deleted. Clamp only MUL: `1000 if mathop == SfpuElwmulInt else None`. `SfpuElwmulInt` stays QSR-only on this pair; BH int MUL is `SfpuMulInt32` in `int_uniform`.

4. **Int `approx_mode`** — perf pins No, matching BH/WH. Do not cartesian Yes/No onto int kernels that ignore the mode. Copy the sweep dict to override a key; a keyword after `**INT_SWEEP` is fine only when that key is not already in the dict.

## After

**QSR test vs perf:** float coverage identical except functional-only `tile_indices` and ignored `implied_math_format` (functional still sweeps No and Yes; perf pins Yes). Perf `approx_mode` is No except atan2. Functional DIV still sweeps Yes/No.

**BH vs QSR perf** (`--cross-arch blackhole quasar --kind perf`)

- Float: `dest_acc` identical. `approx_mode` is No except atan2 on both. `formats` QSR subset (no Bfp8; do not add Bfp8_b to DIV). `mathop` DIFFERENT (shared add/sub/mul; BH rsub/pow/xlogy; QSR div/atan2 — BH already has `div`/`atan2` as separate unmatched families). `bcast_dim` is `[B]`. `implied_math_format` is `[i]`.
- Int: `formats`, `dest_acc`, `approx_mode` identical (`approx_mode` No). `mathop` DIFFERENT (shared add + ElwLt/Gt/Le/Ge; BH sub/shifts; QSR `SfpuElwmulInt`).

**BH vs QSR functional** is the same pairing with `--kind func`. Functional QSR DIV `approx_mode` Yes/No is then a real difference from BH, which pins DIV to No.

Further float mathop greening would be splitting QSR `div`/`atan2` into wrappers named like BH. That is optional packaging, not required for a correct unbundle.

## Kernel facts that blocked fake equality

- QSR `BinaryOp` has ADD/SUB/MUL/DIV/GT/LT/LE/GE/MAX/MIN/QUANT*/ATAN2. No RSHFT/LSHFT/LOGICAL_RSHFT.
- QSR int SUB is not ported (`sub_int_sfpu.h` is WH-only); SUB is float-only in `sfpu_operations_quasar.h`.
- Int ADD/compares/MUL pass `false` for approximation. Perf pins No. Functional QSR DIV Yes is a real kernel path (`_init_reciprocal_`), not packaging.
- `tile_cnt` on the two-operand kernel counts both operand tiles. Per-tile MATH_ISOLATE is about half a result tile versus the old one-operand perf kernel. Do not halve `TILE_COUNT` to hide that. Schema v3 and v5 CSVs are not comparable.
- NONE dvalid mocks post SrcA plus a SrcB zerosrc every face, including `dest_acc=No` (`#1230`).
