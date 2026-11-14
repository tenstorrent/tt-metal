# Eltwise binary SFPU: Blackhole versus Quasar worked example

Canonical walkthrough. Other ops follow the same levers; do not copy these mathop lists blindly.

## Starting mismatches

BH perf families pair by name (`float`, `int`, `div`, `atan2`, `bcast`, …). QSR began as **one** lumped `test_perf_eltwise_binary_sfpu_quasar`, so compare reported unmatched.

On the paired float/int families, BH swept `formats` + `dest_acc` + `mathop` + `approx_mode`. QSR packed `formats_dest_acc` (`QuasarSfpuVariant`) and tuples `binary_op_mathop_approx` / `binary_op_mathop_clamp`. Compare keys by axis name, so `formats.input` never met `formats_dest_acc.formats.input`.

## Fixes (in order)

1. **Split QSR perf** into six wrappers that pair with functional: `int`, `float`, `bf16_rne`, `max_min_float`, `max_min_int32`, `quant`. Leave unmatched QSR-only families unmatched.

2. **Float unbundle** — keep the three `generate_quasar_sfpu_format_variants(SfpuElwadd, …)` representatives; explode to named axes with a dest_acc constraint. Functional `approx_mode` is Yes/No for atan2 and DIV, else No. Perf copies that sweep and pins DIV back to No, leaving atan2 as the only Yes/No pair. Resolve `resolve_quasar_sfpu_variant(SfpuElwadd, formats, dest_acc)` and `binary_op = mathop.cpp_enum_value` inside the test. Do not add `bcast_dim`. `tile_indices` stays functional-only.

3. **Int unbundle** — `formats` + `dest_acc=Yes` + `mathop` using `SfpuElwLt/Gt/Le/Ge`. `SfpuGtInt` / `SfpuLtInt` / `SfpuLeInt` / `SfpuGeInt` remain in `helpers/llk_params.py` as aliases; they are not sweep values. Clamp only MUL: `1000 if mathop == SfpuElwmulInt else None`. `SfpuElwmulInt` stays QSR-only on this pair; BH int MUL is `SfpuMulInt32` in `int_uniform`.

4. **Int `approx_mode`** — perf pins No, matching BH/WH. Do not cartesian Yes/No onto int kernels that ignore the mode. Copy the sweep dict to override a key; a keyword after `**INT_SWEEP` is fine only when that key is not already in the dict.

## After

**QSR test vs perf:** float coverage identical except functional-only `tile_indices`. Perf `approx_mode` is No except atan2. Functional DIV still sweeps Yes/No.

**BH vs QSR perf** (`--cross-arch blackhole quasar --kind perf`)

- Float: `dest_acc` identical. Float perf `approx_mode` is a Blackhole subset of Quasar (`[No]` versus `[No, Yes]`) because atan2 stays in the Quasar float family. `formats` QSR subset (no Bfp8; do not add Bfp8_b to DIV). `mathop` DIFFERENT (shared add/sub/mul; BH rsub/pow/xlogy; QSR div/atan2 — BH already has `div`/`atan2` as separate unmatched families). `bcast_dim` is `[B]`. `iterations` is `[i]`.
- Int: `formats`, `dest_acc`, `approx_mode` identical (`approx_mode` No). `mathop` DIFFERENT (shared add + ElwLt/Gt/Le/Ge; BH sub/shifts; QSR `SfpuElwmulInt`).

**BH vs QSR functional** is the same pairing with `--kind func`. Functional QSR DIV `approx_mode` Yes/No is then a real difference from BH, which pins DIV to No.

Further float mathop greening would be splitting QSR `div`/`atan2` into wrappers named like BH. That is optional packaging, not required for a correct unbundle.

## Kernel facts that blocked fake equality

- QSR `BinaryOp` has ADD/SUB/MUL/DIV/GT/LT/LE/GE/MAX/MIN/QUANT*/ATAN2 and `RSHFT`/`LSHFT`/`LOGICAL_RSHFT`. The Quasar int sweep does not include shifts.
- QSR int SUB is not ported (`sub_int_sfpu.h` is WH-only); SUB is float-only in `sfpu_operations_quasar.h`.
- Int ADD/compares/MUL pass `false` for approximation. Perf pins No. Functional QSR DIV Yes is a real kernel path (`_init_reciprocal_`), not packaging.
- The new two-operand kernel's `tile_cnt` counts both operand tiles, so per-tile MATH_ISOLATE is about half a result tile versus the old one-operand perf kernel, and the schema went from v3 to v5. Neither is a speedup; see `perf-test/references/perf-facts.md`, "Accounting".
- Datacopy and SFPU init are hoisted out of the per-block loop on purpose. Production still reconfigures per tile.
- The NONE stand-ins differ by architecture: per face with a zeroed SrcB on Wormhole and Blackhole, per tile on Quasar. See `perf-test/references/wormhole-blackhole.md` and `quasar.md`.
