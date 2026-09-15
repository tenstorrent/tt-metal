# Final-normalization-only BF16 reciprocal control

`recip8_fullchip.py` retains Q256/K512/D128, two K/V slots, LoFi QK/PV,
BF16 destinations/recurrent state, native approximate exp, and the existing
MAIN/full-FAST/denominator-only layouts. K8/V8 and K8/V4 are supported;
Q is RNE7/BF16, K8/V8 use RNE5/native BFP8 packing, V4 uses group RNE.
The default is the unmodified reciprocal. `--recip8` changes only final
normalization's reciprocal init and calculation. No frozen/production file
is modified. No new CB or data movement is introduced.

The public `recip_tile_init<false>()` and `recip_tile<false>()` arguments select
`legacy_compat=false`; they **do not** override global `APPROX`. With global
approximate math true, Blackhole selects `_calculate_reciprocal_fast_7b_`.
The private wrappers use the same lower-level SFPU macros with explicit
`APPROXIMATION_MODE=false`, `is_fp32_dest_acc_en=false`, `legacy_compat=false`
for both init and calculation. These select `_init_reciprocal_fast_8b_3c_`
and `_calculate_reciprocal_fast_8b_3c_`. The existing implementation applies
BF16-LSB correction. The historical `3c` function name is not a measured
latency claim; current source scheduling must be benchmarked.

The frozen common header, including the original reciprocal API, is parsed
before private definitions. Only `recip_tile_init` / `recip_tile` tokens inside
the selected frozen streaming include are renamed, then immediately undefined.
Each selected Blackhole header has exactly one `<false>` init/calculate pair,
both in final normalization. The Wormhole branch is excluded by the explicit
Blackhole guard. Global APPROX, approximate exp, rescale, matmul fidelities,
scratch/output formats and BF16 pack roundings remain untouched.

The reciprocal initialization still runs per normalized query tile, just as
before. It reestablishes SFPU configuration/addrmods and installs the matching
macro program; the calculation must never be swapped without this init. This
is O(output rows), not an additional operation per score tile. The completed
smoke below finds no accuracy benefit; performance has not been measured.

## Parent-run smoke

```bash
python experiments/sdpa-l2/bfp4-lofi-v2/recip8_fullchip.py \
  --label recip8-fast-control-smoke-v1 --destination fast_bf16 \
  --kv-formats b8_b8 --length 1024 --heads 2 --cores 4 \
  --distributions normal constant_v uniform_constant_v --check-preprocess --iters 0

python experiments/sdpa-l2/bfp4-lofi-v2/recip8_fullchip.py \
  --label recip8-fast-enabled-smoke-v1 --destination fast_bf16 --recip8 \
  --kv-formats b8_b8 --length 1024 --heads 2 --cores 4 \
  --distributions normal constant_v uniform_constant_v --check-preprocess --iters 0
```

Repeat with K8/V4 and optionally `--destination main_bf16`. The inherited
`--denom-only` diagnostic is also available but is not the default FAST point.
N<=1024 checks all Q rows; longer lengths reference the explicitly recorded
sampled rows and all KV. Two combined trace replays run even with `--iters 0`;
output hashes must match bitwise and original CPU/device BF16 input hashes
must remain unchanged. Original-input FP64 L2/PCC and centered residual metrics
are retained. Source pins include reciprocal public API, Blackhole implementation
and SFPU wrapper macros, selected frozen files, actual reader/writer, quantizer
oracles and reference. They are principal dependency pins, not the complete
compiler/firmware closure.

Local static checks passed: Python compilation, identical chunk/CB/global
compute-config AST versus the native grid7 driver's off control, one selected
reciprocal pair per frozen header, matching fixed-false init/calculate flags,
and all 50 source paths present for each destination. C++ JIT/device execution
has not been performed by the implementing agent; the parent owns device tests.

## Completed parent-run negative result

The off-v1 [K8/V8](../recip8-b8_b8-off-smoke-v1.jsonl) and
[K8/V4](../recip8-b8_b4-off-smoke-v1.jsonl) controls and on-v2
[K8/V8](../recip8-b8_b8-on-smoke-v2.jsonl) and
[K8/V4](../recip8-b8_b4-on-smoke-v2.jsonl) runs are complete. Configuration:
N1024/H2/D128, four cores, seed1240, noncausal, full BF16 compensation,
native exp, Q256/K512, two KV slots. All twelve rows pass exact preprocessing,
all-output finiteness, two bitwise combined trace replays, original-input
immutability and unchanged-source checks. Every Q row is compared with the
original-BF16-input FP64 reference. No timed samples were requested (`iters=0`).

Cells below are generated from the JSON ledgers; raw JSON remains authoritative.
Constant reference outputs have undefined PCC, shown as “—”.

| KV | Input | Off L2 % | Recip8 L2 % | Off PCC | Recip8 PCC |
|---|---|---:|---:|---:|---:|
| K8/V8 | normal | 3.092 | 3.094 | 0.999524 | 0.999524 |
| K8/V4 | normal | 12.183 | 12.210 | 0.992615 | 0.992616 |
| Both | constant_v | 0.633 | 0.659 | — | — |
| Both | uniform_constant_v | 0.781 | 0.781 | — | — |

The uniform case is bit-identical off/on, not merely equal after rounding the
reported L2. Normal/constant-V outputs do change. The reciprocal replacement
does not improve any of these L2 measurements and slightly worsens three;
there is no justification here for a long-context performance campaign or
enabling this change. Passing execution/replay gates is not an accuracy pass.

### Failed first compile and source provenance

The enabled-v1 [K8/V8](../recip8-b8_b8-on-smoke-v1.jsonl) and
[K8/V4](../recip8-b8_b4-on-smoke-v1.jsonl) ledgers contain provenance only,
not numerical results or a completion marker. The parent's first JIT failed
because the unguarded `static_assert(APPROX)` was parsed by UNPACK, where
`APPROX` is undeclared. The repair wraps only this assertion in
`#ifdef TRISC_MATH`; the explicit fixed-false init/calculation specializations
are unchanged. These failed historical files are intentionally retained.

All four completed runs have 50 source pins. Comparing off-v1 with on-v2,
the only changed pinned source is `recip_override.hpp`: hash prefix
`f0fbe178f71f` → `ed3fd6292cc8`. The wrapper is not included by the off
configuration, so this compile-only guard repair does not change its executed
control. The off-v1/on-v2 comparison must not be described as identical source
hashes.

## Read-only API and downstream-precision audit

The wrapper's template ordering, Blackhole/BF16 guard, coherent reciprocal
init/calculation, and macro placement agree with the current public and
Blackhole SFPU APIs. The `TRISC_MATH` assertion guard is appropriate for the
three separately compiled kernel threads. No remaining API-selection bug was
found; the enabled-v2 device smokes also compile and execute successfully.

There is an important limitation beyond the reciprocal itself. Final
normalization calls `mul_tiles_bcast_cols(cur_out_cb, scratch_cb, ...)`, which
[safe_rescale.hpp](../safe_rescale.hpp) implements with **HiFi2 ELWMUL**.
Ordinary AB elementwise unpack maps the reciprocal scratch operand to physical
SrcB. HiFi2 phases 0+1 refine SrcA but retain only the high seven significant
bits of SrcB, so the full eight-bit BF16 reciprocal is not consumed by this
multiply. An eight-bit correction can still change a retained high bit through
a carry; therefore changed output is plausible, and bit-identical output was
not required. This is a source-level precision limitation, not a demonstrated
causal decomposition of the measured error.

The independent mismatch between the represented-P denominator and LoFi PV,
plus prior BF16 recurrence rounding, remains unchanged. These results reject
the **reciprocal-only replacement in this pipeline**, not all higher-precision
normalization. A deliberate RNE7-reciprocal control is not implemented. The
following independent final-multiply HiFi4 control is implemented and its
completed 2×2 results are recorded below.

## Independent final-multiply HiFi4 diagnostic

`--final-scale-hifi4` is independent of `--recip8`, allowing a 2×2 control.
It is restricted to FAST (including its denominator-only diagnostic); MAIN
is rejected by both host and private kernel guards. No CB, chunk, input slot,
format, reader, global math fidelity or exp setting changes. Only the final
reciprocal broadcast multiply switches from HiFi2 to HiFi4. Recurrent scaling
still calls the original HiFi2 safe-rescale implementation.

The driver instantiates `sdpa_standard_v2` with `cb_exp_max_diff=14` and
`cb_recip_scratch=5`. In the selected FAST streaming header there are exactly
two broadcast-init sites: recurrent `out_in_cb, bcast_cb`, and final
`cur_out_cb, scratch_cb`. All three recurrent execute sites use `bcast_cb`;
all three `salad_correct_fused` callers pass `cb_exp_max_diff`. The sole final
execute site uses the template-fixed scratch CB. Thus CB5 selects only final
normalization in this driver. Both private init and execute select HiFi4 for
`b==5`, otherwise forwarding to the original safe helper. The final branch
has a compile-time-fixed argument; elimination of any recurrent dynamic branch
is not verified without inspecting emitted code and is not a performance claim.

`final_scale.hpp` is included after common/SFPU definitions, and broadcast macro
renaming surrounds only the selected streaming include. Original safe-rescale
macros are restored immediately afterward. Three CPU-only source-contract tests
pass: CB/callsite mapping, macro scope with matching fidelities, and transitive
source-path presence plus unsupported-MAIN guard. Python compilation passes.
The manifest now additionally pins the private helper and public/Blackhole
ELWMUL/unpack implementation dependencies; older smoke records retain their
original 50-pin provenance.

One remaining limitation is deliberately untouched: final
`matmul_block(cur_sum_cb, col_identity_cb, ...)` still runs at global **LoFi**.
The logical-left BF16 sum is physical SrcB and therefore loses its eighth
significant bit before denominator reduction; each compensation tile is also
subject to the same operand-width limit. Full precision in the subsequent
reciprocal/multiply cannot reconstruct that discarded information. A future
normalization-reduction-only HiFi4 ablation would test it independently; this
2×2 experiment must not attribute every remaining constant-V error to recip.

Suggested parent-run matrix, same existing smoke arguments and fresh labels:

| Reciprocal | Final multiply | Flags |
|---|---|---|
| 7b | HiFi2 | none |
| 8b | HiFi2 | `--recip8` |
| 7b | HiFi4 | `--final-scale-hifi4` |
| 8b | HiFi4 | `--recip8 --final-scale-hifi4` |

Use N1024/H2/C4, full FAST, both KV formats, and normal/constant-V/uniform
constant-V with exact preprocessing and `--iters 0` first. Source revisions
must be pinned separately from the preceding negative reciprocal-only result.

### Completed 2×2 result: no useful repair

The [baseline](../final-scale-r0-f0-smoke-v1.jsonl),
[scale only](../final-scale-r0-f1-smoke-v1.jsonl),
[reciprocal only](../final-scale-r1-f0-smoke-v1.jsonl), and
[both](../final-scale-r1-f1-smoke-v1.jsonl) runs cover **K8/V8 only**,
N1024/H2/C4/D128, seed1240, full BF16 compensation and native exp. All sixteen
result rows are complete. The four ledgers share exactly the same 56 source
hashes and identical original input hashes per distribution. Exact
preprocessing, all-output finiteness, original-input immutability, unchanged
sources and two bitwise combined trace replays pass in every row. Original
BF16 Q/K/V and every Q/KV row define the FP64 reference; no gain correction
is applied to the acceptance metric. These are untimed smokes (`iters=0`),
not throughput results.

| Reciprocal | Final multiply | Normal L2 % | Normal PCC | Constant-V L2 % | Uniform L2 % | Uniform constant-V L2 % |
|---|---|---:|---:|---:|---:|---:|
| 7b | HiFi2 | 3.092 | 0.999524 | 0.633 | 1.775 | 0.781 |
| 7b | HiFi4 | 3.085 | 0.999527 | 0.667 | 1.775 | 0.781 |
| 8b | HiFi2 | 3.094 | 0.999524 | 0.659 | 1.775 | 0.781 |
| 8b | HiFi4 | 3.113 | 0.999527 | 0.814 | 1.775 | 0.781 |

“Uniform” means Q=0 with normal K/V; “uniform constant-V” also sets V=1.
Both uniform cases are **bit-identical across all four configurations**;
uniform PCC is 0.999847, while constant-reference PCC is undefined. The two
HiFi2-multiply controls also exactly reproduce the earlier off-v1/on-v2
reciprocal experiment's output hashes on its three shared distributions.

Higher final-multiply fidelity slightly lowers normal L2 with the old reciprocal
(3.091971→3.085480%), but worsens constant-V error. Enabling both changes
worsens both normal and constant-V L2. The normal fitted gain moves from
0.997790 (baseline) to 1.001019 (scale only), 1.000602 (reciprocal only), and
1.003610 (both); constant-V gain moves from 0.999264 to 1.002705, 1.002056,
and 1.005342. These diagnostics show offsetting biases in the existing
pipeline, not a guarantee that a locally more precise primitive lowers total
error. They do not isolate every upstream source of bias.

Removing final SrcB truncation therefore does **not** uncover a useful 8b
reciprocal win on this matrix. Keep both switches off as the qualified control;
no long-context timing campaign or further kernel change is justified by
these results. The still-LoFi denominator reduction, P/denominator mismatch
and BF16 recurrence remain possible error sources, not established sole causes.
