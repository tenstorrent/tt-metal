# Advisor contribution at decode batch 32

## Full-model estimate: 778,781.2 us -> 777,651.4 us (±371.7 us)

The shipped 109-core MLP gate-times-up product saves an estimated **1,129.8 us per 64-layer model decode**, larger than the conservative model-level band. The estimate is the measured per-layer delta times the model-config layer counts (48 linear-attention + 16 full-attention = 64), applied to the reconciled profile windows and summed. It is an arithmetic full-model estimate, not a direct full-model timing.

| Kind | Layers | Incumbent layer | Confirmed winner | Estimated kind before | Estimated kind after |
|---|---:|---:|---:|---:|---:|
| linear_attention | 48 | 15.844222 ms | 15.825584 ms | 758,472.96 us | 757,578.31 us |
| full_attention | 16 | 1.449491 ms | 1.434794 ms | 20,308.24 us | 20,073.10 us |

Every repeat of each initial winner and fresh-process confirmation beats every repeat of its frozen incumbent. Fresh knob-off runs after the confirmations returned to 15.843234 ms (linear) and 1.449036 ms (full), disconfirming a process-order warm-up explanation.

## What shipped

`OptimizationPolicy.advisor_plan="mlp_product_only"` is enabled in both default layer policies at batch 32. It changes the MLP gate-times-up multiply output from L1 width-sharded 8 cores, `[32,2176]`, to the exact advised L1 width-sharded 109 cores, `[32,160]`. A `self.batch == 32` guard preserves incumbent behavior at other batches. Weight dtypes and compute fidelities are unchanged.

The absolute veto used real weights and the model's existing Hugging Face layer reference at its own PCC bar, 0.995. Candidate and incumbent were equal: linear 0.9981887732, full 0.9980950193. Differential PCC was observational only.

## Whole-plan first and ablations

Candidate #1 applied the maximal executable subset of `advised_plan.ops` after dropping full attention's advisor-unfixable `nlp_concat_heads_decode`: the 11-core block-sharded norms, 80-core residual/add boundary, 109-core MLP product, and requested matmul output layouts. It measured 15.844805 ms (linear) and 1.449963 ms (full), so it lost.

The requested DRAM-sharded matmul outputs did not execute: TTNN warned and substituted the program's computed 8-core layouts. The isolated test in `scripts/isolate_advised_matmul_output.py` found linear 103 cores / `[32,160]` became 8 / `[32,2080]`, and full 90 / `[32,160]` became 8 / `[32,1792]`. These placements are recorded as inexpressible under the incumbent DRAM-sharded matmul program, not as measured advisor losses.

| Candidate | Linear median | Full median | Decision |
|---|---:|---:|---|
| apply-all executable subset | 15.844805 ms | 1.449963 ms | reject |
| 11-core norm only | 15.852609 ms | 1.455297 ms | reject, default-off |
| 80-core residual only | 15.862356 ms | 1.461941 ms | reject, default-off |
| 109-core MLP product only | 15.825584 ms confirmed | 1.434794 ms confirmed | ship |
| residual × MLP product | 15.844347 ms | 1.446797 ms | required product measured; slower than isolate |

There were no `cliff_candidates`, so no legal-ladder cliff sweep was permitted or required. The winner appears in a chain for which reconciliation reports `advisor_removes_per_model_us = 0`; it is an advised regrid, not a predicted conversion removal. Thus the across-kind ranking on that field is a tie, and the same winning placement ships in both kinds; no per-layer ranking was used to choose one kind over the other.

## Reconciliation and profile accounting

Both fresh reconciliations used `--incumbent`, `--ir final_ir.mlir`, and `--evidence`; both close at 100% and are not degraded. Linear's 15,801.520 us window is 33.40% DRAM-resident, 30.90% boundary, 16.96% agrees-with-shipped, 13.88% untraced, and 4.85% chain. Full's 1,269.265 us window is 44.97% agrees-with-shipped, 43.67% chain, 6.13% boundary, 2.37% DRAM-resident, 2.16% untraced, and 0.70% advisor-unfixable.

Before/after category profiles are retained under `profiles/*_ops/`. Linear changes Compute/TM/DM/Other from 6740.97/8411.49/238.99/410.07 us to 6714.64/8410.24/244.62/410.03 us. Full changes 1079.81/75.78/31.00/82.68 us to 1059.37/76.31/35.53/82.61 us.

The profile's conversion ranking is independent of the advice. Linear's leading conversion ops are UntilizeWithUnpadding at 819.343, 819.248, and 819.165 us, then TilizeWithValPadding at 675.364 and 662.267 us. Full's leading conversions are ReshapeView at 13.953 and 12.398 us. The detailed order remains in the reconciliations and incumbent perf CSVs.

Reported but not screened: advisor-agreeing boundaries are 695.919 us/layer linear and 1.596 us/layer full; layer handoff is 1.372 us/layer linear and 1.373 us/layer full; `starved_ops_not_attributable` remain as emitted by reconciliation. None is booked as advisor contribution.

## Scope and provenance

All controls, candidates, confirmations, captures, and oracles use batch 32. The first harness process was discarded, controls were frozen before capture, and `captured_at > measured_at`. Captures used `scripts/capture.py`; their scope records two tracer substitutions: omission of dynamic `memory_config()` guards in `_rms_norm_decode` and `_decode_linear`. The imported tracer fingerprint matches the advisor checkout.

The model config supplies 48 linear-attention and 16 full-attention layers, summing to `num_hidden_layers=64`; those asserted counts are in `incumbent.json`. Raw profiler dumps are ignored, while decision CSVs, reports, final IR, advisor report, and gzipped decision traces are retained.

`could_not_do`: exact advised DRAM-sharded matmul output grids were not executable because TTNN substituted the incumbent layout; `nlp_concat_heads_decode` was advisor-unfixable and was never screened. All losing knobs remain in code default-off.
