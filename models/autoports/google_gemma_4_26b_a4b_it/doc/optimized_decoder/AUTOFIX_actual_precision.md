# AutoFix: actual-input integrated precision

Status: the first integrated policies fail the unchanged **0.995** HF gate on
actual 1025/512 inputs despite passing the actual 4096/128 headline. Isolated
controls identify sliding QKV BFP4 and full output-projection BFP4 as insufficient
on this stream. Passing native/BFP8-cache alternatives exist. Sliding generalized
routing also introduces independent failures; the centered-score composite now
passes all 512 actual checks. The source-equivalent optional centering patch was
applied after the parent confirmed hardware was idle; its default remains off.
Final combined-policy verification is pending. No position is removed from
acceptance.

## Evidence and provenance

Starting reports are `AUTODEBUG_stress.md`, `AUTODEBUG_stress_cpu.md` and
`AUTOFIX_stress.md`. Their Gaussian findings are retained as diagnostics, not
used to veto actual-input wins. This follow-up uses genuine recorded layer inputs
from `actual_text_layer{0,5}_1025_512.pt`, not generated activations. The 512
unique absolute positions are 1025..1536; the harness also records a duplicate
check of position 1025. All completed runs below pass prefill, first-position
repeat equality, runtime prefill/decode audits, and the program-cache guard.
Decode failure is still a failed run even when these other checks pass.

The four integrated and eight isolation runs use runtime SHA256
`e0691dc70a93b1e2d76d3028f022980f137cec4b0ac2ea944a25da425f19ea22`.
Each per-layer pair records the same fixture hash. Exact commands and exit codes
are in `actual_integrated_stress_commands.json` and
`actual_stress_isolation_commands.json`. Timing below is the median of recorded
`traced_decode_host_us`, **host-wall microseconds**, not device profiler time.
Times from failed candidates are observations, not accepted performance wins.

Earlier passing native controls are
`actual_text_native_cache8_stress_layer0.json` (minimum .995626216) and
`actual_text_native_cache8_gate4_stress_layer5.json` (.996953462). Read their
`attention_probe` fields: both enable native full-sync SDPA and BFP8 caches even
though the old base-policy metadata still says BF16 cache. Their attention
weights remain BF16. The full control also uses BFP4 expert gate/up. These older
wrapper controls provide context; current same-source isolation pairs carry the
stronger causal evidence.

## Failed integrated policies

All use native full-sync SDPA, BFP8 cache, generalized gate, all sharded hidden
norm sites, prefill BFP4 gate/up and LoFi prefill attention. Sliding uses QKV
BFP4/output BFP8/down BFP8; full uses QKV BFP8/output BFP4/gate BFP8/down BFP4.
The fast variants additionally change several decode fidelities and activation
or shared-MLP dtypes; the journal is authoritative for these compound policies.

| Exact result file | Prefill PCC | Minimum decode PCC | Failing positions | Host median, us |
| --- | ---: | ---: | --- | ---: |
| `actual_integrated_control_stress_layer0.json` | .997491403 | .993784369 | 1082, 1118, 1241, 1348, 1386, 1511 | 1242.132 |
| `actual_integrated_fast_stress_layer0.json` | .997491403 | .990673317 | 1082, 1118, 1241, 1327, 1348, 1386, 1459, 1511 | 1226.211 |
| `actual_integrated_control_stress_layer5.json` | .998224454 | .993301794 | 1104 | 1310.661 |
| `actual_integrated_fast_stress_layer5.json` | .998224454 | .992604457 | 1104, 1267 | 1295.986 |

## Completed isolation controls

Names encode QKV (`q`), attention output (`o`), expert down (`d`) and expert
gate/up (`g`) storage; 4/8 mean BFP4/BFP8. `no_gate` means the ordinary FP32
top-k router remains. All rows retain native full-sync SDPA and BFP8 cache.

| Exact result file | Prefill PCC | Minimum decode PCC | Failing positions | Host median, us |
| --- | ---: | ---: | --- | ---: |
| `actual_stress_isolate_q4o8d8_no_gate_layer0.json` | .998251106 | .993737384 | 1082, 1118, 1241, 1348, 1386, 1511 | 1256.984 |
| `actual_stress_isolate_q8o8d8_no_gate_layer0.json` | .999202982 | .998308950 | None | 1273.575 |
| `actual_stress_isolate_q8o8d4_no_gate_layer0.json` | .999202982 | .995523980 | None | 1263.712 |
| `actual_stress_isolate_gate_only_layer0.json` | .999205096 | .987116593 | 1313, 1459 | 1349.973 |
| `actual_stress_isolate_q8o8g8_no_gate_layer5.json` | .999916864 | .998536234 | None | 1329.102 |
| `actual_stress_isolate_q8o4g8_no_gate_layer5.json` | .998768880 | .993434071 | 1104 | 1328.198 |
| `actual_stress_isolate_q8o8g4_no_gate_layer5.json` | .999916864 | .996943464 | None | 1326.188 |
| `actual_stress_isolate_gate_only_layer5.json` | .999922490 | .997011430 | None | 1401.449 |

Verified precision boundaries:

- Sliding QKV: the `q4o8d8` to `q8o8d8` pair changes only QKV weight dtype and
  fixes all six failures. This verifies the QKV storage boundary on this stream,
  not an inherent kernel defect. The factory converts both prefill source weights
  and decode lane weights (`tt/optimized_decoder.py`, QKV dtype override), so it
  does not yet distinguish prefill-cache drift from decode projection error.
- Full attention output: the `q8o4g8` to `q8o8g8` pair changes only output weight
  dtype and fixes position 1104. Generalized gating and reduced prefill policy
  are not necessary for this failure. The output dtype also affects both phases.
- Sliding expert down: with QKV/output BFP8, both down BFP4 and BFP8 pass every
  actual step. Down BFP8 improves margin but is not mandated by this evidence.
  The approximately 10 us difference is one recorded timing comparison.
- Full expert gate: `q8o8g4` passes the full stream. Its norm-site override differs
  from the `q8o8g8` all-norm control, so those two are not a one-variable gate
  accuracy comparison. Both are eligible candidates, pending final integration.

## Curve shape and mechanism limits

Sliding integrated-control median PCC is .999143; its six misses are isolated
single-position dips, with immediate neighbors between .998733 and .999437.
The no-generalized-gate QKV-BFP4 control reproduces all six. That is consistent
with an attention perturbation crossing an MoE route boundary, but final-output
PCC alone does not prove an expert changed. Continuous weight error is also
present: median PCC rises from .999149 to .999926 when only QKV is upgraded in
the down-BFP8 pair.

Full output BFP4 similarly has a broad baseline loss (median .998362 versus
.999732 with output BFP8) plus the isolated position-1104 dip. Therefore the
evidence supports a continuous precision loss with possible routing
amplification, not a choice between two mutually exclusive mechanisms.

Sliding generalized-gate-only is more sharply localized: positions 1313 and
1459 fail at .987116593/.994260995, while median PCC is .999499 and the median
per-position difference from the older native/cache control is only +.0000023.
Neighbors of 1313 exceed .99939. Full gate-only passes all 512 positions. These
observations identify a layer/input-specific gate problem, not a blanket native
gate rejection. They do not distinguish BF16 score rounding, tie selection, or
composite softmax differences without the controls below.

There is no shared page/chunk boundary at the failing positions. The current
harness uses 32-token pages, reverse page mapping, and extent 2048 at 1025/512;
the largest rounded native read end is 1664 with its 128-token chunk, within the
allocation. Dtype-only repairs retain the same allocation. This source check
does not replace an exact-shape cache probe if later evidence suggests a cache
defect, but there is no current allocation-overrun evidence.

## Gate localization and refinement results

The composite validates BF16 scores/bias/output and UInt16 indices in
`ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate/device/generalized_moe_gate_device_operation.cpp:65`.
The current router rounds FP32 scores before selection. Close FP32 rank-8 and
rank-9 scores can therefore become equal; this is a hypothesis, not a measured
route result for the failing positions.

`tests/probe_optimized_router_gate.py --center-logits` subtracts the row maximum
in FP32 **before** BF16 conversion, with either `--round-logits-bf16` (ordinary
top-k/softmax) or `--generalized-gate` (composite). A common shift preserves exact
top-k and softmax; it changes BF16 rounding and need not resolve every close gap.
Both new operations remain inside the traced layer and its measured timing.
The original uncentered modes remain available for A/B comparison.

The literal shape limits are explicit: this test wrapper requires a
`BroadcastRouter` with 128 logical experts/top-8, one decode token with logits
shape `[1,1,1,128]`, then pads to 256 with `-inf` for the composite's 16x16 face.
It handles both attention layer kinds; prefill delegates to the original router.
It is not a batched-router validation. Do not combine the wrapper with the
factory's `generalized_router=true`, which it deliberately rejects.

Probe SHA256 after the centering edit:
`0c2237b42db6a57cc8fab6a82d58868775bcf1598addaf4875b5fad6aec4080e`.
CPU fake-op checks verified FP32 max/subtraction before BF16 conversion,
unchanged uncentered behavior, and unchanged prefill delegation. Black and
Python compilation passed. No TTNN import or device execution occurred in this
investigator's verification. The parent subsequently ran the following 18
refinements; exact commands and return codes are retained in
`actual_refine_stress_commands.json`.

All use the same actual 1025/512 fixtures and the passing QKV/output BFP8,
native full-sync/BFP8-cache base. Sliding has expert gate BFP8/down BFP4 and all
sharded norms. Full has expert gate BFP4/down BFP4 and its original norm policy.
Each row changes the named option from that per-layer base; none establishes
correctness of a combination of several successful options.

| Exact result file | Minimum decode PCC | Failing positions | Host median, us |
| --- | ---: | --- | ---: |
| `actual_refine_stress_router_bf16_layer0.json` | .986898615 | 1313, 1459 | 1266.653 |
| `actual_refine_stress_router_center_layer0.json` | .995549871 | None | 1258.372 |
| `actual_refine_stress_router_layer5.json` | .996990382 | None | 1308.276 |
| `actual_refine_stress_qkv_lofi_layer0.json` | .995442457 | None | 1261.252 |
| `actual_refine_stress_qkv_lofi_layer5.json` | .996876294 | None | 1322.082 |
| `actual_refine_stress_sdpa_lofi_layer0.json` | .988700600 | 1257, 1317 | 1262.862 |
| `actual_refine_stress_sdpa_lofi_layer5.json` | .996874807 | None | 1323.313 |
| `actual_refine_stress_expert_act8_layer0.json` | .995505404 | None | 1263.134 |
| `actual_refine_stress_expert_act8_layer5.json` | .996913912 | None | 1311.088 |
| `actual_refine_stress_fused_gelu_layer0.json` | .995523980 | None | 1267.400 |
| `actual_refine_stress_fused_gelu_layer5.json` | .996943464 | None | 1327.489 |
| `actual_refine_stress_lanes1_term1_layer0.json` | .994183442 | 1459 | 1119.355 |
| `actual_refine_stress_lanes8_term1_layer0.json` | .994238889 | 1459 | 1118.383 |
| `actual_refine_stress_lanes16_term1_layer0.json` | .994243097 | 1459 | 1119.116 |
| `actual_refine_stress_lanes1_term1_layer5.json` | .996944219 | None | 1204.933 |
| `actual_refine_stress_lanes8_term1_layer5.json` | .996965522 | None | 1205.831 |
| `actual_refine_stress_lanes16_term1_layer5.json` | .996965522 | None | 1205.821 |
| `actual_refine_stress_shared4_layer5.json` | .996395904 | None | 1316.872 |

The uncentered BF16-score **ordinary** top-k control reproduces sliding gate
failures at 1313 and 1459 without calling the composite. Thus conversion of the
scores to BF16 is sufficient to introduce the failure in an otherwise passing
base; a composite-specific defect is not required. The centered composite passes
prefill (.999202982), all 512 unique positions, deterministic replay, runtime
audits, and the program-cache guard. It repairs accuracy through a specified
FP32 maximum/subtraction before rounding, without changing the threshold,
token set, cache layout, or attention/expert weights.

These runs substantiate the score-rounding boundary and the centered candidate's
correctness. They do not directly measure rank-8/9 ties, nor completely separate
the centering effect from composite selection/softmax: the recorded ordinary
BF16 control is uncentered and the passing centered run uses the composite.
A centered ordinary control or matched uncentered composite on this exact base
would complete that mechanism isolation if needed. The passing candidate's
1258.372 us versus base 1263.712 us is a small single-series host comparison;
final complete-layer profiling must establish any claimed gain.

Full generalized gating combined with QKV/output BFP8 and expert-gate BFP4 also
passes all 512 checks. Full one-term QKV, LoFi QKV, LoFi native SDPA, BFP8 expert
activations and BFP4 shared weights pass independently. Sliding one-term QKV
fails position 1459 at all three tested lane counts, and LoFi native SDPA fails
1257/1317; retain two terms and HiFi4 native attention unless another focused
actual-input control repairs them. Fused GELU has the same reported minimum
PCC but slightly **higher** recorded host medians than its base
(1267.400 versus 1263.712 us sliding; 1327.489 versus 1326.188 us full), so these
measurements do not demonstrate a speed gain. Identical scalar PCC alone does
not establish bitwise output identity.

## Optional production integration

`generalized_router_center.patch` adds factory option
`generalized_router_center=False`, passes it to `GeneralizedRouter`, and records
it in `precision_policy.generalized_router_center`. Enabling it requires
`generalized_router=True`; ordinary defaults remain unchanged. The FP32
dtype-check/max/subtraction AST is identical to the passing probe. The patch
was checked before application, then applied atomically after the parent
confirmed hardware was idle. Applied runtime SHA256:
`ca9eb12ab361b21643d2a2fd8bcc56e80f4eb32237e97befaa77628a355ddd2a`.
Patch SHA256:
`e01e17daf5a9bd5ee5044e329153c1e749ff8fb9967b5d89d9fe00e8d4ec767d`.

Python compilation and patch application checks pass. The patch adds no Black
formatting differences; the pre-existing multiline `decode_gelu_activations`
assignment is still a separate formatting difference in the current runtime.
Production-path and combined-policy hardware verification remain parent-owned.
For deeper route localization, the ordinary-router
`probe_optimized_stress_routes.py` requires saved-output identity and captures
attention/residual/normalization/scores/routes. It intercepts `ttnn.topk` and
must be adapted before observing the composite.

`stress_correct_native_policies.json` records the parent's current passing
candidates. Final selection requires the actual headline, all 512 actual stress
steps, relevant runtime contracts, and measured complete-layer performance.
No failed candidate or Gaussian diagnostic has been relabeled as passing.
