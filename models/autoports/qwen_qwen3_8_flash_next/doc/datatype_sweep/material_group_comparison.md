# Material group comparisons — completed coarse sweep snapshot

CPU-only snapshot initially prepared 2026-11-16T04:53:46.414458+00:00, updated after the coarse queue completed to include both GDN candidates and the chart readability fix. This compares 19 completed `_v3` records only; no selection, stage-review or new device run. Parent-owned combined-policy experiments and remaining work are outside this snapshot.

## Measurement and gate definitions

Every number below is re-derived from the linked immutable result. Speed is the mean warmed **AIME24 chat100 traced teacher-forcing decode tokens/s/user**, accepting only `trace_verified=true` with zero captures/releases in the timing scope. It is not eager speed or post-selection token-out throughput. The baseline has two repeats; other rows have one. Displayed speeds use 12 decimals; source JSON retains the full float.

Top-1/top-5/top-100 are fractions, each the minimum across **both AIME24 chat100 and natural-text 512 positions, prefill and teacher-forcing, and every saved repeat**. The unchanged numerical gate is top-1 ≥0.90, top-5 ≥0.98 and top-100 =1.0 on all rows; reporter `numerical_accuracy` also verifies reference/path coverage and matched counts. Every completed row in this snapshot passes that numerical gate and execution/trace checks. This does not imply P1 or capability acceptance.

Baseline traced repeats: `[9.679690005440683, 9.67923230752992]`; mean `9.679461156485`; observed spread `0.000457697911` t/s/u. Tiny differences need repeat confirmation before final choice.

Policy differences are against `inherited_e4_lofi`: all experts BFP4+LoFi; shared and QSA projections BFP8+HiFi4; GR, GDN Z/A/B and head BF16+HiFi4; activation/residual/CCL/KV/logits/sampling BF16. The explicit `runtime_precision.requested` in each result resolves layer exceptions. The constructed-summary audit is available in `precision_propagation_recheck_v1.json`; it covered baseline/reference/proj8/shared4-LoFi, not every later row.

## Routed experts: BFP4 fidelity pair

| Config / exact result | Changed policy | AIME traced t/s/u | Minimum top-1 / top-5 / top-100 | P1 | Capability |
|---|---|---:|---|---|---|
| [inherited_e4_lofi](inherited_e4_lofi_v3.json) | Expert gate/up/down BFP4 + LoFi, all 48 layers | 9.679461156485 | 0.960000000 / 1.000000000 / 1.000000000 | Pass: initial P1 selected gate | Pass: recorded capability decision |
| [e4_hifi2](e4_hifi2_v3.json) | Same expert BFP4 weights + HiFi2, all 48 layers | 9.670870340389 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |

HiFi2 minus LoFi measured speed: `-0.008590816096` t/s/u (-0.088753%). This pair alone does not establish eligibility.

## Existing BFP8 projection groups: bundled fidelity pair

| Config / exact result | Changed policy | AIME traced t/s/u | Minimum top-1 / top-5 / top-100 | P1 | Capability |
|---|---|---:|---|---|---|
| [proj8_lofi](proj8_lofi_v3.json) | shared/shared_down and QSA QKV/output BFP8 + LoFi | 9.685363109768 | 0.970000000 / 0.998046875 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |
| [proj8_hifi2](proj8_hifi2_v3.json) | Same four BFP8 roles + HiFi2 | 9.689403957899 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |

HiFi2 minus LoFi measured speed: `+0.004040848130` t/s/u (+0.041721%). This pair alone does not establish eligibility.

## Shared expert projections: BFP4 fidelity pair

| Config / exact result | Changed policy | AIME traced t/s/u | Minimum top-1 / top-5 / top-100 | P1 | Capability |
|---|---|---:|---|---|---|
| [shared4_lofi](shared4_lofi_v3.json) | shared/shared_down BFP4 + LoFi in layers 1–46; layers 0/47 BFP8 + HiFi4 | 9.690366404989 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |
| [shared4_hifi2](shared4_hifi2_v3.json) | Same layer/dtype policy; interior shared/shared_down HiFi2 | 9.689034064014 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |

HiFi2 minus LoFi measured speed: `-0.001332340975` t/s/u (-0.013749%). This pair alone does not establish eligibility.

## QSA projections: BFP4 fidelity pair

| Config / exact result | Changed policy | AIME traced t/s/u | Minimum top-1 / top-5 / top-100 | P1 | Capability |
|---|---|---:|---|---|---|
| [qsa4_lofi](qsa4_lofi_v3.json) | QSA QKV/output BFP4 + LoFi in layers 3,7,…,43; layer 47 BFP8 + HiFi4 | 9.686859754653 | 0.910000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |
| [qsa4_hifi2](qsa4_hifi2_v3.json) | Same QSA layer/dtype policy; BFP4 layers use HiFi2 | 9.685994469279 | 0.910000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |

HiFi2 minus LoFi measured speed: `-0.000865285374` t/s/u (-0.008933%). This pair alone does not establish eligibility.

## Routed experts: BFP8 boundary-layer fidelity pair

| Config / exact result | Changed policy | AIME traced t/s/u | Minimum top-1 / top-5 / top-100 | P1 | Capability |
|---|---|---:|---|---|---|
| [e8_edges_lofi](e8_edges_lofi_v3.json) | Expert gate/up/down layers 0/47 BFP8 + LoFi; other 46 BFP4 + LoFi | 9.666055426001 | 0.950000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |
| [e8_edges_hifi2](e8_edges_hifi2_v3.json) | Only layers 0/47 change to BFP8 + HiFi2; other 46 remain BFP4 + LoFi | 9.669162662416 | 0.970000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |

HiFi2 minus LoFi measured speed: `+0.003107236415` t/s/u (+0.032146%). This pair alone does not establish eligibility.

## LM head: BFP8 fidelity pair

| Config / exact result | Changed policy | AIME traced t/s/u | Minimum top-1 / top-5 / top-100 | P1 | Capability |
|---|---|---:|---|---|---|
| [head8_lofi](head8_lofi_v3.json) | LM head BFP8 + LoFi; other groups inherited | 9.702189110140 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |
| [head8_hifi2](head8_hifi2_v3.json) | LM head BFP8 + HiFi2; other groups inherited | 9.706468028874 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |

HiFi2 minus LoFi measured speed: `+0.004278918734` t/s/u (+0.044103%). This pair alone does not establish eligibility.

## GR projections: BFP8 fidelity pair

| Config / exact result | Changed policy | AIME traced t/s/u | Minimum top-1 / top-5 / top-100 | P1 | Capability |
|---|---|---:|---|---|---|
| [gr8_lofi](gr8_lofi_v3.json) | Both GR projections per layer BFP8 + LoFi; other groups inherited | 9.724606206264 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |
| [gr8_hifi2](gr8_hifi2_v3.json) | Same GR weights + HiFi2; other groups inherited | 9.717246726183 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |

HiFi2 minus LoFi measured speed: `-0.007359480081` t/s/u (-0.075679%). This pair alone does not establish eligibility.

## GDN Z/A/B projections: BFP8 fidelity pair

| Config / exact result | Changed policy | AIME traced t/s/u | Minimum top-1 / top-5 / top-100 | P1 | Capability |
|---|---|---:|---|---|---|
| [gdn_zab8_lofi](gdn_zab8_lofi_v3.json) | GDN Z/A/B projections BFP8 + LoFi on all 36 GDN layers; other groups inherited | 9.709247632119 | 0.950000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |
| [gdn_zab8_hifi2](gdn_zab8_hifi2_v3.json) | Same GDN Z/A/B weights + HiFi2; no layer exceptions | 9.701814864534 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |

HiFi2 minus LoFi measured speed: `-0.007432767585` t/s/u (-0.076553%). Both runs completed and passed all four reference/path numerical rows with verified traces. This pair alone does not establish P1 or capability eligibility.

## Activation, CCL and KV candidates

| Config / exact result | Changed policy | AIME traced t/s/u | Minimum top-1 / top-5 / top-100 | P1 | Capability |
|---|---|---:|---|---|---|
| [activation8](activation8_v3.json) | DRAM projection inputs including head BFP8; residual and remaining operation contracts unchanged | 9.653394595070 | 0.940000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |
| [ccl8](ccl8_v3.json) | Model CCL BF16 payload transport BFP8; fixed FP32/integer payloads unchanged | 9.546785963123 | 0.970000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | Pending: no recorded capability decision |
| [kv8](kv8_v3.json) | KV cache BFP8; other policy fields inherited | 9.293589016823 | 0.960000000 / 1.000000000 / 1.000000000 | Pending / not scored in preserved initial P1 | **Fail: full-model KV PCC control** |

## Interpretation and remaining evidence

- Shared and QSA BFP4 groups each have an exact LoFi/HiFi2 counterpart with the same dtype/layer exceptions. The inherited-expert BFP4 pair also changes fidelity alone. These are the completed material BFP4 groups in this snapshot; none of their new BFP4 policies can be accepted from short-reference accuracy alone.
- `proj8_*` changes four BFP8 projection roles together. It provides tested LoFi/HiFi2 coverage for each included role, but its aggregate speed difference cannot be attributed to shared or QSA individually. Conversely, comparing `shared4_*` or `qsa4_*` directly against `proj8_*` is not an isolated BF8-vs-BF4 experiment because non-target group fidelities differ.
- `e8_edges_*` is a mixed BFP8/BFP4 two-boundary-layer experiment, not an all-BFP8 expert model. Its HiFi2 policy changes expert fidelity only on layers 0/47. Do not confuse it with `e8_two_hifi4_v1.json`, whose expert fidelity is HiFi4 globally and which serves as the fitting initial P1 reference.
- The completed GR, head and GDN Z/A/B pairs provide measured BFP8 LoFi/HiFi2 alternatives. Their single-run speed gains are candidates for repeat/P1/combined-policy work, not final wins. The source artifacts above preserve the exact commands, precision policy, hardware, mesh, trace counters and all reference rows. GDN results were unavailable at the initial snapshot and were added only after both completed records existed.
- KV8 passes the two short-reference numerical gates but **fails capability**: `kv_short_prefix_b32_comparison_v2.json:required_gate_passed=false`, minimum phase vocabulary PCC 0.971848264 below 0.99. Matching top-1 does not waive this rejection. `capability_decisions.json` records it explicitly.
- New candidates have no P1 result in `p1_initial_two_v1.json`; its passing selected gate belongs only to `inherited_e4_lofi`. `p1_initial_audit.md` and `p1_decisions_initial.json` explain the reference/HF qualification separately. No inference of transferred P1 acceptance is made.
- Capability is reported from the structured `capability_decisions.json`, including checked JSON gates. Apart from baseline and rejected KV8, these `_v3` policies have no decision there yet. Successful full-capacity execution is useful evidence but does not silently populate that separate gate. Final selection still needs the selected-default validations, max-context/smoke choice, P1, repeat timing and post-selection token-out benchmark.

## Chart readability issue resolved

The initial audit found that the reporter's 11×7-inch figures placed full labels at only three offsets, causing crowding among policies at 100% top-5. The completed coarse snapshot now has **18 of 19 policies with minimum top-5 exactly 100%**, reinforcing the need for the fix.

The reporter now uses short stable point IDs, a speed-ordered side key containing every config/status/value, collision-aware label placement with leader lines, and a detail inset when slow outliers compress the upper timing band. It preserves exact point coordinates, the red selected point, Pareto frontier and dotted 90%/98% threshold. [chart_readability_audit_v1.json](chart_readability_audit_v1.json) records four synthetic 30-policy render checks, including exactly coincident measurements: zero ID/key text overlaps, all measured labels and all policy-key entries retained, and unchanged measured coordinates. All 21 CPU reporter tests and targeted hooks passed; no production sweep charts were overwritten during that work.

At 100% top-5 a frontier may correctly collapse to the fastest point; that is a real lack of an accuracy tradeoff, not a plotting failure. The frontier intentionally includes rejected policies and labels that fact; eligibility remains separately visible. Final measured charts will be generated after the parent finishes combined-policy selection.
