# Optimized multichip decoder — COMPLETE with accepted limitation

Stage5 for `Qwen/Qwen3.6-35B-A3B`, Blackhole chips0–3, true1×4 TP/EP mesh. **The owner accepts the retained linear decode performance as a known limitation and requests no further tuning for that floor.** [Owner decision](owner_decision.md) and [limitations](../eval_readiness/limitations.json) preserve the exact decision, scope and evidence. Full attention meets the 10% floor; linear remains at7.20333%. Correctness and other quality requirements remain unchanged. CPU stage acceptance passes34/34, including one explicit owner-accepted performance limitation. [Independent round3](stage_review_round3.md) returns **pass-with-carried**, with no required work remaining. Stage5 is complete under the owner-revised contract. Scope remains the decoder; full-model and vLLM work have not started.

This evidence snapshot is dated2026-11-14. Every measurement is bound to its JSON and `.json.provenance.json` command/source identity. The tables report the retained default only; the rejected native CCL/recurrence candidates are not substituted into optimized numbers. [Native restoration verification](native_restore_verification.json) confirms exact pre-experiment C++ source restoration and bit-equal prefill/all 128 decode outputs for both retained layer kinds. The rebuilt library has a new binary identity; binary bit-equality is not claimed. Filenames beginning `final_` do not themselves establish stage acceptance or qualify subsequently changed source.

## Before and retained-default measurements

Whole-batch serial1×4 timings below are milliseconds at logical prefill length32, followed by eight changing decode steps. Prefill is the warmed median; decode is the warmed blocking trace median. B32 uses natural activations from32 distinct prompts. The four current-default artifacts pass their existing whole-output PCC checks.

| Layer / batch | Before prefill | Before decode | Current prefill | Current decode | Current evidence |
|---|---:|---:|---:|---:|---|
| Linear /1 | 62.103642 | 1.015803 | 52.603419 | 0.725101 | [final_l0_b1.json](final_l0_b1.json) |
| Linear /32 | 1329.492118 | 21.724882 | 1156.660697 | 16.162265 | [final_l0_b32.json](final_l0_b32.json) |
| Full /1 | 2.216440 | 0.393292 | 2.134202 | 0.346633 | [final_l3_b1.json](final_l3_b1.json) |
| Full /32 | 42.404445 | 6.444862 | 44.724137 | 4.941677 | [final_l3_b32.json](final_l3_b32.json) |

Starting controls are `before_l{0,3}_b{1,32}.json`. Full B32 prefill regresses relative to the starting control, while all four decode rows improve. Current prefill/minimum decode PCC is linear B1 0.999988819/0.999920574, linear B32 0.999966521/0.999904061, full B1 0.999972759/0.999853822, full B32 0.999969649/0.999854346. The separate128-step FUNCTIONAL measurements cannot be substituted into this short-run table.

Current defaults enable batched attention reduction, batched full projections, packed ordered linear inputs and one batched untilize before row extraction. Superseded replicated scan weights are released at setup. The selected exact helpers are FP32 top-k selection, compensated norm tail and tile-worker SiLU; full input normalization uses the existing residual shards. Defaults retain BF16 residual/KV/CCL, BFP8 ordinary and expert weights, HiFi2 ordinary attention/shared matmuls and LoFi active experts. Native norm/projection/recurrence repair variants remain experimental.

Both layer kinds accept and return replicated BF16 hidden2048 residuals. B32 exposes singleton request tensors; internal batching owns row packing/extraction. Stacking these interfaces needs no inter-layer gather, reshard or all-reduce. In-layer collectives use persistent shared workspace. B1 executes indexed top8 TP experts; B32 uses token-grouped EP dispatch and score-weighted combine. Dense all-expert execution is not selected.

## FUNCTIONAL measurements and owner-accepted limitation

The required sequence is one user,128 input tokens and128 changing decode inputs/positions, repeated three times. Latency is the mean of384 blocking trace samples; states reset between sequences. Input/position refresh, presynchronization, host output checks and audit-context setup/teardown are outside the clock. This measures decoder replay, not end-to-end generation.

The corrected useful-DRAM model uses512GB/s **per rank**, sharded weights, replicated work and duplicated KV heads. Linear router weights are BF16; resident L1 output gamma is excluded. Padding, rereads, intermediate traffic and CCL are excluded. The original≥10% performance target corresponds to **≤0.518283125ms/token linear** and **≤0.344432500ms/token full**.

| Current-default evidence | Mean ms/token | Floor fraction | Prefill PCC | Minimum128-step PCC | Outcome |
|---|---:|---:|---:|---:|---|
| [Linear](restored_functional_l0.json) | 0.719504474 | 7.20333% | 0.999991977 | 0.999158705 | Floor missed; owner-accepted limitation |
| [Full](restored_functional_l3.json) | 0.340836807 | 10.10550% | 0.999969297 | 0.999834106 | Pass |

Both rows pass recorded accuracy, saved-state and changing-input eager/trace checks. Linear still needs27.97% less latency to meet its ceiling. The completed AutoFix attempts did not satisfy this floor. The owner explicitly accepts this measured shortfall for stage5; this is not a Stop Rule waiver, an HF-reproduced limitation, or a measured performance pass. No further tuning for this floor is requested. [AUTOFIX_functional.md](AUTOFIX_functional.md) preserves the candidate table, failed attempts and exact-component evidence.

## Extended layer34 sensitivity contract

The additional seed154/S128/D192 stress uses aggregate whole-output PCC≥0.995 and per-step diagnostics floor0.99. Matched-HF default aggregate0.999702065 and diagnostic minimum0.991128479 meet both criteria; selector-only is bit-equal across all192 outputs. **The default does not fail this extended stress and needs no HF waiver.** Earlier contrary work-log interpretations are explicitly superseded. Existing stage4 mandatory per-forward fixtures and their strict checker remain unchanged.

Native norm/projection/recurrence alternatives have below0.99 diagnostic anomalies and remain unaccepted even though their aggregate PCC passes. Token top-1 agreement is N/A at a standalone hidden-state boundary. [Matched HF comparison](layer34_hf_comparison.json) records exact input equality, checkpoint/source provenance, aggregate PCC and per-step diagnostics; it performs only CPU saved-tensor comparisons.

## Evidence and handoff

- [Topology audit](topology_audit.md), [candidate decisions](optimization_decisions.md) and [work log](work_log.md) record family comparisons, rejections, recovery and optional handoffs.
- [CCL AutoDebug](AUTODEBUG.md), [CCL AutoFix](AUTOFIX.md), [FUNCTIONAL AutoDebug](AUTODEBUG_functional.md) and [FUNCTIONAL AutoFix](AUTOFIX_functional.md) preserve diagnoses and repair evidence.
- B1 and B32 profile collection/report generation are complete: [B1 summary](perf_summary.json) and [B32 summary](perf_b32/perf_summary.json) reference the per-rank tables/CSV/provenance and advice. B32 prefill profiling covers one representative request; B32 decode profiling covers the whole batch. These kernel sums are not the warmed wall latencies above. [MoE attribution](moe_share.json) includes the selected FP32 selector and retains the historical TopK controls.
- BF16 KV and head ownership preserve [the context contract](../context_contract.json). Both current-source non-aligned 32K smokes pass; maximum context was not rerun because KV dtype/layout is unchanged.

Current-default validation completed for decoder source `13b403518fb8b71dcf5b502bddeb9be79bcd6acf27b1bb51ab47c4ef902d952f`. CPU consolidation reran the existing strict checks for the following artifacts without changing thresholds or invoking stage review:

| Validation | Evidence | Result |
|---|---|---|
| Non-aligned/public boundaries | `boundary_l{0,3}_s{1,31,65,257}.json` | All 8 pass |
| Evolving 128-step stress with allocation tracking | `stress_l{0,3}_s129_d128.json` | Both pass; minimum decode PCC 0.999768122/0.999834106 |
| Natural B32, 32 distinct teacher-forced prompts | `natural_l{0,3}_b32.json` | Both pass, including activation provenance |
| Watcher B1/B32, both layer kinds | `watcher_l{0,3}_b{1,32}.json` and matching watcher logs | All 4 pass; 2/4/1/2 completed dumps, no failure signatures |
| Watcher non-aligned/resumed and singleton prefill | [watcher_paths_summary.json](watcher_paths_summary.json) | All 3 pass; 12/2/1 completed dumps, no failure signatures |
| Mixed stack/shared cache ownership | [mixed_stack_selected.json](mixed_stack_selected.json) | Pass, including disjoint pages and changing trace replay |
| Allocation-tracker positive control | [positive_control.json](trace_alloc_tracking/positive_control.json) | Unsafe buffer rejected; clean replay passes on all 4 devices |
| Non-aligned 32K context | `context_l{0,3}_32k.json` | Both pass at prefill 32769 with cache capacity 65536 |

The context tests decode two steps after the 32769-token prefix; 65536 is allocated capacity, not a filled-prefix test. Runtime fallback audits in the paired/mixed artifacts pass. This completed subset does not establish overall stage acceptance.

| Layer / batch | MoE kernel-work share | MoE device firmware-span share |
|---|---:|---:|
| Linear /1 | 24.36% | 24.58% |
| Linear /32 | 5.80% | 5.31% |
| Full /1 | 53.58% | 49.57% |
| Full /32 | 18.77% | 14.41% |

Shares are measured within each replay on the reported rank. Firmware spans include gaps; neither column is an exclusive causal fraction of clean host-wall latency. The boundary includes router, routed/shared experts and output collective, excluding post-attention normalization, final residual addition and output splitting.

The final isolated repair controls also fail the floor: [exact recurrence plus FP32 CCL](repair_fp32_ar_exact_l0.json) takes 0.869184612 ms/token (5.96287%), slower than the retained default; [native norm/DRAM projection plus FP32 CCL](repair_fp32_ar_native_l0.json) takes 0.554099732 ms/token (9.35361%). Both pass layer0 accuracy/replay, but the combined sensitive-layer34 candidate still has a below-0.99 diagnostic minimum of 0.987942862. Component success did not produce a qualifying complete decoder. [AutoFix assessment](AUTOFIX_functional.md) records the controls and rejection evidence.

The retained-default linear FUNCTIONAL result is **0.719504474 ms/token versus the original maximum0.518283125ms/token**. The [owner decision](owner_decision.md) accepts this specific result for stage5; it does not lower the target or grant a future-stage exemption. Raw result flags remain false. The checker uses the final restored-native FUNCTIONAL artifacts for both kinds, checks their actual current native build identities, and reports the linear row as `owner-accepted-performance-limitation` with `raw_floor_passed=false`. All PCC, state, trace, mesh and runtime-fallback checks still execute. The prior33/34 result is preserved in [gate_summary_before_owner_acceptance.json](gate_summary_before_owner_acceptance.json).

Profile collection, MoE attribution and restored-native verification are complete. The native CCL experiment is retained only as an unapplied patch. Round1/round2 reports remain historical; the owner decision changes only the remaining performance finding's acceptance. The current [gate_summary.json](gate_summary.json) passes34/34 under the revised owner contract:33 measured/check passes and one owner-accepted performance shortfall. [Gate log](logs/owner_acceptance_gate_check.log) records exit0; six CPU regression tests pass. [Final independent round3](stage_review_round3.md) returns **pass-with-carried**. Handoffs and local acceptance checkpoint SHAs are recorded in [work_log.md](work_log.md). Production formatting is carried in [deferred_formatting.patch](deferred_formatting.patch) to preserve measured source bytes; all newly changed validation files pass formatting/hooks. No successful native repair or push is claimed. Subsequent source/native-library changes require affected qualification.

The bounded serialized validation entrypoint is `tests/run_optimized_multichip_validation.sh` with groups `clean`, `boundary`, `stress`, `natural`, `watcher`, `watcher_paths`, `mixed`, `context`, `functional`. Once evidence is complete, `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.check_optimized_multichip_evidence` writes `gate_summary.json` and `before_after.csv`. The device runner intentionally retains its original performance assertion and still exits nonzero for this accepted raw linear result; the CPU stage-acceptance checker applies the narrowly scoped owner decision.
