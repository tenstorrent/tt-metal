# Optimized multichip decoder

Status: **complete**. All final default gates pass; independent stage review is
`clean-pass` (round 1). Local implementation/evidence checkpoint: `d103c86fc35f59f9d10f1e73daef969f92ffe258`.

This is stage5 for `Qwen/Qwen3.8-Flash-Next`, starting from accepted multichip
checkpoint `79582226b281a19aba1cc5ee3dd4fb80aa87317c`. It runs the actual1×4
Blackhole p300c mesh, devices0–3, p300_x2 descriptor and ring fabric. Attention
remains TP4 and routed experts EP4. No single-chip/replicated fallback, full-model,
vLLM, generation or serving work is included.

## Final default performance

All times below are **milliseconds, before → final default**. Prefill is warmed
eager; decode is warmed blocking trace replay, with input copies/state reset
outside timing. Runs are serialized and separate from watcher/profiler/tracking.
B32 is the existing serial-request attention API with grouped MLP execution;
it is not a concurrent-serving result. The table uses the same32 distinct
checkpoint embedding inputs, reference files and workload options before/after.
Layer0 inputs are actual first-layer HF expansion; later-layer use is real
embedding routing stress, not a claim of recorded full-stack residuals.

| Layer | Prefill33 | B1 decode | B32 decode | B32 improvement |
|---|---:|---:|---:|---:|
| L0 GDN | 21.678 → 21.417 | 2.2673 → 2.0816 | 57.349 → 52.452 | 8.54% |
| L1 PLE/GDN | 26.659 → 25.955 | 3.4332 → 3.1367 | 95.014 → 87.054 | 8.38% |
| L3 QSA | 79.821 → 41.625 | 4.7538 → 2.8589 | 131.110 → 71.598 | 45.39% |

Sources: `real_mesh_before_l*.json`, `real_mesh_final_l*.json` and their command/
source provenance. These numbers come from the final constructors, not the earlier
candidate wrappers. [Performance summary](performance_summary.json) retains the
original fixture baselines and candidate inventory; [candidate CSV](candidate_summary.csv)
and [topology audit](operation_topology_audit.md) contain measured alternatives.

Paired original-fixture chunk-boundary and longer-prefill measurements:

| Layer / logical tokens | Prefill | B1 decode |
|---|---:|---:|
| L0 / 32 | 19.128 → 18.861 | 2.2515 → 2.0658 |
| L0 / 1024 | 597.364 → 591.025 | 2.2727 → 2.0822 |
| L1 / 32 | 23.244 → 22.832 | 3.4483 → 3.1565 |
| L1 / 1024 | 732.860 → 720.543 | 3.4635 → 3.1674 |
| L3 / 32 | 75.530 → 39.118 | 4.7674 → 2.8707 |
| L3 / 1024 | 2495.243 → 1333.318 | 4.8646 → 2.9726 |

Sources: `before_mesh_l*_n*.json`, `final_mesh_l*_n*.json`. Public logical lengths
need no alignment. The final32769-token context smoke is correctness/reservation
evidence; its cold wall time is not substituted for warmed latency.

## Selected implementation

Parallel ordered QSA indexer partials preserve the original FP32 accumulation
order and BF16 output exactly in component tests for all Q1..32 across six HF
windows. PLE striped lookup pipelines32 independent reads per barrier, retaining
exact values and existing scratch storage. All selected projection weights are
prepared during construction; no host conversion or exception fallback occurs
in decoder execution.

| DRAM role | Weight / fidelity | Activation cores / readers per bank / K block | Packing |
|---|---|---|---|
| GR injection | BF16 / HiFi4 | 8 / 3 / 5 | One local scalar projection |
| GDN Z/A/B | BF16 / HiFi4 | 8 / 2 / 2 | Compact logical1560 output |
| Shared gate/up | BFP8 / HiFi4 | 8 / 3 / 5 | Packed; one representation across phases |
| QSA QGate/K/V | BFP8 / HiFi4 | 8 / 3 / 2 | Compact logical3584 output |
| QSA WO | BFP8 / HiFi4 | 8 / 3 / 2 | One rank-local projection |

`tt/multichip_decoder.py:MESH_DRAM_POLICIES` owns these defaults. Projection
activations are BF16, temporarily width-sharded in L1; consumers receive BF16
DRAM output. Remaining QSA attention uses HiFi2 independently of HiFi4 QKV/WO.
Selected groups share immutable prepared weights across Q1 and Q2..32 and own
internal padding/slicing/chunking for larger or unaligned logical inputs.
Inherited routed-expert BFP4/LoFi remains the previously accepted physical-fit
policy. This optimization pass does not newly select BFP4; precision candidates
are evidence for later datatype selection.

## Correctness, trace and watcher

Whole-output PCC threshold remains0.995. Per-row correlations are retained as
diagnostics. The inherited batch probe's per-row0.995 acceptance expression was
corrected to the explicitly requested whole-output contract; finite/state/trace/
tail/stack checks remain unchanged. Final comparisons retain the accepted
single-chip TTNN reference and its existing HF correctness evidence; no new HF
full-model accuracy result is claimed.

| Layer | Short minimum whole PCC | All-tail minimum | B32 whole PCC | Two-layer stack PCC |
|---|---:|---:|---:|---:|
| L0 | 0.99974376 | 0.99954396 | 0.99889207 | 0.99635768 |
| L1 | 0.99973601 | 0.99958992 | 0.99969959 | 0.99877685 |
| L3 | 0.99962932 | 0.99958563 | 0.99983996 | 0.99939001 |

`final_mesh_l*.json` passes16 changing-input B1 and16 alternating-batch B32
replays against exact own-eager outputs and mutable state. PLE token words also
change. Inputs/references are prepared before capture; only fixed buffers are
refreshed during replay. Allocation tracking includes program-cache allocations.
Runtime guards prohibit Torch calls and TTNN host conversions and enable fallback
exceptions. All32 logical tails and the next-layer boundary pass.

`watcher_mesh_l*.json` repeats those paths with watcher10, separately from the
profiler; all three runs pass. Exact all-shape component evidence and unchanged
final custom-kernel source bindings are indexed in
`component_final_source_consistency.json`. The allocation tracker positive control
is a separate diagnostic, not multichip model-performance evidence.

## Residual and context contract

Each rank returns one entire BF16 hyperconnection stream, local
`[1,1,T,2560]`, TILE/DRAM interleaved. Pass this tensor directly to the next layer.
There is **no gather, reshard or all-reduce between decoder layers**; internal GR
and TP/EP collectives stay at their consuming operations. The structured
[boundary contract](inter_layer_residual_contract.json) records ownership,
logical-row handling and future stack requirements.

The [context contract](../context_contract.json) retains262144 autoregressive
context and BF16 page32 KV ownership/layout. Selected DRAM padding adds
506,757,120 bytes/rank across48 layers, including96 GR injections. The final
resident reservation target is31,710,200,640 bytes/rank. Both inherited runtime
planning envelopes remain:1GiB default and2GiB stress. The latter leaves
321,046,720 bytes before IO, so bounded streaming stack IO remains necessary.
No two-whole-slab maximum-context fit or measured full-model runtime claim is made.

The32769-token public prefill plus continuation smoke passes for PLE/GDN and QSA
under that reservation, with whole-output minima0.99984349 and0.99917812.
`final_mesh_l{1,3}_long32769.json` records actual bank allocations and state checks.
This is a decoder smoke with full-stack reservation filler; no full model is
instantiated. KV format did not change, so the stage correctly uses32K–64K smoke
rather than repeating maximum-context runs. See [memory calculations](final_context_memory_plan.md).

## MoE and rejected options

Gate-selected active-expert execution remains in place; no dense all-expert
path was introduced. Final isolated traces of actual decoder MoE intermediates:

| Layer | B1 MoE ms / share | B32 MoE ms / share | B32 expert union / distinct routes |
|---|---:|---:|---:|
| L0 | 0.5425 / 26.06% | 2.1678 / 4.13% | 103 / 32 |
| L1 | 0.5075 / 16.18% | 2.4446 / 2.81% | 151 / 32 |
| L3 | 0.5299 / 18.54% | 2.5566 / 3.57% | 169 / 32 |

The shares include router, active EP experts, shared expert, scores/reduction
and final CCL. They are isolated component traces divided by matched whole-step
latency, not a claim that component traces sum to the whole critical path.

| Family | Evidence and disposition |
|---|---|
| Async/persistent ordered gather | Whole GDN slower than synchronous control; global BFP8 also slower. Explicit buffers/semaphores were measured. |
| Hidden residual + native RS/fused AG-MM | Whole QSA B32 +1.90%/+1.77%; both GR blocks and MoE consumed hidden slices without immediate old-contract restore. |
| Fused MM-RS / AG-MM producer families | Legal shapes/padding measured through residual write and entire next GR at HiFi4/LoFi; best-control traces checked against own eager. Whole-QSA rejection is distinct from component gains. |
| DRAM geometry/packing | Five roles,8/16 shards,1/2/3 readers, legal blocks, packed/split, dtype/fidelity sweeps. QSA block40 L1 failure adapted to legal blocks; GR block40 separately measured slower. |
| GDN padding/concat overhead | First integrated candidate regressed; compact host packing removed the overhead and passed whole decoder. |
| Activation/subgroup CCL BFP8 | Real-input trials pass but give no material matched-path benefit; extra quantization not selected. |
| BFP8 KV | Exact selected-row unpack and whole-decoder checks pass, but B1+12.09% and B32+0.49%; BF16 retained. |
| BFP4 attention | Real short minimum whole PCC0.96450 and B32whole0.98230 fail; not selected. Shared BFP4/LoFi passes but provides no material gain and remains stage8 evidence. |

The starting [operation-topology audit](operation_topology_audit.md),
[anomaly ledger](anomaly_ledger.md), [work log](work_log.md), candidate CSV and
individual JSON/log/provenance files distinguish errors, fixes, measured slower
families and untried extensions. A first TTNN/API error was not used to dismiss a
material optimization.

## Profiles, stop rule and reproduction

Advice-enabled final prefill/decode tables, CSVs and raw provenance are complete
under `tracy/final_mesh_l*`. [Profiler findings](profiler_findings.md) map advice
to measured candidates and stop-rule handoffs. [Policy readback](measured_policy_rows.json)
checks the actual selected dtypes, fidelity and DRAM geometry. Segmented warmed
eager profiler rows explain operator costs; their drain gaps are not headline
traced latency.

Optimize Stop Rule2 fired: the last retained QSA refinements improved B32 by
0.5865% (compact output) and1.2193% (remaining attention HiFi2). All untried
performance extensions are recorded under **Not tried (stop rule)** in the work
log and [handoff notes](stop_rule_handoff.md). This stops tuning, not required
correctness, review or commit work. Full-model-only LM-head, sampling, text
quality,128-in/128-out generation and serving metrics are outside decoder scope;
no full-model FUNCTIONAL throughput claim is made.

Exact commands/environment/source revisions are in `plans/*.json` and each
`*_provenance.json`. Reproduction entry points from the repo root:

```bash
python_env/bin/python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_multichip_stage_jobs models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/plans/final_all.json
python_env/bin/python models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/final_profile_reports.py
python models/autoports/qwen_qwen3_8_flash_next/doc/optimized_multichip_decoder/collect_final_evidence.py
```

Device commands are serialized and bounded. The independent [stage review](stage_review_round1.md)
returns **clean-pass**, with no required work. Local checkpoint SHAs are recorded
in the work log. No push is authorized or performed.
