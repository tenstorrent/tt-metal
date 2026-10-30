# Optimized decoder — stage 03

Status: complete. Independent stage review: **clean-pass**; see stage_review.md.
Scope is a single-device decoder, starting from this run's fused implementation
at b5284798. No multichip, full-model, or vLLM work belongs to this stage.
The standalone `tt/optimized_decoder.py` does not import a candidate, fused, or
functional implementation. All HF loads pin revision
`842da3794eaa0b77d5f08bae87a17459d91ff475`.

## Selected runtime

One Blackhole ASIC, 11x10 available worker grid and eight DRAM banks. Four ASICs
are locally available; all local device commands are serialized. A capture
failure required triage and owner-checked recovery; AUTODEBUG.md and AUTOFIX.md
record that resolved incident. tt-smi is installed in python_env.

All projection weights use BFP4/LoFi, width-sharded across DRAM banks. Norms and
RoPE use BF16/HiFi4 with FP32 accumulation. Decode SDPA uses HiFi2/FP32 and BFP8 paged KV. Prefill uses HiFi2
streaming accumulation without FP32 destination accumulation. Prefill gate/up outputs and their fused-GELU multiply use BFP8; decode
activations and residual/norm tensors use BF16. Output/down projections pad
5376 output channels to5632 and slice before normalization/residual arithmetic.

| Decode role | Input shard cores | K block tiles | Readers/bank |
|---|---:|---:|---:|
| Sliding packed QKV | 28 | 6 | 2 |
| Full packed QK | 8 | 3 | 2 |
| O | 8 | 16 | 2 |
| Gate and up (separate) | 56 | 6 | 2 |
| Down | 8 | 14 | 2 |

Input shard cores are sender cores, not matmul compute cores. Two readers/bank
use16 compute workers. Residuals remain eight-core width-sharded L1 through
norm/add boundaries. Batches above32 use an explicit 2D matmul with the same
precision because the DRAM-sharded factory accepts one physical M tile.
Sliding SDPA uses11x10 above batch8 to match Q shard coordinates; smaller sliding
batches and full DRAM-Q use8x8. Cache update is fused when disjoint K/V core sets
fit, otherwise separate on-device updates preserve the larger-batch interface.

Prefill uses MinimalMatmul M4/K8/N16, subblock1x8 and11x10 workers, chunks1024
with arbitrary logical tails. For <=1024 total physical token rows, a private L1
input clone permits in-place residual and gate multiply, avoiding L1 collision
without mutating caller input. Larger chunks/batches use DRAM intermediates.
Chunk outputs return to DRAM before retention. Sliding attention bounds its SDPA
page view to the preceding1024 tokens plus the current chunk; RoPE and cache
writes retain absolute positions. Continuations not divisible by128 use32-token
SDPA chunks. Decode uses the largest supported chunk no larger than the preferred
64/sliding or32/full that divides valid page-table capacity, avoiding reads of
unspecified aligned padding. This preserves arbitrary logical sequence lengths.

The262144 context contract passes for both layer kinds (`final_context_v2`),
including262143-token nonaligned and exact262144-token prefill and final-position
decode. This probe validates the selected last32 prefill outputs against HF with
all prefix KV present; it does not claim an all-output comparison at that length.
The final full-attention K32 policy is rerun at maximum context in final_runtime;
sliding attention retains its validated maximum-context signature. See
../context_contract.json and memory_ledger.json.

## Evidence and measurement

The initial operation-topology audit and chronological experiments are in
[work_log.md](work_log.md). `candidate_ledger.csv` and `.json` enumerate measured
configurations, PCC, host timing and exact evidence. Failed candidates remain
visible, including adapted retries. Real-input precision checks supersede random
activation screening; synthetic PCC does not veto a real-weight/input win.
`finalist_combinations` compares packing and L1 adaptations under the cumulative
policy. `final_runtime` exercises the delivered default with no policy or geometry
overrides, watcher10, trace allocation tracking1 and skip-program-cache0.
Both kinds pass135 extended B1 replays and129 B32 replays with distinct requests,
changed inputs/positions/page rows, preserved sentinels, resume offsets32/96/128/
1024 and interleaved prefill while the decode trace remains live. One capture is
retained; no request-time release/rebuild is used. The delivered pytest passes
14 cases in53.15s (`pytest_delivered_final.log`), including1/3/5-page capacities and
batches1/8/16/32 in one process. `final_runtime` additionally reruns logical batch boundaries33/56/sliding
and33/34/36/full with distinct HF activations on the final code.

Host timing JSON is candidate ranking, fixed position4096 with128 submissions.
It is not device telemetry. `tracy_candidates_v2` contains provisional whole-layer
device profiles. Final profiles are tracy_final/sliding_attention and
tracy_final_k32/full_attention; prior tracy_final/full_attention K128 is superseded. Device profiles measure
4096 prefill tokens and128 advancing teacher-forced positions4096..4223. This is
a layer workload, not autoregressive token-out model performance. Final harnesses
load recorded HF activations and actual checkpoint weights.

Whole-layer FW-end minus FW-start includes every operation and internal gap.
Decode averages128 complete replay windows, verified against captured op IDs.
Profiler drains and host input refresh occur between windows. Same-run host
latencies are reported separately. Rooflines use useful model FLOPs, estimated
physical DRAM bytes including BFP exponent headers and active KV, and one ASIC's
theoretical663.552TF/s LoFi /512GB/s peaks. No per-op utilization averaging or
matmul-only denominator is used. `peak_basis.json` records sources/assumptions.

Advice-enabled tt-perf-report tables retain recommendations. Readable decode
tables select one complete median-nearest replay; only its incoming external
boundary gap is removed, with original value recorded. Headline measurements
always use all128 windows. The tool undercounts multi-reader compute cores and
fails to recognize explicit MinimalMatmulConfig; its per-op percentages are not
telemetry. Actual geometry and dtype/fidelity remain in profiler rows.

## Correctness and precision tradeoff

| Kind | Fused prefill PCC | Optimized prefill PCC | Fused decode PCC | Optimized decode PCC |
|---|---:|---:|---:|---:|
| sliding_attention | .99998438 | .99948476 | .99998332 | .99966461 |
| full_attention | .99999228 | .99993394 | .99999609 | .99992828 |

Same pinned weights and recorded2049-token HF fixture; before controls are in
final_watch_baseline, delivered after checks in final_runtime. BFP4/LoFi
projections, BFP8 KV and prefill MLP intermediates explain the precision reduction;
all remain above the functional .995 bar. At maximum context, sliding prefill/
decode PCC is .99951649/.99949031 and full .99752344/.99993241. The full prefill
margin narrows with long context but remains above the unchanged acceptance bar.
B32 final prefill/decode is .99937417/.99957934 sliding and .99994105/.99995227 full.

## Final topology audit

| Boundary | Delivered path and evidence |
|---|---|
| Input/norm/residual | Eight-core width-sharded L1 decode, sharded norms and residual adds; avoids repeated DRAM restoration. |
| Attention projections | One packed QKV/sliding or tied-KV QK/full; attention_packing_0, packed_refined and full_refined compare legal tuned separate paths. |
| Head handling | Optimized head split/concat, norms, RoPE, fused cache update and paged SDPA. Full Q DRAM is measured faster than L1. |
| O/down output | Zero-padding5632 enables two readers; interleave/slice/reshard restores5376 logical channels. Cost included and beats unpadded and6144 alternatives. |
| MLP | Shared input reshard to56 cores, separate gate/up, fused GELU multiply; packed family loses including split/layout cost. |
| Prefill | Explicit MinimalMatmul, BF8 gate/up intermediates, private L1 carried input where it fits, bounded sliding SDPA. |
| Runtime host work | No torch/from_torch/to_torch or host fallback in measured forward; setup-only weight conversion. Harness input refresh remains outside layer windows. |

No collective, routed expert, LM head or sampling exists in this dense
single-device decoder. Their checklist items are inapplicable, not deferred
implementation. Remaining split/slice conversions enforce legal TTNN head or
padded projection contracts; measured lower-movement alternatives are recorded.

Local checkpoint IDs are recorded in work_log.md and checkpoint.json. No remote publication is authorized for this goal.

## Final candidate selection

The default is remeasured using the same recorded4096-token HF activation fixture
and128 fixed-position traced submissions as the candidates. This host-ranking
harness is separate from advancing-position device telemetry.

| B1 candidate | Sliding decode us | Full decode us |
|---|---:|---:|
| Tuned separate attention, final precision/padding | 911.697 | 1052.365 |
| Packed MLP42cores/K8/3readers | 908.571 | 1027.641 |
| Selected packed attention/separate MLP, prior K128 full | 907.620 | 1027.059 |
| Full K64, final policy otherwise | — | 1019.474 |
| K32, final policy otherwise | 916.449 | 1017.571 |

Keep K64/sliding and K32/full. Full B32 K32 also beats K64
(2792.975 versus2796.726us); real4223-token B1/B32 PCC passes. The frozen
L1-inplace predecessor has identical sliding decode math and measures907.395us;
the delivered907.620us differs0.025%, not evidence for a different faster policy.
Delivered sliding prefill improves24.947->22.677ms via bounded/streaming SDPA.
Exact trials, controls and source hashes: real_input_final_ranking, final_decode_k,
final_decode_k32 and final_runtime. No synthetic-only precision rejection is used.

[PERFORMANCE.md](PERFORMANCE.md) records all configured batches, before/after
device times, short/tail host timings, rooflines, same-run host/device accounting
and advice dispositions. At33 tokens, host prefill regresses by0.55ms/sliding and
0.78ms/full; this is the explicit tradeoff of the selected4096-target policy.
All measured1024-token and longer boundary/tail requests improve. Long-context
accuracy compares selected output rows with full prefix KV, not all output rows.

Reproduction commands are recorded in each matrix/profile manifest and work_log.
The final profiler command is `python_env/bin/python -m
models.autoports.google_gemma_4_31b_it.tests.run_optimized_profile_queue --layers 5
--artifact-root tracy_final_k32`.
Use a fresh artifact-root when recollecting; existing manifests guard against
mixing different sources. The full-matrix collector defaults to batches1,32,16,8.
Raw profiler captures/logs and weight/activation fixtures remain local and ignored;
compact results, report tables/CSVs, source hashes and exact commands are retained.
