# Optimized Qwen3.6 decoder

Stage 3 implements `tt/optimized_decoder.py` from the completed fused decoder.
All automated qualification gates pass; independent stage review returned **clean-pass**. The erroneous
initial runner cap was revoked at17:14 UTC on2026-11-13; its FAIL receipt is
historical. This stage does not implement a model stack, multichip or vLLM.

## Selected runtime

| Role | Selected policy |
|---|---|
| Routed experts | BFP8, LoFi, exact8-expert indexed sparse execution; L1 intermediates; gate/up block16 on32cores, down block16 on64cores |
| Grouped expert prefill | BFP8/LoFi; existing chunk/padding and routing contract |
| Shared gate/up/scalar and down | BFP8/HiFi2, packed gate/up/scalar; DRAM-sharded decode |
| Full attention Q/K/V/gate and output | BFP8/HiFi2, packed QKVG, three readers per DRAM bank, K block4 |
| Full decode residual | 32-core width-sharded L1 through norms/adds; explicit MoE-output reshard; sharded return |
| Shared/full prefill projections | Explicit2D8x8, K block8, FP32 accumulation,1x1 output subblocks |
| Linear attention/state/router | Inherited fused ordered BF16/FP32 and HiFi4 exceptions |
| KV cache | Unchanged BF16 paged layout; logical context262144 |

All setup packing/conversion occurs in `from_state_dict`. Superseded constructor
weight copies are discarded. Runtime has no torch/from_torch/to_torch fallback;
the real-weight harness wraps measured forward paths in `device_only`.
Inherited fused attention/state methods are shared deliberately; the optimized
class overrides MoE, projections and the full decode residual path. Tests assert
explicit `OptimizedDecoder` selection rather than permitting functional fallback.

The linear precision exceptions preserve the already repaired recurrent
arithmetic; see the previous stage's endurance evidence and
[topology audit](topology_audit.md). BFP4/LoFi is measured as a candidate only.
Stage8 `$datatype-sweep`, including its reasoning-text precision probe, owns the
final lower-precision choice. Grouped expert prefill does not support BFP4 in
this revision and therefore remains BFP8 in that candidate.

## Measurements and selection

Current-source S256 measurements (serial chip0, no Watcher/profiler/allocation
tracking,10 warmed prefill and60 warmed traced decode samples):

| Layer kind | Prefill fused → optimized (ms) | Traced decode fused → optimized (ms) | Decode improvement | Prefill PCC before → after | Decode PCC before → after |
|---|---:|---:|---:|---:|---:|
| Linear, layer0 |478.106 →468.971 |1.447121 →1.346515 |6.95% |.99999361 →.99997592 |.99999607 →.99993438 |
| Full, layer3 |8.975175 →5.876420 |.583035 →.431675 |25.96% |.99989004 →.99977954 |.99992326 →.99990266 |

Artifacts: `baseline_{0,3}.json`, `final_{0,3}.json` and provenance sidecars.
Final reported numbers supersede screening timings. Full prefill improves34.53%;
linear prefill improves1.91% because the inherited sensitive recurrence dominates.
The largest PCC delta is1.105e-4 in full prefill, consistent with the separately
measured BFP8 attention projection change; all whole outputs remain above .995.
Linear decode's6.17e-5 delta comes from the expert/shared policy; its attention
and state arithmetic are unchanged. No numerical acceptance threshold is lowered.

The final full decode is0.17% slower than the original .4309295-ms screening
measurement of the **same selected decode policy**, and faster than the best
other measured layout (.4320665ms). It reproduces the selected policy within
measurement variation; no older faster screening number is claimed as final.
Final prefill5.876ms likewise replaces the screening2D/block8 number5.600ms.
This remains faster than fused. The final prefill samples range5.592–6.712ms
(SD .336ms), overlapping the screen5.556–6.328ms (SD .329ms); the differing
medians are not evidence of a stable prefill gain over that screen. Decode sample
SDs are .00282ms final and .00164ms screen, versus a .00075ms median difference.

The complete search commands, policies, PCC and warmed timings are in
`sparse_search/search.json`, `sparse_search2/search.json`,
`projection_search_retry/search.json`, `residual_search/search.json`, and
`prefill_search/search.json`. `candidate_ranking.json` excludes failing screens
and BFP4 defaults; full acceptance still requires the final gates. Initial
`projection_search` setup failures were repaired and retried, not used to reject
an optimization family.

The last two kept residual decode changes gained0.716% and0.263%, triggering
`$optimize` stop rule2. The already-started prefill comparison completed and
selected2D/block8. Exact rejected options and untried hand-off notes are in
[work_log.md](work_log.md). Packed full/shared projections beat legal separate
controls, including four expert split geometries; larger DRAM K blocks/readers and sparse core/subblock geometries were
measured under the selected precision. Block64 exceeded physical L1 capacity;
smaller adapted blocks passed. Native SDPA/RMSNorm and active-expert execution
remain in the measured path.

## Correctness and contracts

Acceptance uses the user-specified .995 whole-output PCC, determinism, cache and
trace safety. Per-row PCC remains diagnostic. Late-linear synthetic continuation
at row276 has a .9907 outlier: the unchanged fused control reproduces .99055 at
the identical boundary; both pass whole-output PCC. Precision and prefill-config
controls do not remove it. Actual HF-propagated natural inputs pass every row
at the same boundary (minimum .999873). This is not labeled an HF-known limitation
without an HF-versus-HF control. See
[matched diagnostic observations](continuation_diagnosis_observations.md).

The advertised context remains262144. KV dtype, physical page ownership and
recurrent layout do not change; stage3 uses32K smoke, with32767/32768 logical
prefill and traced decode. Public nonaligned lengths and continuation remain
supported. The internal physical chunks remain64 rows for linear attention and
256 for full attention; the planner's4096 hint does not imply4096-row matmuls.

## Reproduction and artifacts

From this checkout, source `python_env/bin/activate` and `../lane-env.sh`.
Use physical chip0 only; unset `TT_VISIBLE_DEVICES`. All device jobs are bounded
and serial, with `fuser` ownership checks. No reset has been needed.

- `bash models/autoports/qwen_qwen3_6_35b_a3b/tests/validate_optimized.sh`
  runs128/1024-step stress, batch32/boundaries, Watcher,32K context, page guards,
  padded trace reuse, real-weight pytest and clean final/boundary latency.
- `bash models/autoports/qwen_qwen3_6_35b_a3b/tests/qualify_optimized_supplement.sh`
  runs natural layer0/3/34 checks, clean128in/128out timing, matched fused
  boundary timings, default and BFP4 profiler captures.
- `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.summarize_optimized_perf`
  produces compact profiler tables, byte-bound accounting and candidate ranking.
- `python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.check_optimized_evidence`
  fails closed on missing or stale required evidence.

Watcher and allocation-tracking checks run separately from clean timing and
profiling. Trace checks include program-cache allocations and repeated/cross-
request replay. Profiler captures contain16 complete decode replays with drains
outside the timed replay. Both phases have `tt-perf-report` CSV and tables with
advice; same-run profiler host/device numbers are distinguished from clean host
latency. Exact commands and source/environment provenance accompany each result.

Independent review: [stage_review_round1.md](stage_review_round1.md), **clean-pass**.
Final gates: `hard_gates.json`, PASS. Local commit receipt is recorded in
[work_log.md](work_log.md). No push is authorized or performed.

## Final evidence summary

`hard_gates.json` records current-source PASS. Stress covers128 and1024
decode steps, three repetitions each, at layers0/3/34; batch32, page ownership,
nonaligned lengths, repeated trace reuse and separate Watcher runs pass.
`latency_comparison.csv` contains matched short and chunk-boundary before/after
measurements. `validation_summary.json` keeps the detailed PCC summaries.

The clean128-in/128-out decoder timings are1.346337ms for linear and .428768ms
for full attention, reaching15.09% and28.04% of the optimistic logical-byte
bandwidth bound. These are per-decoder-layer metrics, not model throughput.
`perf_summary.json` separates clean host timing from same-run profiled
host/kernel accounting. See [profiler conclusions](perf_conclusions.md) for
actual op costs, dtype/fidelity, advice decisions and chunk geometry.

Profiler tables/CSV live under `tracy/layer_{0,3}/` and
`tracy/bfp4_candidate/`; compact `*_op_groups.csv` are versioned. Large raw
captures and expanded reports remain at their original persistent paths and
are listed in `artifact_manifest.json`; they are not compressed or deleted.
BFP4/LoFi remains a measured candidate for stage8; default weights stay BFP8.

Boundary tradeoff: full-attention S127 warmed prefill increases from41.864672
to43.940496ms (+4.96%); all three optimized samples exceed the baseline
samples. This is a measured regression, not dismissed as timing noise. The
retained configuration wins the headline S256 prefill and traced-decode
comparison. The inherited planner executes S127 as one96-token call plus31
singleton calls; S129 as128+1; S4095 as15×256+224+31×1. The latency CSV
records actual call counts/maxima, including these tails. Further tail-specific
config tuning is a stop-rule handoff.
