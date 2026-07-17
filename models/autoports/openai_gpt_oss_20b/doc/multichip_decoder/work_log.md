# GPT-OSS-20B multichip decoder work log

## Scope and baseline

- Branch: `mvasiljevic/model/openai-gpt-oss-20b`.
- Starting HEAD: `3869a0cb95710a55a462081626fc5eeb963a1930`.
- Optimized decoder baseline: `9949cb70f3f82cde84cd725d864bdb092c97ea62`.
- Stage scope: `tt/multichip_decoder.py`, its test, and model-local docs only.
- Explicitly out of scope: full model, generator, vLLM, and serving.
- Pre-existing unrelated edits preserved and excluded from this stage:
  `.agents/skills/forge-functional-decoder-from-ir/SKILL.md` and
  `.agents/skills/multichip/SKILL.md`.

## Hardware and pre-coding decision

`timeout 60 tt-smi -ls --local` found UMD devices 0 and 1, both Blackhole
`p300c`; no other TT devices were available.  A 1x2 P300 ring with TP=2 was
selected before the final implementation.  The complete tensor, activation,
KV-cache, collective, sparse-expert, padding, capacity, and rejected-alternative
plan is in `README.md`.

The runtime repeatedly warns that `/dev/shm` has about 17.5 MiB available for a
16 MiB MPI segment and that motherboard `B850M-C` is unknown.  Both devices
reported healthy DRAM and zero uncorrected GDDR errors throughout final runs.

## Implementation ledger

1. Added a fixed `1x2` `MultichipDecoder(OptimizedDecoder)` and validated mesh,
   GPT-OSS geometry, cache length, and page-table shapes at construction/runtime.
2. Reordered packed QKV shards as `[Q_rank,K_rank,V_rank]`; local ownership is
   32 query heads and four KV heads.
3. Row-sharded O and expert-down K dimensions; rank 0 owns the real bias and
   rank 1 owns zeros so each bias is added once before reduction.
4. Kept router/norm/rotary/page table replicated and expert gate/up column
   sharded over the 1,440-wide local intermediate.
5. Kept the generic exact top-4 `Experts` path for decode.  Replaced the
   prefill grouping path with token-specific sparse batches: gate/up use
   `[S,1,1,2880]`, down uses `[S,32,1,1440]`, and all three receive exactly
   `4*S` active entries.  No dense all-expert runtime path exists.
6. Added 64-token paged local caches `[2048,4,64,64]` per rank at maximum
   context and arbitrary logical-to-physical page-table permutations.
7. Selected explicit sliding decode masking and local-head Q chunk 32 after
   native/explicit comparison; kept public non-aligned logical lengths.
8. Applied 10-core L1 layouts to both decode RMSNorms.  Prefill keeps the
   compiler-prior 45/80-core QKV layout; the final measured decode QKV layout
   is 30 input cores `[32,96]`, 40 output cores `[32,64]`, K block 3, N block
   2, subblock 2, and physical `11x4` extent.  O uses 90 cores / `11x9`.
9. Added independent-cache trace capture/replay with every replay source
   preallocated before capture.
10. After stage-review/AutoFix, split expert programs by mode: prefill uses
    15/30 cores with `1x3` subblocks; decode keeps 45/90 cores with `1x1`
    subblocks.  This combines the measured prefill and decode winners.

## Correctness and capacity commands

Baseline captures use separate device sessions to avoid overlapping mesh
handles:

```bash
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_synthetic_single_chip_optimized_reference
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_real_weight_single_chip_optimized_reference
MULTICHIP_REAL_SEED=3301 pytest -q 'models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_real_weight_single_chip_optimized_reference[blackhole-sliding_attention-mesh_device0-device_params0]'
```

Final TP gates:

```bash
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_real_weight_prefill_decode_matches_single_chip_optimized --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/sharded_input_norm_correctness.junit.xml
pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_warmed_trace_replay_updates_hidden_position_and_paged_cache --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/final_trace_cache_determinism.junit.xml
RUN_MULTICHIP_CONTEXT=1 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_full_context_cache_allocation_and_last_page_update --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/full_context_131072.junit.xml
```

Results:

| Gate | Result |
| --- | --- |
| synthetic S=17/S=33 | PCC 0.99999965 / 0.99999970; local reversed-page cache slices PCC 1.0 |
| exact active prefill control, S=17 | 68 token-specific active entries in each sparse matmul; dense group count refuted |
| real sliding prefill | PCC 0.99936591 |
| real sliding decode, positions 128/129/130 | final PCC 0.99918230 / 0.99909930 / 0.99934228; min attention 0.99999608 |
| real full prefill | PCC 0.99938529 |
| real full decode, positions 128/129/130 | final PCC 0.99917040 / 0.99887053 / 0.99899954; min attention 0.99990406 |
| trace, both layer kinds | PCC 1.0 at positions 128-130; five bit-identical repeats; eager/trace K/V physical pages bit-identical on both ranks |
| maximum context | physical `[2048,4,64,64]` K/V allocation and decode update at position 131071 pass |
| stack layout | first output feeds a second decoder call directly as replicated `[1,1,1,2880]` |
| runtime audit | no host/torch fallback in runtime methods; sparse `Experts` call retained |

The consolidated final gate is retained in
`logs/final_correctness_trace_context.junit.xml`: 9 passed, 13 deselected.
The deterministic near-tie diagnostic uses seed 3301 at position 129.  TP
attention remains near-exact, but legitimate reduction rounding swaps the
fourth/fifth expert.  Exactly four experts remain active on repeated routing.
Feeding exact baseline attention through the TP router/expert path gives route
PCC 1.0 and sparse-expert output PCC 0.99956265.  Canonical gates use declared
layer-specific seeds rather than hiding this discontinuity.

## Trace and watcher recovery ledger

The first trace harness interleaved eager dispatch/allocation with an active
trace and reused one mutable cache for eager and trace.  It was fixed by
preallocating inputs before capture and using independent eager/trace caches.

An all-in-one watcher run passed its first test body, then firmware 19.8.0 left
ACTIVE_ETH core `28-25` without heartbeat during fabric teardown/reinit.  The
exact pytest process was stopped, UMD devices 0 and 1 were reset with
`timeout 180 tt-smi -r 0 1`, and post-reset DRAM/heartbeat checks passed.  A
single watcher test again passed its body but aborted during ACTIVE_ETH
teardown.  The repo-supported scoped workaround was therefore used:

```bash
TT_METAL_WATCHER=10 TT_METAL_WATCHER_DISABLE_ETH=1 RUN_MULTICHIP_CONTEXT=1 timeout 1800 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py -k 'runtime_contract_and_fallback_audit or active_prefill_uses_four_token_specific_sparse_entries or synthetic_non_aligned_prefill_matches_optimized_and_cache_is_head_local or real_weight_prefill_decode_matches_single_chip_optimized or near_tied_router_isolated_to_tp_attention_rounding or warmed_trace_replay_updates_hidden_position_and_paged_cache or full_context_cache_allocation_and_last_page_update' --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/watcher_clean_workers_final.junit.xml
```

Result: 9 passed, 14 deselected, exit 0.  Worker/dataflow watcher remained active
on both devices; only Ethernet-core inspection was disabled.  Retained
`logs/watcher_workers_final.log` has no error/assert/hang match.

The final selected-hybrid watcher run ends with watcher detach from both
devices and clean cluster closure.  Its post-success nanobind reference-leak
diagnostic is classified as framework binding teardown noise, not a device or
decoder failure.

The first stage review requested an explicit post-reset recovery ledger.  The
bounded stage-local control opens, validates, closes, reopens, and closes the
target `MeshShape(1,2)`:

```bash
RUN_MULTICHIP_RECOVERY_SMOKE=1 timeout 300 pytest -q -s models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_recovery_mesh_open_close_smoke --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/device_recovery_mesh_open_close.junit.xml
COLUMNS=240 tt-smi -ls --local
timeout 60 tt-smi -f models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/device_recovery_tt_smi_snapshot.json --snapshot_no_tty
```

Result: mesh smoke 1 passed, exit 0; both P300c devices are listed and
reset-capable.  The final snapshot reports healthy DRAM, zero corrected and
uncorrected GDDR errors, firmware 19.8.0.0, and equal advancing heartbeats.
Failure-time `retrain_count=2` on Device 1 e0,4 is classified as the historical
ACTIVE_ETH watcher/fabric teardown anomaly.  The installed tt-umd ABI prevents
the current triage script from rereading that register, so it is not claimed
to reset to zero; subsequent 9-case worker watcher, repeated fabric workloads,
double mesh open/close, and final telemetry control it as non-active.  Exact
commands, board mapping, and classification are retained in
`logs/device_recovery_final.txt`.

## AutoFix ledger

`$autofix` was invoked for the difficult decode correctness failure and again
for every finding from the first independent stage review.  It used forked
agents as authorized.

| Hypothesis / finding | Isolated result |
| --- | --- |
| QKV local chunk order | proven root cause; `[Q_r,K_r,V_r]` fixed the large decode error |
| 10-wide QKV program on 11-wide physical shard map | proven root cause; `11x8` fixed grid/shard mismatch |
| native vs explicit sliding mask | refuted as cause; explicit was slightly closer and retained |
| identity page table | refuted; reversed physical pages pass |
| column-parallel O | refuted for the required output contract |
| replicated full O | refuted; no correctness improvement |
| disable optimized layouts | refuted |
| FP32 top-k | unsupported by TTNN |
| FP32 attention all-reduce | refuted |
| remaining seed-3301 mismatch | isolated to deterministic sparse top-k discontinuity; exact-attention router/MoE controls pass |
| generic prefill `Experts` grouping | proven incorrect for per-token top-4 semantics; replaced by token-specific sparse batches and a controlled `4*S` activity test |
| fused `moe_compute(compute_only)` | refuted: rolling two-expert double buffer overwrites earlier active experts |
| fused `moe_compute` full mode | refuted: singleton dispatch mesh has no Linear neighbour and Ring rejects self-neighbour |
| `moe_gpt` fused kernel | refuted: hardcodes I=2880 over 12 cores and cannot consume TP-local I=1440 |
| decode post-attention norm one-core path | proven performance finding; moved to advisor 10-core shard and explicit interleaved expert boundary |
| 80-core decode QKV | proven performance finding; precision-locked A/B sweep selected 30 input / 40 output cores, K3/O2/sub2 |
| final exact-active expert geometry | uniform 45/90 `1x1` is slower for prefill; uniform 15/30 `1x3` is slower for decode; selected mode-specific hybrid |

The fused MoE alternatives need a new C++ TP-local-combine kernel, outside the
stage-owned Python decoder/tests/docs scope.  AutoFix therefore exhausted the
in-scope repair space: the final exact active-prefill path is correct but pays
one physical 32-row M tile per token/expert route.  The near-tie diagnostic
remains explicit and non-gating; canonical real-weight PCC gates pass.

## Residual topology comparison

The multichip skill requires a measured contract that does not immediately
reconstruct all-reduce.  The final probe carries O output through
reduce-scatter, a local residual add, distributed post-attention RMSNorm, a
row-sharded router with a 32-logit FP32 all-reduce, and only then a BF16 hidden
gather for the sparse gate/up input:

```bash
RUN_MULTICHIP_TOPOLOGY_PROBE=1 MULTICHIP_TOPOLOGY_REPEATS=20 timeout 1800 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_sharded_residual_topology_candidate --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/residual_topology_candidate.junit.xml
```

Result: distributed norm PCC 1.0, routing PCC 1.0; replicated contract
0.529831 ms, sharded contract 0.733042 ms.  The sharded candidate is 38.4%
slower and still must gather for generic sparse gate/up.  The replicated
residual remains the measured winner.

## Final wall-clock performance

```bash
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=128 MULTICHIP_DECODER_PREFILL_REPEATS=10 MULTICHIP_DECODER_TRACE_REPLAYS=100 timeout 1800 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_capture_single_chip_optimized_perf_reference --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/perf_final_single_chip_wall.junit.xml
RUN_MULTICHIP_DECODER_PERF=1 MULTICHIP_DECODER_PERF_SEQ=128 MULTICHIP_DECODER_PREFILL_REPEATS=10 MULTICHIP_DECODER_TRACE_REPLAYS=100 timeout 1800 pytest -q models/autoports/openai_gpt_oss_20b/tests/test_multichip_decoder.py::test_multichip_decoder_perf --junitxml=models/autoports/openai_gpt_oss_20b/doc/multichip_decoder/logs/perf_final_multichip_wall.junit.xml
```

| Path | 1x1 optimized | 1x2 TP | Speedup | Efficiency |
| --- | ---: | ---: | ---: | ---: |
| S=128 prefill | 13.453845 ms | 49.858989 ms | 0.269838x | 13.492% |
| warmed traced decode | 0.975634 ms | 0.690026 ms | 1.413908x | 70.695% |

The exact token-specific active-prefill repair supersedes the earlier fast but
semantically wrong grouped-expert value.  Its measured regression is retained,
not hidden: sparse matmul can select a batch only, so every active token/expert
route costs a 32-row physical tile.  Decode uses the existing exact top-4
Experts path and improves to 1.414x.

Final-policy expert geometry A/B, all with exact token-specific top-4,
BFP8/LoFi, 10 prefill repeats, and 100 trace replays:

| Program | Prefill | Decode | Result |
| --- | ---: | ---: | --- |
| uniform 45/90 cores, `1x1` | 50.696530 ms | 0.690262 ms | prefill loser |
| uniform 15/30 cores, `1x3` | 49.854757 ms | 0.693085 ms | decode loser |
| mode-specific hybrid | 49.858989 ms | 0.690026 ms | selected |

Artifacts are `multichip_perf_expert_prefill_width1_seq128.json`,
`multichip_perf_expert_subblock_candidate_seq128.json`, the selected result
JSON, and matching `perf_final_exact_active_expert_*.junit.xml`.  The selected
geometry also passes the consolidated 9-case real PCC/trace/context suite.

The final precision-locked QKV A/B sweep kept BF16/HiFi4/FP32 destination:

| Geometry | 10-replay decode |
| --- | ---: |
| old default | 0.712398 ms |
| K3 / K5 / K10 | 0.694127 / 0.691108 / 0.691703 ms |
| O2 / O4 | 0.697210 / 0.695538 ms |
| K3O2 / K5O4 | 0.690411 / 0.690858 ms |

The 100-replay confirmation was K3O2 0.690263 ms, K5O4 0.690447 ms, and K5
0.690927 ms.  K3O2 was selected.  Tracy shows QKV falling from about
60.5-61 us / 80 cores / 243 GB/s to 40-41 us / 40 cores / 362-365 GB/s.
Canonical correctness and trace tests passed with the selected geometry.

## Profiler and shard-advisor provenance

Single-chip Tracy source:
`generated/profiler/reports/2026_07_17_07_54_54/ops_perf_results_2026_07_17_07_54_54.csv`.
Final multichip Tracy source:
`/tmp/gpt_oss_multichip_hybrid/reports/2026_07_17_10_02_15/ops_perf_results_2026_07_17_10_02_15.csv`.
They are copied as `perf/single_chip/ops.csv` and
`perf/multichip/ops.csv`.

`tt-perf-report` text and CSV were generated for all four signpost ranges with
`--active-experts 4`.  The retained 1x1 profile totals are 13,355 us prefill
and 952 us decode.  The final TP2 profile totals 49,770 us for one prefill and
1,984 us for three traced decode executions (about 661.3 us/replay).  TP2
prefill CCL is 121 us; a representative decode replay is about 48 us.  The
prefill bottleneck is 27.197 ms of sparse projections plus 8.431 ms typecast,
7.059 ms reshape, 3.948 ms unary, and 2.037 ms fill/pad work.  The final
prefill sparse rows use valid `1x3` output subblocks and 15/30 cores.
Human tables, advice, CSV, stacked breakdowns, and raw op provenance are
retained under `perf/`.

The stage reran `ttnn-advise` as required, but the installed tt-mlir copy is
ABI-skewed from this checkout and its `_ttnn.so` fails import on undefined
`ttnn::experimental::moe_compute`.  The setup contract forbids rebuilding that
toolchain in the experiment directory.  Exact stderr is retained at
`shard_advise/rerun_stderr.log`; the optimized baseline's valid
`baseline_report.json` and `baseline_final_ir.mlir` are copied into this stage
and support the selected 10-core norms, `11x8` prefill QKV, and `11x9` O
layouts.  The decode-specific `11x4` QKV geometry is the later measured A/B
winner recorded above.

## Overlapping-handle triage

One investigation intentionally serialized a hang caused by opening/running a
single-chip submesh while its parent multi-device handle remained active.
`triage/serialized-overlap-repro.md` records the reproduction, and the retained
triage files contain host/device evidence.  The production tests avoid this
unsupported lifetime pattern by running baseline capture in a separate pytest
process; repeated serialized runs are clean.

## Review and local commits

The first independent `$stage-review` returned `more-work-needed` for two
evidence gaps: the expert subblock candidate predated exact-active prefill, and
the post-reset device ledger lacked an explicit mesh smoke/final snapshot.
AutoFix produced the selected mode-specific expert hybrid and the retained
three-way A/B; the TT-device recovery control then added double mesh
open/close, final device list/telemetry, and retrain classification.

A fresh independent rereview returned `clean-pass` with no required work.  It
confirmed the final selected defaults, 9/9 correctness/trace/context, 9/9
worker watcher, full context contract, current profiler rows, recovery smoke,
and stage-only commit scope.  Local checkpoint SHAs are recorded below after
creation; no push is performed.

- Implementation and evidence clean-pass checkpoint: `8ef1520f95abddb4a0f8220bad079f375dc66afa`.
- Provenance documentation checkpoint: this follow-up commit; no push.
