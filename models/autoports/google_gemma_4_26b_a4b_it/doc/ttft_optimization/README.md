# Gemma-4 post-pipeline TTFT optimization

Status: local validation and independent review pass. Remote qualification is
blocked on vLLM repository write access; see [publication evidence and required
direction](publication.md). No remote success or overall completion is claimed.
This is a separate post-pipeline optimization stage.

## Current measured result

The selected **latency-priority synchronous profile** measures **95.24 ms warmed
S128/O128/C1 TTFT**, versus 419.10 ms before changes (77.3% lower). Two complete
C1 repeats give 95.36 and 95.03 ms. This meets the narrowly scoped warmed primary
target; it is not a guarantee for every prompt, percentile or first-use request.

For that first selected S128/O128/C1 cohort, TTFT median/P99 is
95.241/95.285 ms; TPOT mean/P99 is 20.373/20.768 ms; ITL median/P99 is
20.125/23.163 ms. Mean whole-request latency is 2682.509 ms, output throughput
47.712 tokens/s, and decode rate `1000 / mean TPOT` is 49.084 tokens/s. The latter
is not whole-request output throughput. Three measured requests follow ten
explicit warmups; no serving device time or roofline was measured.

The tradeoff is explicit: C1 decode TPOT rises from about 19.1 ms with async to
20.3–20.5 ms with sync. The separately supported **async throughput profile**
measures 103.38 ms at S128/O16 and 104.29 ms at S128/O128, with essentially
unchanged decode TPOT. It does not meet the 100 ms target in those cohorts.

All results are native streaming HTTP end-to-end measurements, not estimates
from reduced layers or device-only traces. Context remains 262144, max sequences
32, block size 32, selected precision `head4_inner_all4_shared_down4`, canonical
sampling, pinned HF/tokenizer revision
`4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Prefix caching and chunked prefill stay
disabled. No live-serving Tracy/device profiler was used.

Median TTFT, milliseconds; every cohort is retained:

| ISL / OSL / C | Original | Optimized async | Sync first matrix | Sync repeat 1 / 2 |
| --- | ---: | ---: | ---: | ---: |
| 32 / 16 / 1 | 384.323 | 58.343 | 65.087 | 48.974 / 48.763 |
| 33 / 16 / 1 | 395.460 | 75.329 | 76.606 | 61.662 / 63.082 |
| 64 / 16 / 1 | 396.419 | 73.741 | 80.893 | 65.789 / 65.649 |
| 127 / 16 / 1 | 428.949 | 122.988 | 119.404 | 97.888 / 96.516 |
| 128 / 16 / 1 | 424.985 | 103.379 | 107.186 | 97.385 / 95.592 |
| 129 / 16 / 1 | 435.970 | 116.725 | 111.417 | 111.109 / 109.500 |
| 256 / 16 / 1 | 469.483 | 177.567 | 174.165 | 157.074 / 156.897 |
| 128 / 128 / 1 | 419.105 | 104.294 | 95.241 | 95.357 / 95.028 |
| 128 / 32 / 8 | 3745.727 | 3709.733 | 1001.795 | — |
| 100 / 100 / 32 | 14971.166 | 14860.140 | 4124.106 | — |

C1/O16 has 10 measured requests per cohort, O128 has 3, C8 has 8, and C32 has
32. Original C1 uses three explicit native warmups; final C1 uses ten. Original
occupancy has zero warmups; final occupancy warms a full 8/32-request burst.
Native warmups repeat prompt 0. Repeating the full C1 suite also changes process
history and exposes all measured prompts again. The first O16 cohort is not
silently replaced: its S128 median is 107.19 ms. Repeated O16 P99s are
100.28/102.07 ms. S129/S256 remain above 100 ms. Prefix caching stays disabled;
no claim is made that every previously unseen prompt meets the target.

The [matched async control](matched_async_comparison.json) uses identical
warmup counts and original-path restoration switches. Its S128/O16 is 455.92 ms,
but the smaller original baseline remains the conservative headline reference.
Matched C8 optimized async is 2.38% slower in TTFT and 1.00% slower in mean
whole-request latency; C32 whole-request performance is effectively unchanged.

Sync moves first-decode capture into ITL at higher occupancy; it does not remove
that work. Compared with optimized async, C8 mean whole-request latency is
9301.90 versus 9223.29 ms (+0.85%), and C32 is 82197.89 versus 81872.42 ms
(+0.40%). Output throughput falls by the corresponding roughly 0.85%/0.39%.
The larger TPOT increases (C8 271.06 versus 188.65 ms, C32 789.83 versus 680.29 ms)
must be read with this phase shift, not hidden behind the lower TTFT.

Raw cohorts live under `../../readiness_vllm/ttft_optimization/`. Machine
comparisons preserve full texts, lengths, counts, P99, TPOT, E2EL and throughput:
[original versus final](final_sync_original_comparison.json),
[scheduler tradeoff](final_scheduler_comparison.json), and
[repeat 1](acceptance_sync_repeat1_comparison.json) /
[repeat 2](acceptance_sync_repeat2_comparison.json). All 26 sync workload rows
match optimized async outputs and lengths exactly; no requests failed.

The shared suite is a different, zero-warmup protocol: async primary TTFT is
711.19 ms; selected sync is 147.95 ms with mean TPOT 24.49 ms. The native CLI
explicitly skips its optional readiness request when its timeout is zero.
These results remain in [perf_summary.json](perf_summary.json), not overwritten
by warmed measurements. Exact first-use phase attribution is not claimed.
The [experiment ledger](experiment_ledger.md) records the protocol correction,
initial/repeated host-delay evidence, and every rejected candidate.

## Serving profiles

The new `gemma4-autoport` deployment catalog selects `--no-async-scheduling` for
the latency-priority workflow. Canonical Gemma defaults are not replaced.
The local launcher retains the inherited async default; select the measured
latency profile explicitly:

```sh
python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_server.py --no-async-scheduling --output <latency-artifacts>
python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_server.py --output <async-throughput-artifacts>
```

Use the compatible serving container described below, with exclusive device
ownership. Exact executed commands and artifact paths are in [work_log.md](work_log.md).

## Retained implementation

- Generator-owned, one-entry exact-length B1 prefill plus canonical first-token
  sampling traces, bounded to 1..1024 logical tokens. The bound is an optimization
  bound, not a public context cap. Other batches, lengths and sampling modes use
  the existing general path.
- Persistent trace input/output storage; token refresh every request and
  page-table refresh only when contents change. Every logical page-table column
  is retained. Cache ownership, table shape and exact input length invalidate
  incompatible state.
- Compatible decode-graph reuse across compact-prefill/padded-neutral-decode
  sampling representations. Sampled, penalty, logprob and host paths invalidate
  prepared greedy state; canonical sampling still owns seeds and history.
- Prefill-only requests can warm/capture without entering decode. Later decode
  binding releases/rebuilds the graph before allocating incompatible state.
- Same 32-row expert grouping and dtype/fidelity, with gate K22 and down grid 11x4,
  per-core N2. Decode programs and weights are unchanged. Full-model controls
  measured the gate and down changes separately; boundary layers are bit-exact.

`GEMMA4_PREFILL_TRACE=0`, `GEMMA4_PREFILL_GATE_K=11`, and
`GEMMA4_PREFILL_DOWN_CORES=88` restore the original control path. Experimental
wider grouping remains explicitly opt-in via `GEMMA4_PREFILL_EXPERT_BATCH`;
default 32 allocates no extra ownership-index rows. Wider grouping, shared BF16
geometry and both host-yield experiments are not selected. The yield source is
archived disabled, with no active adapter hook.

The speedup combines less eager prefill dispatch and avoiding repeated
first-decode capture/invalidation at B1. It is **not** a claim that kernels alone
became 4x faster. See [diagnosis](AUTODEBUG.md), [fix audit](AUTOFIX.md),
[experiment ledger](experiment_ledger.md), and [chronological commands](work_log.md).

## Correctness, resource and runtime evidence

Current final geometry passes 115 host tests, 68 isolated real-weight layer rows,
12 eager/24 replay-fallback boundary cases, 12 sampling/reset transitions, and 4
prefill-only cases under worker Watcher and allocation tracking. Includes
non-aligned lengths, changed tokens/pages/cache, B2, and 1023/1024/1025 boundaries.
Raw artifacts: `acceptance_host_tests.log`, `expert_down_geometry.*`,
`watcher_down44.*`. Both async and selected sync full live sampling suites pass
72 tests with one expected all-vocabulary logprobs-cap skip. Selected sync takes
1114.96 s; async takes 1135.72 s. All 18 selected-profile greedy chat replays and
six seed-71 sampled dictionaries match the legacy control exactly. See the
[qualitative verdict](qualitative_verdict.md) for formatting, finish metadata,
matched controls and retained concrete quality limitations.
Existing staged accuracy caveats are not waived by TTFT work.

The full-context direct probe allocates 8192 physical cache blocks per layer.
[Context contract](../context_contract.json) records the bounded additional
prefill state (up to 5,181,568 bytes/chip), unchanged reserve and served limit.
Ethernet Watcher could not fit its firmware config buffer in this image;
worker Watcher passes with Ethernet instrumentation disabled. The initial failure,
bounded recovery and healthy-mesh checks remain in raw evidence.

Local wrapper: `/home/mvasiljevic/gemma4-pipeline/vllm-container.sh`. The image tag
starts 0.21, **but the actual installed engine is vLLM 0.26.0+empty**. The tested
plugin is overlaid from monorepo commit
`7f72b1c6e905f5137fe3377f2e7b42738d3f271d`; the monorepo engine is not on PYTHONPATH.
`runtime_versions.json` and server startup logs attest this distinction. The
compiled TTNN binary is image commit 7aee192, not the new source branch. The remote
workflow will build the optimized TT source and explicitly reproduce engine
and plugin provenance; that build/runtime validation is still pending.

## Publication and cleanup

Dedicated TT branch: `mvasiljevic/gemma4-ttft-opt`, based on `e7ae5f184a`.
Workflow/config compatibility changes have dedicated branches documented in
[workflow compatibility audit](workflow_compatibility_audit.md). Exact local
commit SHAs, normal push results and intended dispatch inputs are in
[publication.md](publication.md). The [independent local review](stage_review_local.md)
is clean-pass; no workflow run URL or terminal conclusion exists yet because
the required plugin commit could not be pushed with the available access.
Unrelated `doc/PIPELINE_INTERVENTIONS.md`, nested checkout ownership and foreign
containers are preserved. Generated caches/raw tensor dumps remain local and
ignored. The final owned API/engine stopped with exit code zero; all four device
nodes and port 8000 are clear (`final_server_cleanup.log`). The holder container
is also stopped; [final local cleanup](local_cleanup.md) records the subsequent
device/port check and untouched foreign containers.

### Evidence byte preservation

Before publication, `publication_snapshot.py` snapshots each compact raw JSON
and log as deterministic `.gz` bytes. Its manifest records the original and
compressed SHA256/size. Repository hooks may normalize whitespace in the
browsable originals; finalization checks JSON type-sensitive semantic equality
or log whitespace-only equality and records the normalized SHA256 separately.
Documentation report hashes refer to the browsable files after normalization;
hashes embedded in original raw metadata remain reproducible from the archives.

Three observer originals exceed the repository's 500 KiB file limit and are
published only as `early_handoff/events.json.gz`, `handoff_async/events.json.gz`,
and `selected_async/events_repeat.json.gz`. Their uncompressed originals remain
untouched locally. For any archive, `gzip -dc path.json.gz` reconstructs its exact
original bytes. From the repository root, verify the published evidence with:

```sh
python3 models/autoports/google_gemma_4_26b_a4b_it/doc/ttft_optimization/publication_snapshot.py verify --root models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization
```

The local evidence set is sealed after validation and owned-server cleanup.
Later remote-workflow evidence belongs in sibling
`readiness_vllm/ttft_optimization_remote/`, not in the sealed local set.
