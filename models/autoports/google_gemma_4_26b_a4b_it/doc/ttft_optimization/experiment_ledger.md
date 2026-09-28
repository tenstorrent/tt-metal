# TTFT experiment ledger

Status: **not final accepted**. This is an intermediate evidence snapshot after
the full-model expert geometry sweep and before final native serving validation. Artifact
names such as `final_default` are historical experiment labels, not acceptance.
No new hardware runs were performed to compile this ledger.

## Measurement contract

Native results below are full-model HTTP serving, with S=input tokens,
O=output tokens, C=request concurrency. TTFT values are medians in milliseconds;
TPOT, where quoted, is the artifact's **mean** TPOT. Direct-adapter synchronized
prefill times exclude HTTP, scheduler, transport, and later decode; they are not
interchangeable with native TTFT. Output checks cover the saved requests, not
general model equivalence.

Earlier warmed native runs use three warmups; the async repeat and both handoff
runs use ten. Native readiness/warmup requests repeat only prompt 0, not every
measured prompt. Repeating a suite also changes process history. Initial
distributions are retained rather than replaced by their faster repeats.

C8/C32 baseline runs used zero warmups and skipped the optional initial test;
earlier comparison candidates used three explicit warmups. Neither three warmups nor a single-request
readiness check warms full occupancy. These measurements include first-decode
compilation/capture at the larger occupancy and must not be labeled steady-state
occupancy speedups. The comparison JSON records this protocol mismatch.

## Native serving chronology

Primary shape is S128/O16/C1 (ten completed requests, zero failures in every
listed primary-shape artifact).

| Experiment and raw primary artifact | Median TTFT | Interpretation |
|---|---:|---|
| [baseline_warm](../../readiness_vllm/ttft_optimization/baseline_warm/s128-o16-c1.json) | 424.985 | Three-warmup eager serving reference. |
| [candidate_compact initial](../../readiness_vllm/ttft_optimization/candidate_compact/s128-o16-c1.json) | 143.619 | Bounded prefill/sample tracing with actual scheduler table compaction and decode trace reuse. |
| [candidate_compact matrix](../../readiness_vllm/ttft_optimization/candidate_compact_matrix/s128-o16-c1.json) | 106.710 | Later same-family measurement; initial-to-later difference is not an isolated code speedup. |
| [combined final_default](../../readiness_vllm/ttft_optimization/final_default/s128-o16-c1.json) | 114.837 | Wider short-prefill expert batches plus page-copy optimization: primary regression versus compact matrix. |
| [selected_sync initial](../../readiness_vllm/ttft_optimization/selected_sync/s128-o16-c1.json) | 98.907 | Width32/gate K22, synchronous scheduling; changes the decode tradeoff. |
| [selected_sync matrix](../../readiness_vllm/ttft_optimization/selected_sync_matrix/s128-o16-c1.json) | 99.519 | Primary mean TPOT 20.489, versus async-repeat 18.882; not a free TTFT win. |
| [selected_async initial](../../readiness_vllm/ttft_optimization/selected_async/s128-o16-c1.json) | 137.657 | Three warmups; declining pre/post-runner delays across requests. |
| [selected_async repeat](../../readiness_vllm/ttft_optimization/selected_async_repeat/s128-o16-c1.json) | 105.592 | Ten warmups; unchanged width32/K22 backend. |
| [handoff_async initial](../../readiness_vllm/ttft_optimization/handoff_async/s128-o16-c1.json) | 131.058 | Ten warmups; experimental first-decode `sleep(0)`. |
| [handoff_async repeat](../../readiness_vllm/ttft_optimization/handoff_async_repeat/s128-o16-c1.json) | 106.205 | No consistent primary benefit; hook removed from production. |

The wider-batch combined candidate improved S256 TTFT from
[190.650](../../readiness_vllm/ttft_optimization/candidate_compact_matrix/s256-o16-c1.json)
to [178.241](../../readiness_vllm/ttft_optimization/final_default/s256-o16-c1.json)
while worsening S128 by 8.126 ms. Reduced-layer benefits therefore did not
justify selecting wider batches for the primary full-stack target.

[Combined comparison](combined_candidate_comparison.json) and
[synchronous comparison](sync_comparison.json) each report matching generated
texts, input lengths, and output lengths across all ten saved shapes. They
compare against the original baseline, not exclusively against the preceding
candidate; the large baseline improvement does not erase the S128 regression.

Repeated S128/O128 async control is
[107.092 ms TTFT / 19.107 ms mean TPOT](../../readiness_vllm/ttft_optimization/selected_async_repeat/s128-o128-c1.json);
handoff is
[105.712 / 19.123 ms](../../readiness_vllm/ttft_optimization/handoff_async_repeat/s128-o128-c1.json).
The mixed direction across O16/O128 did not support retaining the hook. Its
[rationale and archived test](experiments/handoff_rejected.md) preserve the
rejected experiment, not an active deployment requirement.

## Full-model direct-adapter controls

Both sweeps below use the same external full-context cache within each run:
262144-token capacity and 8192-column scheduler tables. All saved first-token /
first-decode token pairs match their within-run baseline, including warmups.
These are two-token controls, not full generated-response comparisons.

| Raw artifact | Candidate | Median synchronized prefill (ms) |
|---|---|---:|
| [full_candidates_context](../../readiness_vllm/ttft_optimization/full_candidates_context.json) | width32, cached tables | 92.091 |
| same | width32, forced table copies | 93.063 |
| same | width64, cached tables | 94.034 |
| same | width128, cached tables | 97.782 |
| [full_gate](../../readiness_vllm/ttft_optimization/full_gate.json) | width32, original gate K11 | 92.084 |
| same | gate K22, all layers | 90.037 |
| same | gate K44, all layers | 90.689 |
| same | gate K88, all layers | 91.750 |
| same | gate K22, full-attention layers only | 91.479 |

Gate K22/all layers was selected for subsequent serving experiments. The
historical `full_gate` baseline is K11; the current probe's `32cached` baseline
records the actual current K22 configuration. Names alone do not identify
geometry: consult saved per-layer configurations.

[full_geometry](../../readiness_vllm/ttft_optimization/full_geometry.json) then
passed all 78 saved request rows with exact within-run first/prompt-decode token
pairs, preserving width32/K22 and changing coherent N/core/block geometry:

| Geometry candidate | Median synchronized prefill (ms) |
|---|---:|
| Original gate44/down88 cores | 89.766 |
| Gate22 cores | 89.410 |
| Gate11 cores | 97.539 |
| Down44 cores | 88.225 |
| Down22 cores | 88.855 |
| Gate22 + down44 cores | 88.149 |

Down44 alone is the provisional choice pending boundary checks and native
serving. The combined candidate's additional 0.075 ms is too small to establish
a reliable incremental benefit from this single sweep. These measurements do
not establish an HTTP TTFT improvement or final acceptance.

## Rejected shared-MLP geometry family

The [first full shared probe](../../readiness_vllm/ttft_optimization/full_shared.json)
failed before candidate evaluation because TTNN `Shape` does not support slice
indexing. Converting the shape to a tuple before slicing fixed the probe-only
assertion; the device was closed before retry.

The [retry](../../readiness_vllm/ttft_optimization/full_shared_retry.json) then
failed the first `shared_gate22` token comparison: baseline prompt0 produced
`[127765,127765]`, candidate `[100067,100067]`. This was a real correctness
failure, not a valid performance result; its first warmup timings include
compilation and must not be compared as steady state.

The [isolated actual-input diagnostic](../../readiness_vllm/ttft_optimization/shared_geometry_diagnostic.json)
contains 16 projection rows across layers0/5. Explicit HiFi2 with no custom
program is bit-identical to the original generic BF16 projection on both
layers. Custom gate K22 yields relative L2 errors 0.01580/0.02314; gate K44/K88
yield 0.02566/0.04133. Down44 yields 0.00950/0.00865. Thus the difference persists
with matching actual fidelity, weights, dtypes, approximate-mode, accumulation,
and packer settings; changed program geometry changes numerical results.

Gate22 saves only approximately 1.1/4.4 microseconds per measured projection;
gate88 is slower. Down44 saves approximately 18.1/16.2 microseconds. These
isolated timings do not justify a deployment gain in the face of token changes.
Shared production projections remain unchanged. This classification is distinct
from the exact-output expert down44 candidate above.

## Host-delay evidence and tokenizer control

[Initial](async_transport_initial.json) and [repeated](async_transport_repeat.json)
observer analysis separates pre-runner, runner-prefill, and post-completion
intervals. Initial versus repeated medians were 24.364 / 2.651 ms before the
runner, 93.694 / 94.428 ms inside runner-prefill, and 19.640 / 8.151 ms after
completion. Medians of components need not sum to the median TTFT. Runner time
includes host preparation; it is not a device-kernel-only measurement.

The initial second-request spike therefore does not establish a prefill compute
regression or whole-decode GIL starvation. The broad pre-runner interval also
includes frontend/tokenizer scheduling and engine transport, not just scheduler
queueing. Native warmups repeat prompt 0; exact cause of the decaying delay
remains unproven.

[Fresh-process tokenizer probe](../../readiness_vllm/ttft_optimization/tokenizer_probe.json):
pinned tokenizer, OMP threads 8, ten prompt-0 warm encodes, then ten distinct
saved prompts and a repeat pass. Every token sequence matched. Maximum first-
pass encode was 0.303498 ms; repeat maximum was 0.141399 ms. The cold prompt-0
warmup maximum was 0.965848 ms. Pure BPE/encode cost alone cannot explain a
roughly 53 ms pre-runner delay in this control. This does not measure async
batcher waiting, event-loop scheduling, IPC, or CPU contention.

## Correctness and lifecycle evidence

[watcher_k22](../../readiness_vllm/ttft_optimization/watcher_k22.json) records
`passed=true`, no skipped cases, 12 eager controls, 24 traced/fallback cases,
12 transitions, and four prefill-only cases. This is the reduced real-layer
(0,5) Worker-Watcher/allocation-tracked check, not full30-layer validation.

CPU evidence is complementary, not hardware correctness proof:

| Host test artifact | Result / scope |
|---|---|
| [selected_host_tests](../../readiness_vllm/ttft_optimization/selected_host_tests.log) | 115 passed; selected host contract suite. |
| [trace_contract_tests](../../readiness_vllm/ttft_optimization/trace_contract_tests.log) | 10 passed; capture exception cleanup and sampling invalidation. |
| [page_contract_tests](../../readiness_vllm/ttft_optimization/page_contract_tests.log) | 2 passed; unchanged/changed tables and non-aliasing snapshots. |
| [expert_policy_tests](../../readiness_vllm/ttft_optimization/expert_policy_tests.log) | 11 passed; bounded row/chunk policy. |
| [prefill_only_host_tests](../../readiness_vllm/ttft_optimization/prefill_only_host_tests.log) | 13 passed; includes prefill-only trace lifecycle. |
| [handoff_host_tests](../../readiness_vllm/ttft_optimization/handoff_host_tests.log) | 16 passed for now-rejected hook; archived source, not active production coverage. |

These suites overlap and should not be summed into a unique-test count. Final
acceptance still requires the selected geometry's completed native validation
and review; passing intermediate controls is not stage signoff.

## Final async serving cohorts and anomalies

The complete native matrix in `acceptance_matrix/` retains all requests and
matches original output texts and lengths in all ten workload rows. S128/O16/C1
median TTFT is103.379ms; S128/O128/C1 is104.294ms, with mean TPOT19.123ms.
The separate shared-suite primary S128/O128/C1 result is711.193ms with
TPOT19.092ms (`acceptance_suite/readiness_vllm/vllm_benchmark.json`). It has one
measured request and different request/capture history, rather than the matrix's
ten native prompt0 warmups and three measured requests. Both artifacts and
their logs are retained. The available evidence does not isolate compilation,
capture, transport, and process-history contributions to this difference; it is
an unresolved first-use/protocol anomaly, not a discarded datapoint or a second
estimate of the warmed target statistic.

After the full sampling suite and chat replay, `acceptance_osl1_repeat/` measures
S128/O1 median96.689ms/P99100.934ms, and its separate O128 cohort101.325ms/
TPOT19.201ms. The initial OSL1 cohort remains in `acceptance_osl1/`; its larger
host transient is not erased. OSL1 is not substituted for O128 when discussing
the primary latency target.

The full async sampling suite passes72 tests with one all-vocabulary logprob-cap
skip. Successful ordinary TP4 logprob cases use the established host-compatibility
fallback, not device-logprob support. All18 repeated greedy chat requests match
the official six-prompt controls; seed71 sampled controls are saved for comparison
with the legacy restoration switches. The first metadata utility attempt failed
before generation because chat-template tokenization returned a `BatchEncoding`;
explicit `return_dict=False` fixes JSON serialization. Failed and corrected logs
are both retained; no production change or device reset resulted.

## Selected synchronous profile and protocol correction

The three complete final-geometry C1 cohorts are `acceptance_sync_matrix/`,
`acceptance_sync_repeat1/`, and `acceptance_sync_repeat2/`. S128/O128 medians are
95.241/95.357/95.028ms; P99s are95.285/95.385/97.450ms. S128/O16 first-pass median
107.186ms is retained, followed by97.385/95.592ms in the two complete repeats.
O16 repeat P99 can exceed100ms, and S129/S256 medians remain above100ms. This
supports a narrowly scoped warmed S128/C1 target, not a guarantee for every
novel prompt, first-use request, output length, or percentile.

Synchronous scheduling is selected for the primary TTFT objective, with a
disclosed C1 decode TPOT cost of approximately6–9% versus optimized async.
Async remains available as the throughput profile. C8/C32 mean whole-request
latency is0.85%/0.40% higher than optimized async, despite much lower TTFT:
first-decode capture moves into ITL rather than disappearing. All26 final sync
matrix/repeat rows have exact output text/length parity with async controls,
with no failed requests. See `final_scheduler_comparison.json` and the raw
repeated cohorts; repeats are not substituted for the first cohort.

The selected sync shared-suite primary is147.951ms TTFT,24.486ms mean TPOT,
and3257.728ms mean whole-request latency. Both shared suites use
`num_warmups=0` and `ready_check_timeout_sec=0`. Their native logs explicitly say
`Skipping endpoint ready check` after the misleading `Starting initial single
prompt test run` banner. The installed CLI source confirms no initial request
is sent in that configuration; only explicit warmups count. Its source hash and
relevant code are preserved in `native_protocol_source.log`. Original baseline
C8/C32 likewise had zero warmups, not a single-request warmup. Earlier prose
assuming such a request has been corrected.

Therefore the147.951ms sync and711.193ms async shared-suite results are genuine
no-warmup first-use measurements after qualitative state changes, not the warmed
target protocol. They remain visible in `perf_summary.json`. First-use/capture
cost is consistent with the implementation lifecycle, but detailed ITL arrays
were not saved by the shared runner, so exact phase attribution is not claimed.
