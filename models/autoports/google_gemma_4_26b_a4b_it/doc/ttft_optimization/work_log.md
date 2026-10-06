# Post-pipeline TTFT optimization

Started 2026-09-28 from e7ae5f184a on branch
`mvasiljevic/gemma4-ttft-opt`. Target: warmed end-to-end short-input C1 TTFT,
aiming below 100 ms, retaining the selected precision and full 262144 context,
canonical sampling and C8/C32 support. This is separate from the unfinished
pipeline accuracy evaluation; no accuracy completion is implied.

## Baseline and environment

Existing serving wrapper: `/home/mvasiljevic/gemma4-pipeline/vllm-container.sh`.
Host Docker access uses `sudo -n`; the wrapper runs the container workload as
uid/gid 6002. Owned container started with:

```sh
sudo -n env TT_USE_DEVICE=1 /home/mvasiljevic/gemma4-pipeline/vllm-container.sh -c 'sleep infinity'
sudo -n docker exec 9d99fc35c498 bash -c 'python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_server.py --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/baseline > /tmp/gemma4-ttft-baseline-server.log 2>&1'
sudo -n docker exec 9d99fc35c498 python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_benchmark.py --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/baseline --occupancy
```

The server uses the inherited 32-slot, TP4, P300x2 configuration, async decode,
on-device sampling, and pinned checkpoint/tokenizer revision. Each native vLLM
benchmark constructs an initial test, but skips sending it when the default
`ready_check_timeout_sec=0`; only explicit warmups count. Raw result,
per-request detailed timing, generated text, command/request map and log are in
`readiness_vllm/ttft_optimization/baseline/`. That first baseline had zero explicit
warmups, so its initial compilation/capture is included in the measured requests.

No Tracy, device profiler, or live serving profiler is used, per the serving
skills. Existing reduced/non-serving reports supply kernel context. The baseline
and target concern serving overhead; precision and decoder math are preserved.

## Initial operation audit

| Path | Existing sequence | Candidate | Constraints |
| --- | --- | --- | --- |
| Prefill inputs | Upload hybrid page tables and exact logical tokens | Persistent trace inputs refreshed in place | Per-layer tables, external cache ownership, arbitrary lengths |
| Prefill stack | Embedding; 30 layers of existing TP/EP operations | Bounded exact-length device-only trace | Same projection, CCL, dtype/fidelity and masking; no padded semantic lengths |
| Final head | Slice final hidden row, norm, vocab-sharded head, softcap | Include existing sequence in prefill trace | Already avoids intermediate heads; grouped-head Qwen change has no immediate analogue |
| First sampling | Pad logits to 32 rows, canonical split sampling eagerly | Persistent logits and generator-owned canonical sampling capture | Seed/penalty counts and graph mode must match; no host argmax |
| Decode | Model trace, canonical sampling trace, deferred token read | Preserve existing path and reuse where safe | Scratch allocation lifetime, slot/page refresh and feedback |

Source diagnosis and inspiration review: `AUTODEBUG.md`. Trace lifetime is a
verified source constraint, not yet a measured optimization. No implementation
candidate is accepted before matched hardware measurements and correctness.

## Measured baseline

The first pass exposed cold-variant outliers; retain it as first-use evidence.
The matched warmed baseline uses `ttft_benchmark.py --requests 10 --warmups 3`
and `baseline_warm/`. The native client skips its optional initial test, then sends
three explicit warmup requests before the measured cohort. Its 128-output check uses three measured
requests. Same server, checkpoint, sampling, seed and 32-slot configuration.

| ISL / OSL / C | Requests | TTFT median ms | TTFT P99 ms | Mean TPOT ms |
| --- | --- | --- | --- | --- |
| 32 / 16 / 1 | 10 | 384.323 | 396.244 | 18.915 |
| 33 / 16 / 1 | 10 | 395.460 | 462.216 | 18.893 |
| 64 / 16 / 1 | 10 | 396.419 | 400.806 | 18.793 |
| 127 / 16 / 1 | 10 | 428.949 | 467.425 | 19.217 |
| 128 / 16 / 1 | 10 | 424.985 | 458.008 | 18.858 |
| 129 / 16 / 1 | 10 | 435.970 | 447.090 | 18.871 |
| 256 / 16 / 1 | 10 | 469.483 | 474.806 | 18.958 |
| 128 / 128 / 1 | 3 | 419.105 | 427.166 | 19.079 |
| 128 / 32 / 8 | 8 | 3745.727 | 3746.004 | 188.683 |
| 100 / 100 / 32 | 32 | 14971.166 | 14972.063 | 681.962 |

C8/C32 come from the first pass with no explicit warmup or readiness request; they
are occupancy controls, not identically warmed C1 comparisons. All requests
completed. Native benchmark client teardown emits nanobind reference-leak
warnings already present in the environment; server requests and device
shutdown remain successful. Baseline server log was copied to `baseline/server.log`.
`server_info.json` records the actual full context and runtime configuration.

After the baseline, the owned server PID54 was stopped with SIGINT and exited0;
`fuser` found no accelerator owners. Reduced synchronized phase measurement:

```sh
sudo -n docker exec 9d99fc35c498 bash -c 'OMP_NUM_THREADS=8 python models/autoports/google_gemma_4_26b_a4b_it/tests/check_ttft_prefill.py --baseline-only --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/baseline_phases.json > /tmp/gemma4-ttft-baseline-phases.log 2>&1'
```

The first phase probe failed before model construction because the image's
default `/home/container_app_user/logs/generated/watcher` is not writable as
uid6002. Retained `baseline_phases_path_failure.log`; rerun adds
`TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/phase_runtime`
and writes `/tmp/gemma4-ttft-baseline-phases-retry.log`. This is a logging-path
configuration failure, not a kernel failure; the failed open closed its devices.

## Status

Optimization and final validation in progress. No completion claim yet.
Unrelated operator `doc/PIPELINE_INTERVENTIONS.md` and nested `vllm/` are preserved.

## Phase diagnosis and first trace experiment

Reduced two-real-layer (0 and 5) synchronized phase check passed. Warm2 S128:
configuration0.533ms, model prefill8.217ms, sampling0.841ms, read0.089ms.
Warm2 model times S32/S129/S256:6.122/10.022/11.429ms. These are reduced
diagnostics on repeated-token prompts, not full-model latency estimates or
serving metrics. Full native serving baseline above remains authoritative.

First experiment: one generator-owned exact-length B1 prefill/sampling graph,
bounded at1024 tokens, initially opted in with `GEMMA4_PREFILL_TRACE=1`.
Persistent token/table/logit/output buffers are prepared before decode warmup;
prefill and canonical sampling are captured after decode warmup/capture. Reuse
checks exact length, table geometry and external cache/tensor identity. Changed
contents refresh in place. Other shapes and non-greedy/penalty/logprob modes
keep the eager canonical path. No precision, cache geometry or context change.

The allocation-tracked reduced probe passed all18 candidate cases (nine inputs,
two repetitions each), matching eager first-token plus three-decode-token
streams. Shapes32,33,127,128,129; changed tokens, pages, cache; B2 fallback.
Ten prefill replays and six captures were observed. The1025-token case was
explicitly skipped for this first inner-loop check and remains required.
Allocation tracing/tracebacks distort timing; no speedup is inferred from this
probe. Artifacts: `candidate_probe.json` and `candidate_probe.log`.

```sh
sudo -n docker exec 9d99fc35c498 bash -c 'OMP_NUM_THREADS=8 TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/candidate_probe_runtime TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python models/autoports/google_gemma_4_26b_a4b_it/tests/check_ttft_prefill.py --skip-long --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/candidate_probe.json > /tmp/gemma4-ttft-candidate-probe.log 2>&1'
```

Full-server candidate uses the same `ttft_server.py` command with output
`.../candidate`, log `/tmp/gemma4-ttft-candidate-server.log`, and Docker exec
`-e GEMMA4_PREFILL_TRACE=1`. First matched measurement selects `--lengths128`
(written as separate CLI arguments) with10 requests and3 warmups.

## Serving-boundary repair and first measured win

See `AUTOFIX.md` for the rejected candidate and padded-table diagnosis. The
actual scheduler sends32 page rows for B1 prefill. Compacting those rows enables
the generator-owned trace. `candidate_compact/` uses the same full30-layer,
262144-context server with `GEMMA4_PREFILL_TRACE=1`, no diagnostic timing or
allocation tracker. First S128/O16/C1 run:10/10, median143.619ms, P99170.538ms,
mean TPOT18.169ms. S128/O128 median162.548ms. All10 S128/O16 generated texts
exactly equal the warmed baseline. This run showed elevated/declining request
times after the first request; retain it instead of selecting only the faster run.

Repeated low-ISL matrix on the same server (`candidate_compact_matrix/`) uses:

```sh
sudo -n docker exec 9d99fc35c498 python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_benchmark.py --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/candidate_compact_matrix --lengths 32 33 64 127 128 129 256 --requests 10 --warmups 3 --occupancy
```

C1 medians(ms): S32=57.008,S33=77.257,S64=74.476,S127=121.256,
S128=106.710,S129=121.811,S256=190.650. S128/O128=105.558ms.
C8 S128/O32:8/8, median3772.060ms, TPOT188.741ms (baseline3745.727/188.683).
C32 S100/O100:32/32, median14995.199ms, TPOT680.273ms
(baseline14971.166/681.962); median ITL677.216ms versus677.186ms.
Further experiments and final-default validation remain pending; not stage closure.

Host checks:91 adapter/sampling tests and10 trace-lifecycle tests pass. New
tests cover compact table views and capture-error cleanup/configuration
invalidation. The benchmark driver now fails explicitly on incomplete request
counts, since the native CLI may return zero with zero completed requests.

## Allocation-tracked state/boundary verification

`transitions.json`/`.log`: passed12 eager cases,24 repeated trace/fallback cases,
and12 sampling/reset transitions on real layers0/5. Includes32,33,127,128,129,
1023,1024,1025,changed tokens/pages/cache, B2, greedy↔sampled,penalties→greedy,
host compatibility→greedy,reset→replay. Uses actual32-row scheduler prefill
tables and neutral padded decode parameters. All outputs match eager controls;
allocation-tracked times are not performance claims.

```sh
sudo -n docker exec 9d99fc35c498 bash -c 'OMP_NUM_THREADS=8 TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/transitions_runtime TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_TRACEBACKS=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 python models/autoports/google_gemma_4_26b_a4b_it/tests/check_ttft_prefill.py --wire-sampling --transitions --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/transitions.json > /tmp/gemma4-ttft-transitions.log 2>&1'
```

## Expert batching experiment (pending)

Hypothesis:32-token EP prefill batches repeat weight loads/kernel work four times
at S128;64/128 rows may amortize that cost. The isolated full-layer probe retains
all weight/activation dtypes, math fidelity, collective paths and projection K
blocks. Only expert token batch size, matching M geometry and ownership index
buffers differ. Wider unions can increase inactive expert computation and DRAM
scratch; correctness and replay time decide acceptance.

```sh
sudo -n docker exec 9d99fc35c498 bash -c 'OMP_NUM_THREADS=4 TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/expert_batches_runtime python models/autoports/google_gemma_4_26b_a4b_it/tests/check_ttft_expert_batches.py --lengths 128 --samples 10 --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/expert_batches_s128.json > /tmp/gemma4-ttft-expert-batches.log 2>&1'
```

Initial batch sweep passes with exact layer outputs and exact replay outputs.
S128 layer0:3071.758→2869.919us; layer5:3246.386→3070.547us (32→128 rows).
`expert_batch_matrix.json`42 rows and `expert_batch_tails.json`32 rows all pass
bit-identically. Wider batches lose on S33 and full-layer S65, so the proposed
deployment policy uses128-row expert batches only for physical128..256-row
prefill inputs; tiny and long chunks retain32. No weight/fidelity/collective
changes. Raw `.pt` output tensors remain local under readiness_vllm's inherited
ignore rule; compact comparison JSON/logs are versioned.

Larger legal gate-K sweep (`expert_gate_sweep.json`, S128) passes bit-identically:
sliding K11/22/44 are effectively tied at2868–2871us, K88=2900us; retainK11.
Full attention K11=3067.904us, K22=2925.373us, K44=2924.797us, K88=2974.386us;
selectK22 (K44 difference<1us, no meaningful measured improvement). DownK22,
grid11x4/11x8 and selected dtypes/fidelities unchanged. Boundary verification
for this selected geometry is in progress (`expert_gate_tails.json`).

## Combined candidate: full-server regression, not accepted as final

`expert_gate_tails.json` passed all56 cases bit-identically. Worker-only Watcher
plus allocation tracking passed the selected geometry in `watcher_selected.*`;
Ethernet Watcher remains excluded because its firmware exceeded the runtime
configuration buffer before model execution (see AUTOFIX.md).

The directory `final_default/` is an experimental combined candidate, despite
its original intended name. It enables prefill trace, changed-only prefill table
copies and128-row short expert batches by default. Full30-layer native serving,
unchanged262144 context, selected precision and canonical sampling:
C1 medians(ms) S32=56.757,S33=71.215,S64=73.569,S127=113.013,
S128=114.837,S129=129.876,S256=178.241; S128/O128=111.796.
C8=3509.137ms, TPOT188.522; C32=14013.775ms, TPOT680.902,
ITL677.200. All10 benchmark rows have exact input/output lengths and generated
texts versus the original baseline (`combined_candidate_comparison.json`).
However aligned S128 regresses against trace-only106.710ms, so the combined
configuration is not accepted for the primary target. The full server stopped
cleanly; a single-load full-stack adapter sweep separates the changes next.

```sh
sudo -n docker exec 9d99fc35c498 bash -c 'OMP_NUM_THREADS=4 TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/full_candidates_context_runtime python models/autoports/google_gemma_4_26b_a4b_it/tests/check_ttft_full_candidates.py --cache-context 262144 --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/full_candidates_context.json > /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/full_candidates_context.log 2>&1'
```

The first attempt (`full_candidates.*`, same command without cache-context)
used only64 physical blocks with8192-column tables. Decode validation rejected
that unsupported shape before measurements; normal teardown closed devices.
The retry allocates8192 physical blocks, matching the full-context table width.
This direct-adapter experiment excludes HTTP/scheduler overhead and does not
replace end-to-end acceptance measurements.

## Full-stack isolation and revised selection

`full_candidates_context.json` passes52 request rows, including warmups, with
exact first/prompt-decode token pairs. Same full model/cache across candidates:
32cached92.091ms;32forced93.063ms;64cached94.034ms;128cached97.782ms.
These are synchronized direct-adapter medians, not HTTP TTFT. Thus the earlier
two-layer embedding-fed speedup does not transfer to the actual full-model
activation/routing distribution. Keep32-row experts; accept changed-only page
copies (about0.97ms in this control). Physicalfull-context cache8192 blocks.

`full_gate.json` uses10 native random prompts matching the10-request serving
dataset and compares gate K22/44/88 under the same32-row grouping. All65 request
rows, including warmups, have exact first/prompt-decode token pairs. Medians:
K11=92.084ms,K22=90.037ms,K44=90.689ms,K88=91.750ms;
K22 only on full-attention layers=91.479ms. SelectK22 on all layers; downK22,
grid11x4/11x8, weights, fidelity, output dtypes and canonical sampling unchanged.
The default expert batch is32 again. Wider batches remain an explicit experiment
override, not the deployment default. Decode expert configurations are untouched.

```sh
sudo -n docker exec 9d99fc35c498 bash -c 'OMP_NUM_THREADS=4 TT_METAL_LOGS_PATH=/workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/full_gate_runtime python models/autoports/google_gemma_4_26b_a4b_it/tests/check_ttft_full_candidates.py --candidates 32k22 32k44 32k88 32fullk22 --dataset-requests 10 --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/full_gate.json > /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/full_gate.log 2>&1'
```

Prefill-only requests now capture on their second matching request, without
needing decode warmup. A later decode bind releases that graph before allocating
decode buffers, then recaptures using the established lifecycle. CPU contracts
pass13 tests; current worker-Watcher/allocation run adds three OSL1 requests and
the transition back to decode against eager controls. Final HTTP acceptance is
still pending.

## Workflow integration preparation

The main inference-server catalog selects canonical TTGemma4, so it cannot
attest this autoport by merely passing the HF model name. A separate implementation
`gemma4-autoport` was added on `mvasiljevic/gemma4-ttft-bench` in the owned checkout
`/home/mvasiljevic/gemma4-ttft-inference-server`, based on
`7896680ca8967884fbc9e6126ad7eb6a326c7785`. It preserves the canonical default and
pins the autoport architecture, HF/tokenizer revision, full context, block32,
sampling and fabric settings. Explicit `no-enable-*` flags are required because
the launcher omits booleanfalse arguments.99 catalog/launcher tests pass; raw
`inference_catalog_tests.log`. No refs have been pushed or dispatched yet.

## Scheduler, host transport, and second geometry sweep

Selected K22/32-row worker-Watcher plus allocation tracking passes all12 eager,
24 replay,12 sampling/reset and4 prefill-only cases (`watcher_k22.*`). CPU
contracts pass115 tests (`selected_host_tests.log`). Ethernet Watcher remains
disabled for the documented firmware-size blocker; these are not timing runs.

The synchronous scheduler experiment (`selected_sync/`,
`selected_sync_matrix/`) reaches S128 median98.907/99.519ms, but C1 decode TPOT
increases to20.2–20.6ms versus about19.1ms. It is not selected as an unconditional
win. Its lower C8/C32 TTFT moves first-decode capture into ITL; that is not a
matching warmed-occupancy improvement.

The async control (`selected_async_repeat/`,10 warmups) measures S128/O16
105.592ms and TPOT18.882ms; O128107.092ms and TPOT19.107ms. Initial measurements
on that server were slower (`selected_async/`, median137.657ms). Existing
host-only completion events and native request timestamps isolate a changing
client-to-prefill-dispatch interval, while adapter prefill stays approximately
94ms. See `async_transport_initial.json`, `async_transport_repeat.json`, and
`experiments/handoff_rejected.md`. A fresh pinned-tokenizer CPU probe has all
first-pass encodes below0.304ms, ruling out raw BPE alone. Exact host cause is
not established. Both first-use and repeat evidence are retained. A one-shot
`time.sleep(0)` async handoff trial has no stable primary win and was removed.

The full-model geometry sweep (`full_geometry.*`) preserves current dtype,
fidelity, K22 and all weights.78 request rows match first/decode token pairs.
Direct-adapter medians(ms): baseline89.766; gate22cores89.410;
gate11cores97.539; down44cores88.225; down22cores88.855; combined22/44=88.149.
Down44 is provisional pending native serving and boundary validation; the
combined incremental0.075ms is not established benefit. Shared BF16 candidates
must explicitly preserve default HiFi2: supplying a program config with
`compute=None` would silently select LoFi in TTNN and is not an allowed trial.

## Down44 safety and remaining experiment disposition

`expert_down_geometry.json` passes68 isolated layer rows across logical lengths
2,31,32,33,63,64,65,95,96,97,127,128,129,160,192,224,256. Candidate gateK22,
down44-core/N2 matches the actual gateK22/down88 production baseline bitwise.
The baseline's historical setting label is a sentinel; the recorded actual
program fields, not that label, define the control. Selected default down grid
is now11x4/perN2/outBlockW2/outSubblockW2 at32 expert rows. Decode is unchanged;
`GEMMA4_PREFILL_DOWN_CORES=88` restores the old prefill down program for controls.
`watcher_down44.*` passes12 eager controls,24 replay/fallback cases,12 sampling
transitions and4 prefill-only cases, including1023/1024/1025 boundaries.

Shared-MLP geometry is not selected. `full_shared.*` first failed in Python
shape inspection, fixed by converting the TTNN Shape to a tuple before slicing.
`full_shared_retry.*` then changed the first checked token pair at gateK22.
`shared_geometry_diagnostic.*` isolates actual eager layer0/5 projection inputs:
the generic explicit-HiFi2 control is bit-identical, but custom gate geometries
change BF16 accumulation results (relative L2 approximately1.58–4.13%) for only
small or negative speed changes. Down44 changes results by approximately0.9%
for about17us/op. No shared weight, precision, program, or production code change
is retained. See the experiment ledger for all16 raw comparison rows.

A second one-shot yield experiment moves `time.sleep(0)` to TTModelRunner entry,
before input preparation, retaining async scheduling. `early_handoff/` first
S128/O16 median135.327ms; repeated warmed `early_handoff_repeat/`105.610ms,
TPOT18.874ms; O128103.259ms/19.178ms. It does not establish a useful primary win
against105.592ms before down44, and is removed. Its seven CPU checks passed.
Archived implementation/tests are disabled under `experiments/`; no installed
hook or experiment environment variable remains in the production adapter.

Final-default server evidence is being collected under `acceptance_async/`,
without observer or handoff instrumentation. It starts with native OSL1 checks
under `acceptance_osl1/` so prefill-only capture does not depend on prior decode.

## Final validation and explicit serving-profile decision

`acceptance_matrix/` has ten native workload rows, all output texts and lengths
equal to the original controls (`acceptance_comparison.json`). The async S128/O16
median is103.379ms and mean TPOT18.844ms; S128/O128 is104.294ms/19.123ms.
The stated100ms target is still missed by this throughput-preserving profile.
The original task prioritizes warmed C1 TTFT and requires higher-concurrency
support and regression evidence; it does not explicitly prohibit a disclosed
decode tradeoff. Therefore a final-geometry synchronous latency-priority profile
will be measured separately before deployment selection. The pre-down44 sync
measurements are not represented as final-geometry acceptance evidence.

The full shared sampling suite is running without timing instrumentation.
Repository pre-commit found one new test using `pytest.raises`; it now uses the
repository `expect_error` fixture. The recheck passes, as do all13 trace-contract
tests (`precommit_recheck.log`, `trace_contract_recheck.log`). Runtime source was
not changed by that fix.

The async full sampling suite completes72 passed/1 skipped in1135.72s. The skipped
test is `test_chat_logprobs_all_vocab`, whose skip branch only accepts the server's
all-vocabulary logprobs cap rejection; ordinary logprob tests pass. Shared-suite
qualitative and benchmark artifacts remain separate from the warmed matrix.
The first chat replay attempt failed in metadata serialization before generation:
Transformers5.12 returns a `BatchEncoding` by default from the chat-template API.
The tool now explicitly requests `return_dict=False` for JSON token IDs. The
failed log is retained as `acceptance_chat_replay.log`; the corrected run uses
`acceptance_chat_replay_retry.log`. No runtime/model change or device reset was
required.

All18 corrected chat greedy replays match the official six-prompt outputs;
saved prompt token IDs also match server-reported prompt usage. Seed71 sampled
responses are retained for a fresh legacy control. The warmed OSL1 repeat is
96.689ms median (P99100.934ms); its separate O128 cohort is101.325ms/TPOT19.201ms.
These are explicitly different workloads, not a substitution of OSL1 for the
longer-generation primary metric. Async serverPID34611 exited0 after SIGINT;
the complete log is copied to `acceptance_async/server.log`. All four device
nodes then had no owners. No reset or foreign-process stop was needed.

Matched legacy control uses the current code's restoration switches, which
disable prefill tracing/compact reuse and restore original gate/down programs.
It retains the same async scheduler as the original and optimized async cohorts.
The final sync-versus-async comparison will isolate the scheduler tradeoff on
identical optimized geometry; the implementation control is async-versus-async.

The matched legacy async matrix completes all ten shapes with no failed requests.
`matched_async_comparison.json` confirms exact texts and input/output lengths in
every row. S128/O16 is455.918ms versus103.379ms optimized, and mean TPOT18.849ms
versus18.844ms. S128/O128 is443.567ms versus104.294ms. The original424.985ms
S128/O16 baseline is retained as the conservative headline reference rather than
selecting the larger later control. For C8, optimized async is2.38% slower in
median TTFT and1.00% slower in mean whole-request latency; C32 whole-request
latency is effectively identical (81872.421 versus81874.826ms). Do not label the
original differently-warmed occupancy comparison as proof of an improvement.

The legacy chat replay also passes all18 greedy requests, and all six seed71
sampled responses, usage, finish reasons, and rendered prompt IDs exactly equal
the optimized async controls. Seeded wording/description oddities are therefore
controlled inherited behavior. Unseeded official sampled responses are not
claimed text-identical. The legacy server exits0 after SIGINT, with its complete
log saved as `legacy_async_control/server.log`; all four device nodes have no
remaining owner before the synchronous candidate starts.

Final-geometry synchronous candidate commands (no profiler, observer, handoff,
or kernel experiment override):

```sh
sudo -n docker exec 9d99fc35c498 bash -c 'python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_server.py --no-async-scheduling --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync > /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync_server.log 2>&1'
sudo -n docker exec 9d99fc35c498 python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_benchmark.py --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync_matrix --lengths 32 33 64 127 128 129 256 --requests 10 --warmups 10 --occupancy --warm-occupancy
```

```sh
sudo -n docker exec 9d99fc35c498 bash -c 'GEMMA4_PREFILL_TRACE=0 GEMMA4_PREFILL_GATE_K=11 GEMMA4_PREFILL_DOWN_CORES=88 GEMMA4_PREFILL_EXPERT_BATCH=32 python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_server.py --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/legacy_async_control > /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/legacy_async_control_server.log 2>&1'
sudo -n docker exec 9d99fc35c498 python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_benchmark.py --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/legacy_async_matrix --lengths 32 33 64 127 128 129 256 --requests 10 --warmups 10 --occupancy --warm-occupancy
sudo -n docker exec 9d99fc35c498 python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_qualitative_replay.py --suite models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_suite/readiness_vllm/vllm_qualitative_outputs.json --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/legacy_chat_replay.json
```

```sh
sudo -n docker exec 9d99fc35c498 python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_benchmark.py --output /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_matrix --lengths 32 33 64 127 128 129 256 --requests 10 --warmups 10 --occupancy --warm-occupancy
sudo -n docker exec 9d99fc35c498 bash -c 'PYTHONPATH="${TT_MODEL_BRINGUP_ROOT}/runtime:${PYTHONPATH}" python -m readiness_check.run_vllm_server --stages qualitative,benchmark,sampling --server-url http://127.0.0.1:8000 --model-dir /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_suite --hf-model google/gemma-4-26B-A4B-it --max-num-seqs 32 --max-model-len 262144 --sampling-profile full --benchmark-prompt-len 128 --benchmark-output-len 128 --benchmark-num-requests 1 --benchmark-concurrency 1 > /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_suite.log 2>&1'
```

The two complete final C1 repeats used identical lengths, seeds, request counts
and ten native warmups. No geometry or runtime source changed between cohorts:

```sh
sudo -n docker exec 9d99fc35c498 python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_benchmark.py --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync_repeat1 --lengths 32 33 64 127 128 129 256 --requests 10 --warmups 10
sudo -n docker exec 9d99fc35c498 python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_benchmark.py --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync_repeat2 --lengths 32 33 64 127 128 129 256 --requests 10 --warmups 10
sudo -n docker exec 9d99fc35c498 bash -c 'PYTHONPATH="${TT_MODEL_BRINGUP_ROOT}/runtime:${PYTHONPATH}" python -m readiness_check.run_vllm_server --stages qualitative,benchmark,sampling --server-url http://127.0.0.1:8000 --model-dir /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync_suite --hf-model google/gemma-4-26B-A4B-it --max-num-seqs 32 --max-model-len 262144 --sampling-profile full --benchmark-prompt-len 128 --benchmark-output-len 128 --benchmark-num-requests 1 --benchmark-concurrency 1 > /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync_suite.log 2>&1 && python models/autoports/google_gemma_4_26b_a4b_it/tools/ttft_qualitative_replay.py --suite models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync_suite/readiness_vllm/vllm_qualitative_outputs.json --output models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync_chat_replay.json > /workspace/tt-metal/models/autoports/google_gemma_4_26b_a4b_it/readiness_vllm/ttft_optimization/acceptance_sync_chat_replay.log 2>&1'
```

The three warmed S128/O128 medians are 95.241/95.357/95.028 ms. All 26 sync
workload rows exactly preserve optimized async outputs and token lengths. The
selected deployment is explicitly latency-priority sync, with the async
throughput option retained. C1 TPOT costs approximately 6–9%; C8/C32 mean E2EL
costs 0.85%/0.40%. The first S128/O16 cohort remains 107.186 ms, followed by
97.385/95.592 ms; repeated O16 P99s still exceed 100 ms. These are not hidden.

The final selected sync full suite passes all qualitative/benchmark/sampling
stages. Sampling is 72 passed, one expected all-vocabulary logprobs-cap skip,
in 1114.96 s. This includes structured/plain mixing, sampling variety/seeding,
logprobs through host compatibility, and repetition/presence/frequency penalties.
The source-only compatibility rereview finds no new blocker; fresh CPU checks
again pass 114 inference and six Shield tests. Actual image-build validation
remains the remote workflow gate, not something those CPU checks establish.

The final sync replay completes with all 18 greedy responses and all six
seed-71 sampled text/finish/usage dictionaries exactly equal to legacy. Pinned
rendered IDs and reported prompt counts agree. Fresh finish metadata confirms
four capped greedy responses; concrete inherited quality errors remain in the
final qualitative verdict. No additional hardware experiment is needed for
this scoped optimization decision.

Final API SIGINT exits zero, devices/port clear; the complete log is retained.
The publication utility snapshots 537 JSON/log artifacts, repository hooks
normalize whitespace, and finalize/verify passes for all originals and archives.
Three oversized observer originals remain local with exact-byte gzip publication.
Derived documentation hashes are mechanically updated to normalized browsable
bytes; original raw metadata hashes remain verifiable against gzip originals.
The holder container then stops; [local_cleanup.md](local_cleanup.md) records
the final process/device check. Packaging/review logs after the snapshot stay
outside the sealed raw tree.

Packaging needed one correction: some earlier container-created stage logs
were not writable by the host hook process, so the initial hook run and first
normalization finalization were incomplete. Ownership was repaired only for
the manifest-listed browsable stage files; the utility-owned manifest returned
to `snapshotted` while whitespace/line endings were normalized. Re-finalization
verified all 537 original archives and type-sensitive JSON/whitespace-only log
equivalence. No original gzip or original hash changed. The final host
`pre-commit run` exits zero, as do staged/worktree whitespace checks. No hook
was skipped, no exclusion broadened, and no runtime source changed.

Independent local review returns clean-pass, with remote qualification still
mandatory. Normal commits and pushes publish TT-Metal implementation
`079b9fb26dd7300b61a83f1f02307e8690ea0a2d`, inference
`b6c06944f6d099350f8e9dac3b3e72c58f146112`, Shield
`4f900886b74b82f8fd028d40f1c011c0dc322070`, and QB2
`05415de64c82d91ebb3c41362266c2ba4c076c26`. The QB2 reusable workflow pins the
complete reviewed Shield SHA. No force push, PR or history rewrite occurs.

The exact nested vLLM commit cannot be published with the available account:
push returns 403, repository permissions explicitly say `push:false`, and the
required commit API lookup returns 422. There is no existing accessible personal
fork. No new repository is created without direction. The workflow is not
dispatched with an unavailable plugin SHA, and overall completion is not claimed.
[publication.md](publication.md) records the evidence, exact pending command,
and two unblock options (upstream access/publication or explicit fork creation).
