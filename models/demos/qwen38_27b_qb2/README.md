# Qwen3.8-27B on QB2

TP4 implementation of `Qwen/Qwen3.8-27B` on four Blackhole devices with a
`(1, 4)` mesh. It supports prefill, traced decode, device sampling, and vLLM
serving.

## Experimental Galaxy bring-up

The TP4 validator also accepts a `(1, 4)` submesh of a Blackhole Galaxy. The
Galaxy default is `Topology.Linear`; QB2 keeps its qualified ring default.
Fabric setup, decoder collectives, embedding gather and sampler use the same
choice. Select Ring explicitly only after qualifying the physical wraparound
links, and pass that choice to both `configure_fabric` and `build_generator`.

The target layout is one `(8, 4)` parent mesh with eight independent `(1, 4)`
replicas. Each replica needs its own model, KV/GDN-state pool, traces and CCL
context. Do not share a CCL context between concurrent replicas. The first
hardware test opens the parent and exercises **one** replica; it does not prove
eight-replica scaling or unchanged reference evaluations.

`tests/test_galaxy_smoke.py` runs all layers, then repeats the same prompt and
checks deterministic generation, preserving tokens, decoded text and first/warm
performance in `QWEN_GALAXY_RECEIPT`. Enable it with `QWEN_GALAXY_SMOKE=1` and
set `MODEL_WEIGHTS_DIR` to the pinned checkpoint. Hardware runs must be serialized.
`demo/run_galaxy_qualification.sh` is the bounded-job entrypoint for the isolated
task layout used during this bring-up; it runs the host tests before hardware.
The first hardware attempt hit pytest's inherited 300-second deadline during
weight loading; it did not reach inference. The entrypoint now allows 1800
seconds for a single replica while retaining the device-operation timeout.

For the G0 concurrency gate, set `QWEN_GALAXY_REPLICAS=8` when launching that
entrypoint in a persistent job with a deadline of at least 100 minutes. It
selects `tests/test_galaxy_replicas.py` with a 90-minute test deadline. The test
loads eight independent full models, warms each, then compares five isolated
decode windows per replica against five concurrent windows. It requires exact
greedy token equality and at most 3% median TPOT regression for each replica.
Concurrent decode enqueues traces on all submeshes before waiting for completion;
prefill is outside the timed interval. This is a G0 test, not a serving benchmark
or proof of long-context/batched performance. A two-replica run is available for
bring-up but does not pass the eight-replica gate.

The first full-model Galaxy test passes on `10.228.203.98`: all 64 layers,
prefill, device sampling and traced decode, with identical 128-token greedy
outputs across two runs. Receipts and source hashes are in
`galaxy-evidence/baseline-v2/`; 66 host tests also pass (plus 40 subtests).

| Single TP4 replica, 63-token prompt / 128 output tokens | Measured |
| --- | ---: |
| Model/generator setup | 325.94 s |
| First-use TTFT, including compilation | 49.44 s |
| Warm TTFT | 62.13 ms |
| Warm decode, 127 steps plus final history readback | 38.89 tokens/s/user |
| Warm TPOT | 25.72 ms |

The decoded answer is coherent through its first EOS. This fixed-length probe
intentionally continues after EOS, so its trailing output is not a serving EOS
test. These are short-context B1 results; they do not establish long-context,
high-batch, eight-replica or reference-evaluation performance. The eight-replica
test has been launched as a separate persistent job and is not yet qualified.

`demo/build_galaxy_fabric_tests.sh` successfully built `test_tt_fabric` in a
separate build tree. `demo/run_galaxy_fabric_tests.sh` queues the upstream torus
neighbor-exchange tests behind the same device lock, with 1000 packets per
sender and a 15-minute execution limit. The allocated-host job is
`qwen38-fabric-torus-neighbors-20261006.service`; it is queued behind the replica
test, so the wraparound links are not yet qualified.

The first eight-replica attempt loaded all eight copies, then failed before the
first inference: Transformers returned a `BatchEncoding` from tokenized chat
templating, while the test expected a list of token IDs. Writing that object to
JSON obscured the original type error and skipped explicit cleanup. Both smoke
and replica tests now prepare the same plain token list before opening devices;
six host regressions pass, including the actual pinned tokenizer matching the
passing baseline's 63 IDs. Replica receipts now preserve loading progress,
physical device IDs and failure details, and cleanup runs even if saving fails.

The first fabric job also failed before running its selected test: the filter
needed the `name.` prefix. The corrected argument is
`--filter name.2DTorusXYNeighborExchange`. This was a launch-argument error, not
a measured link failure. The following safe runner reset the Galaxy and began
the TP4 sweep. Corrected fabric and eight-replica jobs are queued behind that
sweep as `qwen38-fabric-torus-neighbors-v2-20261006.service` and
`qwen38-metal-galaxy-eight-replicas-v2-20261006.service`. Neither gate is passed
yet. Raw failure/host-test receipts are retained in `galaxy-evidence/`.

## Measured input-length and concurrency sweep

The requested artifact runs one TP4 replica first, then the full Galaxy. The
initial grid is ISL 128 / 8,192 / 32,768 / 55,000 / 131,072 / 262,016 and
concurrency 1 / 2 / 4 / 8 / 16 per replica. A 128-token output budget leaves the
largest cell inside the 262,144-token context limit. The existing 1,179,648-token
per-replica KV allocation guard excludes three cells, leaving 27 to measure;
excluded cells are not reported as measured out-of-memory failures.

`tests/sweep_report.py --init --replicas 1 --output RESULTS` creates the plan and
the standalone HTML, PNG, SVG, PDF, CSV and JSON artifact. Run
`demo/run_galaxy_perf_sweep.sh TASK_ROOT RESULTS` in a persistent job with
`MODEL_WEIGHTS_DIR` set. The test is serialized against other device jobs and
allows three hours of execution. An optional `QWEN_WAIT_FOR_UNIT` waits for a
specified user service to finish before entering the hardware queue; include
that wait in the enclosing service deadline.

Each cell uses the full model and a deterministic repeated-text input, one
unscored warmup and three measured repetitions. It requires exact repeatability
and no trace capture in measured windows. Graphs show tokens/s/user, aggregate
decode tokens/s and p50 TTFT. Raw JSON/CSV also preserve p90 TTFT, end-to-end
throughput, counters and cold timing. Prefill is fresh, without prefix-cache
reuse; output delivery after the first token is deferred to a final history
readback. These are native generator timings, excluding HTTP/router overhead.
Default model knobs match the initial Galaxy baseline. This sweep does not
qualify model accuracy or the future optimized configuration.

Completed job: `qwen38-perf-sweep-tp4-v1-20261006.service`, measuring the TP4 grid
after the first replica/fabric attempts. Results are under
`TASK_ROOT/perf-sweep-tp4-v1/`; `index.html` is
the artifact entrypoint and updates after each completed cell. Five host tests
validate the metric accounting and reject cold captures as warm measurements.
All twenty-seven supported cells are measured and preserved in
[`galaxy-evidence/perf-sweep-tp4-v1/index.html`](galaxy-evidence/perf-sweep-tp4-v1/index.html)
(HTML plus PNG/SVG/PDF/CSV/JSON). The hardware test passed in 2 h 3 min,
including setup, compilation, warmups and repeated full prefills. At ISL 128, C=1/2/4/8/16 measured
38.79/31.08/24.85/20.48/12.15 tokens/s/user, with aggregate decode throughput
38.79/62.16/99.39/163.84/194.40 tokens/s. This baseline has substantial batch
overhead and does not meet the optimized high-batch targets.

At 8K, C1 delivers 37.98 tokens/s/user with 1.236 s TTFT; C16 delivers
11.67 tokens/s/user (186.76 aggregate) with 19.98 s TTFT. At 32K, C1 delivers
36.59 tokens/s/user with 5.316 s TTFT. This baseline prefills users serially,
which explains the near-linear TTFT growth with concurrency. The code also
splits B16 GDN into two scan launches, pads single-token decode to a 32-row
chunk, and copies the resulting recurrent state back. These are profiling
targets; source inspection does not establish their measured cost.
At 55K, C1 measures 30.57 tokens/s/user and 9.65 s TTFT; C8 measures
16.33 tokens/s/user (130.64 aggregate) and 78.14 s TTFT.
At 128K, C1/C2/C4/C8 measure 31.87/26.12/19.19/13.75 tokens/s/user;
C4 delivers 76.76 aggregate with 113.14 s TTFT and C8 delivers 109.99 aggregate
with 226.45 s TTFT. Near-256K C1/C2/C4 measured 27.24/22.22/15.60 tokens/s/user,
or 27.24/44.44/62.40 aggregate decode tokens/s, with p50 TTFT
75.19/150.71/301.46 s. Higher-batch cells marked capacity guard were not run.

Initialize with `--replicas 8`
for the follow-up: throughput is measured on all eight replicas rather than
inferred by multiplying the TP4 result.

### Long-context optimization priority and reduced P0 profile

The user prioritizes 128K–256K context and aggregate throughput at useful
concurrency, accepting some short-context slowdown. Select candidates using
those long-context measurements, while retaining short-context controls to
quantify the tradeoff. Accuracy thresholds and context semantics are unchanged:
do not truncate, use sliding-window attention, or lower precision merely to
claim a long-context win. The existing KV-pool guard is a configured limit,
not proof of the maximum physical concurrency.

`demo/run_galaxy_layer_profile.sh TASK_ROOT NEW_RESULTS` uses the same baseline
knobs, the pinned checkpoint, one TP4 submesh, and layers 0/3 (GDN/GQA). The
matrix is 8K at B1/B16, 128K at B1/B8, and 262,016 tokens at B1/B4. The latter
leaves room for decode within the 262,144-token limit. After repeated warmup it profiles an eager decode step with
Tracy signposts around layer, norm, attention and FFN/residual stages. It runs
through the Metal safe pytest wrapper and the shared device lock. The launcher
requires the actual passing JUnit, completed receipt and per-op CSV because the
Tracy wrapper can mask pytest failures. Collection and shell syntax validation
pass; device results are still pending.

The persistent `qwen38-layer-profile-v2-20261006.service` waits for the corrected
eight-replica qualification job. Receipts and Tracy reports go to
`TASK_ROOT/layer-profile-v2/`. The enclosing deadline is eight hours including
queue time; the test itself allows sixty minutes. This reduced eager trace is
for operation attribution. Its host time includes profiling and Python dispatch
and must not be presented as full-model traced TPOT. P0 still needs measured
operation totals reconciled with full-model TPOT and TP8 collective costs.

`tests/layer_profile_report.py` selects only the diagnostic signpost windows,
keeps parallel devices separate, and distinguishes inclusive from exclusive
nested stages. It writes per-device, per-stage and per-op CSV/JSON plus a ranked
Markdown report under `analysis/`. Missing device timings are explicit. Nine
host tests protect against warmup contamination, duplicate rows, trace replay,
unbalanced signposts and incomplete rank coverage. Their receipt is
`galaxy-evidence/long-context-profile-host.xml`. The profile and serving v1
wrappers were verified to be waiting and stopped before replacing their queue;
the running TP4 sweep and other hardware jobs were not interrupted.

Another tuning candidate is cache/chunk alignment. `_ensure_cache` rounds a
fresh allocation to 32 tokens, while `_full_decode` halves `sdpa_k` until it
divides the page-table extent. With the native sweep's 128-token output budget,
55,000 input tokens allocate 55,136 positions and select a 32-token chunk;
131,072 inputs allocate 131,200 positions and retain the TP4 default chunk of
128. This is a source-derived explanation to test for the non-monotonic C1
timings, not a demonstrated cause. Benchmark aligned cache extents and larger
chunks at unchanged precision. Do not simply delete the divisibility guard:
the paged reader reads whole chunks through the page table, so a partial last
chunk needs valid mapped backing even though causal attention masks its tail.

### Attention chunk experiment

`demo/run_long_context_attention.sh TASK_ROOT NEW_RESULTS` runs the independent
TP4 experiment after the profile. Seven geometries cover 8K B1/B16, 55K B1,
128K B1/B8 and near-256K B1/B4. It tests chunks 32/64/128/256/512 on identical
512-aligned cache extents. Query/KV formats and local head geometry match Qwen
(BF16/BFP8, six Q heads, one KV head, head dimension 256). The synthetic inputs
are replicated on four chips; every rank is checked. These are attention-op
measurements, not checkpoint-backed full-layer or full-model qualification.

Random physical page mapping, independent per-user positions and large future
value sentinels exercise causal masking. The FP32 CPU reference reads the actual
quantized device cache. Each user/rank must reach PCC >=0.999 and normalized RMS
error <=0.02. Timing uses five warm samples of 100 trace replays, excluding
reference calculation and readback. This is traced wall cost including dispatch,
not pure device-kernel time. A repeated baseline flags >3% timing drift; fast
but inaccurate candidates cannot win. Nothing is automatically promoted into
the model or serving policy.

Four host tests pass, recorded in `galaxy-evidence/attention-tuning-host.xml`.
The persistent job is `qwen38-attention-tuning-v1-20261006.service`, waiting for
profile-v2, with an eight-hour enclosing deadline, 128 GiB host-memory limit,
eight-core quota and thirty-minute test timeout. Results will be saved to
`TASK_ROOT/attention-tuning-v1/attention.json`. Hardware results remain pending.

### Persistent eight-engine serving and reference evaluation

`demo/run_galaxy_serving.sh TASK_ROOT NEW_RESULTS G0_RECEIPT` waits for an
optional `QWEN_WAIT_FOR_UNIT`, requires that predecessor to succeed, and checks
the completed eight-replica G0 receipt before acquiring the shared device lock.
The actual model Python and precision hashes must match qualification. The
launcher uses the already-prepared `serving_env` and `eval_env`; it neither
installs packages nor changes the preserved native build.

The pinned plugin's serialized standard-DP placement fields carry the exact
qualified chip groups into eight independent TP4 workers. They are an internal
contract of plugin `b7e4292e4193cba20abe9c7c68ce489201b2e36b`, not a generic
cross-version interface. Actual worker log bindings must match all eight
groups. Each engine has `max_num_seqs=16`, for 128 total slots. The configuration
uses the existing CI optimization knobs (compact MLP/attention/RoPE, batched
prefill, decode buckets), Linear fabric, FP32 recurrence and a 1,050,592-token KV
pool per replica. This differs from the initial native baseline; reference
accuracy and live performance for these Galaxy settings remain unqualified.

The supervisor binds `127.0.0.1:8000`, waits up to 90 minutes for readiness,
then checks health, model listing, streaming/nonstreaming greedy agreement,
multiturn chat history and a 128-request burst. It runs all 198 GPQA-D questions
at concurrency 128 with a 32,768-token budget and the recorded 0.892 threshold.
The evaluator distinguishes per-engine capacity from endpoint capacity and has
128 HTTP connections rather than silently queueing behind httpx's default 100.
The four-hour evaluation deadline includes warmup. A completed below-threshold
score remains a failure and is retained for diagnosis.

After GPQA completes, the same resident engines run a whole-Galaxy HTTP sweep
at total concurrency 8/16/32/64/128 across the input lengths above. Its graph
shows median client decode speed, **aggregate end-to-end throughput**, and TTFT.
Independent engines overlap prefill and decode, so this graph does not invent a
global decode-only phase or multiply TP4 speed into a measured Galaxy number.
Client decode timing ends at stream completion, including any suppressed
special-token tail. This fixes a measurement bug that could otherwise inflate
fixed-length throughput by stopping at the final visible text.

The persistent job is `qwen38-galaxy-serving-v4-20261006.service`, queued behind
`qwen38-attention-tuning-v1-20261006.service`, which follows profile-v2. It has a 48-hour enclosing deadline
including queue time, a 256 GiB host-memory limit and a 32-core CPU quota. It
keeps the endpoint resident after the evaluations and sweep. Stop this owned
job with `systemctl --user stop qwen38-galaxy-serving-v4-20261006.service`;
weights and caches remain. The next device job resets a pessimistic dirty
marker through the normal safe-runner path.

Receipts are under `TASK_ROOT/galaxy-serving-v4/`: `deployment.json`,
`server.log`, `api.json`, `gpqa/`, and `http-sweep/`. Connect after readiness
using `ssh -L 8000:127.0.0.1:8000 ttuser@10.228.203.98`; the served model is
`Qwen/Qwen3.8-27B` at `/v1/chat/completions`.

Before starting the resident server, v4 runs the isolated
[`experiments/gdn_step`](experiments/gdn_step/README.md) candidate via
`QWEN_GDN_STEP_EXPERIMENT_DIR=TASK_ROOT/gdn-step-candidate-v1`. It removes the
32-token padded recurrence and temporary DRAM state copy in a standalone
FP32/SFPU kernel. Twelve CPU scheduling/accuracy-gate tests pass; device
compilation, long-horizon accuracy, and latency remain pending. The candidate
is never promoted automatically and a failed candidate does not stop baseline
GPQA. The waiting v3 wrapper alone was replaced; live hardware jobs were not
interrupted. Baseline G0/model-source checks still gate serving.

Fabric v2 completed all eight 2D-torus neighbor-exchange variants (one/two
links and four packet types), with zero failed tests. Its enclosing service
nevertheless exited 1: the upstream script's EXIT cleanup used a final false
conditional when no rank file existed. This was reproduced without hardware
and fixed with an explicit `if`; six host cases cover successful/failed runs
with empty/present/absent rank files. The historical service status remains
failed, while its raw test log records success. Eight-replica qualification
then used the normal safe-runner recovery and started loading at 05:41 UTC.

All 41 launch, benchmark, mocked-HTTP and real-plugin host tests pass, including
the actual vLLM argument parser. The full DP8 dataset preparation preserves the
same 198 input hashes as the earlier preparation. Evidence is in
`galaxy-evidence/serving-launch-host.xml` and
`galaxy-evidence/gpqa-full-dp8-prepared-v1/`. These are preparation receipts;
no live eight-engine API or GPQA score is claimed yet.

## Capacity

- Maximum supported context: 262,144 tokens.
- Largest tested input sequence length: 261,892 tokens with 252 output tokens.
- Up to 16 concurrent users.
- Decode uses batch sizes 1, 8, and 16. Other active-user counts are padded to
  the next supported batch size: 2–8 use batch 8, and 9–16 use batch 16.

## Performance

Batch-1 vLLM performance on QB2/P300x2:

| Input / output tokens | Tokens/s/user | TTFT |
| --- | ---: | ---: |
| 128 / 128 | 39.4 | 67.9 ms |

## Evaluation

Evaluation results:

- GPQA Diamond: 9/10 (90%).
- Terminal-Bench 2.1: 4/5 (80%).
- SWE-bench Verified: 3/5 (60%).

These inherited demo results are small subsets; they are not full Galaxy
qualification. The evaluator now accepts a full 198-question Diamond run while
retaining the ten-question CI default:

```bash
python models/demos/qwen38_27b_qb2/tests/benchmark.py \
  --mode gpqa --base-url http://127.0.0.1:8000 \
  --server-capacity 16 --gpqa-count 198 --gpqa-concurrency 16 \
  --gpqa-max-tokens 32768 --gpqa-threshold 0.892 \
  --output-dir /path/to/new-gpqa-receipts
```

This pins the existing dataset and scoring harness, uses choice-shuffle seed
42 and thinking sampling (temperature 1, top-p 0.95, top-k 20), and records the
exact question selection and protocol. `gpqa-progress.json` updates after every
completed question; the JSONL preserves per-question scores, truncations,
timings, usage and response hashes. Interrupted runs keep completed receipts,
and a rerun refuses to overwrite them. Prompts and generated evaluation content
are not written to these publication artifacts. Seventeen host tests pass,
including synthetic full-dataset accounting, interrupted streams and bounded
concurrency; these tests do not measure model accuracy.

An existing authorized CSV cache can be used with `--gpqa-csv PATH`. The loader
requires its Git blob ID to equal the `gpqa_diamond.csv` entry at the pinned Hub
revision (`7589e3e467d69a1dceb126a60c4108d6d4f1d166`) and records its SHA256. It
then uses the same harness processing, seeded choice shuffle, prompt and scorer.
This path successfully prepared all 198 questions on the Galaxy host after the
Hub download reported `DatasetNotFoundError`; the existing Kimi CSV was verified
byte-for-byte against the pinned Hub blob. Prepared protocol and input hashes
are in `galaxy-evidence/gpqa-prepared-v2/`. This is dataset preparation, not a
completed model evaluation. The CSV and question text are not committed.

The 0.892 threshold above is the published
[model-card GPQA-D score](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/README.md).
It requires at least 177/198 correct in this single seeded run. The card does
not specify an equivalent GPQA harness and output budget, so meeting that
numeric threshold alone does not establish protocol equivalence. The default
CI threshold remains 0.9. Report truncations when comparing output budgets.

For the isolated Galaxy task layout, `demo/prepare_galaxy_serving_env.sh`
prepares `serving_env` and `eval_env` without installing into the running native
test environment. Put the vLLM plugin at `TASK_ROOT/vllm-plugin` checked out at
`b7e4292e4193cba20abe9c7c68ce489201b2e36b` first. The entrypoint requires the
base uv environment to be relocatable, copies it using separate inodes, keeps
the Torch/Transformers/NumPy pins, installs the plugin's pinned vLLM empty
target, and installs the existing evaluation requirements separately. It
refuses to replace existing environments and does not launch a server or open
devices. This setup completed successfully on `.98`; 35 adapter host tests and
18 subtests pass in `serving_env`. Hardware serving and reference evaluations
remain to be qualified.

## Run the demo

Build tt-metal and activate its Python environment:

```bash
./build_metal.sh --enable-ccache
source python_env/bin/activate
export PYTHONPATH="$PWD:$PWD/ttnn:$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_WEIGHTS_DIR=/path/to/Qwen3.8-27B/snapshot
```

Run the model:

```bash
python models/demos/qwen38_27b_qb2/demo/text_demo.py \
  --full --length 128 --generate 128 --output /tmp/qwen38.json
```

For vLLM serving, set `EXTRA_MODELS_DIR` to `models/demos`. The model is
registered as `TTQwen38ForCausalLM`.
