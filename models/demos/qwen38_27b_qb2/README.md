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

Current job: `qwen38-perf-sweep-tp4-v1-20261006.service`, loading the TP4 model
after the first replica/fabric attempts. Results are under
`TASK_ROOT/perf-sweep-tp4-v1/`; `index.html` is
the artifact entrypoint and updates after each completed cell. Five host tests
validate the metric accounting and reject cold captures as warm measurements.
No sweep cells have been measured at publication. Initialize with `--replicas 8`
for the follow-up: throughput is measured on all eight replicas rather than
inferred by multiplying the TP4 result.

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
