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

At initial publication, 66 host tests pass (plus 40 subtests). The full model
hardware test is running; performance, eight concurrent replicas, long-context
serving and reference evaluation results remain unqualified on Galaxy.

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
