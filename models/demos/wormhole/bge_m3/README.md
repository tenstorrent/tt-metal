# BGE-M3

Tenstorrent implementation of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3),
a multilingual embedding model supporting dense, sparse (lexical), and ColBERT
(multi-vector) retrieval.

## Low-level model creation

Use `create_tt_model()` when you want the raw TT encoder model.

```python
import ttnn

from models.demos.wormhole.bge_m3.tt.common import create_tt_model

device = ttnn.open_device(device_id=0)

model_args, tt_model, state_dict = create_tt_model(
    mesh_device=device,
    max_batch_size=1,
    max_seq_len=128,
    dtype=ttnn.bfloat16,
    hf_model_name="BAAI/bge-m3",
    pooling=None,  # see "Pooling methods" below
)
```

You can then tokenize with `model_args.encode_prompts(...)` and pass `input_ids`, `attention_mask`, and `token_type_ids` to `tt_model`.

### Pooling methods

The `pooling=` argument selects which head the model applies to the encoder's
last hidden state. The model returns the **raw head output**; all downstream
post-processing (CLS crop, attention masking, L2 normalization, vocabulary
scatter, scoring) is the caller's responsibility — this mirrors how the
`BgeM3ForEmbedding` vLLM wrapper consumes the `colbert_linear` / `sparse_linear`
heads.

In the output shapes below, `B` = batch size, `S` = sequence length (number of
tokens), and `D` = hidden dimension (1024 for BGE-M3).

| `pooling`   | Output shape   | Description |
|-------------|----------------|-------------|
| `None`      | `[B, 1, S, D]` | Full last hidden state — no pooling. Use when you want the raw token embeddings. |
| `"cls"`     | `[B, 1, 1, D]` | Dense sentence embedding taken from the first (CLS) token. Normalize for cosine similarity / dense retrieval. |
| `"mean"`    | `[B, 1, 1, D]` | Dense sentence embedding from a mask-weighted mean over valid tokens. |
| `"colbert"` | `[B, 1, S, D]` | Per-token ColBERT projection (`colbert_linear`) for multi-vector / late-interaction retrieval. Caller crops the CLS token, masks padding, and L2-normalizes. |
| `"sparse"`  | `[B, 1, S, 1]` | Per-token sparse (lexical) weights (`sparse_linear`, ReLU applied inside the head). Caller scatters the weights into a `[B, vocab_size]` vector (max over repeated tokens) and zeroes special tokens. |

```python
# Dense (CLS) sentence embeddings
model_args, tt_model, _ = create_tt_model(
    mesh_device=device, max_batch_size=2, max_seq_len=512,
    dtype=ttnn.bfloat8_b, hf_model_name="BAAI/bge-m3", pooling="cls",
)
```

> **Note:** `"colbert"` and `"sparse"` require the M3 head weights
> (`colbert_linear.pt` / `sparse_linear.pt`). These are loaded automatically by
> `ModelArgs.load_state_dict()` when you let `create_tt_model` build the
> state_dict (i.e. pass `state_dict=None`); a state_dict built only from
> `AutoModelForCausalLM` will not contain them.
>
> For an end-to-end worked example of each pooling mode (dense / sparse /
> ColBERT) driven through `create_tt_model(pooling=...)`, see
> `tests/pcc/test_model_pooling.py`.

## Running the model: eager vs trace

`model.forward(...)` runs the encoder. It takes one required argument,
`input_ids`, plus optional `attention_mask`, `token_type_ids`, and
`position_ids` (each defaults to `None`, and the model derives sensible values
internally). The `mode` keyword selects how the forward runs:

- `mode="eager"` (default) — run the program directly. Best for one-off calls,
  debugging, or correctness checks.
- `mode="trace"` — capture the program once, then replay it on every subsequent
  call. Best latency for repeated inference. All the capture/replay bookkeeping
  is handled for you, so the call site is identical to eager.

```python
import ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

device = ttnn.open_device(device_id=0, trace_region_size=50_000_000, num_command_queues=1)

model_args, model, _ = create_tt_model(
    mesh_device=device, max_batch_size=1, max_seq_len=512,
    dtype=ttnn.bfloat8_b, hf_model_name="BAAI/bge-m3",
)

# Eager: run the program directly.
inputs = model_args.encode_prompts(["What is BGE-M3?"], prompt_length=512)["model_inputs"]
output_dev = model.forward(**inputs)                  # mode="eager" is the default

# Trace: the FIRST call warms up + captures; every later call replays.
for prompt in ["First query.", "Second query.", "Third query."]:
    inputs = model_args.encode_prompts([prompt], prompt_length=512)["model_inputs"]
    output_dev = model.forward(**inputs, mode="trace")
    # ... read output_dev back to torch and extract embeddings

ttnn.close_device(device)
```

`encode_prompts(...)["model_inputs"]` returns the four device tensors
(`input_ids`, `attention_mask`, `token_type_ids`, `position_ids`) already
converted to the dtypes/layouts the model expects, so `model.forward(**inputs)`
just works.

See `demo/demo_single_chip.py` for a complete runnable example (batch sizes
1/8/16/32, including the optimized device→host readback).

## How trace works

A *trace* is a recording of the exact sequence of device operations a forward
pass performs. Capturing it once and replaying it skips the host-side overhead
(Python dispatch, op-by-op command building) on every later call, which is what
gives trace mode its latency advantage.

The first time you call `model.forward(**inputs, mode="trace")`, three things
happen automatically:

1. **Warmup** — an eager forward runs first to JIT-compile all kernels. (Kernel
   compilation synchronizes the device, which is not allowed *during* capture,
   so it must happen before.)
2. **Capture** — `ttnn.begin_trace_capture` records the program. The captured
   trace is bound to the **fixed device memory addresses** of the input tensors
   it read during capture.
3. **Replay** — `ttnn.execute_trace` runs the recorded program.

On every subsequent call, the model copies your new inputs into those same fixed
input addresses (a fast device→device copy) and replays the trace — no
recompilation, no recapture. Because the trace is fixed-shape, the model must be
built (`max_batch_size` / `max_seq_len`) for the shape you intend to run.

If you prefer the low-level API, the same three steps are also exposed directly
as `model.capture_trace(...)`, `model.execute_trace(...)`, and
`model.release_trace()` — see `demo/demo_v2.py`.

## Single-chip demo

`demo/demo_single_chip.py` is a complete, runnable example of the trace API on a
single device. It builds the model, encodes prompts, and runs one forward pass
per batch size (1, 8, 16, 32) using `model.forward(**inputs, mode="trace")`, then
reads the result back with the optimized `copy_device_to_torch` device→host path
and prints the resulting embeddings.

```bash
TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_single_chip.py
```

Each batch size builds its own model and trace (a captured trace is fixed-shape).
To run a single batch size, edit `BATCH_SIZES` at the top of the file.

## Multi-chip data-parallel benchmark

After validating the single-chip demo, use `dp_multiprocess.py` to benchmark
BGE-M3 across many chips with one process per chip. Global batch is
`--batch-size` (per chip) × `--num-devices`; the report shows the H2D / Forward /
D2H breakdown plus throughput. Do not set `TT_VISIBLE_DEVICES` — the script
assigns chips itself.

```bash
# Batch 1 per chip, 32 chips (global batch 32)
python models/demos/wormhole/bge_m3/tests/perf/dp_multiprocess.py --batch-size 1 --num-devices 32

# Batch 32 per chip, 32 chips (global batch 1024)
python models/demos/wormhole/bge_m3/tests/perf/dp_multiprocess.py --batch-size 32 --num-devices 32
```

Set `--num-devices` to the number of connected chips you want to use on the
machine.

## Performance benchmarks

### Optimization guidelines

The [`GUIDELINES/`](./GUIDELINES/) folder documents the optimization work behind this
demo: best practices for optimizing BGE-style encoder transformers (BERT / XLM-R) on
Tenstorrent hardware (TT-NN). It covers normalization, QKV projection, attention/SDPA,
MLP, fusion/residuals, profiling and op analysis, and the overall methodology — distilled
from the BGE-M3 optimization campaign (B1 5.7→4.30 ms; B32 194.9→60.55 ms, 3.22×).
Start with [`00_README.md`](./GUIDELINES/00_README.md) or
[`AGENT_INDEX.md`](./GUIDELINES/AGENT_INDEX.md).

Two benchmark scripts live in `models/demos/wormhole/bge_m3/tests/perf/`.

### `perf.py` — Latency and throughput

Measures trace-replay latency for B1 and B32 at S512. Each iteration copies fresh random inputs to device before replaying the trace, timing only the device execution.

```bash
# Batch 1
TT_VISIBLE_DEVICES=0 pytest "models/demos/wormhole/bge_m3/tests/perf/perf.py::test_embedding_perf[blackhole-device_params0-batch1]" -s

# Batch 32
TT_VISIBLE_DEVICES=0 pytest "models/demos/wormhole/bge_m3/tests/perf/perf.py::test_embedding_perf[blackhole-device_params0-batch32]" -s

# All batch sizes (1, 8, 16, 32)
TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_embedding_perf -s
```

### Accuracy evaluation (MTEB)

`demo/mteb_eval_minimal.py` evaluates embedding quality on standard
[MTEB](https://github.com/embeddings-benchmark/mteb) tasks (`STSBenchmark`,
`SICK-R`, `ArguAna`) and can compare the TT model against the HuggingFace
reference side by side.

Install the MTEB requirements first:

```bash
source python_env/bin/activate
uv pip install -r models/demos/wormhole/bge_m3/demo/requirements_mteb.txt
```

Run STSBenchmark on a single chip (TT only):

```bash
TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py \
    --task STSBenchmark --mode tt --batch-size 32 --max-seq-len 512
```

To also run the HuggingFace reference and print a side-by-side comparison, use
`--mode both` (the default):

```bash
TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py \
    --task STSBenchmark --mode both
```

The TT model scores ~0.846 (Spearman) on STSBenchmark, matching the BGE-M3
reference. Results are written to `./mteb_eval_results/comparison.json`.

### `tracy_perf.py` — Kernel-level profiling

Runs a single forward pass inside Tracy signposts for device-level op reports. Requires `TT_METAL_DEVICE_PROFILER=1` — the test will error if it's not set.

```bash
# Batch 1
TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r --no-runtime-analysis -v -m pytest models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py -k "batch1" -sv

# Batch 32
TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r --no-runtime-analysis -v -m pytest models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py -k "batch32" -sv
```

Reports are saved to `generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv` with per-kernel device timing, core utilization, and memory layout.

To generate a human-readable summary from the CSV report, first install `tt-perf-report` if you haven't already:

```bash
source python_env/bin/activate
uv pip install tt-perf-report
```

Then run:

```bash
tt-perf-report generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv --start-signpost start --end-signpost stop 2>&1 | tee bge_m3_tracy_report.log
```

## Embedding API

For dense, sparse, and ColBERT-style embeddings, use `BgeM3ForEmbedding`.

```python
import torch
import torch.nn.functional as F
import ttnn

from models.demos.wormhole.bge_m3.demo.generator_vllm import BgeM3ForEmbedding
from models.demos.wormhole.bge_m3.demo.m3_scores import (
    compute_colbert_score_torch,
    compute_dense_score_torch,
    compute_sparse_score_torch,
)

device = ttnn.open_device(device_id=0)

sentences_1 = ["What is BGE M3?", "Definition of BM25"]
sentences_2 = [
    "BGE M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction.",
    "BM25 is a bag-of-words retrieval function that ranks documents based on matching query terms.",
]

model = BgeM3ForEmbedding(
    device=device,
    max_batch_size=2,
    max_seq_len=512,
    tt_data_parallel=1,
    dtype=ttnn.bfloat8_b,
    model_name="BAAI/bge-m3",
    sentence_pooling_method="cls",
    return_dense=True,
    return_sparse=True,
    return_colbert=True,
)
model._initialize_model()
model_args = model.model_args
```

Notes:

- The current generator path is single-device.
- `sentence_pooling_method` controls how `dense_vecs` are produced from the last hidden state.
- The default is `"mean"`, which averages token embeddings across the non-padded tokens in the prompt.
- `"cls"` pools from the first token and matches the reference setup used in `tests/pcc/test_generator_vllm.py`.
- `"last_token"` pools from the last valid token in each prompt.
- The returned tensors are padded to `max_batch_size`, so slice back to your real batch size.

## Dense pooling modes

`BgeM3ForEmbedding` currently supports these `sentence_pooling_method` values:

- `"mean"`: (default) averages token embeddings using the attention mask.
- `"cls"`: returns the embedding from the first token position.
- `"last_token"`: returns the embedding from the last non-padding token.

Example with the default behavior:

```python
model = BgeM3ForEmbedding(
    device=device,
    max_batch_size=2,
    max_seq_len=512,
    model_name="BAAI/bge-m3",
    return_dense=True,
)
```

## Run inference (Example)

```python
def encode(sentences,model_args,model):
    encoded = model_args.encode_prompts(sentences)
    outputs = model.forward(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        token_type_ids=encoded.get("token_type_ids", torch.zeros_like(encoded["input_ids"])),
    )

    seq_len = encoded["input_ids"].shape[1]
    batch_size = len(sentences)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "dense_vecs": outputs["dense_vecs"][:batch_size].to(torch.float32),
        "dense_vecs_norm": F.normalize(outputs["dense_vecs"][:batch_size].to(torch.float32), dim=-1),
        "sparse_vecs": outputs["sparse_vecs"][:batch_size].to(torch.float32),
        "colbert_vecs": outputs["colbert_vecs"][:batch_size, : seq_len - 1].to(torch.float32),
        "colbert_vecs_norm": F.normalize(outputs["colbert_vecs"][:batch_size, : seq_len - 1].to(torch.float32), dim=-1),
    }

embeddings_1 = encode(sentences_1, model_args, model)
embeddings_2 = encode(sentences_2, model_args, model)
```

## Dense retrieval

`dense_vecs` are sentence embeddings. Normalize them before computing similarity.

```python
similarity = compute_dense_score_torch(
    embeddings_1["dense_vecs_norm"],
    embeddings_2["dense_vecs_norm"],
)
print(similarity)
```

## Sparse retrieval

`sparse_vecs` are lexical-weight vectors over the vocabulary. Use them for sparse matching.

```python
sparse_scores = compute_sparse_score_torch(
    embeddings_1["sparse_vecs"],
    embeddings_2["sparse_vecs"],
)
print(sparse_scores)
```

## ColBERT / multi-vector retrieval

`colbert_vecs` are token-level multi-vector embeddings. Normalize them before scoring.

```python
colbert_scores = compute_colbert_score_torch(
    embeddings_1["colbert_vecs_norm"],
    embeddings_2["colbert_vecs_norm"],
    q_mask=embeddings_1["attention_mask"],
)
print(colbert_scores)
```

The ColBERT path skips the first token internally, which is why the examples slice token vectors to `: seq_len - 1`.

## Reference examples

- `models/demos/wormhole/bge_m3/tests/pcc/test_generator_vllm.py`
