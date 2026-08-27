# BGE-M3

Tenstorrent implementation of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3),
a multilingual embedding model supporting dense, sparse (lexical), and ColBERT
(multi-vector) retrieval.

## Long-context serving: batch 12, sequence length 8192 (N300)

This is the optimized long-context prefill configuration for a **single
Wormhole N300 chip (64 cores, 8×8 grid)**: global batch **12**, input
sequence length (ISL) **8192**, weights and activations in **bfloat8_b**.

### Requirements to launch this configuration

- **Hardware:** one Wormhole N300 (single chip; the demo/perf tests expect
  exactly one device — do **not** set `TT_VISIBLE_DEVICES` to more than one id).
- **Model weights:** `BAAI/bge-m3` (downloaded automatically from HuggingFace
  on first run, or point to a local checkout via `hf_model_name`).
- **Data type:** `ttnn.bfloat8_b`.
- **Device launch parameters** (must match across demo + perf tests):
  - `trace_region_size = 50_000_000` (holds the captured 24-layer encoder program)
  - `num_command_queues = 1`
- **Fixed shapes:** `max_batch_size = 12`, `max_seq_len = 8192`. Prompts are
  padded/truncated to 8192; the batch dimension is exactly 12.

### Run the demo

`demo_traced.py` builds the model, runs one warmup forward to compile the
kernels, captures the trace, and replays it to embed the prompts. Use
`--data-parallel` to run the batch across both chips of an N300, which is the
configuration the performance and evaluation numbers use.

```bash
# Batch 12, sequence length 8192, both chips (the serving shape)
TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_traced.py \
  --batch 12 --seq-len 8192 --data-parallel

# Batch 12, sequence length 8192, one chip
TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_traced.py --batch 12 --seq-len 8192

# Batch 1, sequence length 512, one trace replay per prompt
TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_traced.py --batch 1
```

Batch 1 allows sequence length 512, and batch 12 allows sequence length 8192.
The output is the encoder hidden state `[12, 1, 8192, 1024]` plus one pooled
CLS embedding per prompt.

### Measure prefill performance (`perf.py`)

`test_n300_dp` captures the trace and times only the trace replay
(`execute_trace(blocking=True)`), reporting avg / best ms and embeddings/s for
the B12/S8192 DP=2 shape on one N300. It has two variants:

```bash
# Full 8192-token attention (headline wall-clock latency)
TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_n300_dp -k nomask -s

# Masked serving: compact valid-length masking swept over
# 128 / 512 / 1024 / 2048 / 4096 tokens (padded to 8192). A short request
# skips the all-padding key blocks and finishes faster than the full pass.
TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_n300_dp -k masked -s

# Both variants
TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_n300_dp -s
```

### Run MTEB evaluation (`mteb_eval_minimal.py`)

Evaluates the B12/S8192 DP=2 model against the HF/CPU reference on MTEB
retrieval tasks. `--mode both` scores HF and TT and prints the delta; `hf` or
`tt` run a single backend.

```bash
# HF vs TT on STSBenchmark (full set)
TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py \
  --mode both --task STSBenchmark --output-dir ./mteb_eval_results

# TT only, quick smoke over a few samples
TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py \
  --mode tt --task STSBenchmark --smoke-samples 50
```

Install the eval dependencies once inside `python_env`:

```bash
uv pip install --python python_env/bin/python mteb
```

### Kernel-level profiling (`tracy_perf.py`)

`test_n300_dp_tracy` runs a single B12/S8192 DP=2 forward (no trace capture —
Tracy needs the individual device ops) inside Tracy signposts. Requires
`TT_METAL_DEVICE_PROFILER=1`:

```bash
TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r \
  --no-runtime-analysis -v -m pytest \
  models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py \
  -k "n300_dp_tracy" -sv
```

Reports are saved to
`generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv`. Then
summarize the CSV with `tt-perf-report` (install it once with
`uv pip install --python python_env/bin/python tt-perf-report`; the console
script lands in `python_env/bin/`):

```bash
python_env/bin/tt-perf-report generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv \
  --start-signpost start --end-signpost stop 2>&1 | tee bge_m3_n300_dp_tracy_report.log
```

The stacked report at the end gives the total device kernel time per chip (both
DP replicas run the same program concurrently, so per-chip sets the wall). A
reference run produced:

| Total % | Op | Device time sum |
|--------:|----|----------------:|
| 76.7 % | GenericOpDeviceOperation (encoder SDPA + head-split + concat-heads) | 667.8 ms |
| 18.5 % | MinimalMatmulDeviceOperation (QKV, Wi, Wo, attention output) | 160.6 ms |
|  4.6 % | LayerNormDeviceOperation | 39.9 ms |
|  0.2 % | EmbeddingsDeviceOperation | 2.0 ms |
| **100 %** | **Total device kernel time** | **≈ 870 ms** |

This is the untraced forward (pure device-kernel time), so it sits just under
the traced wall time from `test_n300_dp` (≈ 985 ms); the difference is host
dispatch overhead that trace replay hides. Attention (the `GenericOp` path)
dominates at ~77 %. `tt-perf-report` prints "Unclassified operation" warnings
for the model-local `GenericOp`/`MinimalMatmul` ops — cosmetic, totals are
unaffected.

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

## Trace capture for repeated inference

Trace capture records the model's program once on device and replays it without recompilation, giving the best latency for repeated inference. When using trace capture, follow the warmup → capture → replay pattern:

```python
import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.tt.common import create_tt_model

device = ttnn.open_device(device_id=0, trace_region_size=50_000_000, num_command_queues=1)

model_args, model, _ = create_tt_model(
    mesh_device=device, max_batch_size=1, max_seq_len=512,
    dtype=ttnn.bfloat8_b, hf_model_name="BAAI/bge-m3",
)

# 1. Warmup (JIT compile)
encoded = model_args.encode_prompts(["warmup"], prompt_length=512)
staged = encoded["model_inputs"]
warmup_out = model(**staged)
ttnn.synchronize_device(device)
ttnn.deallocate(warmup_out)

# 2. Capture trace (records the program at fixed device memory addresses)
output_dev = model.capture_trace(
    input_ids=staged["input_ids"],
    attention_mask=staged["attention_mask"],
    token_type_ids=staged["token_type_ids"],
    position_ids=staged["position_ids"],
    mesh_device=device, cq_id=0,
)

# 3. For each new prompt: overwrite device tensors in-place, then replay
for prompt in ["First query.", "Second query.", "Third query."]:
    enc = model_args.encode_prompts([prompt], prompt_length=512)

    # copy_host_to_device_tensor writes new data to the SAME device address
    # the trace reads from — this is how new inputs reach the captured program.
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(enc["input_ids"].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        staged["input_ids"],
    )
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(enc["attention_mask"].bfloat16(),
                        dtype=model_args.attention_mask_dtype, layout=ttnn.TILE_LAYOUT),
        staged["attention_mask"],
    )
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(enc["token_type_ids"].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        staged["token_type_ids"],
    )
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(enc["position_ids"].int(), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT),
        staged["position_ids"],
    )

    model.execute_trace(blocking=True)
    hidden_states = to_torch_auto_compose(output_dev, device=device)
    # ... extract embeddings from hidden_states

model.release_trace()
ttnn.close_device(device)
```

See `demo/demo_traced.py` for a complete runnable example.

## Performance benchmarks

Two benchmark scripts live in `models/demos/wormhole/bge_m3/tests/perf/`.

### `perf.py` — Latency and throughput

Measures trace-replay latency for B1 and B32 at S512. Each iteration copies fresh random inputs to device before replaying the trace, timing only the device execution.

```bash
# Batch 1
TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py -k "batch1" -s

# Batch 32
TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py -k "batch32" -s

# Both
TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py -s
```

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
uv pip install --python python_env/bin/python tt-perf-report
```

Then run:

```bash
python_env/bin/tt-perf-report generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv --start-signpost start --end-signpost stop 2>&1 | tee bge_m3_tracy_report.log
```

## Galaxy multi-chip measurement (data parallel)

`dp_multiprocess.py` benchmarks BGE-M3 across many chips (e.g. a 32-chip
Blackhole Galaxy) with one process per chip. Global batch is `--batch-size` (per
chip) × `--num-devices`; the report shows the H2D / Forward / D2H breakdown plus
throughput. Do not set `TT_VISIBLE_DEVICES` — the script assigns chips itself.

```bash
# Batch 1 per chip, 32 chips (global batch 32)
python models/demos/wormhole/bge_m3/tests/perf/dp_multiprocess.py --batch-size 1 --num-devices 32

# Batch 32 per chip, 32 chips (global batch 1024)
python models/demos/wormhole/bge_m3/tests/perf/dp_multiprocess.py --batch-size 32 --num-devices 32
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
