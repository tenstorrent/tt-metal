# BGE-M3

Tenstorrent implementation of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3),
a multilingual embedding model supporting dense, sparse (lexical), and ColBERT
(multi-vector) retrieval.

> **This branch (`bge_n300_tp_optimizations`) targets the two-chip
> sequence-parallel (SP) B12/S8192 prefill path.** Its perf and Tracy
> instructions are the section immediately below. The single-chip section that
> follows is retained for reference.

## Two-chip sequence-parallel: batch 12, sequence length 8192 (N300, TP2/SP)

This branch provides a **two-chip sequence-parallel (SP)** prefill path for
B12/S8192. The sequence dimension is split across the two ASICs of one Wormhole
N300 (a `(2, 1)` mesh); each chip computes its local `S/2 = 4096` query rows and
the attention K/V are shared across chips with an `all_gather` over the mesh
axis. The tests are labelled `tp2` for historical reasons but the parallelism is
sequence-parallel, not classic tensor-parallel.

### Requirements to launch this configuration

- **Hardware:** one Wormhole N300 exposed as **two devices** (local + remote
  chip). Select a single N300 with `TT_VISIBLE_DEVICES=0`; the `(2, 1)` mesh then
  maps to that board's two chips. Do **not** enumerate more than one board.
- **Fabric:** SP requires the 1D fabric for the cross-chip K/V `all_gather`, so
  the device parameters add `fabric_config = ttnn.FabricConfig.FABRIC_1D` on top
  of the single-chip parameters (`trace_region_size = 50_000_000`,
  `num_command_queues = 1`).
- **Data type / shapes:** same as the single-chip path (`bfloat8_b`,
  `max_batch_size = 12`, `max_seq_len = 8192`).

### Measure prefill performance (`tp2_perf.py`)

Builds the SP model, warms up, captures the trace, and times trace replays
(device compute only); reports avg / best ms and tokens/s:

```bash
TT_VISIBLE_DEVICES=0 pytest \
  models/demos/wormhole/bge_m3/tests/perf/tp2_perf.py::test_embedding_perf_b12_s8192_tp2 -s
```

### Kernel-level profiling (`tracy_perf.py`, TP2/SP case)

Profiles a single untraced sequence-parallel forward between Tracy signposts
(no trace capture — Tracy needs the individual device ops). Requires
`TT_METAL_DEVICE_PROFILER=1`. **Before profiling, clear any stale profiler logs**
so the report is not contaminated by a previous run:

```bash
TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r \
  --no-runtime-analysis -m pytest \
  models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py::test_bge_m3_tracy_perf_b12_s8192_tp2 -sq
```

Summarize the resulting CSV with `tt-perf-report`:

```bash
tt-perf-report generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv \
  --start-signpost start --end-signpost stop 2>&1 | tee bge_m3_b12_s8192_tp2_tracy_report.log
```


**Reading the SP profile:** unlike the single-chip path, the SP breakdown
contains `AllGatherDeviceOperation` ops — the cross-chip K/V gather. On device 0
this is roughly `SDPA` (~784 ms) + `AllGather` (~340 ms, the SP tax) +
`MinimalMatmul` (~230 ms) + LayerNorm/head ops. SP prefill measures ~1559 ms
best (~62.9k tokens/s), slower than the single-chip and data-parallel paths
because of the ~340 ms all-gather tax (one ethernet link per mesh axis).

## Long-context serving: batch 12, sequence length 8192 (N300)

This is the optimized long-context prefill configuration for a **single
Wormhole N300 chip (64 cores, 8×8 grid)**: global batch **12**, input
sequence length (ISL) **8192**, weights and activations in **bfloat8_b**.

### Requirements to launch this configuration

- **Hardware:** one Wormhole N300 (single chip; the demo/perf tests expect
  exactly one device — do **not** set `TT_VISIBLE_DEVICES` to more than one id).
- **Model weights:** `BAAI/bge-m3` (downloaded automatically from HuggingFace
  on first run, or point to a local checkout via `hf_model_name`).
- **Data type:** `ttnn.bfloat8_b` (holds the 0.94 hidden-state PCC gate at
  B12/S8192 — measured 0.961).
- **Device launch parameters** (must match across demo + perf tests):
  - `trace_region_size = 50_000_000` (holds the captured 24-layer encoder program)
  - `num_command_queues = 1`
- **Fixed shapes:** `max_batch_size = 12`, `max_seq_len = 8192`. Prompts are
  padded/truncated to 8192; the batch dimension is exactly 12.

### Run the demo

Builds the model, runs a single warmup forward (JIT compile), captures the
trace, then does a single trace replay (one forward pass) over 12 prompts:

```bash
TT_VISIBLE_DEVICES=0 python models/demos/wormhole/bge_m3/demo/demo_long_seq.py
```

Output is the encoder hidden state `[12, 1, 8192, 1024]` plus one pooled
(CLS + L2-normalized) embedding per prompt.

### Measure prefill performance (`perf.py`)

Trace-capture latency/throughput for B12/S8192. Each iteration copies fresh
random inputs to device and times only the trace replay (device compute);
reports avg / best ms, embeddings/s, and tokens/s:

```bash
TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/perf/perf.py -k "b12_s8192" -s
```

### Kernel-level profiling (`tracy_perf.py`)

Runs a single forward pass (no trace capture — Tracy needs the individual
device ops) inside Tracy signposts. Requires `TT_METAL_DEVICE_PROFILER=1`:

```bash
TT_VISIBLE_DEVICES=0 TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r \
  --no-runtime-analysis -v -m pytest \
  models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py \
  -k "b12_s8192" -sv
```

Reports are saved to
`generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv`. Then
summarize the CSV with `tt-perf-report` (install it once with
`uv pip install tt-perf-report` inside `python_env`; see the
**`tracy_perf.py` — Kernel-level profiling** section below for details):

```bash
tt-perf-report generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv \
  --start-signpost start --end-signpost stop 2>&1 | tee bge_m3_b12_s8192_tracy_report.log
```

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

See `demo/demo_v2.py` for a complete runnable example.

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
source python_env/bin/activate
uv pip install tt-perf-report
```

Then run:

```bash
tt-perf-report generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv --start-signpost start --end-signpost stop 2>&1 | tee bge_m3_tracy_report.log
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
