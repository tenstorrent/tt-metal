# BGE-M3 User Guide

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
)
```

You can then tokenize with `model_args.encode_prompts(...)` and pass `input_ids`, `attention_mask`, and `token_type_ids` to `tt_model`.

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
Blackhole Galaxy) using **one process per chip**. Each worker isolates a single
chip via `TT_VISIBLE_DEVICES`, builds its own model and trace, and is pinned to a
dedicated CPU core range. Workers rendezvous on a barrier after warmup so build /
compile of slower workers never overlaps the measurement window. This sidesteps
the single-process mesh-dispatch serialization and gives near-linear scaling.

The global batch is `--batch-size` (per chip) × `--num-devices`. Each worker runs
the full H2D → Forward (trace) → D2H pipeline and reports per-stage timing; the
orchestrator aggregates per-chip medians into global throughput (gated by the
slowest chip, since all chips run in lockstep after the barrier).

```bash
# Batch 1 per chip, 32 chips -> global batch 32
python models/demos/wormhole/bge_m3/tests/perf/dp_multiprocess.py \
    --batch-size 1 --num-devices 32

# Batch 32 per chip, 32 chips -> global batch 1024
python models/demos/wormhole/bge_m3/tests/perf/dp_multiprocess.py \
    --batch-size 32 --num-devices 32
```

Useful flags (defaults match `perf.py`): `--seq-len` (512), `--iterations` (10
measured), `--warmup` (3 trace warmups), `--num-devices` (32). Reduce
`--num-devices` (e.g. 4 or 8) for a quick smoke test before the full run.

The report prints, per chip and in aggregate, the H2D / Forward / D2H breakdown
plus throughput in embeddings/s and tokens/s. Do **not** set
`TT_VISIBLE_DEVICES` when launching — the script assigns each worker its own
chip internally.

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
