# BGE-M3

Tenstorrent implementation of [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3),
a multilingual embedding model with dense, sparse (lexical) and ColBERT
(multi-vector) outputs.

This branch is tuned for **one Blackhole chip at sequence length 512** (for
example one chip of a Blackhole Galaxy) at batch 1, 8, 16 and 32, with weights
and activations in `bfloat8_b`.

## Setup

Run every command from the tt-metal root, inside `python_env`, on one chip:

```bash
export TT_METAL_HOME=$PWD PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=0          # one chip; any single chip id works
```

The weights (`BAAI/bge-m3`) download from HuggingFace on the first run.

## Performance (`tests/perf/perf.py`)

`test_perf` builds the model, captures the 24-layer encoder in a trace, and
times the trace replay (10 iterations, average and best). It reports latency,
embeddings/s and tokens/s.

Each test ID has three parts:

| part | values | meaning |
|---|---|---|
| batch | `b1_s512`, `b8_s512`, `b16_s512`, `b32_s512` | batch size at sequence length 512 |
| mask | `nomask`, `masked` | `nomask`: all 512 tokens are valid. `masked`: the model applies the padding mask (the serving path) |
| mode | `forward`, `2cq`, `h2d_d2h` | what the timer covers, see below |

- `forward`: the trace replay only (device time, no host cost).
- `2cq`: the next input uploads on a second command queue while the trace runs.
- `h2d_d2h`: one full request: input upload, trace, and output download to the host.

```bash
# One batch, device time
pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_perf -k "forward and b8_s512 and nomask" -s

# Every batch, device time
pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_perf -k "forward" -s

# Everything (4 batches x 2 mask settings x 3 modes)
pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_perf -s
```

Reference `forward` results, `nomask`, one Blackhole Galaxy chip (12x10 compute
grid, stock 130 W power limit):

| batch | latency (ms) | embeddings/s | tokens/s |
|---|---|---|---|
| 1  | 3.37  | 297  | 152k |
| 8  | 11.99 | 667  | 342k |
| 16 | 22.48 | 712  | 364k |
| 32 | 45.21 | 708  | 362k |

## Kernel profiling (`tests/perf/tracy_perf.py`)

`test_bge_m3_tracy_perf` runs one forward (no trace, so Tracy sees each device
op) between the signposts `start` and `stop`. It needs
`TT_METAL_DEVICE_PROFILER=1`. Select the batch and mask with the exact test ID:

```bash
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r --no-runtime-analysis -v -m pytest \
  "models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py::test_bge_m3_tracy_perf[device_params0-batch8-nomask]" -sv
```

IDs: `batch1`, `batch8`, `batch16`, `batch32`, each with `-nomask` or `-masked`.
If Tracy cannot connect (for example when the hostname resolves to an address
that is not on the machine), give it a port with `-t 8086`.

The report is
`generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv`. For a
per-op summary, install `tt-perf-report` once and run it on the CSV:

```bash
uv pip install --python python_env/bin/python tt-perf-report
python_env/bin/tt-perf-report generated/profiler/reports/<timestamp>/ops_perf_results_<timestamp>.csv \
  --start-signpost start --end-signpost stop
```

`tt-perf-report` shows "Unclassified operation" for the model-local ops
(`GenericOp`). This does not change the totals.

## Accuracy (MTEB, `demo/mteb_eval_minimal.py`)

Install the eval packages once. Keep the pydantic pin of
`tt_metal/python_env/requirements-dev.txt` (uv then selects mteb 2.12.10):

```bash
uv pip install --python python_env/bin/python mteb 'pydantic==2.9.2'
```

Score the model on one chip at S512, at each batch size (masked path, CLS
pooling, cosine similarity):

```bash
python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py --mode tt \
  --batch 1 8 16 32 --task STSBenchmark ArguAna --output-dir ./mteb_eval_results/s512
```

The script writes one folder per batch (`tt_b<B>/`) and a summary
`tt_single_chip_scores.json`. Add `--smoke-samples 50` for a quick check.

Reference scores, one Blackhole Galaxy chip:

| task | B1 | B8 | B16 | B32 | HF at 512 tokens | HF at 8192 tokens |
|---|---|---|---|---|---|---|
| STSBenchmark (Spearman) | 84.70 | 84.61 | 84.67 | 84.58 | — | 84.87 |
| ArguAna (nDCG@10) | 54.49 | 54.24 | 54.33 | 54.23 | 54.05 | 53.99 |

mteb's HF reference for BAAI/bge-m3 uses 8192 tokens. ArguAna has documents
longer than 512 tokens, so compare S512 scores with HF at 512 tokens.

## Long context on Wormhole N300 (batch 12, sequence length 8192)

The same model also runs batch 12 at sequence length 8192 across both chips of
a Wormhole N300 (data parallel, `trace_region_size=50_000_000`). On an N300,
`perf.py` selects this shape automatically.

```bash
# Demo: embed prompts with trace replay
python models/demos/wormhole/bge_m3/demo/demo_traced.py --batch 12 --seq-len 8192 --data-parallel

# Performance
pytest models/demos/wormhole/bge_m3/tests/perf/perf.py::test_perf -k nomask -s

# Kernel profile
TT_METAL_DEVICE_PROFILER=1 python -m tracy -p -r --no-runtime-analysis -v -m pytest \
  models/demos/wormhole/bge_m3/tests/perf/tracy_perf.py -k "n300_dp_tracy" -sv

# MTEB, HF on CPU against TT
python models/demos/wormhole/bge_m3/demo/mteb_eval_minimal.py --mode both --task STSBenchmark \
  --output-dir ./mteb_eval_results
```

# Python API

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
