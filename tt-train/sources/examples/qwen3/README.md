# Qwen3 — Python Implementation with ttml Ops

Implements [Qwen3](https://huggingface.co/docs/transformers/model_doc/qwen3) in Python using `ttml` ops.
Three main scripts share a consistent CLI and cover the full workflow: correctness validation (forward & backward) and training.

## Models

| File | Description |
|------|-------------|
| `model_qwen3.py` | Single-device model, config, weight loading from HF |
| `model_qwen3_distributed.py` | Tensor-parallel model (ColumnParallel / RowParallel) |

Both models are selected automatically based on `--mesh_shape`.

## Main Scripts

All three scripts load a pretrained HuggingFace Qwen3 model and accept the same core flags:

| Flag | Description | Default |
|------|-------------|---------|
| `--model_path` | HuggingFace model (e.g. `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-8B`) | required |
| `--max_seq_len` | Maximum sequence length | `128` |
| `--mesh_shape ROWS COLS` | Device mesh `[DP, TP]` | `1 1` |
| `--sharded_loss` | Keep LM head output vocab-sharded (distributed cross-entropy) | off |
| `--checkpoint` | Gradient checkpointing (activation recomputation) | off |
| `--scatter_intermediates` | Scatter saved activations across TP devices | off |
| `--track_memory [N]` | Memory tracking (snapshot every N-th layer) | off |

### 1. `generate.py` — Forward / Generation Correctness

Compares autoregressive generation between HuggingFace (CPU) and ttml (Tenstorrent device).
Reports per-step logit PCC, top-1 match rate, and per-token timing.

```bash
# Single device
python generate.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \
    --max_tokens 32 --max_seq_len 128

# Tensor parallelism
python generate.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \
    --max_tokens 32 --max_seq_len 128 --mesh_shape 1 8

# Data parallelism
python generate.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \
    --max_tokens 32 --max_seq_len 128 --mesh_shape 4 1

# DP + TP
python generate.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \
    --max_tokens 32 --max_seq_len 128 --mesh_shape 4 8

# On-device sampling (no D2H logits transfer)
python generate.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \
    --max_tokens 32 --max_seq_len 128 --mesh_shape 1 8 --no_logits

# Sharded loss + on-device sampling
python generate.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \
    --max_tokens 32 --max_seq_len 128 --mesh_shape 1 8 --sharded_loss --no_logits

# Batched generation
python generate.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \
    --max_tokens 32 --max_seq_len 128 --mesh_shape 2 1 --batch_size 4

# KV cache (prefill + decode)
python generate.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \
    --max_tokens 32 --max_seq_len 128 --kv_cache
```

**Extra flags:** `--prompt`, `--max_tokens`, `--temperature`, `--batch_size`, `--kv_cache`, `--no_logits`

### 2. `gradients.py` — Backward / Gradient Correctness

Compares per-parameter gradients (forward + backward with cross-entropy loss) between HuggingFace (CPU) and ttml.
Reports AbsDiff, RelDiff, CosSim, and gradient norms for every parameter.

```bash
# Single device
python gradients.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \
    --max_seq_len 128

# Tensor parallelism
python gradients.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \
    --max_seq_len 128 --mesh_shape 1 8

# TP + gradient checkpointing
python gradients.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \
    --max_seq_len 128 --mesh_shape 1 8 --checkpoint

# TP + checkpointing + scattered intermediates
python gradients.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \
    --max_seq_len 128 --mesh_shape 1 8 --checkpoint --scatter_intermediates

# Data parallelism
python gradients.py --model_path Qwen/Qwen3-0.6B --prompt "Once upon a time" \
    --max_seq_len 128 --mesh_shape 4 1

# DP + TP
python gradients.py --model_path Qwen/Qwen3-8B --prompt "Once upon a time" \
    --max_seq_len 128 --mesh_shape 4 8 --checkpoint --scatter_intermediates
```

**Extra flags:** `--prompt`, `--batch_size`

### 3. `train.py` — Training / Fine-tuning

Fine-tunes a Qwen3 model on a next-token-prediction objective.
Supports full fine-tuning, LoRA, checkpointing, and HF-compatible export.

```bash
# Single device, wikitext
python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \
    --dataset wikitext --steps 500 --lr 1e-4

# Tensor parallelism
python train.py --model_path Qwen/Qwen3-8B --max_seq_len 128 \
    --dataset wikitext --steps 500 --lr 1e-4 --mesh_shape 1 8

# TP + gradient checkpointing + scattered intermediates
python train.py --model_path Qwen/Qwen3-8B --max_seq_len 128 \
    --dataset wikitext --steps 500 --lr 1e-4 --mesh_shape 1 8 \
    --checkpoint --scatter_intermediates

# LoRA fine-tuning
python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \
    --dataset wikitext --steps 500 --lr 1e-4 --lora_rank 8

# Save checkpoints + TensorBoard
python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \
    --dataset wikitext --steps 500 --lr 1e-4 \
    --save_dir ./output --save_every 100

# Resume from checkpoint
python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \
    --dataset wikitext --steps 500 --lr 1e-4 \
    --save_dir ./output --resume_from ./output/checkpoints/step_200

# Export HF-compatible model after training
python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \
    --dataset wikitext --steps 500 --lr 1e-4 \
    --export_hf_dir ./my_finetuned_model

# Train on local text file
python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \
    --dataset /path/to/corpus.txt --steps 1000 --lr 5e-5

# Train on C/C++ source code
python train.py --model_path Qwen/Qwen3-0.6B --max_seq_len 128 \
    --dataset sourcecode:/path/to/dir --steps 1000 --lr 5e-5
```

**Extra flags:** `--dataset`, `--steps`, `--lr`, `--batch_size`, `--gradient_accumulation_steps`, `--clip_grad_norm`, `--warmup_steps`, `--lr_schedule`, `--freeze_embeddings` / `--no-freeze_embeddings`, `--lora_rank`, `--lora_alpha`, `--lora_targets`, `--save_dir`, `--save_every`, `--resume_from`, `--export_hf_dir`, `--eval_every`, `--gen_every`, `--timings`

## Tensor Parallelism

| Component | Strategy | Communication |
|-----------|----------|---------------|
| Embedding (untied) | Sharded along hidden dim | all-gather after lookup |
| Embedding (tied) | VocabParallel (Megatron-LM) | all-reduce hidden vectors |
| Q, K, V projections | ColumnParallel | broadcast input |
| O projection | RowParallel | all-reduce output |
| MLP gate/up | ColumnParallel | broadcast input |
| MLP down | RowParallel | all-reduce output |
| RMSNorm, QK-Norm | Replicated | None |
| LM head | ColumnParallel + gather | broadcast + all-gather |

**2 all-reduces per layer.** TP requires all head counts, hidden, and intermediate sizes divisible by `tp_size`.

## Weight Loading

- Q/K weight/bias unpermutation (HF `[real_half, imag_half]` → ttml interleaved)
- QK-Norm weight unpermutation
- Shape adaptation (HF 2D → ttml 4D)
- Vocabulary padding to tile alignment
- Distributed: sharded weights via `shard_tensor_to_mesh_mapper`

## Project Structure

```
qwen3/
├── generate.py                  # Forward / generation correctness
├── gradients.py                 # Backward / gradient correctness
├── train.py                     # Training / fine-tuning
├── model_qwen3.py               # Single-device model
├── model_qwen3_distributed.py   # Tensor-parallel model
└── utils/
    ├── checkpoint.py            # Gradient checkpointing (activation recomputation)
    ├── context_managers.py      # empty_init context manager
    ├── dataset.py               # TextDataset, SourceCodeDataset loaders
    ├── device_setup.py          # Device/mesh initialization
    ├── dist_helpers.py          # Sharded/replicated tensor constructors
    ├── distributed_ops.py       # AllGather/Scatter autograd ops, VocabParallelEmbedding
    ├── kv_cache.py              # KV cache, causal/decode masks
    ├── lora.py                  # LoRA adapter injection
    ├── memory.py                # Memory usage tracking
    ├── model_factory.py         # Unified model creation (auto-selects single/distributed)
    ├── param_utils.py           # Weight mapping, permutation transforms
    ├── save_load.py             # Checkpoint save/load, HF export
    ├── sharded_loss.py          # Distributed cross-entropy (vocab-sharded)
    └── tensor_utils.py          # Tensor creation, padding, mesh gather helpers
```

## Hardware / Descriptor Notes

This example was developed and tested on a **Wormhole (WH) LoudBox** (T3K).

Device initialisation lives in `utils/device_setup.py` and is shared by all
three entry-point scripts (`generate.py`, `gradients.py`, `train.py`).
It contains a hardcoded lookup table (`_MGD_TABLE`) that maps mesh shapes to
mesh-graph descriptor files and fabric configs:

- `(2, 4)` → `t3k_mesh_graph_descriptor.textproto`, `FABRIC_2D`
- `(1, 8)` → `t3k_1x8_mesh_graph_descriptor.textproto`, `FABRIC_2D_TORUS_XY`

Any shape not in the table falls back to `FABRIC_2D` with auto-discovery.
To run on a different machine or topology you will likely need to update
`_MGD_TABLE` (or set `TT_MESH_GRAPH_DESC_PATH` manually).

The Python-side table was introduced to avoid the hardcoded fallback to
`t3k_mesh_graph_descriptor` that the C++ path
(`ttml/ttnn_fixed/distributed/tt_metal.cpp`) uses for 8-device setups, that
hangs on LoudBox.

## Nice-to-Have TODOs

- **Slow generation** — decode-step attention masks and KV cache updates
  involve host-to-device communication every token. Same applies to
  sampling/generation with sharded loss / scattered last block.

- **VocabParallelEmbedding & sharded loss** — `_vocab_parallel_embedding`
  (`utils/distributed_ops.py`) and `sharded_cross_entropy_loss`
  (`utils/sharded_loss.py`) are composite ops with NumPy host-side logic.
  Implement as proper device kernels.

- **AllGatherFwdScatterBwd** — (`utils/distributed_ops.py`) is a workaround
  custom autograd op (all_gather forward, scatter backward). Replace with the
  appropriate built-in op when one becomes available.
