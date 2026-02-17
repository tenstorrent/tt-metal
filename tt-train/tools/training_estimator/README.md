# Training Estimator

Analytical model for estimating **step time** and **per-chip peak memory** when training Llama 3 8B on Tenstorrent Galaxy hardware with various distributed training configurations.

## Purpose

Before committing multi-Galaxy hardware to a long training run, this tool lets you:

1. **Predict step time** for any combination of TP, DDP, gradient accumulation, gradient checkpointing, and 2-tier multi-host training — after running a small set of benchmark experiments on a single Galaxy.
2. **Predict per-chip memory** to determine maximum batch size before OOM.
3. **Run what-if analysis** to estimate the impact of future optimizations (better MFU, communication overlap, flash attention, fused FFN) without implementing them first.
4. **Find the optimal configuration** (TP, batch, grad accum, number of workers) that maximizes tokens/sec/dollar.

## Files

| File | Description |
|------|-------------|
| `config.py` | `EstimatorConfig` dataclass with all model, measured, and tuning parameters |
| `estimator.py` | `estimate_step_time()`, `estimate_memory_per_chip()`, report/comparison printing |
| `EXPERIMENTS.md` | Detailed instructions for every benchmark experiment |
| `README.md` | This file — methodology, assumptions, formulas |

## Quick Start

```bash
# 1. Fill in measured values in config.py (see EXPERIMENTS.md)
# 2. Run
cd tt-train/tools/training_estimator
python3 estimator.py
```

For programmatic what-if analysis:

```python
from config import EstimatorConfig
from estimator import estimate_step_time, estimate_memory_per_chip, print_comparison
from dataclasses import replace

cfg = EstimatorConfig(
    fwd_time_s=2.0, bwd_time_s=4.0, opt_time_s=0.5,
    tp_ccl_fwd_s=0.3, tp_ccl_bwd_s=0.5,
    ddp_ar_time_s=0.2,
    two_tier_comm_per_worker_s=1.0,
    tp=8, local_batch_size=2, grad_accum_steps=4,
    n_workers=3, use_2tier=True,
)

# Compare configurations
print_comparison([
    ("Baseline",     cfg),
    ("Better MFU",   replace(cfg, mfu_scale=0.7)),
    ("Overlap DDP",  replace(cfg, ddp_ccl_scale=0.0)),
    ("All improved", replace(cfg, mfu_scale=0.7, tp_ccl_scale=0.5,
                             ddp_ccl_scale=0.0, inter_host_ccl_scale=0.5)),
])
```

---

## Step Time Model

### Overview

One optimizer step consists of these sequential phases (no overlap in the current implementation):

```
┌─────────────────────────────────────────────────────┐
│  Worker Compute (repeated for each micro-batch)     │
│  ┌─────────────────────┐ ┌────────────────────────┐ │
│  │ Forward + Fwd CCL   │ │ Backward + Bwd CCL     │ │
│  │ (×2 if checkpointed)│ │                        │ │
│  └─────────────────────┘ └────────────────────────┘ │
│           × grad_accum_steps                        │
├─────────────────────────────────────────────────────┤
│  DDP Gradient Sync (within Galaxy)                  │
├─────────────────────────────────────────────────────┤
│  2-Tier Communication (if multi-host)               │
├─────────────────────────────────────────────────────┤
│  Optimizer Step                                     │
└─────────────────────────────────────────────────────┘
```

### Formula

```
step_time = worker_compute + ddp_sync + two_tier_comm + optimizer_time
```

Where each component is defined below.

#### 1. Forward Compute (per micro-batch)

```
fwd_compute = (fwd_time × (1 - tp_perf_perc) + fwd_time × tp_perf_perc / TP) × B × MFU_SCALE
```

- `fwd_time`: Measured forward pass time at TP=1, DDP=1, batch=1
- `tp_perf_perc`: Fraction of compute that parallelizes with TP (~0.85 for transformers: linear layers dominate, but LayerNorm, softmax, embedding, loss don't shard)
- `TP`: Tensor parallel degree
- `B`: Local batch size per chip
- `MFU_SCALE`: Compute efficiency tuning knob (1.0 = current, <1.0 = better)

**Why this decomposition?** With Megatron-style TP, linear layers (Q, K, V, O projections, MLP gate/up/down) are split across TP devices — each device computes 1/TP of the work. Non-linear operations (LayerNorm, softmax, embedding lookup, loss computation) are NOT split and run at full cost on every device.

#### 2. Forward TP CCL (per micro-batch)

```
fwd_ccl = tp_ccl_fwd × B × TP_CCL_SCALE
```

- `tp_ccl_fwd`: Measured total CCL time during forward at batch=1 (aggregate of all all-reduce + all-gather ops across all blocks)
- Scales linearly with batch because the data volume per CCL op is `B × S × D × dtype`

**What CCL ops happen?** In Megatron-style TP, each transformer block has:
- **Attention**: ColumnParallelLinear (broadcast fwd / all-reduce bwd) → RowParallelLinear (all-reduce fwd / broadcast bwd)
- **MLP**: ColumnParallelLinear → RowParallelLinear (same pattern)

So the forward pass has ~2 all-reduce ops per block × num_blocks total.

#### 3. Backward Compute + CCL (per micro-batch)

Same decomposition as forward. Measured separately because:
- Backward typically costs ~2× forward (compute gradient w.r.t. both activations and weights)
- CCL ops in backward are the "reverse" of forward (all-reduce ↔ broadcast), which may have different performance

```
bwd_compute = (bwd_time × (1 - tp_perf_perc) + bwd_time × tp_perf_perc / TP) × B × MFU_SCALE
bwd_ccl     = tp_ccl_bwd × B × TP_CCL_SCALE
```

#### 4. Worker Compute (all micro-batches)

```
checkpoint_mult = 2 if use_grad_checkpoint else 1

worker_compute = (
    (fwd_compute + fwd_ccl) × checkpoint_mult + (bwd_compute + bwd_ccl)
) × grad_accum_steps
```

**Gradient checkpointing** at block boundaries means the forward pass is computed **twice**: once during the actual forward, and once during backward (to recompute activations before backpropagating through each block). This doubles the forward cost but drastically reduces activation memory.

**Gradient accumulation** repeats the forward+backward loop for `grad_accum_steps` micro-batches, accumulating gradients, before performing one optimizer step.

#### 5. DDP Gradient Sync

```
ddp_sync = ddp_ar_time × use_ddp × DDP_CCL_SCALE
```

- `ddp_ar_time`: Measured gradient all-reduce time within one Galaxy
- Happens **once** per optimizer step (after all micro-batches)
- Ring all-reduce time ≈ `2 × (n-1)/n × data / bandwidth`, which is roughly **constant** for n ≥ 2 (not proportional to DDP degree)
- `DDP_CCL_SCALE = 0` models full overlap with backward compute (a future optimization)

**Important**: Without 2-tier, DDP all-reduce runs on the worker Galaxy. With 2-tier, it runs on the aggregator Galaxy (workers send raw gradients from all DDP replicas without local reduction). Either way, the wall-clock contribution is the same.

#### 6. 2-Tier Inter-Host Communication

```
two_tier_comm = two_tier_comm_per_worker × n_workers × INTER_HOST_CCL_SCALE
```

- `two_tier_comm_per_worker`: Measured time for the aggregator to process one worker (recv gradients + elementwise add + send weights back)
- Scales **linearly** with `n_workers` because the current implementation processes workers **sequentially** (see `aggregator_worker.cpp` — a for-loop over workers)
- `INTER_HOST_CCL_SCALE` models improvements like parallel recv/send via async fabric

**What happens on the aggregator per step:**
1. Recv gradients from worker 0, add to running sum
2. Recv gradients from worker 1, add to running sum
3. ... (repeat for all n_workers)
4. Divide by n_workers (average)
5. DDP all-reduce on aggregator (if enabled)
6. Optimizer step
7. Send updated weights to worker 0
8. Send updated weights to worker 1
9. ... (repeat for all n_workers)

Steps 1-3 and 7-9 are captured by `two_tier_comm_per_worker × n_workers`. Steps 5-6 are captured by `ddp_sync` and `optimizer_time`.

#### 7. Optimizer Step

```
optimizer_time = opt_time × params_fraction_per_chip × MFU_SCALE
```

- `opt_time`: Measured optimizer step time at TP=1 (full model on one chip)
- `params_fraction_per_chip = (1 - tp_mem_shard) + tp_mem_shard / TP`: per-chip parameter count as a fraction of total
- Optimizer is **elementwise** (AdamW updates m, v, then weights) — runs per-chip in parallel across all 32 devices. Wall-clock time is determined by the per-chip workload, NOT the total across the Galaxy.

### Throughput Metrics

```
effective_batch_size = local_batch × DDP × grad_accum × n_data_galaxies
tokens_per_step     = effective_batch_size × seq_len
tokens_per_sec      = tokens_per_step / step_time
```

Where `n_data_galaxies = n_workers` if 2-tier, else 1. Total Galaxies = n_data_galaxies + 1 (aggregator) if 2-tier.

---

## Memory Model

### Overview

Memory is modeled per-chip for two roles:

- **Worker**: runs forward + backward, stores weights + gradients + activations + (optionally) optimizer states
- **Aggregator** (2-tier only): receives/averages gradients, runs optimizer, stores weights + gradient buffers + optimizer states — **no activations**

### Worker Peak Memory

Peak memory occurs during the backward pass of a transformer block:

```
peak_worker = weights + gradients + optimizer_states + checkpoints + activations + embedding + logits + misc
```

#### Weights

```
weights_per_chip = total_params × params_fraction_per_chip × dtype_bytes
```

With TP=8 and tp_mem_shard=0.85: `params_fraction = 0.15 + 0.85/8 = 0.256`, so ~2.05B params per chip → 4.10 GB in BF16.

#### Gradients

This is the key insight: **gradient memory depends on whether gradient accumulation is used**.

**Without gradient accumulation** (`grad_accum_steps = 1`):

Gradients are consumed (applied by optimizer or sent to aggregator) and **freed per-block** during backward. Only the current block's parameter gradients are live at any time.

```
grads_per_chip = per_block_params_per_chip × dtype_bytes
```

Per-block parameters are computed from the architecture:
- Linear projections (TP-sharded): Q, K, V, O, gate, up, down = `2D² + 2D·D_kv + 3D·D_ff`
- Norm parameters (replicated): 2 × RMSNorm = `2D`
- Per-chip: `linear_params / TP + norm_params`

For Llama 8B with TP=8: ~30.4M params/chip/block → ~61 MB in BF16.

**With gradient accumulation** (`grad_accum_steps > 1`):

Gradients must persist across micro-batches (accumulated in-place), so **all** parameter gradients are live:

```
grads_per_chip = total_params × params_fraction_per_chip × dtype_bytes
```

Same size as weights. For Llama 8B with TP=8: ~4.10 GB.

**Impact**: Without grad accum, gradient memory is ~61 MB (1 block). With grad accum, it's ~4.10 GB (full model). This ~4 GB difference directly affects how much batch size you can fit.

#### Optimizer States

AdamW maintains two state tensors per parameter: first moment (m) and second moment (v).

```
opt_states_per_chip = 2 × total_params × params_fraction_per_chip × opt_dtype_bytes
```

- `opt_dtype_bytes = 2` (BF16) or `4` (FP32), controlled by `use_fp32_optimizer_state`
- BF16 states: 8.20 GB per chip (TP=8). May need Kahan summation for accuracy.
- FP32 states: 16.40 GB per chip (TP=8). More accurate but doubles memory.

**Critical**: In 2-tier mode, optimizer states live on the **aggregator only**. Workers store **zero** optimizer state, freeing significant memory for larger batches.

#### Activation Checkpoints

With gradient checkpointing at each block boundary:

```
checkpoint_mem = num_blocks × B × S × D × dtype_bytes
```

Each checkpoint is the block input: `[B, S, D]` in BF16. These are **NOT** TP-sharded (they are the fully-reduced output of the previous block's row-parallel all-reduce).

For Llama 8B: 32 blocks × `B × 2048 × 4096 × 2` = `B × 537 MB`.

Without checkpointing: no checkpoint memory, but all block activations remain live (much worse — see below).

#### Live Activations (Peak Working Set)

With checkpointing, only **one block** is processed at a time. During backward of block `i`:
1. Recompute forward of block `i` from checkpoint
2. Store activations needed for backward
3. Run backward, computing parameter gradients
4. Free activations

Peak ≈ 2× one block's forward activations (recomputed forward + backward intermediates overlap).

Without checkpointing: all `L` blocks' activations are live simultaneously = `(L+1) × per_block_activations`.

**Per-block activation breakdown:**

| Component | Shape | TP-sharded? | Size formula |
|-----------|-------|-------------|-------------|
| Attention RMSNorm (input + output) | 2 × [B, S, D] | No | `2 × B × S × D × dt` |
| Q, K, V projections | [B, S, (D+2·D_kv)/TP] | Yes | `B × S × (D + 2·D_kv) / TP × dt` |
| **Attention scores** | [B, H/TP, S, S] | Yes | `B × (H/TP) × S² × dt` |
| Attention output | [B, S, D/TP] | Yes | `B × S × D/TP × dt` |
| MLP RMSNorm (input + output) | 2 × [B, S, D] | No | `2 × B × S × D × dt` |
| **MLP intermediates (SwiGLU)** | 3 × [B, S, D_ff/TP] | Yes | `3 × B × S × D_ff/TP × dt` |
| MLP output | [B, S, D/TP] | Yes | `B × S × D/TP × dt` |

The two **bold** entries are the ones affected by feature flags:

- **Flash attention** (`use_flash_attention`): Eliminates the `B × (H/TP) × S × S` attention score matrix. Instead stores only logsumexp statistics: `B × (H/TP) × S`. For S=2048, TP=8, B=1: saves 33.5 MB/block → 67 MB total (2 live blocks with checkpointing). Savings grow **quadratically** with sequence length.

- **Fused FFN** (`use_fused_ffn`): Eliminates 2 of the 3 SwiGLU intermediate tensors (gate, up, silu(gate)×up). The fused kernel recomputes them during backward. Reduces from 3 to 1 stored tensor. For D_ff=14336, TP=8, B=1: saves 14.7 MB/block → 29.4 MB total.

#### Other Activations

- **Embedding**: `B × S × D × dt` (output of embedding lookup)
- **Output logits**: `B × S × V × dt` (can be significant: 131 MB for V=32000 at B=1)
- **Misc buffers**: `2 × B × S × D × dt` (residuals, masks, temporaries)

### Aggregator Memory (2-Tier)

```
aggregator_mem = weights + gradient_buffers + optimizer_states + misc
```

- **Weights**: Same per-chip fraction as worker (holds weights for broadcasting)
- **Gradient buffers**: 2× weights size (running accumulation sum + incoming recv buffer)
- **Optimizer states**: Full m + v (this is where they live in 2-tier mode)
- **No activations**: Aggregator doesn't run forward/backward

**Warning**: With FP32 optimizer states, aggregator memory can be tight. At TP=8: weights (4.1 GB) + grad buffers (8.2 GB) + opt states FP32 (16.4 GB) = 28.7 GB out of 32 GB (89.9%).

---

## Scale Factors

The four scale factors enable what-if analysis without re-running experiments:

| Scale Factor | Applied to | Example use case |
|---|---|---|
| `mfu_scale` | Forward compute, backward compute, optimizer | "What if we optimize ops to 70% peak?" → set to 0.7 |
| `tp_ccl_scale` | Forward TP CCL, backward TP CCL | "What if we overlap 50% of TP comm?" → set to 0.5 |
| `ddp_ccl_scale` | DDP gradient all-reduce | "What if we fully overlap DDP with backward?" → set to 0.0 |
| `inter_host_ccl_scale` | 2-tier communication | "What if we implement parallel recv/send?" → set to 0.5 |

All default to 1.0 (current measured performance). Set to 0.0 to fully eliminate a component. Set between 0.0 and 1.0 to model partial improvement.

---

## Feature Flags

| Flag | Affects | Description |
|---|---|---|
| `use_flash_attention` | Memory only | Eliminates `[B, H/TP, S, S]` score matrix per block. Replaces with `[B, H/TP, S]` logsumexp. Assume perf neutral (tune via `mfu_scale`). |
| `use_fused_ffn` | Memory only | Eliminates 2 of 3 SwiGLU intermediate tensors per block. Fused kernel recomputes during backward. Assume perf neutral (tune via `mfu_scale`). |

These flags only affect `estimate_memory_per_chip()`. Step time is unchanged because we assume the fused/flash implementations match the unfused performance (any difference can be tuned via `mfu_scale`).

---

## Key Assumptions

1. **Compute scales linearly with batch size.** This assumes the hardware is saturated even at batch=1. In practice, small batches may not fully utilize the hardware (sublinear scaling), and memory-bandwidth-bound ops (LayerNorm, softmax) don't scale perfectly. **Validate with Experiment 1** (Phase 1).

2. **TP CCL scales linearly with batch size.** The data volume per CCL operation is `B × S × D × dtype`, which is proportional to batch size. This is exact for bandwidth-dominated transfers and a good approximation for latency-dominated ones at non-trivial batch sizes.

3. **DDP all-reduce time is constant (independent of DDP degree).** Ring all-reduce time ≈ `2 × (n-1)/n × data / bandwidth`, which converges quickly for n ≥ 2. For DDP ∈ {4, 8}, the factor ranges from 0.75 to 0.875 — close enough to treat as constant. **Validate with Experiments 11-13** (Phase 3).

4. **2-tier communication scales linearly with n_workers.** This follows directly from the sequential recv/send loop in the aggregator implementation. If the implementation changes to parallel processing, use `inter_host_ccl_scale` to model the improvement.

5. **No overlap between compute and communication.** The current implementation runs compute, then CCL, then DDP, then 2-tier, then optimizer — all sequentially. Scale factors can model partial overlap (e.g., `ddp_ccl_scale = 0` for full overlap with backward).

6. **Gradient checkpointing doubles forward compute.** With checkpointing at block boundaries, each block's forward is recomputed during backward. Total = 2× forward + 1× backward. This is exact for full recomputation.

7. **Without gradient accumulation, gradients are freed per-block.** During backward, each block's parameter gradients are consumed (applied to optimizer or sent to aggregator) and freed before processing the next block. Peak gradient memory = 1 block's worth.

8. **Optimizer time scales with per-chip parameter count.** AdamW is elementwise — all 32 devices work in parallel. The wall-clock time is proportional to the per-device workload, not the total across the Galaxy.

---

## Limitations

- **No pipeline parallelism** in the model. PP would add pipeline bubble overhead and change the memory/compute trade-off.
- **Tile alignment overhead** is not modeled. Tenstorrent hardware operates on 32×32 tiles; non-aligned shapes cause padding waste.
- **Data loading time** is not modeled (assumed negligible or overlapped).
- **Host-device synchronization** overhead (graph management, kernel launch) is not modeled separately — it's baked into the measured baseline times.
- **Embedding and output head** are treated as having the same TP behavior as transformer blocks via the global `tp_perf_perc` and `tp_mem_shard` fractions. In reality, they may be handled differently (e.g., vocabulary-parallel output head).
- **Activation memory is approximate.** The per-block breakdown captures the main components but may miss small framework-internal buffers. Compare estimates against measured memory (Experiment 4) to calibrate.
