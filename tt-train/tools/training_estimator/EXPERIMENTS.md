# Experiments Guide

Step-by-step instructions for collecting the measured parameters needed by the training estimator. Experiments are organized in phases with dependencies — earlier phases feed into later ones.

## Prerequisites

- One Tenstorrent Galaxy (32 chips, 4×8 mesh) with tt-metal and tt-train installed
- For Phase 4: a second Galaxy with inter-host networking configured
- Profiler setup (see `tt-train/docs/PROFILER.md`)
- wandb configured (optional, for tracking results)

## Quick Reference: Parameters to Measure

| Parameter | Config field | Phase | Description |
|---|---|---|---|
| Forward time (batch=1) | `fwd_time_s` | 1 | Forward pass, TP=1, DDP=1 |
| Backward time (batch=1) | `bwd_time_s` | 1 | Backward pass, TP=1, DDP=1 |
| Optimizer time | `opt_time_s` | 1 | AdamW step, TP=1, DDP=1 |
| TP perf fraction | `tp_perf_perc` | 2 | Fraction of compute that speeds up with TP |
| TP CCL forward | `tp_ccl_fwd_s` | 2 | Total CCL time in forward, batch=1 |
| TP CCL backward | `tp_ccl_bwd_s` | 2 | Total CCL time in backward, batch=1 |
| TP memory shard | `tp_mem_shard` | 2 | Fraction of params that are TP-sharded |
| DDP all-reduce time | `ddp_ar_time_s` | 3 | Gradient sync within Galaxy |
| 2-tier comm/worker | `two_tier_comm_per_worker_s` | 4 | Per-worker aggregator round-trip |

---

## Phase 1: Single-Device Baselines

**Goal**: Measure forward, backward, and optimizer time on a single chip with no parallelism.

**Hardware**: 1 chip (mesh_shape: [1, 1])

**Note**: If the full model doesn't fit in single device memory, see **Appendix A** at the end of this document for how to measure with reduced models and extrapolate.

### Experiment 1.1: Forward + Backward Time vs Batch Size

**What to measure**: Wall-clock time for forward pass and backward pass separately, at batch sizes 1, 2, 4, 8.

**How to run**:

```bash
# Configure for single device, no TP, no DDP
# Create or modify a training config YAML:
#   device_config:
#     enable_tp: false
#     enable_ddp: false
#     mesh_shape: [1, 1]
#   training_config:
#     batch_size: 1  # vary: 1, 2, 4, 8
#     model_config: "configs/model_configs/llama8b.yaml"

# Run with profiler enabled
TT_METAL_PROFILER=1 ./build/sources/examples/nano_gpt/nano_gpt \
    --config configs/training_configs/your_single_device_config.yaml
```

**How to extract the numbers**:

The training script already has profiler markers for each phase of a training step (see `main.cpp`):
- `forward_pass_done` — end of forward pass
- `backward_pass_done` — end of backward pass
- `gradient_sync_done` — end of gradient synchronization
- `optimizer_step_done` — end of optimizer step
- `compilation_finished` — marks the end of the first (compilation) iteration

Use the profiler CSV output and the visualization notebook `notebooks/profiler_results.ipynb` to extract per-phase timing. The notebook loads the profiler CSV and can compute time between markers. Profiler output is written to `generated/profiler/reports/<timestamp>/`.

**What to record**:

| Batch Size | Forward (s) | Backward (s) | Total (s) |
|---|---|---|---|
| 1 | → `fwd_time_s` | → `bwd_time_s` | |
| 2 | | | |
| 4 | | | |
| 8 | | | |

**Validation**: Check that time scales roughly linearly with batch size. If it's significantly sublinear for small batches, the model's linearity assumption is weak — consider fitting `time(B) = fixed + marginal × B` instead.

**Config fields to fill**: `fwd_time_s` (batch=1 value), `bwd_time_s` (batch=1 value)

---

### Experiment 1.2: Optimizer Step Time

**What to measure**: Wall-clock time for `optimizer.step()` only.

**How to run**: Same setup as Experiment 1.1. The profiler markers already bracket the optimizer step (`backward_pass_done` → `optimizer_step_done`). Extract the time between these markers from the profiler CSV or the visualization notebook.

**What to record**: Optimizer time in seconds.

**Validation**: Optimizer time should be roughly **constant** regardless of batch size (it's elementwise on parameters, not on activations). Run at batch=1 and batch=4 to confirm.

**Config field to fill**: `opt_time_s`

---

### Experiment 1.3: Memory Profiling vs Batch Size

**What to measure**: Peak DRAM usage per chip at batch sizes 1, 2, 4, 8.

**How to run**: The training script automatically captures memory snapshots via `MemoryUsageTracker` during the first iteration. It takes snapshots at key points (`FORWARD_PASS`, `BACKWARD_PASS`) and prints a full memory usage summary after the first iteration completes (`print_memory_usage()`). Just run the training script and look at the output — no extra instrumentation needed.

**What to record**:

| Batch Size | Peak Memory (GB) | Estimated (from tool) | Delta |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 4 | | | |
| 8 | | | |

The "Estimated (from tool)" column comes from the memory estimator:
```python
from config import EstimatorConfig
from estimator import estimate_memory_per_chip, fmt_bytes
from dataclasses import replace

cfg = EstimatorConfig(
    # Fill in your model architecture params
    local_batch_size=1,  # vary: 1, 2, 4, 8
    # ...
)
mem = estimate_memory_per_chip(cfg)
print(fmt_bytes(mem['total_worker_bytes']))
```

**Validation**: Compare measured peak memory against `estimate_memory_per_chip()` output. The delta should be consistent (a constant offset from framework overhead not captured in the model). Use this to calibrate the `misc_bytes` estimate.

**Config field to fill**: None directly — this validates the memory model.

---

### Experiment 1.4: Gradient Checkpointing Overhead

**What to measure**: Forward+backward time WITH vs WITHOUT gradient checkpointing.

**How to run**: Run Experiment 1.1 twice — once with `runner_type: memory_efficient` (checkpointing) and once with `runner_type: default` (no checkpointing).

```yaml
# In llama8b.yaml:
transformer_config:
  runner_type: memory_efficient  # or remove for default
```

**What to record**:

| Mode | Forward (s) | Backward (s) | Total (s) | Peak Memory (GB) |
|---|---|---|---|---|
| No checkpoint | | | | |
| With checkpoint | | | | |

**Validation**:
- Total time with checkpointing should be ~3× forward + 1× backward (≈ fwd × 2 + bwd), where the extra forward is due to recomputation during backward.
- Memory with checkpointing should be dramatically lower (only 1 block's activations live vs all L blocks).

---

## Phase 2: TP Characterization

**Goal**: Measure TP communication overhead and determine what fraction of compute parallelizes.

**Hardware**: 4-8 chips for TP (mesh_shape: [1, 4] or [1, 8])

### Experiment 2.1: TP Forward + Backward Time

**What to measure**: Total forward and backward time with TP enabled at batch=1.

**How to run**:

```yaml
# Config for TP=8:
device_config:
  enable_tp: true
  enable_ddp: false
  mesh_shape: [1, 8]
training_config:
  batch_size: 1
```

```bash
TT_METAL_PROFILER=1 ./build/sources/examples/nano_gpt/nano_gpt \
    --config configs/training_configs/your_tp8_config.yaml
```

Repeat for TP=4 (`mesh_shape: [1, 4]`).

**What to record**:

| TP | Fwd Total (s) | Bwd Total (s) | Fwd Compute (s) | Fwd CCL (s) | Bwd Compute (s) | Bwd CCL (s) |
|---|---|---|---|---|---|---|
| 1 | (from 1.1) | (from 1.1) | | 0 | | 0 |
| 4 | | | | | | |
| 8 | | | | | | |

### Experiment 2.2: Isolating CCL Time

**What to measure**: Time spent in CCL operations (all-reduce, all-gather) during forward and backward.

**How to extract**:

1. **Profiler approach** (preferred): Run with `TT_METAL_PROFILER=1`. In the profiler output, identify CCL operations (all_reduce, all_gather, reduce_scatter). Sum their durations separately for forward and backward regions.

2. **Subtraction approach**: If profiler data isn't granular enough:
   ```
   tp_ccl_fwd = fwd_time_tp8 - (fwd_time_tp1 × (1 - tp_perf_perc) + fwd_time_tp1 × tp_perf_perc / 8)
   ```
   This requires knowing `tp_perf_perc` first — solve the system of equations from TP=4 and TP=8 measurements:
   ```
   fwd_tp4 = fwd × (1 - p) + fwd × p / 4 + ccl_fwd_tp4
   fwd_tp8 = fwd × (1 - p) + fwd × p / 8 + ccl_fwd_tp8
   ```
   With two unknowns (`p` and `ccl_fwd`), two equations are enough if you assume CCL per block is similar for TP=4 and TP=8 (roughly true for bandwidth-dominated all-reduce).

**Config fields to fill**: `tp_ccl_fwd_s`, `tp_ccl_bwd_s` (at the TP degree you plan to use)

### Experiment 2.3: Deriving tp_perf_perc

**How to compute**: Using measured times at TP=1 and TP=N (batch=1, ignoring CCL):

```
pure_compute_tpN = fwd_time_tpN - tp_ccl_fwd_tpN

tp_perf_perc = (fwd_time_tp1 - pure_compute_tpN) / (fwd_time_tp1 × (1 - 1/N))
```

For example, with TP=8:
```
tp_perf_perc = (fwd_tp1 - (fwd_tp8 - ccl_fwd_tp8)) / (fwd_tp1 × 7/8)
```

**Validation**: Should be ~0.80-0.90 for transformers (linear layers are ~85% of compute, but this varies with model architecture and sequence length).

**Config field to fill**: `tp_perf_perc`

### Experiment 2.4: TP Batch Scaling

**What to measure**: Forward time at TP=8 with batch=1, 2, 4.

**Validation**: Check that CCL time scales linearly with batch (data volume ∝ batch). If it doesn't scale linearly, the model's assumption needs adjustment.

### Experiment 2.5: TP Memory Shard Fraction

**What to measure**: Fraction of model parameters that are TP-sharded.

**How to extract**:

1. **From code inspection**: In the model definition, identify which layers use `ColumnParallelLinear` / `RowParallelLinear` (TP-sharded) vs regular `Linear` / `LayerNorm` (replicated). Sum parameter counts for each.

2. **From runtime**: Use the `get_number_of_parameters()` function in `main.cpp` which already distinguishes TP-sharded vs replicated parameters:
   ```cpp
   // In main.cpp, the function checks for "fc", "linear", "mlp/w" in param names
   // to identify TP-sharded parameters and multiplies by TP size
   ```
   Compare `num_params_with_tp_correction` vs `num_params_without_correction`:
   ```
   tp_mem_shard = 1 - (replicated_params / total_params)
   ```

3. **From memory measurement**: Compare per-chip memory at TP=1 vs TP=8:
   ```
   tp_mem_shard ≈ 1 - (weights_tp8 / weights_tp1)  ×  TP / (TP - 1)
   ```

**Config field to fill**: `tp_mem_shard`

---

## Phase 3: DDP Characterization

**Goal**: Measure gradient all-reduce time within one Galaxy.

**Hardware**: Full Galaxy, 32 chips (various TP/DDP splits)

### Experiment 3.1: DDP Gradient All-Reduce Time

**What to measure**: Time for the gradient all-reduce step only (not including forward, backward, or optimizer).

**How to run**:

```yaml
# TP=8, DDP=4:
device_config:
  enable_tp: true
  enable_ddp: true
  mesh_shape: [4, 8]
```

**How to extract**: The DDP all-reduce happens between backward completion and optimizer step. Use the profiler markers: the gradient sync time is the duration between `backward_pass_done` and `gradient_sync_done` markers. Extract from the profiler CSV using `notebooks/profiler_results.ipynb` (which reports this as the "Gradient Sync" phase).

**What to record**:

| Config | DDP | TP | All-Reduce Time (s) |
|---|---|---|---|
| [4, 8] | 4 | 8 | |
| [8, 4] | 8 | 4 | |
| [32, 1] | 32 | 1 | |

### Experiment 3.2: DDP Scaling Validation

**Validation**: All-reduce time should be roughly constant across DDP=4, 8, 32 (ring all-reduce). If it varies significantly, the model's constant-time assumption needs adjustment.

**What to check**: The data volume per DDP all-reduce equals the per-chip gradient size (model_params × params_fraction_per_chip × dtype). For TP=8: ~4.1 GB. For TP=1: ~16 GB. So the all-reduce time will differ across TP configurations (more data with lower TP), but within a given TP, it should be roughly constant across DDP degrees.

**Config field to fill**: `ddp_ar_time_s` (measured at your target TP/DDP configuration)

---

## Phase 4: 2-Tier Communication

**Goal**: Measure inter-host communication overhead for the 2-tier training setup.

**Hardware**: 2+ Galaxies with inter-host networking (MPI or fabric sockets)

### Experiment 4.1: Single-Worker Round-Trip

**What to measure**: Time for the aggregator to receive one worker's gradients, perform the elementwise add, and send weights back.

**How to run**:

```yaml
# 2-tier config with 1 worker:
multihost_config:
  enabled: true
  num_workers: 1
  socket_type: fabric  # or mpi
device_config:
  enable_tp: true
  enable_ddp: true
  mesh_shape: [4, 8]
```

Launch with 2 ranks: 1 worker + 1 aggregator_optimizer.

**How to extract**:

Add timing to the aggregator loop in `aggregator_worker.cpp` or `trainer.py`:

```python
# In the 2-tier aggregator loop:
t_start = time.time()
# recv grads from worker 0 + add + send weights to worker 0
t_end = time.time()
per_worker_time = t_end - t_start
```

Or measure the total aggregator cycle time and subtract optimizer + DDP time:

```
two_tier_comm_per_worker = (total_cycle - opt_time - ddp_time) / n_workers
```

**What to record**:

| n_workers | Total Cycle (s) | Opt (s) | DDP (s) | Comm/Worker (s) |
|---|---|---|---|---|
| 1 | | | | → `two_tier_comm_per_worker_s` |

### Experiment 4.2: Multi-Worker Scaling

**What to measure**: Total aggregator cycle time with 2, 3, 4 workers.

**How to run**: Same as 4.1 but with more worker ranks.

**Validation**: Total communication time should scale linearly with n_workers (since the aggregator processes workers sequentially). Plot total_comm vs n_workers — it should be a straight line through the origin.

| n_workers | Total Comm (s) | Expected (comm/worker × n) | Error |
|---|---|---|---|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |

If it's **not** linear, there may be:
- Fixed overhead per round (setup, synchronization)
- Bandwidth sharing/congestion effects
- Parallelism in the fabric that partially offsets the sequential loop

Adjust the model accordingly (e.g., add a fixed offset: `two_tier_comm = fixed + per_worker × n_workers`).

### Experiment 4.3: Inter-Host Bandwidth Characterization

**What to measure**: Raw bandwidth between hosts for known tensor sizes.

**How to run**: Send tensors of various sizes between two hosts and measure throughput:

```python
import time
# Create a tensor of known size
tensor = ttnn.empty(shape, device=device, dtype=ttnn.bfloat16)

t_start = time.time()
socket_manager.send(tensor, ctx, target_rank)
t_end = time.time()

bytes_sent = tensor.volume() * 2  # BF16
bandwidth_GBps = bytes_sent / (t_end - t_start) / 1e9
```

**What to record**: Bandwidth in GB/s. This helps understand whether the inter-host link is shared (total volume matters) or parallel (per-device volume matters).

Test with:
- A single small tensor (latency-dominated)
- A single large tensor (bandwidth-dominated)
- A full model's worth of tensors (realistic workload)

---

## Phase 5: End-to-End Validation

**Goal**: Validate the estimator against real training step times.

### Experiment 5.1: Full Training Step Measurement

**What to measure**: Total wall-clock time for one optimizer step with the target configuration.

**How to run**:

```yaml
# Your target config, e.g.:
device_config:
  enable_tp: true
  enable_ddp: true
  mesh_shape: [4, 8]
training_config:
  batch_size: 2
  gradient_accumulation_steps: 4
```

Run for 10-20 steps, discard the first 2-3 (warmup), average the rest.

**How to validate**:

```python
from config import EstimatorConfig
from estimator import estimate_step_time

cfg = EstimatorConfig(
    fwd_time_s=<from 1.1>,
    bwd_time_s=<from 1.1>,
    opt_time_s=<from 1.2>,
    tp_ccl_fwd_s=<from 2.2>,
    tp_ccl_bwd_s=<from 2.2>,
    tp_perf_perc=<from 2.3>,
    ddp_ar_time_s=<from 3.1>,
    tp=8, local_batch_size=2, grad_accum_steps=4,
    use_grad_checkpoint=True, use_ddp=True,
)

predicted = estimate_step_time(cfg)
print(f"Predicted: {predicted['total_step_time_s']:.3f}s")
print(f"Measured:  {measured_step_time:.3f}s")
print(f"Error:     {abs(predicted['total_step_time_s'] - measured_step_time) / measured_step_time * 100:.1f}%")
```

**Acceptable error**: <15% for initial calibration. If error is larger:
- Check which component is off by comparing per-component breakdown against profiler data
- The linearity assumption (Experiment 1.1) may need correction
- Framework overhead (graph management, kernel launch, host-device sync) may be significant

### Experiment 5.2: Grad Accumulation Scaling

**What to measure**: Step time with grad_accum = 1, 2, 4, 8.

**Validation**: The worker_compute portion should scale linearly with grad_accum. The DDP sync, 2-tier comm, and optimizer portions should be constant.

### Experiment 5.3: Batch Size Scaling

**What to measure**: Step time with local_batch_size = 1, 2, 4 (adjusting grad_accum to keep effective batch constant if desired).

**Validation**: Worker compute should scale linearly with batch size. If sublinear, consider fitting a two-parameter model: `time(B) = overhead + marginal × B`.

### Experiment 5.4: Memory Validation

**What to measure**: Peak per-chip DRAM at the target configuration.

**Validation**: Compare against `estimate_memory_per_chip()`:
- The **relative** differences between configurations should be accurate (e.g., memory savings from checkpointing, from 2-tier, from larger batch)
- The **absolute** value may have a constant offset from framework overhead
- Use the delta to calibrate `misc_bytes` if needed

---

## Phase 6: Optimal Configuration Search

After calibrating the model with Phases 1-5, sweep over configurations to find the optimum.

### What to Sweep

| Parameter | Values to try |
|---|---|
| `tp` | 4, 8, 32 |
| `local_batch_size` | 1, 2, 4, ..., max that fits in memory |
| `grad_accum_steps` | 1, 2, 4, 8, 16 |
| `n_workers` | 1, 2, 4, 8 (if multi-host) |
| `use_grad_checkpoint` | True, False |
| `use_fp32_optimizer_state` | True, False |

### Optimization Targets

1. **Maximum tokens/sec/galaxy** — best throughput efficiency per Galaxy
2. **Maximum tokens/sec/dollar** — factor in n_workers + aggregator cost
3. **Minimum step time** — for latency-sensitive training
4. **Target batch size** — some training recipes require specific global batch sizes

### Example Sweep Script

```python
from config import EstimatorConfig
from estimator import estimate_step_time, estimate_memory_per_chip
from dataclasses import replace

base = EstimatorConfig(
    fwd_time_s=2.0, bwd_time_s=4.0, opt_time_s=0.5,
    tp_ccl_fwd_s=0.3, tp_ccl_bwd_s=0.5,
    tp_perf_perc=0.85, tp_mem_shard=0.85,
    ddp_ar_time_s=0.2,
    two_tier_comm_per_worker_s=1.0,
)

results = []
for tp in [4, 8]:
    for batch in [1, 2, 4, 8]:
        for ga in [1, 2, 4, 8]:
            for n_w in [1, 2, 4]:
                cfg = replace(base,
                    tp=tp,
                    local_batch_size=batch,
                    grad_accum_steps=ga,
                    n_workers=n_w,
                    use_2tier=(n_w > 1),
                )
                mem = estimate_memory_per_chip(cfg)
                if mem['worker_utilization_pct'] > 95:
                    continue  # would OOM
                step = estimate_step_time(cfg)
                results.append({
                    'config': f"TP={tp} B={batch} GA={ga} W={n_w}",
                    'step_time': step['total_step_time_s'],
                    'tok_sec': step['tokens_per_sec'],
                    'tok_sec_gal': step['tokens_per_sec_per_galaxy'],
                    'mem_pct': mem['worker_utilization_pct'],
                    'eff_batch': step['effective_batch_size'],
                })

# Sort by tokens/sec/galaxy (cost efficiency)
results.sort(key=lambda x: -x['tok_sec_gal'])
for r in results[:10]:
    print(f"{r['config']:<30} {r['tok_sec']:>10,.0f} tok/s  "
          f"{r['tok_sec_gal']:>10,.0f} tok/s/gal  "
          f"mem={r['mem_pct']:.0f}%  batch={r['eff_batch']}")
```

---

## Troubleshooting

### Measured time is much higher than predicted
- **Framework overhead**: Graph management, kernel launch, host-device sync add fixed time per step. Measure "empty" step overhead (no model, just framework calls) and add to the model.
- **Memory bandwidth bottleneck**: LayerNorm, softmax, activation functions are memory-bandwidth-bound and may not scale linearly with batch. Use profiler to check utilization.
- **Tile padding**: Non-tile-aligned shapes (not multiples of 32) cause padding overhead. Check that batch × seq_len and hidden dimensions are tile-aligned.

### Measured time is much lower than predicted
- **Model double-counts**: Check that `tp_perf_perc` isn't too low (would overcount non-parallelizable time). Verify CCL measurement doesn't overlap with compute measurement.
- **Warmup effects**: First few steps are slower (JIT compilation, memory allocation). Discard warmup steps.

### Memory estimate is off
- **Constant offset**: If the delta between estimated and measured is constant across batch sizes, it's framework overhead — adjust `misc_bytes`.
- **Scales with batch**: If the delta grows with batch, an activation component is missing from the model. Use the profiler's memory view to identify which tensors are unexpected.
- **Fragmentation**: Device memory fragmentation can cause OOM earlier than the theoretical limit. Leave 10-15% headroom.

---

## Appendix A: Measuring Models That Don't Fit in Memory

When the full model exceeds single-device DRAM (e.g., Llama 8B needs ~50GB vs 32GB available), reduce `num_blocks` to create smaller models that fit, measure those, and extrapolate.

### Step 1: Create Reduced Model Configs

Copy your model config and reduce `num_blocks`. Keep everything else the same. Pre-made configs are provided in `configs/model_configs/`:

- `llama8b_4blocks.yaml` — 4 blocks (~1.3B params, ~17GB)
- `llama8b_2blocks.yaml` — 2 blocks (~0.8B params, ~15GB)

Matching training configs in `configs/training_configs/llama8b/`:

- `llama_8b_4blocks_baseline.yaml`
- `llama_8b_2blocks_baseline.yaml`

### Step 2: Measure

Run both configs. Discard the first 2-3 warmup steps and average the rest.

```bash
./build/sources/examples/nano_gpt/nano_gpt \
    --config configs/training_configs/llama8b/llama_8b_4blocks_baseline.yaml
# Record: fwd_4, bwd_4, opt_4

./build/sources/examples/nano_gpt/nano_gpt \
    --config configs/training_configs/llama8b/llama_8b_2blocks_baseline.yaml
# Record: fwd_2, bwd_2
```

To isolate embedding and output projection (lm_head) time, run one of the above again with `vocab_size: 96` in the model config. The difference gives you the embedding + lm_head contribution.

### Step 3: Extrapolate

Use the differential method to separate per-block time from overhead (embedding, output projection, final RMSNorm):

```
fwd_per_block = (fwd_4 - fwd_2) / 2
bwd_per_block = (bwd_4 - bwd_2) / 2

overhead_fwd = fwd_2 - 2 * fwd_per_block
overhead_bwd = bwd_2 - 2 * bwd_per_block

fwd_time_s = overhead_fwd + 32 * fwd_per_block
bwd_time_s = overhead_bwd + 32 * bwd_per_block
```

Optimizer time scales linearly with parameter count. Use actual parameter counts printed by the training script:

```
opt_time_s = opt_4 * (params_32blocks / params_4blocks)
```

### Step 4: Validate

Optionally run an 8-block model to confirm linear scaling:

```
fwd_8_predicted = overhead_fwd + 8 * fwd_per_block
error = abs(fwd_8_measured - fwd_8_predicted) / fwd_8_measured
# Should be < 5%
```

Fill the extrapolated values into `config.py` as `fwd_time_s`, `bwd_time_s`, `opt_time_s`.

---

## Appendix B: Extrapolating TP Results from Reduced Models

Running the full model with TP and profiler enabled can be slow. Use a reduced-block model to measure TP overheads faster, then extrapolate.

### Approach

TP adds per-block CCL ops (all-reduce / reduce-scatter / all-gather). The number of CCL ops scales linearly with `num_blocks`, and per-op cost is independent of block count. So:

1. Measure with a small block count (e.g., 4 blocks) at TP=1 and TP=N
2. Compute per-block TP overhead from the difference
3. Extrapolate to 32 blocks

### Steps

1. Run single-device baseline with 4 blocks (reuse from Appendix A):
   ```
   fwd_tp1_4blocks, bwd_tp1_4blocks
   ```

2. Run TP=8 with 4 blocks:
   ```bash
   TT_METAL_PROFILER=1 ./build/sources/examples/nano_gpt/nano_gpt \
       --config configs/training_configs/llama8b/llama_8b_tp_4blocks.yaml
   ```
   Record: `fwd_tp8_4blocks`, `bwd_tp8_4blocks`

3. Compute CCL time for 4 blocks (difference between TP=8 total and TP=8 pure compute):
   ```
   # From profiler: sum all CCL op durations (all_reduce, reduce_scatter, all_gather)
   # between forward_pass_done and backward_pass_done markers
   tp_ccl_fwd_4blocks = <from profiler>
   tp_ccl_bwd_4blocks = <from profiler>
   ```

4. Extrapolate to 32 blocks — CCL time scales linearly with block count:
   ```
   tp_ccl_fwd_s = tp_ccl_fwd_4blocks * (32 / 4)
   tp_ccl_bwd_s = tp_ccl_bwd_4blocks * (32 / 4)
   ```

5. Derive `tp_perf_perc` (this is a ratio, independent of block count — measure at any size):
   ```
   pure_compute_tp8 = fwd_tp8_4blocks - tp_ccl_fwd_4blocks
   tp_perf_perc = (fwd_tp1_4blocks - pure_compute_tp8) / (fwd_tp1_4blocks * (1 - 1/8))
   ```

### Validation

Run with 8 blocks at TP=8. Predict:
```
tp_ccl_fwd_8blocks_predicted = tp_ccl_fwd_4blocks * 2
fwd_tp8_8blocks_predicted = (overhead_fwd + 8 * fwd_per_block_tp8) + tp_ccl_fwd_8blocks_predicted
```
Compare against measured. Error should be < 10%.

---

## Appendix C: Extrapolating DDP Results from Reduced Models

DDP all-reduce time is proportional to gradient data volume, which is proportional to parameter count, which scales linearly with `num_blocks`. Measure with fewer blocks and scale up.

### Steps

1. Run DDP with 4 blocks:
   ```bash
   TT_METAL_PROFILER=1 ./build/sources/examples/nano_gpt/nano_gpt \
       --config configs/training_configs/llama8b/llama_8b_ddp_4blocks.yaml
   ```
   Extract `gradient_sync` time from profiler (time between `backward_pass_done` and `gradient_sync_done` markers).

2. Extrapolate to 32 blocks:
   ```
   ddp_ar_time_s = gradient_sync_4blocks * (params_32blocks / params_4blocks)
   ```
   This works because ring all-reduce is bandwidth-bound and time is proportional to data volume.

3. For combined TP+DDP, the gradient volume per chip is reduced by TP (each chip holds fewer parameters). Scale accordingly:
   ```
   ddp_ar_time_tp8 = gradient_sync_4blocks * (params_32blocks / params_4blocks) / tp_reduction_factor
   ```
   where `tp_reduction_factor ≈ params_fraction_per_chip_tp1 / params_fraction_per_chip_tp8`.

### Validation

Run with 8 blocks at the same DDP config. Predict:
```
gradient_sync_8blocks_predicted = gradient_sync_4blocks * (params_8blocks / params_4blocks)
```
Compare against measured. Error should be < 10%.
