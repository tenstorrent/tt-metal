# Training Profiling Toolkit

Tools for running profiling experiments on Tenstorrent hardware, extracting timing/memory data, and enriching results with roofline and bandwidth analysis.

## Quick Start

```bash
# 1. Run experiments (submits to Slurm)
./tt-train/tools/profiling/run_experiments.sh --partition bh_pod_4x32_B45 --phases 1 2

# 2. Monitor progress
watch -n1 cat tt-train/tools/profiling/slurm_logs/profiling_<job_id>.out

# 3. Extract results
python3 tt-train/tools/profiling/results_json/extract_results.py \
  tt-train/tools/profiling/experiments/<run_dir>/ \
  -o results.json

# 4. (Optional) Add roofline analysis (includes CCL bandwidth/utilization)
python3 tt-train/tools/profiling/results_json/add_roofline.py results.json /path/to/tt-train-roofline
```

## Scripts

| Script | Purpose |
|---|---|
| `run_experiments.sh` | Slurm wrapper — submits experiments or runs locally |
| `run_experiments.py` | Generates configs, resets board, runs each experiment |
| `results_json/extract_results.py` | Parses profiler CSVs and stdout logs → JSON |
| `results_json/add_roofline.py` | Adds roofline %, MFU, and CCL bandwidth/utilization to results JSON |

## Step 1: Running Experiments

### Via Slurm (default)

```bash
# Run all experiments
./tt-train/tools/profiling/run_experiments.sh --partition bh_pod_4x32_B45

# Run specific phases
./tt-train/tools/profiling/run_experiments.sh --partition bh_pod_4x32_B45 --phases 1 2 3

# Target a specific node
./tt-train/tools/profiling/run_experiments.sh \
  --partition bh_pod_4x32_B45 --nodelist bh-glx-b08u02 \
  --phases 1 2
```

### Local mode

If you want to run this locally, without slurm present, you can do this with `--local` flag.

```bash
# Then on the node:
./tt-train/tools/profiling/run_experiments.sh --local --phases 1 2
```

### Custom model

Override default llama 8B templates for a different model:

```bash
./tt-train/tools/profiling/run_experiments.sh \
  --partition bh_pod_4x32_B45 \
  --name tinyllama \
  --model-template tt-train/configs/model_configs/tinyllama.yaml \
  --training-template tt-train/configs/training_configs/training_shakespeare_tinyllama.yaml \
  --max-steps 12
```

### Key flags

| Flag | Description |
|---|---|
| `--partition <name>` | Slurm partition |
| `--nodelist <node>` | Target node |
| `--local` | Run on current node (skip sbatch) |
| `--name <name>` | Experiment set name (used in directory and file names) |
| `--model-template <path>` | Model config YAML template |
| `--training-template <path>` | Training config YAML template |
| `--binary <path>` | Training binary (default: nano_gpt) |
| `--max-steps <n>` | Steps per experiment (default: 8) |
| `--phases <p1 p2 ...>` | Run only specific phases |
| `--experiments <name ...>` | Run only named experiments |
| `--dry-run` | Generate configs and print plan without running |
| `--skip-reset` | Skip board reset between experiments |

### Monitoring

```bash
# Watch Slurm output in real time
watch -n1 cat tt-train/tools/profiling/slurm_logs/profiling_<job_id>.out
```

### Handling failures

Experiments can fail due to hardware instability. Check the summary at the end of the Slurm log for `[FAIL]` entries, then rerun only those:

```bash
./tt-train/tools/profiling/run_experiments.sh \
  --partition bh_pod_4x32_B45 \
  --experiments p2_b1_blk4_tp4_ddp1_ga1_default p3_b1_blk4_tp1_ddp8_ga1_default_naive
```

## Step 2: Extracting Results

Parse experiment logs into a single JSON:

```bash
# Single run
python3 tt-train/tools/profiling/results_json/extract_results.py \
  tt-train/tools/profiling/experiments/run_20260224_152552/ \
  -o results.json

# Merge multiple runs (later runs override earlier on name collision)
python3 tt-train/tools/profiling/results_json/extract_results.py \
  experiments/run_1/ experiments/run_2/ experiments/run_3/ \
  -o all_results.json

# Merge into an existing JSON
python3 tt-train/tools/profiling/results_json/extract_results.py \
  experiments/run_rerun/ \
  --merge all_results.json \
  -o all_results.json

# Parse a standalone profiler CSV
python3 tt-train/tools/profiling/results_json/extract_results.py \
  --csv /path/to/ops_perf_results.csv
```

Important flag: `--seq-len <n>` (default 2048).

## Step 3 (Optional): Roofline Analysis

Adds per-phase roofline percentage, MFU, and CCL utilization for all experiments:

```bash
python3 tt-train/tools/profiling/results_json/add_roofline.py \
  results.json \
  /path/to/tt-train-roofline \
  -o results_roofline.json \
  --hardware bh_glx
```

## Output JSON Structure

Each experiment entry in the results JSON:

```json
{
  "name": "p1_b1_blk4_tp1_ddp1_ga1_default",
  "command": "...",
  "profiler": true,
  "experiment": {
    "phase": "1",
    "local_batch": 1,
    "num_blocks": 4,
    "tp": 1,
    "ddp": 1,
    "grad_accum": 1,
    "runner_type": "default"
  },
  "mesh_shape": [1, 1],
  "total_devices": 1,
  "csv_path": "/path/to/ops_perf_results.csv",
  "num_parameters": 1134596096,

  "memory": {
    "overall_dram_peak_mb": 10322.75,
    "final_dram_usage_mb": 8926.09,
    "segments": {
      "model_creation":        { "segment_peak_mb": ..., "allocations_mb": ..., "deallocations_mb": ..., "segment_change_mb": ..., "cumulative_peak_mb": ..., "cumulative_current_mb": ... },
      "optimizer_creation":    { ... },
      "forward_pass":          { ... },
      "backward_pass":         { ... },
      "first_iteration_complete": { ... }
    }
  },

  "timings": {
    "device_0": {
      "iterations": [
        { "iteration": 2, "forward_ms": 88.6, "backward_ms": 139.9, "gradient_sync_ms": 0.0, "optimizer_ms": 84.1, "other_ms": 0.0, "total_ms": 312.5,
          "fwd_rs_ms": 0.0, "fwd_ag_ms": 0.0, "bwd_rs_ms": 0.0, "bwd_ag_ms": 0.0, "sync_rs_ms": 0.0, "sync_ag_ms": 0.0, "opt_rs_ms": 0.0, "opt_ag_ms": 0.0 },
        ...
      ],
      "average": { "forward_ms": 88.6, "backward_ms": 139.9, ... },
      "num_steady_iterations": 6
    }
  },

  "naive_timings": {
    "device_host": {
      "iterations": [ { "iteration": 3, "forward_ms": 289.1, "backward_ms": 636.5, ... }, ... ],
      "average": { ... },
      "num_steady_iterations": 5
    }
  },

  "step_timings": {
    "step_times": {
      "iterations": [ { "iteration": 1, "total_step_ms": 54126.2 }, { "iteration": 2, "total_step_ms": 140.3 }, ... ],
      "average": { "total_step_ms": 372.0 },
      "num_steady_iterations": 5
    }
  },

  "throughput": {
    "tokens_per_step": 2048,
    "step_time_ms": 312.5,
    "tokens_per_sec": 6553.6,
    "tokens_per_sec_per_device": 6553.6
  },

  "roofline": {
    "checkpointing_adjusted": false,
    "forward":  { "roofline_ms": 37.2, "flops_tflops": 0.77, "measured_ms": 88.6, "roofline_perc": 42.0, "mfu_perc": 5.4 },
    "backward": { ... },
    "optimizer": { ... },
    "total":    { "roofline_ms": 134.5, "flops_tflops": 2.31, "measured_ms": 312.5, "roofline_perc": 43.0, "mfu_perc": 4.6 },
    "ccl": {
      "grad_sync_theoretical_ms": 62.5, "grad_sync_measured_ms": 363.1, "grad_sync_bw_GBs": 8.26, "grad_sync_util_perc": 17.2,
      "fwd_ccl_theoretical_ms": 10.4, "fwd_ccl_measured_ms": 15.2, "fwd_ccl_util_perc": 68.9,
      "bwd_ccl_theoretical_ms": 19.3, "bwd_ccl_measured_ms": 29.4, "bwd_ccl_util_perc": 65.8,
      "total_ccl_theoretical_ms": 29.8, "total_ccl_measured_ms": 44.6, "total_ccl_util_perc": 66.8
    }
  }
}
```

### Timing sources (priority order)

| Field | Source | When available |
|---|---|---|
| `timings` | Tracy profiler device cycles | `profiler=True` and CSV generated |
| `naive_timings` | `[NAIVE_PROFILER]` host timestamps in stdout | `profiler="naive"` |
| `step_timings` | `Full step time` wall clock from stdout | Always (but skip for tracy — includes tracy overhead) |

- **`timings`**: Per-device, per-iteration, per-phase breakdown from device firmware cycles. Most accurate for device compute. Includes CCL op breakdown (`fwd_rs_ms`, `fwd_ag_ms`, etc.).
- **`naive_timings`**: Host-side timestamps with device sync. ~5% accuracy vs tracy. Single "device_host" entry (no per-device split). Good for multi-device experiments where tracy times out.
- **`step_timings`**: Raw wall-clock from training binary stdout. Includes host overhead. Skips iterations 1-2 for averaging.
- **`throughput`**: Computed from the best available step time (tracy device total → naive total → wall clock). Uses `tokens_per_step = local_batch × seq_len × ddp × grad_accum`.

### Enrichment fields (optional)

- **`roofline`**: Added by `add_roofline.py`. Per-phase roofline time, FLOPs, `roofline_perc` (% of theoretical minimum achieved), `mfu_perc` (model FLOP utilization vs peak BF16).
- **`roofline.ccl`**: DDP gradient sync bandwidth/utilization and TP CCL utilization (theoretical vs achieved) for forward, backward, and total.

## Experiment Parameters

Each experiment is defined by calling `_exp(...)` in `run_experiments.py` with the following parameters:

| Parameter | Key | Default | Description |
|---|---|---|---|
| Phase | `phase` | (required) | Experiment group ("1", "2", "3", "5", "6") |
| Local batch size | `local_batch` | (required) | Per-device micro-batch size |
| Num blocks | `num_blocks` | (required) | Number of transformer blocks (use reduced counts like 2, 4, 8 to fit in memory, then extrapolate) |
| Tensor parallel | `tp` | 1 | TP degree (1, 2, 4, 8). Mesh uses first dim for TP-only, second dim for TP in TP+DDP |
| Data parallel | `ddp` | 1 | DDP degree (1, 2, 4, 8, 32). Mesh uses first dim for DDP-only, first dim for DDP in TP+DDP |
| Grad accumulation | `grad_accum` | 1 | Gradient accumulation steps before optimizer step |
| Runner type | `runner_type` | `"default"` | `"default"` (no checkpointing) or `"memory_efficient"` (gradient checkpointing at block boundaries) |
| Profiler | `profiler` | `True` | `True` (tracy — per-op device cycles), `"naive"` (host timestamps with device sync), `False` (no profiling, wall clock only) |

Example experiment definitions:

```python
# Single device, 4 blocks, batch 1, tracy profiler
_exp(phase="1", local_batch=1, num_blocks=4, runner_type="default")

# TP=8, 4 blocks, tracy profiler
_exp(phase="2", local_batch=1, num_blocks=4, tp=8, runner_type="default")

# DDP=8 only, naive profiler (gives gradient_sync directly)
_exp(phase="3", local_batch=1, num_blocks=4, ddp=8, runner_type="default", profiler="naive")

# Batch scaling, memory efficient, naive profiler
_exp(phase="5", local_batch=16, num_blocks=4, runner_type="memory_efficient", profiler="naive")

# TP+DDP end-to-end with grad accumulation
_exp(phase="6", local_batch=2, num_blocks=4, tp=8, ddp=4, grad_accum=4, profiler="naive")
```

### Mesh shape rules

- **TP only**: `mesh_shape = [tp, 1]` — first dimension
- **DDP only**: `mesh_shape = [ddp, 1]` — first dimension
- **TP + DDP**: `mesh_shape = [ddp, tp]` — first dim DDP, second dim TP
- Total devices = `tp × ddp` (must be ≤ 32 for one Galaxy)

### Profiler modes

| Mode | Flag | What you get | Overhead | Use when |
|---|---|---|---|---|
| Tracy | `profiler=True` | Per-device per-op device cycle timings, CCL breakdown | Very high (~50x wall clock) | Single device or TP-only |
| Naive | `profiler="naive"` | Per-phase host timestamps (fwd/bwd/sync/opt) | Minimal | Multi-device (TP+DDP), DDP experiments |
| None | `profiler=False` | Wall clock step time only | None | Quick sanity checks |

## Experiment Naming Convention

Names are auto-generated from parameters:

```
p{phase}_b{batch}_blk{blocks}_tp{tp}_ddp{ddp}_ga{grad_accum}_{runner}[_noprof|_naive]
```

Examples:
- `p1_b1_blk4_tp1_ddp1_ga1_default` — Phase 1, batch 1, 4 blocks, no TP/DDP, default runner, tracy profiler
- `p2_b1_blk4_tp8_ddp1_ga1_default` — Phase 2, TP=8, tracy profiler
- `p3_b1_blk4_tp1_ddp8_ga1_default_naive` — Phase 3, DDP=8, naive profiler
- `p5_b4_blk4_tp1_ddp1_ga1_memeff_naive` — Phase 5, batch 4, memory efficient (gradient checkpointing), naive profiler

## Visualization

Two Jupyter notebooks are provided:

- **`profiling_analysis_multiple_exp.ipynb`** — Full pipeline across all experiments: extract → enrich → DataFrame → batch/DDP/TP scaling plots. Edit the first cell to set your run directories and configuration.
- **`profiling_analysis_single_exp.ipynb`** — Deep-dive into a single profiler CSV: per-op time breakdowns (pie charts, bar charts), interactive per-operation plots, anomaly detection, and per-phase training step timing.

```bash
jupyter notebook tt-train/tools/profiling/profiling_analysis_multiple_exp.ipynb
jupyter notebook tt-train/tools/profiling/profiling_analysis_single_exp.ipynb
```

The visualization scripts can also be run standalone from the command line, taking a CSV as input:

```bash
# Convert JSON → CSV first
python3 tt-train/tools/profiling/results_visualization/results_to_csv.py results.json results.csv

# Then generate plots
python3 tt-train/tools/profiling/results_visualization/plot_batch_scaling.py results.csv output_dir/
python3 tt-train/tools/profiling/results_visualization/plot_ddp_scaling.py results.csv output_dir/
python3 tt-train/tools/profiling/results_visualization/plot_tp_scaling.py results.csv output_dir/
```

| Script | Plots |
|---|---|
| `plot_batch_scaling.py` | Tokens/s, fwd/bwd/opt/step time vs batch size (with ideal lines + MFU) |
| `plot_ddp_scaling.py` | Tokens/s, tokens/s/device, step time, gradient sync vs DDP degree |
| `plot_tp_scaling.py` | Tokens/s, tokens/s/device, fwd/bwd/opt/step time vs TP (per block count, with CCL overlay) |

## Experiment Phases

| Phase | Goal | Profiler |
|---|---|---|
| 1 | Single-device baselines (fwd/bwd/opt at multiple block counts) | Tracy |
| 2 | TP characterization (TP=2,4,8 at multiple block counts for differential method) | Tracy |
| 3 | DDP characterization (DDP=2,4,8,32 with TP=1) | Naive |
| 5 | Scaling verification (batch size, block count) | Naive |
| 6 | End-to-end validation (TP+DDP, grad_accum sweep) | Naive |
