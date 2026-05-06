# Training Log Comparison Analysis

> **Quick start:**
> 1. Run your training binary multiple times with different configurations
> 2. Save the logs to text files (e.g., `baseline.txt`, `optimized.txt`)
> 3. Run `python tt-train/scripts/plot_training_comparison.py --baseline baseline.txt --compare optimized.txt`

---

## Table of Contents

1. [Purpose](#purpose)
2. [Prerequisites](#prerequisites)
3. [Log Format](#log-format)
4. [Usage](#usage)
5. [Output](#output)
6. [Example Workflow](#example-workflow)

---

## Purpose

This tool helps evaluate the impact of kernel optimizations, fusion strategies, or configuration changes in tt-train by:

- **Comparing training loss curves** across multiple runs
- **Visualizing loss differences** relative to a baseline
- **Analyzing step time performance** to measure speedups
- **Computing summary statistics** (mean step time, speedup factors, final loss)

Use this when you've made changes to kernels (e.g., fusing operations) and want to verify:
1. The optimization doesn't degrade training quality (loss should match or improve)
2. The optimization improves performance (step time should decrease)

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Via `create_venv.sh` |
| NumPy | - | Included in dev dependencies |
| Matplotlib | - | Included in dev dependencies |

> **Note:** All dependencies are installed automatically when you run `create_venv.sh`.

---

## Log Format

The script expects log files from tt-train's main training binary (e.g., `nano_gpt`). The logs should contain lines in the following format:

```
Step: 1, Loss: 11.0234375
Full step time 703.141 ms
Step: 2, Loss: 10.8765432
Full step time 698.234 ms
...
```

The script extracts:
- **Loss values**: From lines matching `Step: \d+, Loss: ([\d.]+)`
- **Step times**: From lines matching `Full step time ([\d.]+) ms`

---

## Usage

### Basic Comparison

Compare a baseline run against an optimized version:

```bash
python tt-train/scripts/plot_training_comparison.py \
    --baseline run_baseline.txt \
    --compare run_optimized.txt
```

### Multiple Comparisons with Labels

Compare multiple optimization strategies:

```bash
python tt-train/scripts/plot_training_comparison.py \
    --baseline fw_only.txt \
    --compare fw_bw_3_packs.txt fw_bw_4_packs.txt \
    --labels "Forward Only" "FW+BW 3 Packs" "FW+BW 4 Packs"
```

### Customization Options

```bash
python tt-train/scripts/plot_training_comparison.py \
    --baseline baseline.txt \
    --compare optimized.txt \
    --output-dir ./plots \
    --warmup-steps 20 \
    --max-steps 5000 \
    --title-prefix "NanoLlama SiLU "
```

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--baseline` | (required) | Path to baseline log file |
| `--compare` | `[]` | Paths to log files to compare |
| `--labels` | filenames | Custom labels for each run |
| `--output-dir` | `.` | Directory for output plots |
| `--warmup-steps` | `15` | Steps to skip for timing analysis |
| `--max-steps` | all | Limit steps in loss plots |
| `--title-prefix` | `""` | Prefix for plot titles |

---

## Output

The script generates three plots:

### 1. `losses.png` - Loss Comparison
Shows training loss curves for all runs overlaid. Useful for verifying that optimizations don't degrade convergence.

### 2. `losses_diff.png` - Loss Difference
Shows the difference in loss between each compared run and the baseline. Values near zero indicate equivalent training quality.

### 3. `step_time.png` - Step Time Comparison
Shows per-step execution time for all runs. Lower is better.

### Console Output

The script also prints summary statistics:

```
SUMMARY STATISTICS
============================================================

Mean Step Times:
  baseline: 703.14 ms (std: 12.34 ms)
  optimized: 650.22 ms (std: 10.56 ms)

Speedup relative to 'baseline':
  optimized: 1.081x

Final Loss (last 100 steps average):
  baseline: 3.456789
  optimized: 3.456123
```

---

## Example Workflow

### Evaluating a SiLU Kernel Fusion

1. **Run baseline training:**
   ```bash
   ./build/tt-train/sources/examples/nano_gpt/nano_gpt > baseline.txt 2>&1
   ```

2. **Apply your kernel optimization and rebuild:**
   ```bash
   ./build_metal.sh -b Release --build-tt-train
   ```

3. **Run optimized training:**
   ```bash
   ./build/tt-train/sources/examples/nano_gpt/nano_gpt > optimized.txt 2>&1
   ```

4. **Compare results:**
   ```bash
   python tt-train/scripts/plot_training_comparison.py \
       --baseline baseline.txt \
       --compare optimized.txt \
       --labels "Baseline" "SiLU Fused" \
       --output-dir ./silu_comparison \
       --title-prefix "SiLU Fusion: "
   ```

5. **Review:**
   - Check `losses.png` - curves should overlap closely
   - Check `losses_diff.png` - differences should be near zero
   - Check `step_time.png` - optimized should be faster
   - Review printed speedup factor

---

## See Also

- [PROFILER.md](PROFILER.md) - Detailed kernel-level profiling
- [MEMORY_TRACKING.md](MEMORY_TRACKING.md) - Memory usage analysis
