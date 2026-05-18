# MoE Expert Balance Logging

> **Quick start:**
> 1. Add `--log-expert-activations moe_activation.csv` to your DeepSeek training invocation
> 2. After training (or mid-run), run `python tt-train/scripts/plot_expert_activation.py --input moe_activation.csv --output moe_activation.png`
> 3. Open `moe_activation.png` — one heatmap per MoE layer, x = step, y = expert, color = P(activation)

---

## Table of Contents

1. [Purpose](#purpose)
2. [Prerequisites](#prerequisites)
3. [Training Argument](#training-argument)
4. [CSV Schema](#csv-schema)
5. [Plotting](#plotting)
6. [Output](#output)
7. [Example Workflow](#example-workflow)
8. [See Also](#see-also)

---

## Purpose

Mixture-of-Experts routes each token to a small subset of specialized experts,
enabling models to scale capacity without proportionally scaling compute.
For this to work well, the routing must stay balanced across experts: if a small
subset absorbed most tokens while others went unused, the MoE model would
collapse silently.
This tool makes that visible by logging per-expert activation probabilities at training time and rendering them as heatmaps.

Use it to:

- **Detect routing collapse** early (one or two experts dominate across all steps)
- **Verify load balancing** is working as intended (activation probability close to `n_activated / num_experts` across experts)
- **Track routing dynamics** — watch how expert utilization evolves over the course of training

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Via `create_venv.sh` |
| pandas | Included in dev dependencies |
| matplotlib | Included in dev dependencies |
| numpy | Included in dev dependencies |

> **Note:** All dependencies are installed automatically when you run `create_venv.sh`.

---

## Training Argument

```
--log-expert-activations <path>
```

Appends per-step per-expert activation probabilities to a CSV file. **DeepSeek models only.**

| Property | Value |
|----------|-------|
| Argument | `--log-expert-activations <csv_path>` |
| Default | `None` (disabled) |
| Model support | DeepSeek only |
| Log schedule | Steps 1–10 (inclusive), then every 100th step |
| File behaviour | Created fresh on step 1; appended on subsequent steps |

The sparse schedule (dense early, sampled later) keeps file size manageable while capturing routing behaviour at both the start of training and throughout.

### Example invocation

```bash
python tt-train/sources/examples/nano_gpt/train_nanogpt.py \
    --config configs/nano_deepseek.yaml \
    --log-expert-activations moe_activation.csv
```

---

## CSV Schema

One row per `(step, layer, expert)`:

```
step,layer,expert,prob
1,0,0,0.031250
1,0,1,0.062500
...
```

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Global training step |
| `layer` | int | MoE layer index (0-based) |
| `expert` | int | Expert index (0-based) |
| `prob` | float | Fraction of tokens that selected this expert during the step (`[0, 1]`) |

The fully-balanced target is `n_activated_experts / num_experts` (e.g. `2/64 ≈ 0.031` for DeepSeek-style routing).

---

## Plotting

```bash
python tt-train/scripts/plot_expert_activation.py \
    --input moe_activation.csv \
    --output moe_activation.png
```

### All options

| Option | Default | Description |
|--------|---------|-------------|
| `--input` | (required) | Path to activation CSV |
| `--output` | (required) | Destination PNG path |
| `--layer` | all layers | If set, plot only this MoE layer index |
| `--cmap` | `viridis` | Matplotlib colormap name |
| `--dpi` | `150` | Figure DPI |

### Single-layer plot

```bash
python tt-train/scripts/plot_expert_activation.py \
    --input moe_activation.csv --output layer0.png --layer 0
```

---

## Output

The script generates a single PNG heatmap where each subplot (= MoE layer) is a heatmap in which:
- **x-axis** — training step (up to 16 tick labels; steps are sampled evenly if there are more)
- **y-axis** — expert index
- **color** — P(activation) in `[0, 1]`
- **title** — layer index and the fully-balanced lower bound (`~ 1 / num_experts`)

A colorbar on the right of each subplot maps color to activation probability.

Healthy routing looks like a roughly uniform color across all experts. A single bright horizontal band or a mostly-dark grid with one or two bright rows indicates routing collapse.

---

## Example Workflow

1. **Run training with logging enabled:**
   ```bash
   python tt-train/sources/examples/nano_gpt/train_nanogpt.py \
       --config configs/nano_deepseek.yaml \
       --log-expert-activations runs/exp1/moe_activation.csv \
       > runs/exp1/train.log 2>&1
   ```

2. **Generate the heatmap after training completes (or mid-run):**
   ```bash
   python tt-train/scripts/plot_expert_activation.py \
       --input runs/exp1/moe_activation.csv \
       --output runs/exp1/moe_activation.png
   ```

3. **Inspect the output:**
   - Uniform color across experts → balanced routing
   - One or two experts with significantly higher activation → potential routing collapse
   - Dead experts (always dark) → experts that never get selected

4. **Drill into a single layer if needed:**
   ```bash
   python tt-train/scripts/plot_expert_activation.py \
       --input runs/exp1/moe_activation.csv \
       --output runs/exp1/layer3.png \
       --layer 3
   ```

---

## See Also

- [TRAINING_LOG_COMPARISON.md](../TRAINING_LOG_COMPARISON.md) — compare loss curves and step times across runs
- [PROFILER.md](../PROFILER.md) — kernel-level profiling
- [MEMORY_TRACKING.md](../MEMORY_TRACKING.md) — memory usage analysis
