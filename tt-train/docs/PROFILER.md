# TT‑Metal Profiler Guide

> **Quick start:**
>
> 1. Build with `build_metal.sh`
> 2. Run `tt-train/run_profiler.sh` with the appropriate binary
> 3. Open `tt-train/notebooks/profiler_results.ipynb` notebook and run using the generated `ops_perf_results_<timestamp>.csv`

---

## Table of Contents

- [TT‑Metal Profiler Guide](#ttmetal-profiler-guide)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Build \& Install](#build--install)
  - [Running the Profiler](#running-the-profiler)
    - [What do those variables mean?](#what-do-those-variables-mean)
  - [Output Artifacts](#output-artifacts)
  - [Analysing Results](#analysing-results)
  - [Profiler Markers](#profiler-markers)
    - [API](#api)
      - [Low-level: `profiler_marker`](#low-level-profiler_marker)
    - [Insertion patterns](#insertion-patterns)
      - [Sequential pipeline (single consumer)](#sequential-pipeline-single-consumer)
      - [Parallel branches (multiple consumers of one input)](#parallel-branches-multiple-consumers-of-one-input)
      - [Loop over the same input (fan-out)](#loop-over-the-same-input-fan-out)
      - [Forward-only markers (training loop)](#forward-only-markers-training-loop)
    - [Quick reference](#quick-reference)
    - [Avoiding common pitfalls](#avoiding-common-pitfalls)
  - [Problems](#problems)

---

## Features

- **One‑line build**
- **Environment‑variable toggles**—no code changes required
- Generates **comma‑separated log files** ready for pandas / Excel and **.tracy** file for the Tracy GUI
- Companion **Jupyter Notebook** for rich visualisation

---

## Prerequisites


| Requirement  | Version    | Notes                                                |
| ------------ | ---------- | ---------------------------------------------------- |
| Ubuntu / WSL | 20.04 +    | Tested on 22.04 as well                              |
| Python       | 3.9 +      | `./create_venv.sh && source python_env/bin/activate` |
| CMake        | ≥ 3.20     |                                                      |
| Ninja        | (optional) | Faster parallel builds                               |


> **Tip:** GPU drivers / SDK versions follow the standard TT‑Metal requirements—see the main project README.

---

## Build & Install

```bash
./build_metal.sh -b Release --build-tt-train
```

The flags do the following:


| Flag                   | Purpose                                                                                |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `-b Release`           | Compile with full optimisation.                                                        |
| ~~`--enable-profile`~~ | *(No longer needed)* Injects profiler hooks into every kernel, now enabled by default. |
| `--build-tt-train`     | Builds the `tt-train` helper used by the NanoGPT example.                              |


After completion the relevant binaries live under `${TT_METAL_HOME}/build/tt-train/`.

---

## Running the Profiler

Activate python environment (created in tt-metal) and run following command:

```bash
env -u TT_METAL_DPRINT_CORES \
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DEBUG_DELAY=10 \
TT_METAL_READ_DEBUG_DELAY_CORES=0,0 \
TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0 \
TT_METAL_READ_DEBUG_DELAY_RISCVS=BR \
TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR \
python -m tracy -r -v -p ${TT_METAL_HOME}/build/tt-train/sources/examples/nano_gpt/nano_gpt
```

Alternatively you can use this script, passing your full training command (binary or python script) as the argument:

```bash
./tt-train/run_profiler.sh ./tt-train/sources/examples/nano_gpt/train_nanogpt.py --config ./tt-train/configs/training_configs/training_shakespeare_nanogpt.yaml
# or for a compiled binary:
./tt-train/run_profiler.sh ./build/tt-train/sources/examples/nano_gpt/nano_gpt --config tt-train/configs/training_configs/training_shakespeare_tinyllama.yaml
```

### What do those variables mean?


| Variable                                 | Default | Description                                                                         |
| ---------------------------------------- | ------- | ----------------------------------------------------------------------------------- |
| `TT_METAL_DPRINT_CORES`                  | `0,0`   | DPRINT and profiler cannot be enabled at the same time.                             |
| `TT_METAL_WATCHER_NOINLINE`              | `0`     | Forces watchdog helpers to stay **out‑of‑line** for clearer flame graphs.           |
| `TT_METAL_WATCHER_DEBUG_DELAY`           | `0` ms  | Extra delay (ms) after each kernel for debugger attachment.                         |
| `TT_METAL_READ/WRITE_DEBUG_DELAY_CORES`  | `"0,0"` | Comma‑separated *(x,y)* coordinates of cores for which to inject read/write delays. |
| `TT_METAL_READ/WRITE_DEBUG_DELAY_RISCVS` | `BR`    | RISC‑V side delays. Use `BR` for **B**oot & **R**untime, `B`, `R`, or leave empty.  |


> **Can I omit some options?** Yes—only set what you actually need; unset variables fallback to defaults.

---

## Output Artifacts

After the run finishes, all profiler outputs are stored under:
`${TT_METAL_HOME}/generated/profiler/reports/<timestamp>/`

The directory contains the following files:


| File                               | Description                                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------ |
| `ops_perf_results_<timestamp>.csv` | The file contains one row per kernel launch with timestamps, core coordinates, cycle counts etc. |
| `profile_log_device.csv`           | Device-side profiling data captured during execution.                                            |
| `tracy_profile_log_host.tracy`     | Host-side Tracy profiler log, which can be opened with the Tracy GUI.                            |


Custom markers can be emitted via `ctx().get_profiler().read_results(device, "my_custom_marker", dump_results=True)` for a final or periodic flush.
At least one call with `dump_results=True` is required to produce or update the device-side profiling artifacts.

---

##  Analysing Results

Install pandas, matplotlib and ipywidgets to use companion jupyter notebook. Open the companion notebook and point it to your freshly‑generated CSV:

```bash
jupyter lab tt-train/tools/profiling/profiling_analysis_single_exp.ipynb
```

The notebook walks you through:

1. **Aggregating time‑per‑op** across the whole training step.
2. Analyzing anomalies inside metrics per operation

Feel free to fork / extend the notebook for your own workflows.

---

## Profiler Markers

Profiler markers let you annotate regions of your model with `[START]`/`[END]` tags
that appear as noop device ops in the CSV trace. The companion notebook uses these
to build a hierarchical flamegraph of your training step.

### API

```python
from ttml.common.profiler_utils import profiler_marker_start, profiler_marker_end
```

**`profiler_marker_start(x, name, dump_results=False)`** / **`profiler_marker_end(x, name, dump_results=False)`**

These are the recommended functions for annotating model regions. They
automatically add the `[START]`/`[END]` prefix required by the visualisation
notebook:

```python
x = profiler_marker_start(x, "[Block] Attn")
x = self.attn(x, mask)
x = profiler_marker_end(x, "[Block] Attn")
```

Zones with the same name are accumulated — you don't need per-layer indices for
inner components. The outermost loop can use indices if needed:

```python
for i, block in enumerate(self.blocks):
    x = profiler_marker_start(x, "[Model] Block")
    x = block(x, mask)
    x = profiler_marker_end(x, "[Model] Block")
```

| Argument | Description |
|----------|-------------|
| `x` | Autograd tensor to pass through, or `None` for a forward-only marker. |
| `name` | Zone name (e.g. `"[Block] Attn"`). `[START]`/`[END]` is added automatically. |
| `dump_results` | If `True`, flush device profiling data to disk. Expensive — use sparingly (e.g. once per block). |

Returns a **new autograd tensor** wrapping the same device data. The caller
**must use the returned value** — discarding it drops the backward node.

- **Forward:** immediately fires a device noop tagged `[FWD] [START] <name>` or
  `[FWD] [END] <name>`.
- **Backward:** fires `[BWD] [START] <name>` or `[BWD] [END] <name>` when
  gradients reach this node.
- **`x=None`:** forward-only marker (no backward node).

#### Low-level: `profiler_marker`

```python
from ttml.common.profiler_utils import profiler_marker
```

**`profiler_marker(x, name, dump_results=False)`**

The underlying function used by `profiler_marker_start`/`profiler_marker_end`.
Emits `[FWD] <name>` and `[BWD] <name>` without any `[START]`/`[END]` prefix.
Use for training-loop phase markers that don't represent zones:

```python
profiler_marker(None, "forward_pass_done")
profiler_marker(None, "backward_pass_done")
```

### Insertion patterns

#### Sequential pipeline (single consumer)

When a tensor flows through one operation at a time, chaining on `x` is safe:

```python
h = profiler_marker_start(h, "[Block] Attn")
h = self.attn(h, mask)
h = profiler_marker_end(h, "[Block] Attn")
```

Forward trace: `[FWD] [START] → attn ops → [FWD] [END]`
Backward trace: `[BWD] [END] → attn backward → [BWD] [START]`

#### Parallel branches (multiple consumers of one input)

When the same input feeds multiple independent paths, each start marker must
**branch from the original tensor** into a separate variable. Do **not** chain
`x = profiler_marker_start(x, ...)` repeatedly — that builds a linear autograd
chain and backward markers will fire in the wrong zones.

```python
# CORRECT — independent branches from x_in:
x_in = profiler_marker_start(x, "[MoE]")

x_r = profiler_marker_start(x_in, "[MoE] Routing")
logits = self.gate(x_r)
scores = profiler_marker_end(scores, "[MoE] Routing")

x_e = profiler_marker_start(x_in, "[MoE] Experts")
# ... expert loop ...
output = profiler_marker_end(output, "[MoE] Experts")

x_s = profiler_marker_start(x_in, "[MoE] SharedExp")
shared = self.shared_experts(x_s)
shared = profiler_marker_end(shared, "[MoE] SharedExp")
```

```python
# WRONG — linear chain corrupts backward zone boundaries:
x = profiler_marker_start(x, "[MoE] Routing")
x = profiler_marker_start(x, "[MoE] Experts")
x = profiler_marker_start(x, "[MoE] SharedExp")
```

Why: each call wraps `x` with a new autograd node. Chaining means backward for
SharedExp must flow through Experts and Routing backward nodes first, causing
their `[BWD]` markers to fire inside the wrong zone.

#### Loop over the same input (fan-out)

When a loop runs N computations on the same input, each per-iteration start
must branch from a single parent — **not** chain through `x`:

```python
# CORRECT — all experts branch from x_e:
x_e = profiler_marker_start(x_in, "[MoE] Experts")
for i in range(num_experts):
    x_exp = profiler_marker_start(x_e, "[MoE] Expert")
    out = self.experts[i](x_exp)
    out = profiler_marker_end(out, "[MoE] Expert")
    # ... accumulate ...
output = profiler_marker_end(output, "[MoE] Experts")
```

```python
# WRONG — chains x through every iteration:
for i in range(num_experts):
    x = profiler_marker_start(x, "[MoE] Expert")
    out = self.experts[i](x)
    out = profiler_marker_end(out, "[MoE] Expert")
```

Why: the wrong version builds `x_e → exp0 → exp1 → ... → expN`. Any later
consumer of `x` (e.g. shared experts) would drag all N expert backward nodes
into its backward path.

#### Forward-only markers (training loop)

For markers outside the model's autograd graph, pass `None` to the low-level
`profiler_marker`:

```python
profiler_marker(None, "dataloader_step_done")
loss = model(tokens, mask)
profiler_marker(None, "forward_pass_done")
loss.backward()
profiler_marker(None, "backward_pass_done")
optimizer.step()
profiler_marker(None, "optimizer_step_done")
```

These are unprefixed (no `[FWD]`/`[BWD]`) and define the top-level training
phases in the flamegraph.

### Quick reference

| Situation | Pattern | Variable |
|-----------|---------|----------|
| Single pipeline | `x = profiler_marker_start(x, ...)` / `profiler_marker_end(x, ...)` | Reuse `x` |
| Multiple paths from one input | `x_a = profiler_marker_start(x, ...)` | Separate vars |
| Loop over same input | `x_i = profiler_marker_start(x_parent, ...)` | Per-iteration var |
| Training loop event | `profiler_marker(None, ...)` | No tensor |

### Avoiding common pitfalls

- **Shared modules in multiple contexts:** if the same module (e.g. `DeepSeekMLP`)
  is called from both a dense FFN path and as a shared expert inside MoE, do not
  put markers inside the shared module. The same marker name would fire in
  different nesting contexts, and the visualization cannot place them correctly.
  Instead, put markers at each call site.
- **`dump_results=True`:** flushes profiler data to disk. Use at most once per
  block to prevent device memory overflow, typically on the outermost END marker.
- **Integer tensors:** markers on non-float tensors (e.g. token IDs) are
  forward-only — the function detects integer dtypes and skips the backward node.

---

## Problems

Unfortunately, we can't pass command arguments in tracy command, that's why you need to hardcode correct config path in cpp file before building tt-train.
