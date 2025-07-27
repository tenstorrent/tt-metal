# TT‑Metal Profiler Guide

> **Quick start**: Build with profiling enabled, run your experiment with a single command, then open the companion notebook to explore the generated `.csv`.

---

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Build & Install](#build--install)
4. [Running the Profiler](#running-the-profiler)
5. [Output Artifacts](#output-artifacts)
6. [Analysing Results](#analysing-results)
7. [Problems](#problems)

---

## Features

- **One‑line build** with all profiler hooks enabled.
- **Environment‑variable toggles**—no code changes required.
- Generates a **single comma‑separated log (**``**)** ready for pandas / Excel.
- Companion **Jupyter Notebook** for rich visualisation.

---

## Prerequisites

| Requirement  | Version    | Notes                                             |
| ------------ | ---------- | ------------------------------------------------- |
| Ubuntu / WSL | 20.04 +    | Tested on 22.04 as well                           |
| Python       | 3.9 +      | `python -m venv venv && source venv/bin/activate` |
| CMake        | ≥ 3.20     |                                                   |
| Ninja        | (optional) | Faster parallel builds                            |

> **Tip:** GPU drivers / SDK versions follow the standard TT‑Metal requirements—see the main project README.

---

## Build & Install

```bash
./build_metal.sh -b Release \
                 --enable-profile \
                 --build-tt-train
```

The flags do the following:

| Flag               | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| `-b Release`       | Compile with full optimisation.                           |
| `--enable-profile` | Injects profiler hooks into every kernel.                 |
| `--build-tt-train` | Builds the `tt-train` helper used by the NanoGPT example. |

After completion the relevant binaries live under `build/tt-train/`.

---

## Running the Profiler

Enable python environment (created in tt-metal) and run following command:

```bash
TT_METAL_WATCHER_NOINLINE=1 \
TT_METAL_WATCHER_DEBUG_DELAY=10 \
TT_METAL_READ_DEBUG_DELAY_CORES=0,0 \
TT_METAL_WRITE_DEBUG_DELAY_CORES=0,0 \
TT_METAL_READ_DEBUG_DELAY_RISCVS=BR \
TT_METAL_WRITE_DEBUG_DELAY_RISCVS=BR \
python -m tracy -r -v -p build/tt-train/sources/examples/nano_gpt/nano_gpt
```

### What do those variables mean?

| Variable                                 | Default | Description                                                                         |
| ---------------------------------------- | ------- | ----------------------------------------------------------------------------------- |
| `TT_METAL_WATCHER_NOINLINE`              | `0`     | Forces watchdog helpers to stay **out‑of‑line** for clearer flame graphs.           |
| `TT_METAL_WATCHER_DEBUG_DELAY`           | `0` ms  | Extra delay (ms) after each kernel for debugger attachment.                         |
| `TT_METAL_READ/WRITE_DEBUG_DELAY_CORES`  | `"0,0"` | Comma‑separated *(x,y)* coordinates of cores for which to inject read/write delays. |
| `TT_METAL_READ/WRITE_DEBUG_DELAY_RISCVS` | `BR`    | RISC‑V side delays. Use `BR` for **B**oot & **R**untime, `B`, `R`, or leave empty.  |

> **Can I omit some options?** Yes—only set what you actually need; unset variables fallback to defaults.

---

## Output Artifacts

After the run finishes, you will find a **single CSV log** in the current working directory:

```text
trace_<timestamp>.csv
```

The file contains one row per kernel launch with timestamps, core coordinates, cycle counts, and any custom markers you emit via `ctx().get_profiler().mark(...)`.

---

## Analysing Results

Install pandas, matplotlib and ipywidgets to use companion jupyter notebook. Open the companion notebook and point it to your freshly‑generated CSV:

```bash
jupyter lab notebooks/profiler_results.ipynb
```

The notebook walks you through:

1. **Aggregating time‑per‑op** across the whole training step.
2. Analyzing anomalies inside metrics per operation

Feel free to fork / extend the notebook for your own workflows.

---

## Problems

Unfortunately, we can't pass command arguments in tracy command, that's why you need to hardcode correct config path in cpp file before building tt-train.
