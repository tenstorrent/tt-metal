# TT‑Metal Profiler Guide

> **Quick start:**
> 1. Build with `build_metal.sh`
> 2. Run `tt-train/run_profiler.sh` with the appropriate binary
> 3. Open `tt-train/notebooks/profiler_results.ipynb` notebook and run using the generated `ops_perf_results_<timestamp>.csv`

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

- **One‑line build**
- **Environment‑variable toggles**—no code changes required
- Generates **comma‑separated log files** ready for pandas / Excel and **.tracy** file for the Tracy GUI
- Companion **Jupyter Notebook** for rich visualisation

---

## Prerequisites

| Requirement  | Version    | Notes                                             |
| ------------ | ---------- | ------------------------------------------------- |
| Ubuntu / WSL | 20.04 +    | Tested on 22.04 as well                           |
| Python       | 3.9 +      | `./create_venv.sh && source python_env/bin/activate` |
| CMake        | ≥ 3.20     |                                                   |
| Ninja        | (optional) | Faster parallel builds                            |

> **Tip:** GPU drivers / SDK versions follow the standard TT‑Metal requirements—see the main project README.

---

## Build & Install

```bash
./build_metal.sh -b Release --build-tt-train
```

The flags do the following:

| Flag               | Purpose                                                   |
| ------------------ | --------------------------------------------------------- |
| `-b Release`       | Compile with full optimisation.                           |
| ~~`--enable-profile`~~ | *(No longer needed)* Injects profiler hooks into every kernel, now enabled by default. |
| `--build-tt-train` | Builds the `tt-train` helper used by the NanoGPT example. |

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

### What do those variables mean?

| Variable                                 | Default | Description                                                                         |
| ---------------------------------------- | ------- | ----------------------------------------------------------------------------------- |
| `TT_METAL_DPRINT_CORES`                  | `0,0`   | DPRINT and profiler cannot be enabled at the same time. |
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
| File | Description |
|------|--------------|
| `ops_perf_results_<timestamp>.csv` | The file contains one row per kernel launch with timestamps, core coordinates, cycle counts etc. |
| `profile_log_device.csv` | Device-side profiling data captured during execution. |
| `tracy_profile_log_host.tracy` | Host-side Tracy profiler log, which can be opened with the Tracy GUI.|

Custom markers can be emitted via `ctx().get_profiler().read_results("my_custom_marker")`

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
