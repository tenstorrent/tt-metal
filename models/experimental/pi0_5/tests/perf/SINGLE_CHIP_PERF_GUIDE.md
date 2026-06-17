# PI0.5 Single-Chip Stage Perf Tests — Run Guide

Three standalone perf tests isolate one pi0.5 stage on **a single Blackhole
chip** (no cross-chip D2D). Each has two modes:

- **Trace capture (default)** — captures the stage as one TTNN trace and times
  steady-state replay → wall-clock latency.
- **Tracy (`EAGER=1`)** — runs the stage *eagerly* (no trace) wrapped in
  signposts so Tracy can attribute per-op device time. Tracy cannot profile a
  traced replay, so device-op breakdowns must use this mode.

| File | Stage | Signpost window | Trace capture? |
|---|---|---|---|
| `test_pi0_5_siglip_only_single_chip.py`  | SigLIP encoder (27 blocks) | `PHASE_siglip → PHASE_end`   | yes |
| `test_pi0_5_prefill_only_single_chip.py` | VLM prefill (18 Gemma-2B blocks) | `PHASE_prefill → PHASE_end` | yes |
| `test_pi0_5_denoise_only_single_chip.py` | Denoise loop (N steps) | `PHASE_denoise → PHASE_end`  | yes (also PCC-checked) |

All three:
- auto-apply `_bench_runs/pi05_production.env` (prod flags on by default),
- assert the device handle spans exactly **1 chip** (guards "no D2D"),
- **skip** if the checkpoint isn't found.

---

## Setup

```bash
cd <repo>/tt-metal
# Set PI05_CHECKPOINT_DIR to a local pi0.5 checkpoint (machine-specific).
# Replace /path/to/pi05_weights below with your checkpoint dir.
```

---

## 1. Trace capture (wall-clock latency)

Default mode — just run the test. Reports avg/min/max ms over 20 replays.

```bash
# Denoise
PI05_CHECKPOINT_DIR=/path/to/pi05_weights python_env/bin/python -m pytest -svq \
  models/experimental/pi0_5/tests/perf/test_pi0_5_denoise_only_single_chip.py

# SigLIP
PI05_CHECKPOINT_DIR=/path/to/pi05_weights python_env/bin/python -m pytest -svq \
  models/experimental/pi0_5/tests/perf/test_pi0_5_siglip_only_single_chip.py

# VLM prefill
PI05_CHECKPOINT_DIR=/path/to/pi05_weights python_env/bin/python -m pytest -svq \
  models/experimental/pi0_5/tests/perf/test_pi0_5_prefill_only_single_chip.py
```

---

## 2. Tracy device-op profile (`EAGER=1`)

Wrap the same test with the tracy harness and set `EAGER=1`. Each run prints an
`OPs csv generated at: .../ops_perf_results_<ts>.csv` path at the end.

```bash
# Denoise
EAGER=1 PI05_CHECKPOINT_DIR=/path/to/pi05_weights python_env/bin/python -m tracy -p -r -v \
  --op-support-count 100000 -m pytest -svq \
  models/experimental/pi0_5/tests/perf/test_pi0_5_denoise_only_single_chip.py

# SigLIP
EAGER=1 PI05_CHECKPOINT_DIR=/path/to/pi05_weights python_env/bin/python -m tracy -p -r -v \
  --op-support-count 100000 -m pytest -svq \
  models/experimental/pi0_5/tests/perf/test_pi0_5_siglip_only_single_chip.py

# VLM prefill
EAGER=1 PI05_CHECKPOINT_DIR=/path/to/pi05_weights python_env/bin/python -m tracy -p -r -v \
  --op-support-count 100000 -m pytest -svq \
  models/experimental/pi0_5/tests/perf/test_pi0_5_prefill_only_single_chip.py
```

Then filter the CSV to the stage's signpost window:

```bash
tt-perf-report <csv_path> --start-signpost PHASE_siglip  --end-signpost PHASE_end   # SigLIP
tt-perf-report <csv_path> --start-signpost PHASE_prefill --end-signpost PHASE_end   # prefill
tt-perf-report <csv_path> --start-signpost PHASE_denoise --end-signpost PHASE_end   # denoise
```

---

## Pipeline-parallel chunk mode (SigLIP + prefill)

`PI05_CHUNK_TOKENS=N` replicates one **PP chunk** on a single chip: it feeds an
N-token synthetic hidden sequence straight through the stage's transformer
blocks (skipping patch-embed / SigLIP / lang-embed). Models the per-stage
workload when 768 image tokens are split into chunks (e.g. 768/24 = 32).

- Asserted `N <= 768` (the image-token budget).
- **Synthetic input** — this is a shape/perf microbench, not a correctness run
  (no PCC check in chunk mode).

```bash
# SigLIP, 32-token chunk, Tracy
EAGER=1 PI05_CHUNK_TOKENS=32 PI05_CHECKPOINT_DIR=/path/to/pi05_weights python_env/bin/python -m tracy -p -r -v \
  --op-support-count 100000 -m pytest -svq \
  models/experimental/pi0_5/tests/perf/test_pi0_5_siglip_only_single_chip.py

# VLM prefill, 32-token chunk, Tracy
EAGER=1 PI05_CHUNK_TOKENS=32 PI05_CHECKPOINT_DIR=/path/to/pi05_weights python_env/bin/python -m tracy -p -r -v \
  --op-support-count 100000 -m pytest -svq \
  models/experimental/pi0_5/tests/perf/test_pi0_5_prefill_only_single_chip.py
```

---

## Env knobs

| Var | Default | Applies to | Meaning |
|---|---|---|---|
| `PI05_CHECKPOINT_DIR` | repo `weights/pi05_libero_upstream` | all | checkpoint path (test skips if missing) |
| `EAGER` | unset | all | `1` = eager + signposts (Tracy); unset = trace capture |
| `PI05_CHUNK_TOKENS` | `0` | siglip, prefill | `>0` = N-token synthetic PP chunk (≤768); `0` = full real input |
| `PI05_LANG_SEQ_LEN` | `256` | prefill, denoise | language input length (full path only) |
| `PI05_NUM_DENOISE_STEPS` | `5` | denoise | flow-matching Euler steps |
| `PI0_NUM_CAMERAS` | `3` | siglip, prefill, denoise | image slots (full path only) |
| `PI05_TRACE_NUM_ITERS` | `20` | all | timed trace replays |
| `PI05_TRACE_NUM_WARMUP` | `2` | all | JIT warmup iters |
| `PI05_DENOISE_PCC` | `0.99` | denoise | PCC pass threshold (traced vs eager) |

> Notes
> - SigLIP full-path token count is fixed by image patches (256/image); it is
>   **not** affected by `PI05_LANG_SEQ_LEN`. Use `PI05_CHUNK_TOKENS` to vary its
>   sequence length.
> - Tracy profiles **eager** execution only; the trace-capture path is for
>   wall-clock latency, not op attribution.
