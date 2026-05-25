# sweeps/

Op-sweep runner + lightweight result browser for reproducible TTSim experiments.

```
sweeps/
├── record.py         # run a sweep, write one JSON per (op, arch, workers) to runs/
├── build_report.py   # read runs/*.json → emit report.html
├── runs/             # committed result archive, one JSON per experiment
└── report.html       # static page, opens with file:// (Plotly loaded via CDN)
```

## Reproducing on another machine

Prereqs: the repo's `python_env` virtualenv with pytest + pytest-xdist installed, and
either a TTSim shared object at `~/sim/{bh,wh}/libttsim.so` (auto-detected) or real
hardware.

```bash
# Activate the repo's venv
source python_env/bin/activate

# Single run
sweeps/record.py sdpa -n 4 -m "not slow" -k decode_sweep

# Scaling sweep (one JSON per worker count)
sweeps/record.py sdpa --scaling "1 2 4 8" \
    -m "not slow" \
    -k "decode_sweep or (prefill_sweep and (1536 or mqa_2k))"

# Rebuild the report
sweeps/build_report.py
```

## Viewing the report

The HTML renders the summary tables server-side (always visible) and layers
Plotly charts on top via CDN. Open it however you like:

```bash
# Direct (works in most browsers; fine for VS Code's built-in "open with browser")
open sweeps/report.html         # macOS
xdg-open sweeps/report.html     # Linux

# Or serve over HTTP if your browser blocks file:// scripts:
python -m http.server -d sweeps 8000
# then visit http://localhost:8000/report.html
```

If the Plotly CDN is blocked, the charts won't render but the tables and the
per-test breakdown (in a `<details>` block) still have all the numbers.

Supported ops (first positional arg to `record.py`):

| op | test file |
|---|---|
| `sdpa` | `tests/ttnn/unit_tests/operations/sdpa/test_sdpa_sweep.py` |
| `conv` | `tests/ttnn/unit_tests/operations/conv/test_conv2d_sweep.py` |
| `layernorm` | `tests/ttnn/unit_tests/operations/fused/test_layer_norm_sweep.py` |

Everything after the op/`-n`/`--scaling` flags is passed through to pytest, so the
usual filters work: `-k EXPR`, `-m "not slow"`, `--timeout=...`, etc.

## What's in a run JSON

One file per `(op, arch, workers)` invocation, ~5–40 KB:

```jsonc
{
  "id":   "20260525T140737_sdpa_blackhole_n8",
  "ts":   "2026-05-25T14:07:37Z",
  "op":   "sdpa",
  "arch": "blackhole",         // detected from TT_METAL_SIMULATOR path
  "workers": 8,
  "dist": "worksteal",
  "test_path":   "tests/.../test_sdpa_sweep.py",
  "pytest_args": ["-m", "not slow", "-k", "decode_sweep or ..."],
  "host":   { "cores": 8, "hostname": "x" },
  "result": { "wall_s": 211.0, "passed": 38, "failed": 0, "skipped": 0 },
  "per_test": [
    { "id": "test_decode_sweep[gpt2_medium-kv_bf16]", "call_s": 12.34 },
    ...
  ]
}
```

Filename: `<UTC-timestamp>_<op>_<arch>_n<workers>.json`.

## How the report groups runs

`build_report.py` buckets runs by `(op, arch, hostname)`. Each bucket gets a section
with:

- a **scaling chart** (workers vs wall, workers vs speedup-relative-to-n=1)
- a **per-test bar chart** from the n=1 baseline, sorted descending
- a **summary table** with wall, speedup, efficiency, pass/fail counts

No theoretical overlays (Amdahl/linear) — just the observed numbers. If you want
those, add them in `build_report.py`.

## Adding a new op

1. Write a `test_*_sweep.py` with real-model presets (see existing files for the
   shape). Use `pytest.param(..., id="readable-name")` so the report tooltips are
   self-explanatory.
2. Add the op to the `OPS` dict in `record.py`.
3. `sweeps/record.py <new_op> -n 1` to smoke-test.

## What's checked in

- `record.py`, `build_report.py`, `README.md`
- `runs/*.json` — yes, commit your runs; that's the archive.
- `report.html` — yes, the built page is committed too, so anyone browsing the
  repo can open it directly without re-running anything.
