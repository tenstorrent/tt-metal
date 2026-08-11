# tt-power-sidecar

A standalone sidecar tool for measuring live power consumption on Tenstorrent
Wormhole and Blackhole hardware during test runs or benchmarks.

## Quick start

```bash
# Wrap any command — no code changes needed
python3 tt_power_sidecar.py -- pytest tests/ops/test_matmul.py

# Custom interval and output path
python3 tt_power_sidecar.py --interval 50 -o power.json -- ./build/test_add

# Monitor specific devices
python3 tt_power_sidecar.py --devices 0,1 -v -- sleep 10
```

The sidecar runs the given command as a subprocess, polls device power in a
background thread, and writes a JSON report when the command exits.  The
wrapped command's exit code is propagated.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--interval` | 100 | Poll interval in milliseconds (must be > 0) |
| `--out` / `-o` | `power_report.json` | Output JSON file path |
| `--devices` | all | Comma-separated device indices to monitor |
| `--backend` | `auto` | Power measurement backend: `auto`, `sysfs`, or `pyluwen` |
| `--verbose` / `-v` | off | Print each sample to stderr as it arrives |

## Backends

Three backend modes are available via `--backend`:

| Mode | Behaviour |
|------|-----------|
| `auto` | Tries sysfs first; falls back to pyluwen if sysfs finds nothing |
| `sysfs` | sysfs hwmon only — reads `/sys/class/hwmon/hwmonN/power1_input` (µW) exposed by `tt-kmd`.  Zero Python dependencies.  **Safe for multi-chip systems.** |
| `pyluwen` | pyluwen firmware telemetry only.  ⚠️ **Do not use on T3000 or galaxy systems** — see Known Limitations. |

> **CI recommendation:** use `--backend sysfs` explicitly on any CI runner.
> It has zero Python dependencies, never contacts firmware, and is safe regardless
> of system topology (single-chip, T3000, or galaxy).

## JSON output format

```json
{
  "command": ["pytest", "test_matmul.py"],
  "exit_code": 0,
  "duration_s": 12.34,
  "poll_interval_ms": 100,
  "devices": {
    "0": {
      "energy_J": 48.2,
      "energy_Wh": 0.013389,
      "avg_power_W": 39.1,
      "peak_power_W": 75.0,
      "min_power_W": 15.0,
      "sample_count": 123,
      "backend": "sysfs"
    }
  }
}
```

Energy is computed via trapezoidal integration of the power samples over
wall-clock time.  `energy_Wh = energy_J / 3600`.

## Pytest fixture

`conftest_power.py` provides a `power_monitor` fixture for in-process
monitoring without the sidecar launcher:

```python
def test_matmul(power_monitor):
    # ... run workload ...
    pass
    # power_monitor.report is populated after the test body
    # power_monitor.report_path points to the JSON file
```

Copy or symlink `conftest_power.py` into your test directory, or reference
it via `pytest_plugins`.

## Requirements

- Python 3.10+
- Linux with `tt-kmd` driver loaded (for sysfs backend)
- Optional: `pyluwen` Python package (for fallback backend)

## Known Limitations

### pyluwen backend on multi-chip systems (T3000 / galaxy)

**Do not use `--backend pyluwen` explicitly on T3000 or galaxy systems.**

`pyluwen` reads power telemetry by sending ARC messages over Ethernet
(`RemoteCommunicationLegacyFirmware::read_non_mmio`).  On a T3000 with 8
chips, concurrent polling messages from a single host saturate the per-chip ARC
response queues (`RESPONSE_Q out of sync`), which then causes
`Timeout waiting for Ethernet core service remote IO request` failures in any
test that initialises the mesh fabric — even tests that are unrelated to the
sidecar.

**`--backend auto` is safe on T3000.**  When `tt-kmd` is loaded, `auto` detects
≥2 local sysfs devices and selects the sysfs path — no ARC traffic is generated.
The T3000 fabric and CCL jobs in CI (`t3000-fast-tests-impl.yaml`) use
`--backend auto` for exactly this reason.

**Workaround for explicit control:** use `--backend sysfs` on any multi-chip
runner to guarantee sysfs is used regardless of what pyluwen might otherwise
detect.
