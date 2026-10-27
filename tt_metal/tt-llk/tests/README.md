# Test Infrastructure Guide

To run the test infrastructure, ensure your system has one of the following devices installed:

- **Blackhole** (or any derivative)
- **Wormhole** (or any derivative)

> ⚠️ The device must be flashed with the original firmware.

When the testing environment is correctly initialized, it will auto-detect the underlying hardware.

---

## Steps to Run Tests

### 1. Set Up the Environment

- **If using the `tt-llk` Docker image**, run:

    ```bash
    ./setup_testing_env.sh
    ```

- **If you are an external developer or not using the Docker image**, run:

    ```bash
    source ./setup_external_testing_env.sh
    ```

> 🔄 **Note**: Always use `source` to ensure the environment is activated in your current shell session.

---

### 2. Navigate to the Python Test Directory

```bash
cd python_tests
```

### 3. Run any test using `pytest`

```bash
pytest <test_file_name>
```

Replace <test_file_name> with the specific test script you want to execute, e.g., test_eltwise_binary_sfpu.py.

### 4. Run all tests using `pytest`

```bash
pytest
```

### Running on the ttsim functional simulator

See [TTSIM.md](TTSIM.md) for running the LLK suite on
[ttsim](https://github.com/tenstorrent/ttsim) — no silicon required.

---

## Row-pack performance comparison

After setting up the LLK environment, run the matching pack-isolate sweeps from
`tests/python_tests` on their respective targets:

```bash
CHIP_ARCH=quasar PERF_RUN_TAG=pack-rows-quasar pytest -q quasar/perf_pack_rows_to_addr_quasar.py
CHIP_ARCH=blackhole PERF_RUN_TAG=pack-rows-blackhole pytest -q perf_pack_rows_to_addr.py
```

For an emulator, add `--run-simulator` with its usual simulator environment.
Use silicon or a cycle-faithful emulator for timing; functional ttsim timing
does not measure hardware performance.

Both sweeps cover BF16, FP32, and plain E4M3 FP8 output at 1, 8, 15, 16, 17, 32,
33, 50, and 64 rows. They use 32-bit DEST, SyncHalf, DEST tile 0, and a ring of
16 output slots. After warming all slots, each sample measures 1024 pack calls,
including address setup, loop overhead, and the final packer drain. Init and
warm-up are excluded from the measured loop. Five samples provide mean and
standard deviation. This measures pack execution; use the functional tests to
check output data.

Once both raw reports are available in the same checkout, compare them from
`tests/python_tests`:

```bash
python ../../perf/compare_pack_rows_to_addr.py \
  --quasar ../../perf_data/runs/pack-rows-quasar/perf_pack_rows_to_addr_quasar/perf_pack_rows_to_addr_quasar.csv \
  --blackhole ../../perf_data/runs/pack-rows-blackhole/perf_pack_rows_to_addr/perf_pack_rows_to_addr.csv
```

The helper normalizes the raw timings to cycles per pack call and prints the
Blackhole/Quasar ratio. Values greater than 1 favor Quasar. It rejects incomplete
sweeps, mismatched configurations, and already-normalized `.post.csv` inputs.

## Logging

Tests use [loguru](https://github.com/Delgan/loguru) for structured, leveled
logging. Import the shared logger and use `{}` placeholders:

```python
from helpers.logger import logger

logger.info("Running op={} on {}", op_name, arch)
```

Control verbosity with `--logging-level=DEBUG` on the CLI or by setting the
`LOGURU_LEVEL` environment variable. Logs are written to `test_run.log` (session)
and `test_errors.log` (persistent, ERROR+ only).

See [LOGGING.md](LOGGING.md) for the
full guide including log levels, output destinations, and pytest-xdist behaviour.

## SFPU accuracy

Each SFPU op declares how closely its output must match its golden, in
`python_tests/helpers/sfpu_accuracy_budget.yaml`: a tolerance, or a step budget. Most
unary step budgets are measured by `python_tests/test_unary_sfpu_ulp.py` over every
normal 16-bit value and a stride of float32; the binary, ternary and scalar rows, and
the `Signbit`, isinf/isnan and threshold-family rows, over their own drivers' stimuli.
See [SFPU_ULP.md](SFPU_ULP.md) for how to enrol an op and the rules it enforces.
