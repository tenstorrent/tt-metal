# DIDT Tests

This directory contains DIDT tests for validating the behavior, performance, and power consumption characteristics of TT-Metal operations across various hardware configurations.

## Available Tests

- **`test_binary_mul.py`**: Binary element-wise multiplication for power consumption analysis
- **`test_deepseek_v3_128k_matmul.py`**: Deepseek V3 prefill matmuls with 128k sequence length
- **`test_duty_cycle.py`**: Duty cycle testing with alternating matmul/non-matmul workloads
- **`test_ff1_matmul.py`**: Feed-forward layer matmul with/without GELU activation
- **`test_lm_head_matmul.py`**: Language model head matmul operations
- **`test_minimal_matmul.py`**: Basic matmul operations for baseline testing
- **`test_mla_sdpa.py`**: Multi-Layer Attention SDPA with submesh support
- **`test_mm_after_non_mm.py`**: Matmul operations following non-matmul workloads
- **`test_resnet_conv.py`**: ResNet convolution operations
- **`test_sdpa_op.py`**: Scaled Dot Product Attention operations
- **`test_sdxl_conv.py`**: Stable Diffusion XL convolution tests (UNet and VAE)
- **`test_sdxl_conv_1280x1280_upsample.py`**: SDXL upsample convolution tests
- **`test_sdxl_matmul.py`**: SDXL-specific matmul operations
- **`test_sharded_ff1.py`**: Legacy sharded FF1 tests

## didt_runner

`didt_runner.py` runs the DIDT ND/hang screens of one config on one Blackhole galaxy, with one command.
It writes one `status.json` that names each failing chip by hardware identity (`bdf tray asic asic_id kmd`) and,
when the test logs it, the failing cores. A FAIL or HANG in one step never stops the next step.

### Setup

- Build as usual: `./build_metal.sh`, then `./create_venv.sh` and `source python_env/bin/activate`.
- Run from the repo root, on the galaxy itself. Never on a login node.
- Needs a Blackhole galaxy (32 chips) and permission to run `tt-smi -glx_reset_auto`: the runner resets before every attempt.
- Under SLURM `sbatch`: launch with `srun --propagate=NONE` and wrap the command in `sg slurmusers -c '...'`.
  Without them the job gets `ulimit -u 512` (too few processes for metal) and the reset fails with `Operation not permitted`.

### Run

| command | what it does |
|---|---|
| `python tests/didt/didt_runner.py` | every step of the weekly config |
| `python tests/didt/didt_runner.py --list` | print step keys, test ids and fallbacks, then exit |
| `python tests/didt/didt_runner.py --co` | collect gate only: every test id must still collect; no device work |
| `python tests/didt/didt_runner.py --only nd:ff1,nd:mcast-ack` | only these step keys (comma-separated) |
| `python tests/didt/didt_runner.py --skip nd:core-sweep` | every step except these |
| `python tests/didt/didt_runner.py --config daily` | a shipped tier config by name, or `--config <file.json>`; default `weekly` |
| `python tests/didt/didt_runner.py --out <dir>` | parent of the run dir; default `generated/didt_runner` |

Exit code: `0` every step PASS or WARN, `1` any step FAIL or HANG, `2` config/usage error or no chips found.

Shipped configs, in `tests/didt/runner_configs/`:

| tier | steps | used by |
|---|---|---|
| `daily` | `nd:minimal` with `throttle` and `pin_core` | exabox health check `daily` |
| `weekly` | every step | exabox health check `weekly` and `full`; the default |
| `install` | same as `weekly` today; kept apart because the two will diverge | deployment tests `test_runner.sh` |

### Results

Each run makes a new dir `<out>/LOG_<YYYY_MM_DD_HH_MM>_<host>_<commit>/` (suffix `_2`, `_3`, ... if it exists).

| path | content |
|---|---|
| `status.json` | the only result: chips, steps, attempts, failing devices; rewritten after every change |
| `run.log` | one timestamped line per attempt start and end |
| `collect/` | collect gate logs |
| `nd/<name>/` | first attempt of step `nd:<name>`: `reset.log`, `log` (test output), `tt-smi.json`, `crash_triage.csv` on a hang, `chips/chip<N>/` (`each_chip` steps: that chip's `pytest.log`, `crash_triage.csv` on a hang) |
| `nd/<name>_throttle/` | `throttle` fallback: same test with `TT_MM_THROTTLE_PERF=5` |
| `nd/<name>_pin_core_dev<N>/` | `pin_core` fallback: the per-chip test on failing device `N` alone, twice the iterations |
| `nd/<name>_throttle_hang/` | `throttle_hang` fallback (`each_chip` steps): every chip again, `TT_MM_THROTTLE_PERF=5`, `MATMUL_ITERS=100` |

Fallbacks run only after a FAIL or HANG that their `on` list names (default both), in config order, each in its own
`<dir>_<name>` folder.
Status values: `PENDING RUN PASS WARN FAIL HANG SKIP`. A step's status is its first attempt; fallbacks never change it.
The `status.json` fields are listed in [runner/CLAUDE.md](runner/CLAUDE.md).

### How chips run and fallbacks

| attempt | chips |
|---|---|
| pytest step (minimal .. core-sweep), first attempt | one pytest over the full 32-chip mesh |
| pytest step, `throttle` | the same, full mesh |
| pytest step, `pin_core` | the per-chip test on one failing chip; failing chips one after another |
| `each_chip` step (mcast-ack, mcast-ack-duty), first attempt | one pytest per chip, every chip at once |
| `each_chip` step, `throttle_hang` | the same, every chip at once |

| step type | fallback | runs after | knobs | reset before it |
|---|---|---|---|---|
| pytest | `throttle` | FAIL or HANG | `TT_MM_THROTTLE_PERF=5` | normal galaxy reset |
| pytest | `pin_core` | FAIL or HANG, per hung or ND device | that chip alone, twice the iterations, at least the first loop time | ND: normal; hang: `tt-smi -r` |
| `each_chip` | `throttle_hang` | HANG only; ND alone is FAIL with no fallback | `TT_MM_THROTTLE_PERF=5`, `MATMUL_ITERS=100` (first: 10) | normal galaxy reset |

`each_chip` steps have no `pin_core`: their first attempt already runs each chip in its own process.

| `throttle_hang` result | status | reason |
|---|---|---|
| every chip passes | PASS | `32/32 chips passed`: throttle cures the hang |
| ND on some chips, no hang | FAIL | `failed on k/32 chips`, ND device rows |
| any chip hangs | HANG | `hang persists at throttle 5`, hung device rows from each chip's triage |

| `pin_core` result (pytest steps) | status | reason |
|---|---|---|
| the chip hangs again | HANG | `triage <csv>`, hung device row |
| the chip shows ND again | FAIL | `ND reproduced on chip alone: k/c checks` |
| the chip passes | WARN | `not reproduced on chip alone` |
| another chip fails | adds | `WARNING failure on device N, outside the pinned chip` |

Gap: `throttle` on a pytest step gives one verdict for the whole mesh; only the first attempt's failing chips get a
`pin_core`.

### Env knobs

All opt-in: unset, the test runs as before. The runner passes your shell environment to every step (except
`TT_VISIBLE_DEVICES`, `TT_METAL_VISIBLE_DEVICES`, `TT_MESH_GRAPH_DESC_PATH`, `TT_MM_THROTTLE_PERF`, the dispatch
timeout variables and the `pin_core` knobs `DIDT_DEVICE_ID`, `DIDT_MIN_LOOP_SECONDS`), so an exported knob changes
every step that reads it. A step that needs a mesh graph descriptor sets it in its config `env`.

| variable | read by | effect |
|---|---|---|
| `DIDT_DEVICE_ID=<dev>` | `didt_chip.py` (per-chip tests) | pick the chip by device id, not by submesh index; `pin_core` sets it |
| `DIDT_MIN_LOOP_SECONDS=<s>` | `OpTestBase.run_op_test` | keep iterating until the loop has run `<s>` seconds; `pin_core` sets it |
| `DIDT_MINIMAL_M=<M>` | `test_minimal_matmul.py` | override M (default 16384); M > N transposes the core grid |
| `DIDT_GRID_X=<x>`, `DIDT_GRID_Y=<y>` | `test_ff1_matmul.py` | shrink the compute grid, anchored at core (0,0) |
| `DIDT_MATH_FIDELITY=<name>` | `test_minimal_matmul.py`, `test_ff1_matmul.py` | `LoFi`, `HiFi2`, `HiFi3` or `HiFi4` |
| `DIDT_IN0_DTYPE`, `DIDT_IN1_DTYPE`, `DIDT_OUT_DTYPE` | `test_ff1_matmul.py` | `ttnn.DataType` name, e.g. `BFLOAT4_B`; defaults `BFLOAT16`, `BFLOAT8_B`, `BFLOAT16` |
| `MCAST_ACK_WORK_DIR=<dir>` | `test_mm_mcast_write_ack_failure.py` | per-chip logs dir of the standalone scan (default `/tmp/mm_mcast_write_ack_failure`) |
| `MCAST_ACK_CACHE_DIR=<dir>` | `test_mm_mcast_write_ack_failure.py` | kernel cache dir of the standalone scan (default: the work dir); keep it on local disk |

### Unit tests

No device needed:

```bash
python -m pytest -q --noconftest -p no:cacheprovider tests/didt/runner tests/didt/test_didt_chip.py \
  tests/didt/test_mcast_ack_scan_dirs.py tests/didt/test_op_test_base_min_loop.py tests/didt/test_nd_core_map.py
```

Expected: 158 passed, 5 skipped. The skipped tests render `status.json` with the sweep page; they run only when they
find the `build-gwdt-sweep` skill (set `GWDT_SWEEP_DIR` to the skill's folder).

### Changing the runner

Read [runner/CLAUDE.md](runner/CLAUDE.md) first: it has the rules, the `status.json` shape and where each change goes.
PR and commit rules: [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Analysis Tools

- **`process_profiler_output.py`**: Tool for processing profiler CSV output
- **`sync_analysis.py`**: Synchronization analysis utilities

## Usage Examples

```bash
# Matrix multiplication tests
pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and all" --didt-workload-iterations 100
pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "with_gelu and all" --didt-workload-iterations 100
pytest tests/didt/test_lm_head_matmul.py::test_lm_head_matmul -k "all" --didt-workload-iterations 100
pytest tests/didt/test_minimal_matmul.py::test_minimal_matmul -k "all" --didt-workload-iterations 100

# Convolution tests
pytest tests/didt/test_resnet_conv.py::test_resnet_conv -k "all" --didt-workload-iterations 100
pytest tests/didt/test_sdxl_conv.py::test_sdxl_conv -k "unet_resnet_1280x1280 and all" --didt-workload-iterations 100

# Attention operations
pytest tests/didt/test_sdpa_op.py::test_sdpa_op -k "bf16_HiFi2 and all" --didt-workload-iterations 100
pytest tests/didt/test_mla_sdpa.py::test_mla_sdpa -k "all" --didt-workload-iterations 100

# Power and performance analysis
pytest tests/didt/test_binary_mul.py::test_binary_mul -k "without_gelu and all" --didt-workload-iterations 100
pytest tests/didt/test_duty_cycle.py::test_duty_cycle -k "duty-3 and rep-1000x and all" --didt-workload-iterations 100

# Advanced model tests
pytest tests/didt/test_deepseek_v3_128k_matmul.py::test_deepseek_v3_mla_matmul -k "1chips" --didt-workload-iterations 100
pytest tests/didt/test_deepseek_v3_128k_matmul.py::test_deepseek_v3_gate_matmul -k "galaxy" --didt-workload-iterations 100
```

## System Support

Supported hardware configurations:
- **N150, N300**: Single and dual-chip systems
- **T3000**: 8-chip systems
- **6U Galaxy**: 32-chip systems (8×4 mesh)
- **Blackhole**: Single-chip and multi-chip configurations

Parametrization IDs: `1chips`, `2chips`, `8chips`, `galaxy`, `all`

Most tests support both Wormhole and Blackhole architectures. Some tests are Blackhole-specific.

## Configuration Options

```bash
# Custom iteration count (default: 100000)
pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips" --didt-workload-iterations 5000000

# Determinism checking (check every N iterations)
pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips" --didt-workload-iterations 5000 --determinism-check-interval 50

# Target specific device by ID
pytest tests/didt/test_ff1_matmul.py::test_specific_chip_ff1_matmul -k "8chips and 8chips_chip_3"

# Target specific board (T3000)
pytest tests/didt/test_ff1_matmul.py::test_specific_board_ff1_matmul -k "8chips and board_id_2"

# Submesh testing
pytest tests/didt/test_mla_sdpa.py::test_mesh_size_mla_sdpa -k "4x2 and 0-0"

# Duty cycle testing (adjust compute/non-compute ratio)
pytest tests/didt/test_duty_cycle.py -k "duty-3 and rep-1000x"  # 3 non-MM ops per MM op
pytest tests/didt/test_duty_cycle.py -k "duty-6 and rep-10000x"  # Lower duty cycle
```

Math fidelity levels: LoFi (faster), HiFi2/HiFi3/HiFi4 (higher precision).
Data types: BFLOAT16, BFLOAT8_B, BFLOAT4_B (support varies by test).

## Blackhole Features

Default compute grid: runs on full device grid (optimized for Blackhole architecture).
Supports single-chip (`1chips`) and Galaxy (`galaxy`) configurations.
Some tests require a full 11×10 grid (Deepseek V3, MLA SDPA).

```bash
# Reduced grid size testing (maintains compute per core)
pytest tests/didt/test_ff1_matmul.py::test_grid_size_ff1_matmul -k "2chips and 6x8"
```

## Analysis Tools

```bash
# Process Tracy profiler CSV output for timing analysis
python tests/didt/process_profiler_output.py --input-file <profiler-csv>
```

This generates per-core timing differences, standard deviation analysis, and synchronization statistics.
The `sync_analysis.py` tool provides utilities for analyzing device synchronization patterns.

## Creating New Tests

All DIDT tests inherit from `OpTestBase` which provides iteration management, memory management, determinism checking, device synchronization, and flexible configuration support.
To add a new test, create a new file under the same directory, and then either:
- instantiate object of the base class in case you don't need to change any behavior, just populate dimensions, configs etc (example in `test_ff1_matmul.py`)
- extend the base class to override any behavior that needs to be changed (for now we allow to change the way we generate activations & weights, and setting the seed), and then instantiate object of the new class

Simple test creation:
```python
from tests.didt.op_test_base import OpTestBase, OpParameter

test = OpTestBase(
    mesh_device,
    OpParameter(input_shape, dtype, layout, mem_config),  # activations
    [OpParameter(weight_shape, dtype, layout, mem_config)],  # inputs
    output_mem_config, output_dtype, program_config, compute_config
)
test.run_op_test()
```

Custom test with overrides:
```python
class CustomTest(OpTestBase):
    def run_device_operation(self):
        return custom_ttnn_operation(self.activations, self.inputs[0], ...)

    def generate_torch_activations(self, shape):
        return custom_tensor_generation(shape)
```

Available method overrides: `run_device_operation()`, `generate_torch_activations()`, `generate_torch_input()`, `set_seed()`, `deallocate_activations()`

## Legacy Commands

```bash
# Legacy FF1 without GELU
pytest models/experimental/falcon_7b/tests/test_reproduce_hang_matmul.py -k "test_reproduce_matmul_2d_hang and ff1-hang and 8chips"

# Legacy FF1 with GELU
pytest tests/didt/test_sharded_ff1.py -k "test_reproduce_matmul_2d_hang and 8chips"

# Legacy LM head
pytest models/demos/falcon7b/tests/test_falcon_hang.py -k "test_reproduce_lm_head_nd_32 and 8chips"
```
