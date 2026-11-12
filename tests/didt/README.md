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

## MaxUtil Power Virus (`unit_tests_didt_max_util`, Blackhole)

A C++ gtest (`tests/didt/max_util_workload/`) that holds constant, controllable power in each
power domain at the same time. Every knob is an environment variable.

| Domain | Generator | Knobs |
|---|---|---|
| FPU | TRISC MVMUL loop | `MAX_UTIL_FPU_UTILIZATION_PCT` (0 = off, 1–92, default 92) |
| DRAM (all boards) | NCRISC on every worker, reads round-robin across all banks over NOC0 and NOC1 | `MAX_UTIL_TDRAM_*` (see below) |
| DRAM via ETH cores | Active ERISC, one bank per ETH core | `MAX_UTIL_DRAM_UTILIZATION_PCT` (0 = off), `MAX_UTIL_DRAM_PAGE_SIZE_BYTES` |
| Ethernet SerDes | Active ERISC0, bidirectional traffic on every link that is up between dispatched chips | `MAX_UTIL_ETH_LINK_*` (see below) |
| Duty cycle / run length | Interleaved slow cos() program | `MAX_UTIL_DUTY_CYCLE_PCT`, `MAX_UTIL_NUM_WL_LOOPS`, `MAX_UTIL_NUM_ITERATIONS`, `MAX_UTIL_SUPER_SYNC` |

The Tensix-DRAM (`MAX_UTIL_TDRAM_`) and ETH-link (`MAX_UTIL_ETH_LINK_`) generators share one knob scheme.
Its raw knobs mirror the syseng ttx NOC power virus:

| Knob | Meaning | Default |
|---|---|---|
| `<prefix>_XFER_BYTES` | Bytes per transfer, a power of two. TDRAM: 64–16384. ETH link: 64–8192. 0 turns the generator off | 8192 |
| `<prefix>_BURST` | Transfers per burst. 0 means continuous | 0 |
| `<prefix>_IDLE_512` | Idle time after each burst, in units of 512 wall-clock ticks | 0 |
| `<prefix>_UTILIZATION_PCT` | Phase-locked duty cycle: traffic only while `(wall_clock/512) % PERIOD < PERIOD*pct/100`, so all generators toggle together. 0 turns the generator off. Ignored if `BURST` or `IDLE_512` is set | 100 |
| `<prefix>_PERIOD_512` | Phase period in units of 512 ticks, a power of two | 256 (~97 µs) |

ETH-link-only knobs:
- `MAX_UTIL_ETH_LINK_NUM_MSGS` sets the payloads per side per dispatch. By default it is derived from the compute duration.
- `MAX_UTIL_ETH_LINK_STARTUP_DELAY_TICKS` adds a delay before the per-dispatch handshake.

Ethernet link traffic runs only in `MaxUtilWorkload_AllDevices`, because both ends of a link must be dispatched together.
Smoke and Stress log a skip.
Links that leave the cluster, such as Galaxy exit links, are skipped. A single-chip board (p100/p150) has no links up, so it gets FPU and DRAM load only.
Active-ETH kernels require slow dispatch.

```bash
export TT_METAL_SLOW_DISPATCH_MODE=1
# Single chip: full FPU + DRAM
MAX_UTIL_NUM_WL_LOOPS=1000000 MAX_UTIL_NUM_ITERATIONS=100 \
  ./build/test/didt/unit_tests_didt_max_util --gtest_filter=*Stress*
# Galaxy / multi-chip: FPU + DRAM + all internal ETH links
MAX_UTIL_NUM_WL_LOOPS=1000000 MAX_UTIL_NUM_ITERATIONS=100 \
  ./build/test/didt/unit_tests_didt_max_util --gtest_filter=*AllDevices*
# Isolate one domain, e.g. Ethernet only at 50%
MAX_UTIL_FPU_UTILIZATION_PCT=0 MAX_UTIL_TDRAM_UTILIZATION_PCT=0 MAX_UTIL_DRAM_UTILIZATION_PCT=0 \
  MAX_UTIL_ETH_LINK_UTILIZATION_PCT=50 ./build/test/didt/unit_tests_didt_max_util --gtest_filter=*AllDevices*
```

## Legacy Commands

```bash
# Legacy FF1 without GELU
pytest models/experimental/falcon_7b/tests/test_reproduce_hang_matmul.py -k "test_reproduce_matmul_2d_hang and ff1-hang and 8chips"

# Legacy FF1 with GELU
pytest tests/didt/test_sharded_ff1.py -k "test_reproduce_matmul_2d_hang and 8chips"

# Legacy LM head
pytest models/demos/falcon7b/tests/test_falcon_hang.py -k "test_reproduce_lm_head_nd_32 and 8chips"
```
