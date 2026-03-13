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
# Custom iteration count (default: 1000)
pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips" --didt-workload-iterations 5000000

# Determinism checking (check every N iterations)
pytest tests/didt/test_ff1_matmul.py::test_ff1_matmul -k "without_gelu and 2chips" --didt-workload-iterations 5000 --determinism-check-interval 50

# Target specific device by ID
pytest tests/didt/test_ff1_matmul.py::test_specific_chip_ff1_matmul -k "8chips and logical_chip_3_"

# Target specific board (T3000)
pytest tests/didt/test_ff1_matmul.py::test_specific_board_ff1_matmul -k "8chips and board_id_2"

# Submesh testing
pytest tests/didt/test_mla_sdpa.py::test_mesh_size_mla_sdpa -k "4x2 and 0-0"

# Duty cycle testing (adjust compute/non-compute ratio)
pytest tests/didt/test_duty_cycle.py -k "duty-3 and rep-1000x"  # 3 non-MM ops per MM op
pytest tests/didt/test_duty_cycle.py -k "duty-6 and rep-10000x"  # Lower duty cycle
```

Math fidelity levels: LoFi (faster), HiFi2/HiFi3/HiFi4 (higher precision)
Data types: BFLOAT16, BFLOAT8_B, BFLOAT4_B (support varies by test)

## Blackhole Features

Default compute grid: runs on full device grid (optimized for Blackhole architecture)
Supports single-chip (`1chips`) and Galaxy (`galaxy`) configurations
Some tests require a full 11×10 grid (Deepseek V3, MLA SDPA)

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
