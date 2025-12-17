# PyTorch Reference Outputs - Usage Guide

This guide explains how to generate and use PyTorch reference outputs for cross-machine testing.

## Generating Reference Outputs

### Basic Usage (CPU)

```bash
# Generate on CPU (default)
python generate_torch_reference_outputs.py
```

### GPU Usage

```bash
# Generate on default CUDA device (cuda:0)
python generate_torch_reference_outputs.py --device cuda

# Generate on specific CUDA device
python generate_torch_reference_outputs.py --device cuda:0
python generate_torch_reference_outputs.py --device cuda:1
```

### Options

- `--device DEVICE`: Device to use for computation
  - `cpu` - Use CPU (default)
  - `cuda` - Use default CUDA GPU
  - `cuda:0`, `cuda:1`, etc. - Use specific GPU

## Generated Files

The script generates the following structure:

```
torch_reference_outputs/
├── system_info.json          # System information (Python, PyTorch, device info)
├── config.json                # Test configuration
├── README.md                  # Auto-generated documentation
├── bfloat16/                  # BFloat16 test cases
│   ├── k0072_ic008_conv2d_input.npy
│   ├── k0072_ic008_conv2d_weight.npy
│   ├── k0072_ic008_conv2d_bias.npy
│   ├── k0072_ic008_conv2d_output.npy
│   ├── k0072_ic008_matmul_A.npy
│   ├── k0072_ic008_matmul_B.npy
│   ├── k0072_ic008_matmul_output.npy
│   ├── k0072_ic008_metadata.json    # Shapes, dtypes, hashes
│   └── ...
└── float32/                   # Float32 test cases
    └── ...
```

## Loading Reference Outputs

### Python API

```python
from load_torch_reference_outputs import ReferenceDataLoader

# Initialize loader
loader = ReferenceDataLoader("torch_reference_outputs")

# Print summary
loader.print_summary()

# Load a specific test case
data = loader.load_test_case(dtype="bfloat16", input_channels=8)

# Access tensors
conv2d_input = data['conv2d']['input']
conv2d_weight = data['conv2d']['weight']
conv2d_bias = data['conv2d']['bias']
conv2d_output = data['conv2d']['output']

matmul_A = data['matmul']['A']
matmul_B = data['matmul']['B']
matmul_output = data['matmul']['output']

# Access metadata
shapes = data['metadata']['shapes']
dtypes = data['metadata']['dtypes']
hashes = data['metadata']['input_hashes']
```

### Verify Data Integrity

```python
# Verify that loaded data matches expected hashes
verification = loader.verify_hashes(dtype="bfloat16", input_channels=8)

# Check results
if all(verification['conv2d'].values()) and all(verification['matmul'].values()):
    print("✓ All inputs verified - same data generated on both machines")
else:
    print("✗ Hash mismatch - different data generated")
```

### List Available Test Cases

```python
# Get all available test cases
cases = loader.list_available_cases()

for dtype, input_channels in cases:
    k = loader.ic_to_k[input_channels]
    print(f"dtype={dtype}, input_channels={input_channels}, K={k}")
```

## Cross-Machine Testing Workflow

### Machine 1 (Generate Reference)

```bash
# Generate reference outputs (use --device cuda if GPU available)
python generate_torch_reference_outputs.py --device cpu

# Compress for transfer
tar -czf torch_reference_outputs.tar.gz torch_reference_outputs/
```

### Machine 2 (Verify and Compare)

```bash
# Transfer and extract
scp machine1:torch_reference_outputs.tar.gz .
tar -xzf torch_reference_outputs.tar.gz

# Verify data integrity
python load_torch_reference_outputs.py

# Or use in your own script
python -c "
from load_torch_reference_outputs import ReferenceDataLoader

loader = ReferenceDataLoader('torch_reference_outputs')

# Verify all test cases
for dtype in ['bfloat16', 'float32']:
    for ic in [8, 16, 32, 64, 128, 256]:
        verification = loader.verify_hashes(dtype=dtype, input_channels=ic)
        all_pass = all(verification['conv2d'].values()) and all(verification['matmul'].values())
        status = '✓' if all_pass else '✗'
        print(f'{status} {dtype} ic={ic}')
"
```

## Hash Verification

The input hashes allow you to verify that the same random inputs were generated on both machines:

- **SHA256 hashes** are computed for all input tensors
- Hashes are stored in `metadata.json` files
- Use `verify_hashes()` to confirm identical generation

**Note**: For bfloat16 tensors, hashes are computed on the float32 representation (since NumPy doesn't support bfloat16 natively). This ensures cross-platform compatibility.

## Configuration

The script uses the following fixed configuration:

- **Seed**: 42 (for reproducibility)
- **Batch size**: 1
- **Input size**: 32×32
- **Output channels**: 32
- **Kernel**: 3×3
- **Padding**: (1, 1)
- **Stride**: (1, 1)
- **Bias**: Enabled
- **Data types**: bfloat16, float32
- **Input channels sweep**: [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]

K values are automatically calculated as: K = input_channels × 3 × 3

## Device Selection Impact

### CPU vs GPU

The device choice affects:
- **Execution speed**: GPU is typically faster for large tensors
- **Numerical results**: May have slight differences due to different hardware implementations
- **Compatibility**: CPU results are most portable across systems

### Recommended Practices

1. **For cross-machine comparison**: Use the same device type (both CPU or both GPU)
2. **For maximum reproducibility**: Use CPU on both machines
3. **For performance testing**: Use GPU if available, but document in system_info.json

### System Information

The `system_info.json` file records:
- Python version
- PyTorch version
- Platform (OS, architecture)
- Device used (cpu, cuda:0, etc.)
- CUDA version (if GPU used)
- GPU name (if GPU used)
- NumPy version

This information helps identify sources of numerical differences when comparing results across machines.

## Example: Compare TTNN vs PyTorch

```python
import torch
import ttnn
import numpy as np
from load_torch_reference_outputs import ReferenceDataLoader

# Load reference data
loader = ReferenceDataLoader("torch_reference_outputs")
data = loader.load_test_case(dtype="bfloat16", input_channels=32)

# Convert to torch tensors
input_tensor = torch.from_numpy(data['conv2d']['input']).to(torch.bfloat16)
weight_tensor = torch.from_numpy(data['conv2d']['weight']).to(torch.bfloat16)
bias_tensor = torch.from_numpy(data['conv2d']['bias']).to(torch.bfloat16)

# Get PyTorch reference output
pytorch_output = torch.from_numpy(data['conv2d']['output']).to(torch.bfloat16)

# Run TTNN conv2d
device = ttnn.CreateDevice(0)
ttnn_output = run_ttnn_conv2d(input_tensor, weight_tensor, bias_tensor,
                               padding=(1,1), stride=(1,1), device=device)
ttnn.close_device(device)

# Compare results
from models.common.utility_functions import ulp
ulp_errors = ulp_error(ttnn_output, pytorch_output)
print(f"Mean ULP error: {ulp_errors.mean().item():.4f}")
print(f"Max ULP error: {ulp_errors.max().item():.4f}")
```

## Troubleshooting

### Hash Mismatch

If hashes don't match across machines:
1. Verify the same PyTorch version is used
2. Check that the same seed (42) is being used
3. Ensure no external factors affect random number generation
4. Confirm NumPy versions are compatible

### Memory Issues

For large input_channels (e.g., 8192):
- Use GPU if available (`--device cuda`)
- Process dtypes separately
- Monitor memory usage

### CUDA Errors

If CUDA is requested but unavailable:
- Script automatically falls back to CPU
- Check CUDA installation with `torch.cuda.is_available()`
- Verify GPU drivers are installed
