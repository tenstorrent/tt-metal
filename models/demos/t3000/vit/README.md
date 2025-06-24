# ViT on T3000

This demo shows how ViT Base patch16-224 runs on T3000 using data parallel execution across multiple devices.

## Setup

Ensure you have the required dependencies and have followed the setup instructions.

### Running the Performance Tests

To run the performance tests:

```bash
# Trace + 2CQs optimization (best performance)
pytest models/demos/t3000/vit/tests/test_perf_e2e_vit.py::test_perf_trace_2cqs
```

## Data Parallel Implementation

This implementation uses data parallel execution across T3000 devices:

- `device_batch_size`: The batch size per device
- `batch_size = device_batch_size * num_devices`: Total batch size across all devices
- Images are distributed across devices for parallel processing
- Results are aggregated for final output

## Model Details

- **Model**: google/vit-base-patch16-224
- **Input Resolution**: 224x224
- **Patch Size**: 16x16
- **Architecture**: Vision Transformer with 12 transformer blocks
- **Parameters**: ~86M parameters
