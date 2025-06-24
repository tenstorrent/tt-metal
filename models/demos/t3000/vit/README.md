# ViT on T3000

This demo shows how ViT Base patch16-224 runs on T3000 using data parallel execution across multiple devices.

## Setup

Ensure you have the required dependencies and have followed the setup instructions.

### Running the Performance Tests

To run the performance tests:

```bash
# Trace + 2CQs optimization (best performance)
pytest models/demos/t3000/vit/demo/demo_vit_performant_imagenet_inference.py::test_run_vit_trace_2cqs_inference
```

## Data Parallel Implementation

This implementation uses data parallel execution across T3000 devices:

- `device_batch_size`: The batch size per device = 8
- `batch_size = device_batch_size * num_devices`: Total batch size across all devices
- Images are distributed across devices for parallel processing
- Results are aggregated for final output
