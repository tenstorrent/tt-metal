# ViT on T3000

This demo shows how ViT Base patch16-224 runs on T3000 using data parallel execution across multiple devices.

## Setup

Ensure you have the required dependencies and have followed the setup instructions.

## How to run

Prepare your ImageNet validation or test split dataset. We expect the dataset to be in the ImageNet format inside a folder named `ImageNet_data`. The implementation uses the same dataset loading approach as other ViT demos in this repository.

**Dataset Setup:**
- Place your ImageNet validation dataset in a folder named `ImageNet_data/`
- The dataset should contain images in the standard ImageNet format
- If no local dataset is found, the implementation will fall back to using HuggingFace's imagenet-1k dataset

### Expected Performance

The ViT model achieves the following expected performance targets for different execution modes:

- **Basic**: device_batch_size=16, expected_inference_time=0.0120s
- **Trace**: device_batch_size=16, expected_inference_time=0.0070s
- **2CQs**: device_batch_size=16, expected_inference_time=0.0125s
- **Trace+2CQs**: device_batch_size=16, expected_inference_time=0.0043s

### Running the Performance Tests

To run the performance tests:

```bash
# Basic performance test
pytest models/demos/t3000/vit/tests/test_perf_e2e_vit.py::test_perf

# Trace optimization
pytest models/demos/t3000/vit/tests/test_perf_e2e_vit.py::test_perf_trace

# 2 Command Queues optimization
pytest models/demos/t3000/vit/tests/test_perf_e2e_vit.py::test_perf_2cqs

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
