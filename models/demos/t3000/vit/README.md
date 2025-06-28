# ViT on T3000

This demo shows how ViT Base patch16-224 runs on T3000 using data parallel execution across multiple devices.

## Setup

Ensure you have the required dependencies and have followed the setup instructions.

## Tests Overview

This demo includes two performance tests:

### 1. Full ImageNet Inference Test (`test_run_vit_trace_2cqs_inference`)
- **Purpose**: Complete end-to-end inference on ImageNet-1k validation dataset
- **Features**:
  - Processes actual ImageNet images with preprocessing
  - Calculates accuracy metrics
  - Provides detailed per-sample predictions
  - Measures both inference time and accuracy
- **Use Case**: Validation and accuracy verification

### 2. Performance Benchmark Test (`test_run_vit_trace_2cqs_inference_with_random_inputs`)
- **Purpose**: Pure performance benchmarking with synthetic data
- **Features**:
  - Uses random inputs for maximum throughput measurement
  - Focuses on inference speed and FPS
  - No data preprocessing overhead
  - Optimized for performance measurement
- **Use Case**: Performance benchmarking and optimization

## Running the Tests

### Full ImageNet Inference Test
```bash
# Run with ImageNet validation data (requires ImageNet dataset)
pytest models/demos/t3000/vit/demo/demo_vit_performant_imagenet_inference.py::test_run_vit_trace_2cqs_inference
```

### Performance Benchmark Test
```bash
# Run with random inputs for performance measurement
pytest models/demos/t3000/vit/demo/demo_vit_performant_imagenet_inference.py::test_run_vit_trace_2cqs_inference_with_random_inputs
```


## Data Parallel Implementation

This implementation uses data parallel execution across T3000 devices:

- `device_batch_size`: The batch size per device = 8
- `batch_size = device_batch_size * num_devices`: Total batch size across all devices (64 for 8 devices)
- Images are distributed across devices for parallel processing
- Results are aggregated for final output
- Performance metrics account for the total system throughput across all devices

## Performance Metrics

Both tests report:
- **Inference Time**: Average time per iteration (wall-clock time)
- **FPS**: Frames per second (total samples processed per second)
- **Compile Time**: Time for initial model compilation
- **Accuracy**: (Full test only) Classification accuracy on ImageNet validation set
