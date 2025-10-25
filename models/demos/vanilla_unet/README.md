# Vanilla UNet

## Model Overview

Vanilla UNet is a convolutional neural network architecture designed for biomedical image segmentation. The model uses an encoder-decoder structure with skip connections, enabling it to capture both high-level semantic information and fine-grained spatial details.

### Architecture

The UNet architecture consists of:
- **Encoder Path**: Four downsampling blocks, each with two 3x3 convolutions followed by batch normalization, ReLU activation, and 2x2 max pooling
- **Bottleneck**: Two 3x3 convolutions with batch normalization and ReLU activation
- **Decoder Path**: Four upsampling blocks using transposed convolutions, concatenated with skip connections from the encoder, followed by two 3x3 convolutions
- **Output Layer**: 1x1 convolution to produce the final segmentation mask

### Model Details

- **Input**: RGB images of size 480x640
- **Output**: Single-channel segmentation mask of size 480x640
- **Task**: Binary segmentation (identifying regions of interest in brain MRI scans)
- **Precision**: BFloat16 for activations and weights

## Getting Started

### Single Image Demo

Run inference on a single brain MRI image:

```bash
pytest models/demos/vanilla_unet/demo/demo.py::test_unet_demo_single_image
```

This will:
1. Load a test brain MRI image from `models/demos/vanilla_unet/demo/images/`
2. Run inference using the TT-NN implementation
3. Generate a visualization with:
   - **Red outline**: Model prediction
   - **Green outline**: Ground truth segmentation
4. Save the result to `models/demos/vanilla_unet/demo/pred/result_ttnn_1.png`

## Performance Testing

### Device Performance Benchmarks

Run device performance benchmarks:

```bash
pytest models/demos/vanilla_unet/tests/test_unet_perf.py::test_vanilla_unet_perf_device
```

### End-to-End Performance

Run end-to-end performance tests including compile time and inference throughput:

```bash
pytest models/demos/vanilla_unet/tests/test_unet_perf.py::test_vanilla_unet_perf_e2e
```
