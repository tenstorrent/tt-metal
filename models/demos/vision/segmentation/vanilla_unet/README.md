# Vanilla UNet

## Platforms:
    Wormhole (N150, N300), Blackhole (P100)

## Introduction
This directory runs a UNet for single-class segmentation of brain MRI slices on device, at
batch size 1 and a 480x640 input: the four-level encoder, the bottleneck, the four-level decoder
with its transposed convolutions and skip connections, and the final 1x1 convolution.

UNet is an encoder-decoder convolutional network whose skip connections carry the encoder's
features of each resolution to the decoder level of the same resolution.
https://arxiv.org/abs/1505.04597

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

The tests and the demo download the pretrained weights on first use with
`weights_download.sh`, to `models/demos/vision/segmentation/vanilla_unet/unet.pt`.

## Demo

From the repository root, run inference on the sample image (Wormhole only for now):

```sh
pytest models/demos/vision/segmentation/vanilla_unet/demo/demo.py::test_unet_demo_single_image
```

The prediction is outlined in red and the ground truth in green, in
`models/demos/vision/segmentation/vanilla_unet/demo/pred/result_ttnn_1.png`.

## Performance

- Trace + 2CQ end-to-end throughput at batch 1, on one device and on every device of a mesh:

```sh
pytest models/demos/vision/segmentation/vanilla_unet/tests/test_unet_perf.py::test_vanilla_unet_perf_e2e
pytest models/demos/vision/segmentation/vanilla_unet/tests/test_unet_perf.py::test_vanilla_unet_perf_e2e_multi_device
```
- Device kernel throughput; its target is a Wormhole one, so it runs on Wormhole only:

```sh
pytest models/demos/vision/segmentation/vanilla_unet/tests/test_unet_perf.py::test_vanilla_unet_perf_device
```

## Testing

- `test_unet_model.py` runs the whole model on a random 480x640 input and checks its output
  against the PyTorch reference; `test_unet_model_multi_device.py` does the same on every device
  of a mesh:

```sh
pytest --disable-warnings models/demos/vision/segmentation/vanilla_unet/tests/test_unet_model.py
```

### Reduced grid

The model derives its plan from the device's grid. On a grid without an 8x8 block of worker cores
it runs from DRAM: the input stays in DRAM, every convolution, pool and transposed convolution
slices its DRAM input by height, the skip connections are concatenated in DRAM and the final 1x1
convolution runs as a DRAM matmul. The end-to-end test passes on two worker cores:

```sh
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=1,0 pytest models/demos/vision/segmentation/vanilla_unet/tests/test_unet_model.py
```

The override is the inclusive end coordinate of the worker grid, so `1,0` is a 2x1 grid.

The trace + 2CQ performance tests reshard their input into the L1 plan, so they skip on such a grid.

### Reduced layer count

`VANILLA_UNET_NUM_LAYERS=N` (1 to 4) keeps the first `N` encoder levels and the matching `N`
decoder levels; below the full depth, encoder `N + 1` stands in for the bottleneck. The reference is
reduced the same way. The tests and the demo read it:

```sh
VANILLA_UNET_NUM_LAYERS=1 pytest models/demos/vision/segmentation/vanilla_unet/tests/test_unet_model.py
```

## Project Structure

```
models/demos/vision/segmentation/vanilla_unet/
├── weights_download.sh    # Downloads the pretrained weights
├── demo/
│   ├── demo.py            # Single-image demo
│   └── demo_utils.py      # Image loading and visualization
├── reference/model.py     # PyTorch reference, its depth set by VANILLA_UNET_NUM_LAYERS
├── tests/
│   ├── test_unet_model.py               # E2E PCC
│   ├── test_unet_model_multi_device.py  # E2E PCC on a mesh
│   └── test_unet_perf.py                # Device and end-to-end throughput
└── tt/
    ├── common.py          # Weight loading, parameter preprocessing, VANILLA_UNET_NUM_LAYERS
    ├── config.py          # Per-layer configurations, from the device's grid
    └── model.py           # TTNN implementation
```

## Model Structure

`UNet` from `reference/model.py`, the repeated blocks folded:

```
UNet(
  (encoder1..4): 4 x Sequential(                 3 -> 32, 32 -> 64, 64 -> 128, 128 -> 256 channels
    (0): Conv2d(in, out, kernel_size=(3, 3), padding=(1, 1), bias=False) + BatchNorm2d + ReLU
    (3): Conv2d(out, out, kernel_size=(3, 3), padding=(1, 1), bias=False) + BatchNorm2d + ReLU
  )
  (pool1..4): 4 x MaxPool2d(kernel_size=2, stride=2)  (after each encoder)
  (bottleneck): Sequential(
    (0): Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1), bias=False) + BatchNorm2d + ReLU
    (3): Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1), bias=False) + BatchNorm2d + ReLU
  )
  (upconv4..1): 4 x ConvTranspose2d(in, in // 2, kernel_size=(2, 2), stride=(2, 2))  512 -> 256 ... 64 -> 32
  (decoder4..1): 4 x Sequential(                 512 -> 256, 256 -> 128, 128 -> 64, 64 -> 32 channels
    (0): Conv2d(2 * out, out, kernel_size=(3, 3), padding=(1, 1), bias=False) + BatchNorm2d + ReLU
    (3): Conv2d(out, out, kernel_size=(3, 3), padding=(1, 1), bias=False) + BatchNorm2d + ReLU
  )
  (conv): Conv2d(32, 1, kernel_size=(1, 1))
)
```

Each decoder's input is its upconv's output concatenated with the skip connection of the encoder
of the same resolution.

Indented under each file are the modules from the tree it implements, with the class or function
in parentheses.

```
tt/model.py
    UNet (TtUNet)
    encoder1..4 with pool1..4 (TtUNetEncoder)
    upconv4..1 (transpose_conv2d)
    skip-connection concatenation (concatenate_skip_connection)

tt/config.py
    every layer's configuration, from the device's grid (TtUNetConfigBuilder)

tt/common.py
    parameters, with each BatchNorm2d folded into its Conv2d (create_unet_preprocessor)
```

## Details
- Entry points are `create_unet_configs_from_parameters` in `tt/config.py`, given the device so
  the plan fits its grid, and `create_unet_from_configs` in `tt/model.py`.
- Batch size: 1 per device
- Input: 3x480x640, bfloat16; output: 1x480x640.
- Parameters: 7.8M; weights: 31 MB.
