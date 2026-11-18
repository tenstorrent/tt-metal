# RetinaNet

## Platforms:
    Blackhole (P100)

## Introduction
This directory runs torchvision's RetinaNet (`retinanet_resnet50_fpn_v2`) on device: the
ResNet-50 backbone with its FPN, both heads and the box decoding, with the score threshold and
NMS on the host, at batch size 1.

RetinaNet is a one-stage object detector that scores and refines nine anchors at every location
of five pyramid levels.
https://arxiv.org/abs/1708.02002

## Prerequisites

- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)

## Demo

From the repository root, run inference on an image:

```sh
python models/experimental/retinanet/demo/demo.py --input <path/to/image>
```

The image is resized to 512x512, without keeping its aspect ratio. `--threshold` (default 0.5)
sets the lowest score that is drawn; scores below torchvision's 0.05 are dropped before that.
The annotated result is written to `result.jpg` in `--output`, which defaults to
`models/experimental/retinanet/resources/outputs/`. The demo opens device 0.

## Performance

- Device kernel throughput at batch 1, measured over `tests/pcc/test_retinanet.py`:

```sh
pytest models/experimental/retinanet/tests/perf/test_perf.py
```

## Testing

- Use the following command to run every PCC test:

```sh
pytest --disable-warnings models/experimental/retinanet/tests/pcc
```
- Every test compares against the torchvision model, and no test reads an image file. The
  backbone and E2E tests run on a seeded random 512x512 image (`tests/pcc/synthetic_input.py`):
  - Backbone: `test_resnet50_stem.py`, `test_resnet50_bottleneck.py` (every bottleneck of every
    stage), `test_resnet50_fpn.py` and `test_resnet50_backbone.py` for the five pyramid levels.
  - Heads: `test_classificationhead.py` and `test_regressionhead.py`, on random features of the
    five pyramid levels' shapes.
  - Postprocessing: `test_postprocess.py` checks the device's candidates against torchvision's
    selection and box decoding, and the host's detections against torchvision's
    `postprocess_detections`, on random head outputs.
- `test_retinanet.py` runs the whole model and checks the five pyramid levels and both head
  outputs, and the device's candidates on those head outputs against torchvision's selection and
  box decoding of them. Run it independently with:

```sh
pytest --disable-warnings models/experimental/retinanet/tests/pcc/test_retinanet.py
```

### Reduced grid

The model derives its grids from the device. On fewer than 64 worker cores it switches to a plan
that keeps activations in DRAM and slices the convolutions that do not fit L1. The whole PCC
suite passes on two worker cores:

```sh
TT_METAL_CORE_GRID_OVERRIDE_TODEPRECATE=1,0 pytest models/experimental/retinanet/tests/pcc
```

The override is the inclusive end coordinate of the worker grid, so `1,0` is a 2x1 grid.

### Reduced layer count

`RETINANET_NUM_LAYERS=N` keeps `N` repeats of every stack of identical layers: the repeated
bottlenecks of each ResNet stage and the conv blocks of both heads. Each stage also keeps its
first block, which projects the shortcut and, from layer2 on, strides. The reference is reduced
the same way, so the PCC thresholds still apply, but the reduced model is not the trained
detector. The E2E, backbone and head tests support it:

```sh
RETINANET_NUM_LAYERS=1 pytest models/experimental/retinanet/tests/pcc/test_retinanet.py
```

Every kernel of the full model still runs with `N=1`.

## Project Structure

```
models/experimental/retinanet/
├── num_layers.py          # RETINANET_NUM_LAYERS: reduces the reference model
├── demo/demo.py           # Image CLI: runs the model and draws its detections
├── resources/             # Demo image and outputs
├── tests/
│   ├── pcc/               # Module and E2E PCC tests against torchvision
│   └── perf/test_perf.py  # Device kernel throughput
└── tt/                    # TTNN implementation
```

## Model Structure

`RetinaNet` from `torchvision.models.detection.retinanet_resnet50_fpn_v2`, the repeated blocks
folded.

```
RetinaNet(
  (backbone): BackboneWithFPN(
    (body): IntermediateLayerGetter(
      (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
      (bn1): BatchNorm2d(64)
      (relu): ReLU(inplace=True)
      (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1)
      (layer1): Sequential(
        (0-2): 3 x Bottleneck(
          (conv1): Conv2d(256, 64, kernel_size=(1, 1), bias=False) + BatchNorm2d  (64 in, block 0)
          (conv2): Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1), bias=False) + BatchNorm2d
          (conv3): Conv2d(64, 256, kernel_size=(1, 1), bias=False) + BatchNorm2d
          (relu): ReLU(inplace=True)
          (downsample): Conv2d(64, 256, kernel_size=(1, 1), bias=False) + BatchNorm2d  (block 0)
        )
      )
      (layer2): Sequential(
        (0-3): 4 x Bottleneck(
          1x1 256/512 -> 128, 3x3 128 -> 128 (stride 2 in block 0), 1x1 128 -> 512
          (downsample): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2)) + BatchNorm2d  (block 0)
        )
      )
      (layer3): Sequential(
        (0-5): 6 x Bottleneck(
          1x1 512/1024 -> 256, 3x3 256 -> 256 (stride 2 in block 0), 1x1 256 -> 1024
          (downsample): Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2)) + BatchNorm2d  (block 0)
        )
      )
      (layer4): Sequential(
        (0-2): 3 x Bottleneck(
          1x1 1024/2048 -> 512, 3x3 512 -> 512 (stride 2 in block 0), 1x1 512 -> 2048
          (downsample): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2)) + BatchNorm2d  (block 0)
        )
      )
    )
    (fpn): FeaturePyramidNetwork(
      (inner_blocks): ModuleList(
        (0): Conv2d(512, 256, kernel_size=(1, 1))
        (1): Conv2d(1024, 256, kernel_size=(1, 1))
        (2): Conv2d(2048, 256, kernel_size=(1, 1))
      )
      (layer_blocks): ModuleList(
        (0-2): 3 x Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
      )
      (extra_blocks): LastLevelP6P7(
        (p6): Conv2d(2048, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        (p7): Conv2d(256, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
      )
    )
  )
  (anchor_generator): AnchorGenerator()
  (head): RetinaNetHead(
    (classification_head): RetinaNetClassificationHead(
      (conv): Sequential(
        (0-3): 4 x Conv2dNormActivation(
          (0): Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1), bias=False)
          (1): GroupNorm(32, 256, eps=1e-05, affine=True)
          (2): ReLU(inplace=True)
        )
      )
      (cls_logits): Conv2d(256, 819, kernel_size=(3, 3), padding=(1, 1))
    )
    (regression_head): RetinaNetRegressionHead(
      (conv): Sequential(
        (0-3): 4 x Conv2dNormActivation(...)  (as classification_head)
      )
      (bbox_reg): Conv2d(256, 36, kernel_size=(3, 3), padding=(1, 1))
    )
  )
  (transform): GeneralizedRCNNTransform(
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      Resize(min_size=(800,), max_size=1333, mode='bilinear')
  )
)
```

Indented under each file are the modules from the tree it implements, with the class or function
in parentheses.

```
tt/tt_backbone.py
    backbone (TTBackbone)

tt/tt_stem.py
    backbone.body.conv1, bn1, relu, maxpool (resnet50Stem)

tt/tt_bottleneck.py
    backbone.body.layer1-4 Bottleneck, every conv with its BatchNorm folded in (TTBottleneck)

tt/tt_fpn.py
    backbone.fpn, with its nearest upsampling and P6/P7 (resnet50Fpn)

tt/tt_classification_head.py
    head.classification_head (ttnn_retinanet_classification_head)

tt/tt_regression_head.py
    head.regression_head (ttnn_retinanet_regression_head)

tt/utils.py
    Conv2d (TTConv2D), upsampling (TTUpsample)
    head Conv2dNormActivation, with GroupNorm on device (Conv2dNormActivation)

tt/tt_postprocess.py
    anchor_generator, computed once at init (TTDetectionDecoder)
    per-level top-k candidates and box decoding, on device (TTDetectionDecoder)
    score threshold and NMS, on the host (select_detections)

tt/custom_preprocessor.py
    every module's parameters, taken from the torchvision model

demo/demo.py, on the host
    transform's normalization; a fixed 512x512 resize in place of its aspect-preserving one
```

## Details
- `TTBackbone` in `tt/tt_backbone.py` runs the backbone and FPN; the heads are the functions in
  `tt/tt_classification_head.py` and `tt/tt_regression_head.py`, and `TTDetectionDecoder` with
  `select_detections` in `tt/tt_postprocess.py` turn their outputs into detections. `demo/demo.py`
  shows how they compose.
- Batch size: 1
- Input: 3x512x512, bfloat16.
- Pyramid levels P3-P7: strides 8 to 128 (64x64 down to 4x4 at 512x512); 9 anchors per location;
  91 COCO category slots in `cls_logits` (819 = 9 x 91).
