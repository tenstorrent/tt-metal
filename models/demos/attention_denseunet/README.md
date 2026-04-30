# Attention DenseUNet (TTNN)

## Platforms

- Wormhole (N150 / N300 class devices)

## Introduction

This demo implements **Attention DenseUNet** using TTNN APIs for segmentation workloads on Tenstorrent hardware.

The model combines:

- DenseNet-style encoder blocks
- Attention-gated skip connections
- U-Net decoder with transposed convolutions

Tracking issue / bounty: [#30863](https://github.com/tenstorrent/tt-metal/issues/30863)

## Repo Layout

```text
models/demos/attention_denseunet/
├── README.md
├── eval/evaluate_iou_dice.py       # IoU/Dice evaluator
├── reference/model.py              # PyTorch reference model
├── tt/common.py                    # constants + parameter preprocessing
├── tt/config.py                    # TTNN configs + optimization presets
├── tt/model.py                     # TTNN model implementation
├── demo/demo.py                    # direct CLI demo + pytest demo entry
└── tests/test_attention_denseunet.py
```

## Setup

```bash
source python_env/bin/activate
```

If needed:

```bash
pip install pillow requests
```

## Run the Model

### 1) PyTorch reference sanity

```bash
python models/demos/attention_denseunet/reference/model.py
```

### 2) Direct demo script

PyTorch path (no device):

```bash
python models/demos/attention_denseunet/demo/demo.py --pytorch --output-name demo_pytorch.png
```

TTNN path (device):

```bash
python models/demos/attention_denseunet/demo/demo.py --output-name demo_ttnn.png --optimization-level stage2
```

By default, the demo uses:
`http://images.cocodataset.org/val2017/000000039769.jpg`

You can still override with any URL or local path:

```bash
python models/demos/attention_denseunet/demo/demo.py --image /path/to/image.jpg
```

Outputs are written to:

- `models/demos/attention_denseunet/demo/pred/*_prob.png`
- `models/demos/attention_denseunet/demo/pred/*_mask_0p5.png`
- `models/demos/attention_denseunet/demo/pred/*_mask_adaptive.png`
- `models/demos/attention_denseunet/demo/pred/*_overlay.png`

### 3) Tests

```bash
# Main correctness tests (init + inference PCC)
pytest -v models/demos/attention_denseunet/tests/test_attention_denseunet.py

# Demo tests (TTNN and PyTorch parametrized)
pytest -v models/demos/attention_denseunet/demo/demo.py
```

### 4) IoU/Dice evaluation (dataset-backed)

Quick way to prepare a small real dataset (Oxford-IIIT Pet subset):

```bash
python models/demos/attention_denseunet/eval/prepare_oxford_pets_subset.py \
  --max-images 20 \
  --height 256 --width 256
```

This creates:

```text
models/demos/attention_denseunet/eval/datasets/oxford_pets_small/
├── images/
└── masks/
```

Dataset used for the README evaluation flow:

- Oxford-IIIT Pet segmentation dataset (small subset created by `prepare_oxford_pets_subset.py`)
- Foreground masks are derived from trimaps (`class 1` as foreground, others as background)

Expected dataset layout:

```text
<dataset_root>/
├── images/
│   ├── sample1.png
│   └── sample2.png
└── masks/
    ├── sample1.png
    └── sample2.png
```

Run evaluator:

```bash
python models/demos/attention_denseunet/eval/evaluate_iou_dice.py \
  --image-dir models/demos/attention_denseunet/eval/datasets/oxford_pets_small/images \
  --mask-dir models/demos/attention_denseunet/eval/datasets/oxford_pets_small/masks \
  --height 256 --width 256 \
  --optimization-level stage2 \
  --output-json models/demos/attention_denseunet/eval/results/oxford_pets_small_metrics.json
```
Note:- oxford_pets_small dataset was installed by me through a script i am not adding that here because you can use any dataset
You can still use your own dataset by replacing `--image-dir` and `--mask-dir` with
`<dataset_root>/images` and `<dataset_root>/masks`.

What this reports:

- TTNN vs PyTorch reference IoU/Dice (always)
- TTNN vs GT and PyTorch vs GT IoU/Dice (when `--mask-dir` is provided)

If you have trained weights compatible with `reference/model.py`, add:

```bash
--checkpoint /path/to/checkpoint.pt
```

## Current Validation Coverage

- Model initialization on device
- End-to-end TTNN inference
- Shape parity vs PyTorch
- PCC check vs PyTorch reference (`>= 0.97`)

## Optimization Presets in Code

Defined in `tt/config.py`:

- `stage1`: baseline auto-sharding
- `stage2`: preferred profile currently used by tests
- `stage3`: aggressive profile for deeper tuning

Current tests run with `OptimizationLevel.STAGE2`.

## What Is Implemented vs Pending

### Implemented (code + tests)

- TTNN model bring-up and demo
- PCC-based correctness test against PyTorch
- Direct single-image demo script path
- Profiling/visualizer compatible run path


## References

- [TTNN model bring-up tech report](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md)
- [CNN optimization guide](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/cnn_optimizations.md)
- [DenseNet paper](https://arxiv.org/abs/1608.06993)
- [Attention U-Net paper](https://arxiv.org/abs/1804.03999)
