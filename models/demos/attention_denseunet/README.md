# Attention DenseUNet (TTNN)

## Platforms

- Wormhole (N150 / N300 class devices)

## Introduction

This demo implements **Attention DenseUNet** using TTNN APIs for segmentation workloads on Tenstorrent hardware.

Tracking issue / bounty: [#30863](https://github.com/tenstorrent/tt-metal/issues/30863)

## Repo layout

```text
models/demos/attention_denseunet/
├── README.md
├── eval/
│   ├── prepare_coco_val_subset.py
│   └── evaluate_iou_dice.py
├── reference/model.py
├── tt/common.py
├── tt/config.py
├── tt/model.py
├── demo/demo.py
└── tests/test_attention_denseunet.py
```

## Setup

---

## Performance results

Profiling results on **N300 (Wormhole B0, 64 cores)**, input **256×256, batch 1**, `OptimizationLevel.STAGE2` (L1 height-sharded).
Numbers from Tracy profiler CSV (`generated/profiler/reports/attention_denseunet/2026_04_27_12_27_38/ops_perf_results_*.csv`).

| Metric | DRAM Interleaved | L1 Sharded | Vision SDPA
|--------|-----------------:|-----------:|--------------
| **Device time** | 20.40 ms | **17.57 ms** | 17.57 ms
| **Matmul time** | 2.50 ms | **2.15 ms** | 2.15 ms
| **Attn time** | 0.71 ms | **0.66 ms** | 0.66 ms


### Regenerate profiler CSV

```bash
source python_env/bin/activate
python -m tracy -p -r -v -m pytest \
  models/demos/attention_denseunet/tests/test_attention_denseunet.py -k "test_ttnn" -s
# CSV lands in: generated/profiler/reports/<timestamp>/ops_perf_results_*.csv
```

---

## COCO IoU / Dice (reproducible)

### Download COCO 2017 val + annotations

```bash
export COCO_ROOT=/path/to/coco2017
mkdir -p "$COCO_ROOT" && cd "$COCO_ROOT"
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip -q val2017.zip && rm -f val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip -q annotations_trainval2017.zip && rm -f annotations_trainval2017.zip
```

Layout: `$COCO_ROOT/val2017/*.jpg` and `$COCO_ROOT/annotations/instances_val2017.json`.

### Build subset (default: `person`, 50 images, 256×256)

```bash
python models/demos/attention_denseunet/eval/prepare_coco_val_subset.py \
  --coco-root "$COCO_ROOT" \
  --out-dir models/demos/attention_denseunet/eval/datasets/coco_val2017_person_subset \
  --max-images 50 --height 256 --width 256
```

### Evaluate

```bash
python models/demos/attention_denseunet/eval/evaluate_iou_dice.py \
  --image-dir models/demos/attention_denseunet/eval/datasets/coco_val2017_person_subset/images \
  --mask-dir models/demos/attention_denseunet/eval/datasets/coco_val2017_person_subset/masks \
  --height 256 --width 256 \
  --optimization-level stage2 \
  --output-json models/demos/attention_denseunet/eval/results/coco_metrics.json
```

Optional: `--checkpoint /path/to.pt`.

---

## Stage 2 optimizations (demonstrated in code)

**Preset:** `stage2` (`OptimizationLevel.STAGE2`) — default for tests, demo, and eval above.

| What you need to show | What the code does |
|------------------------|-------------------|
| **Conv sharding** | For spatial size ≥ 32×32, convs use **height sharding** (`HeightShardedStrategyConfiguration`, `reshard_if_not_optimal=True`, `act_block_h_override=32`). Smaller tensors stay **auto** sharded. See `_select_conv_sharding_strategy` when `optimization_level == STAGE2`. |
| **Upsample (transposed conv) + sharding** | `transpose_conv2d` sets `Conv2dConfig.shard_layout=HEIGHT_SHARDED` **when the decoder input is already sharded**, so upsampling follows the sharded activation path instead of forcing DRAM interleaved. See `tt/model.py` (`transpose_conv2d`). |
| **Interleaving vs sharding** | Skip / dense paths use **`concatenate_features(..., use_row_major_layout=True)`**: both branches are moved to **DRAM**, **ROW_MAJOR**, concatenated, then retiled — explicit **interleaved** concat to avoid mixed sharded/interleaved concat errors. The `use_row_major_layout=False` branch can keep **height-sharded L1** concat when both inputs share sharding. See `tt/model.py` (`concatenate_features`). |
| **L1 beyond default** | (1) **Input tensor** is placed in **`L1_MEMORY_CONFIG`** via `configs.l1_input_memory_config`. (2) Device opens with **`ATTENTION_DENSEUNET_L1_SMALL_SIZE`** (larger L1 carveout for dense blocks). See `tt/config.py` (`build_configs`), `tt/common.py`, `demo/demo.py` / `evaluate_iou_dice.py`. |
| **Fused Conv + ReLU** | After BN fold, many layers set `activation=UnaryWithParam(RELU)` on **`Conv2dConfiguration`**, so ReLU runs in the TT CNN op where supported. See `_create_conv_config_from_params` (`activation` block) and `TtTransitionDown` (conv then pool). |


## Stage 3 optimizations (implemented)

Stage 3 adds stronger sharding choices and attention-gate tuning on top of Stage 2 (see `tt/config.py`, `tt/model.py`).

| Focus area | Stage 3 implementation |
|------------|------------------------|
| **Encoder/decoder parallelism** | `_select_conv_sharding_strategy` biases **decoder path** convs toward **block sharding** on larger maps where applicable. |
| **Attention gates** | Stage-aware gate config; upsample mode can use `nearest` to reduce TM/memory vs bilinear; intermediate tensors deallocated where applicable. |
| **Upsample path** | `UpconvConfiguration` / `transpose_conv2d` carry Stage 3 transposed-conv hints. |

### Known issues

- Some boards may hit **L1 circular-buffer clashes** or **OOM** on certain presets; fall back to `stage2` for benchmarking until resolved.
- `ttnn::tilize ... legacy sharded optimized program factory` warnings may appear on current stack.

---

## Run demo & tests

```bash
python models/demos/attention_denseunet/demo/demo.py --output-name out.png --optimization-level stage2
pytest -v models/demos/attention_denseunet/tests/test_attention_denseunet.py
pytest -v models/demos/attention_denseunet/demo/demo.py
```

---

## References

- [TTNN model bring-up](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md)
- [CNN optimization guide](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/CNNs/cnn_optimizations.md)
- [DenseNet](https://arxiv.org/abs/1608.06993) · [Attention U-Net](https://arxiv.org/abs/1804.03999)
