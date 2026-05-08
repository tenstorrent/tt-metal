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
│   ├── evaluate_iou_dice.py
│   ├── perf_benchmark_core.py
│   ├── generate_perf_sheet.py      # Stage 2 perf CSV by default
│   └── benchmark_stage2_inference.py
├── reference/model.py
├── tt/common.py
├── tt/config.py
├── tt/model.py
├── demo/demo.py
└── tests/test_attention_denseunet.py
```

## Setup

```bash
source python_env/bin/activate
pip install pillow requests loguru pycocotools   # pycocotools only for COCO prep
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

### Benchmarks (Stage 2)

**Headline latency** (device-synchronized, mean of N inferences after warmup):

```bash
python models/demos/attention_denseunet/eval/benchmark_stage2_inference.py \
  --optimization-level stage2 --height 256 --width 256 --warmup 3 --iterations 10
```

**Perf sheet CSV** (same column names as `models/perf/perf_utils.prep_perf_report`; **Stage 2 only** by default):

```bash
python models/demos/attention_denseunet/eval/generate_perf_sheet.py \
  --output-dir models/demos/attention_denseunet/eval/results
```

Produces e.g. `eval/results/perf_attention_denseunet_stage2_YYYY_MM_DD.csv`.
`Expected *` columns are seeded to measured values; tighten them when you freeze a baseline.

**Kernel-level profiling:** use the TTNN visualizer / Tracy flow described in [TTNN model bring-up](https://github.com/tenstorrent/tt-metal/blob/main/tech_reports/ttnn/TTNN-model-bringup.md) around the same `demo/demo.py` or eval inference path.

## Stage 3 optimizations (implemented)

Stage 3 is now implemented with additional parallelism and attention-gate tuning on top of Stage 2:

| Focus area | Stage 3 implementation |
|------------|------------------------|
| **Encoder/decoder parallelism** | `_select_conv_sharding_strategy` now biases **decoder path convs** (`decoder*`, `upconv*`, `attention*`, `conv_out`) to **block sharding** earlier (from 32×32 maps, non-pointwise) using smaller `act_block_h_override` to increase core-level parallel work. |
| **Attention gate conv parallelism** | `_create_attention_gate_config` now uses a dedicated stage-aware hook. For Stage 3 on current WH constraints, gate conv sharding is kept conservative (`AutoSharded`) to avoid L1 circular-buffer clashes while decoder path stays aggressively parallelized. |
| **TM / memory overhead in gates** | Attention gate upsample mode is stage-aware: Stage 3 uses `nearest` (config-driven) in `TtAttentionGate` to reduce resize overhead vs bilinear. Intermediate gate tensors (`theta_x`, `phi_g`, `f`, `attention`, `y`) are explicitly deallocated after use to reduce peak activation footprint. |
| **Upsample path tuning** | `UpconvConfiguration` now carries Stage 3-specific transposed-conv tuning (`act_block_h_override`, force height-sharded layout hint) applied in `transpose_conv2d`. |

### Stage 3 benchmark / perf sheet

Generate a Stage 3-only perf sheet:

```bash
python models/demos/attention_denseunet/eval/generate_perf_sheet.py \
  --stages stage3 \
  --height 256 --width 256 \
  --warmup 3 --iterations 10 \
  --output-dir models/demos/attention_denseunet/eval/results
```

This writes:

- `eval/results/perf_attention_denseunet_stage3_YYYY_MM_DD.csv`

Optionally generate side-by-side Stage2/Stage3:

```bash
python models/demos/attention_denseunet/eval/generate_perf_sheet.py \
  --stages stage2 stage3 \
  --height 256 --width 256 \
  --warmup 3 --iterations 10 \
  --output-dir models/demos/attention_denseunet/eval/results
```

### Advanced tuning notes / known issues

- Stage 3 gate upsample uses `nearest` to reduce TM/memory overhead; this may slightly shift logits vs Stage 2 bilinear on edge-heavy masks.
- A frequent runtime warning may appear (`ttnn::tilize ... legacy sharded optimized program factory`) on current stack; this is known and does not invalidate perf-sheet generation.
- Performance can vary by board firmware/JIT cache state; keep warmup/iteration counts fixed when comparing Stage 2 vs Stage 3.

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
