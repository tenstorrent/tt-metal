# RT-DETR on Tenstorrent Wormhole

## Platforms
- Wormhole (N300)

## Introduction

This repository implements **RT-DETR** (Real-Time DEtection TRansformer) using TTNN APIs for high-performance object detection on Tenstorrent Wormhole hardware.

RT-DETR is an end-to-end object detector that overcomes the slow inference speed of standard DETR models via an efficient hybrid architecture:

- **CNN Backbone** — PResNet-50 for multi-scale feature extraction
- **Hybrid Encoder** — AIFI transformer on the coarsest scale (20×20) + CCFM neck (FPN top-down + PAN bottom-up) with CSPRepLayer blocks
- **Transformer Decoder** — 6-layer decoder with deformable cross-attention for bounding box and class prediction

Reference: [DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069) (Zhao et al., 2023)
Official repo: [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR)

---

## Model Architecture

### Backbone (PResNet-50)
| Parameter | Value |
|---|---|
| Stem | 3× Conv3×3 + MaxPool |
| Bottleneck layout | [3, 4, 6, 3] |
| Extracted features | S3 (80×80), S4 (40×40), S5 (20×20) |
| BN folding | Folded into conv weights at load time |

### Encoder (HybridEncoder)
| Parameter | Value |
|---|---|
| Hidden dimension | 256 |
| AIFI attention heads | 8 |
| AIFI sequence length | 400 tokens (20×20) |
| FFN dimension | 1024 |
| Neck | FPN top-down + PAN bottom-up, CSPRepLayer blocks |

### Decoder
| Parameter | Value |
|---|---|
| Layers | 6 |
| Queries | 300 |
| Self-attention | On-device (TTNN) |
| Cross-attention | CPU fallback (deformable attention) |
| FFN | On-device (TTNN) |

---

## Directory Structure

```
rt_detr/
├── demo/
│   ├── demo.py                  # Demo inference script
│   ├── demo_images/             # Input images
│   └── demo_output/             # Annotated output images
├── tt/
│   ├── resnet_blocks.py         # conv_block, residual_block
│   ├── resnet_backbone.py       # PResNet-50 forward pass
│   ├── rtdetr_encoder.py        # AIFI encoder layer
│   ├── hybrid_encoder.py        # HybridEncoder (AIFI + CCFM)
│   ├── rtdetr_decoder.py        # Transformer decoder
│   ├── weight_utils.py          # Weight extraction and BN folding
│   └── attention.py             # Attention utilities
├── tests/
│   ├── test_end_to_end_pcc.py   # End-to-end PCC validation
│   └── evaluate_coco.py         # COCO mAP evaluation
├── weights/
│   └── rtdetr_r50vd.pth         # Downloaded by setup.sh
├── data/
│   └── coco/                    # COCO val2017 (optional, for evaluation)
├── RT-DETR/                     # Lyuwenyu reference repo (cloned by setup.sh)
│   └── rtdetr_pytorch/
├── setup.sh
└── requirements.txt
```

---

## Setup

### 1. Prerequisites

- Tenstorrent Wormhole N300 device (1×2 mesh)
- TT-Metalium with TTNN — follow [INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md) and make sure your environment variables are sourced
- Python 3.10+

### 2. Run setup.sh

`setup.sh` handles everything in one shot: installs Python dependencies, clones the Lyuwenyu reference repo, and downloads the model checkpoint.

**Always run it with `source`, not `bash`:**

```bash
source setup.sh
```

> **Why `source`?**
> `bash setup.sh` spawns a child process — any `export` inside it is thrown away when the script exits, so `PYTHONPATH` never makes it back to your shell. `source` (equivalently `. setup.sh`) runs the script inside your current shell so the export sticks for the rest of the session.

### 3. Make PYTHONPATH permanent

To avoid re-sourcing on every new terminal, add the export to your shell profile once:

```bash
echo "export PYTHONPATH=\"$(pwd)/RT-DETR/rtdetr_pytorch:\$PYTHONPATH\"" >> ~/.bashrc
source ~/.bashrc
```

### 4. Verify the setup

```bash
python -c "from src.core import YAMLConfig; print('OK')"
```

---

## Running the Demo

Place images in `demo/demo_images/` (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp` are all supported), then:

```bash
python demo/demo.py
```

Annotated images are written to `demo/demo_output/` with bounding boxes, class labels, and confidence scores drawn on them. A `detections.json` summary is also saved there.

Example terminal output:

```
Found 3 image(s)

── street.jpg
   person                0.921  [120,80,310,450]
   car                   0.876  [400,200,650,380]
   traffic light         0.743  [290,60,330,140]
   → demo/demo_output/street_detected.jpg

── sample.jpg
   dog                   0.954  [80,120,420,390]
   couch                 0.811  [30,200,600,480]
   → demo/demo_output/sample_detected.jpg
```

---

## Running Tests

### End-to-End PCC Validation

Runs the full TT pipeline against the PyTorch reference and checks per-channel correlation (PCC) at every stage — backbone outputs, encoder FPN/PAN intermediates, decoder layer outputs, and final logits/boxes.

```bash
pytest tests/test_end_to_end_pcc.py -v
```

In order to run the unit tests, for example is we want to run the unit test for hybrid_encoder then
```bash
pytest tests/unit/test_hybrid_encoder.py -v
```

Expected results:

| Test | Threshold | Result |
|---|---|---|
| `test_pred_logits_pcc` | PCC ≥ 0.90 | pass |
| `test_pred_boxes_pcc` | PCC ≥ 0.90 | pass |
| `test_top5_labels_and_scores` | score diff < 0.05 | pass |
| `test_box_iou_fired_detections` | IoU > 0.90 | pass |

The test loads `demo/demo_images/sample.jpg` as the input image. If that file is absent it falls back to a zero tensor.

### COCO mAP Evaluation

Evaluates the full TTNN pipeline over the COCO 2017 validation set (5000 images). Supports checkpoint/resume — if interrupted it picks up where it left off.

Download the data first:

```bash
cd data/coco
wget http://images.cocodataset.org/zips/val2017.zip && unzip val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip && unzip annotations_trainval2017.zip
cd ../..
```

Then run:

```bash
python tests/evaluate_coco.py
```

### Device Performance

To replicate the reported device performance numbers using the Tracy profiler:

```sh
# Manually inspect on-device ops (generates a CSV perf report)
./tools/tracy/profile_this.py -n rt_detr -c "pytest --disable-warnings models/demos/wormhole/rt_detr/tests/test_end_to_end_pcc.py -v"
```

> **Note:** The deformable cross-attention step runs on CPU and is excluded from the device trace. See [issue #17076](https://github.com/tenstorrent/tt-metal/issues/17076).

**End-to-end accuracy on COCO val2017:**

| Metric | Value |
|---|---|
| mAP (AP @ IoU=0.50:0.95) | **49.8** |
| AP @ IoU=0.50 | 67.9 |
| AP @ IoU=0.75 | 53.6 |
| AR @ 100 dets | 66.0 |
| PCC vs PyTorch (logits) | 0.92 |
| PCC vs PyTorch (boxes) | 0.97 |

---

## Performance

Performance was measured on the Wormhole B0 N300 device using a batch size of 1.

| Metric | Latency | Throughput |
| :--- | :--- | :--- |
| **Device Execution (TTNN Trace)** | 61.11 ms | 16.36 FPS |
| **End-to-End Wall-Clock Time** | ~300.0 ms | 3.3 FPS |

**Note on Hybrid Execution & Fallbacks:**
The End-to-End wall-clock time includes PyTorch CPU fallbacks. While the ResNet Backbone, Hybrid Encoder, Self-Attention, FFNs, and LayerNorms execute natively on the Wormhole chip in `HiFi4` and `HiFi2` precision, the **Deformable Cross-Attention** remains on the Host CPU. Forcing Deformable Attention onto the device using current ops caused massive TM overhead due to uncoalesced memory scatter/gather operations, so the host fallback was maintained to preserve optimal end-to-end pipeline speed.

### Summary

| Metric | Value | Notes |
|:---|:---|:---|
| Hardware | Wormhole N300 (B0) | Tenstorrent accelerator |
| Batch size | 1 | Single image inference |
| Weight precision | BF16 | HiFi4 math fidelity |
| Device latency | **61.11 ms** | On-device FW duration |
| Host latency | 139.95 ms | Includes dispatch + cold cache |
| Device throughput | **~16 FPS** | 1000 / 61.11 ms |
| Ops traced | 1,860 | On-device ops only — excludes CPU cross-attention fallback |
| Core utilisation | 56% at max (64 cores) | 1,042 / 1,860 ops at full 64 cores |
| Top bottleneck | Matmul — 27.5% | Of total device kernel time |

### Latency

| Metric | Value |
|---|---|
| End-to-end host time | 139.95 ms |
| Device compute time (FW) | 61.11 ms |
| Total device kernel time | 59.05 ms |
| FW overhead | ~2.06 ms (~3.4%) |

> **Note:** The host time of 139.95 ms was measured on a first-run (cold cache) trace and includes kernel compilation overhead. Device kernel times are representative of steady-state execution as they measure raw Tensix cycles independent of cache state. A cache-warm trace will yield a lower host-side baseline.

### Operations on Device

The backbone and encoder execute fully on-device. In the decoder, self-attention and FFN run on-device, but the deformable cross-attention step in each of the 6 decoder layers falls back to PyTorch CPU — the query tensor is transferred host↔device once per layer. This cross-attention fallback executes outside the TTNN dispatch graph and is therefore not captured in the profiler trace above; the reported host time of 139.95 ms reflects the on-device portion only and does not include the CPU cross-attention cost.

### Device Time by Operation Class

| Operation | Calls | Device Kernel Time | Share |
|---|---|---|---|
| MatmulDeviceOperation | 206 | 16.23 ms | 27.5% |
| TilizeWithValPaddingDeviceOperation | 194 | 11.97 ms | 20.3% |
| Conv2dDeviceOperation | 66 | 6.53 ms | 11.1% |
| BinaryNgDeviceOperation | 278 | 6.31 ms | 10.7% |
| CopyDeviceOperation | 154 | 3.45 ms | 5.8% |
| UnaryDeviceOperation | 122 | 2.73 ms | 4.6% |
| SDPAOperation | 14 | 0.73 ms | 1.2% |
| LayerNormDeviceOperation | 40 | 0.60 ms | 1.0% |
| Other | 166 | 10.50 ms | 17.8% |

---

## Implementation Notes

**BatchNorm folding** — All conv+BN pairs are folded at weight-load time in `weight_utils.py`, removing BN from the inference graph with no accuracy loss.

**Weight layout** — Conv weights stay on host in the format `ttnn.conv2d` expects. Linear weights are transposed and uploaded to DRAM in TILE_LAYOUT at load time.

**Mesh device** — Targets a 1×2 Wormhole mesh. Weights are replicated via `ReplicateTensorToMesh`; outputs are pulled back with `ConcatMeshToTensor`.

**Decoder cross-attention** — Deformable multi-scale attention is not yet natively implemented in TTNN for Wormhole (tracked in [#17076](https://github.com/tenstorrent/tt-metal/issues/17076)). The cross-attention step in each of the 6 decoder layers falls back to PyTorch CPU, with one host↔device query transfer per layer. This is the primary remaining opportunity to move computation fully on-device.

**Memory management** — All intermediate TTNN tensors are explicitly deallocated with `ttnn.deallocate` to prevent L1/DRAM fragmentation across the backbone→encoder→decoder pipeline.

---

## References

- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [RT-DETR Official Implementation](https://github.com/lyuwenyu/RT-DETR)
- [TT-Metalium INSTALLING.md](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
