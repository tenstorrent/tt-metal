# Facebook / SAM 2 (Hiera-tiny) — Tenstorrent `ttnn` Native Port ($1,500 Bounty #48311)

This directory contains the Tenstorrent native `ttnn` implementation of **Segment Anything Model 2 (SAM 2) (`facebook/sam2-hiera-tiny`)** for promptable visual segmentation in still images (`1024x1024` input mode).

> **Note on Scope (`Issue #48311`):** Tenstorrent explicitly excluded video object tracking (`memory bank`, `memory encoder`, `temporal attention blocks`) to focus strictly on **still-image segmentation (`Image Mode`)**.

## 📁 Repository Structure

```text
models/demos/sam2/
├── reference/
│   ├── __init__.py
│   └── sam2_reference.py         # PyTorch baseline wrapping facebook/sam2-hiera-tiny
├── tt/
│   ├── __init__.py
│   └── hiera_image_encoder.py    # Multi-scale window Vit in native ttnn ops
├── tests/
│   ├── __init__.py
│   ├── test_sam2_reference.py    # PyTorch baseline Stage 1 multi-scale verification suite
│   └── test_sam2_accuracy.py     # Stage 1: Pearson Correlation (assert pcc >= 0.999 mismatch test)
├── demo/
│   └── demo_segmentation.py      # End-to-end sample image execution ($ python demo/...)
├── conftest.py                   # Pytest shared configuration and device hooks
├── requirements.txt              # PyTorch + Torchvision + Pytest dependencies
├── PERF.md                       # Stage 2/3: L1 Sharding & Core Utilization benchmark metrics
└── README.md                     # This documentation
```

## 🚀 Stage 1 Verification Suite (PyTorch Baseline Equivalence)

We verify our hierarchical downsampling feature maps (`4x`, `8x`, `16x`, `32x` resolutions matching channel depths `96, 192, 384, 768`) and two-way mask decoder against deterministic PyTorch baselines before and during `ttnn` operator conversion.

### Running the Stage 1 Self-Check Harness:

```bash
# Standalone execution check
python3 models/demos/sam2/tests/test_sam2_reference.py

# Or via pytest
pytest models/demos/sam2/tests/test_sam2_reference.py -v
```

### Expected Stage 1 Verification Output:
```text
Execution Stage 1 Verification Self-Check...
✅ All hierarchical features and mask decoding checks passed cleanly (4x, 8x, 16x, 32x).
```

## 🏗️ Architectural & Sharding Specifications (Wormhole N150 / N300 / Blackhole)

- **Tensor Layout:** All multi-scale feature buffers enforce `ttnn.TILE_LAYOUT` (`32x32` blocks).
- **L1 Memory Config:** Intermediate windowed multi-head cross-attention blocks reside entirely in `ttnn.L1_MEMORY_CONFIG` to prevent DRAM thrashing.
- **Sharding Strategy:** `HEIGHT_SHARDED` layout across grid cores (`x=8, y=8` bounding box on N150) for uniform compute footprint (`TFLOPS` maximization). See `PERF.md` for exact profiling curves.
