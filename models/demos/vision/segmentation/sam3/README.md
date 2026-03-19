# SAM3

## Platforms:
    Blackhole (p150)

## Introduction
SAM3 (Segment Anything with Concepts) is a vision-language image segmentation model that combines a ViT image backbone, an FPN neck, a text encoder, and a transformer encoder+decoder to perform open-vocabulary instance and semantic segmentation. It takes an image (and optionally text prompts) as input and outputs pixel-wise segmentation masks identifying arbitrary concepts in the scene.

## Prerequisites
- Cloned [tt-metal repository](https://github.com/tenstorrent/tt-metal) for source code
- Installed: [TT-Metalium™ / TT-NN™](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md)
- `sam3` Python package installed in your venv (or pointed to via `SAM3_VENV_PATH`)
- BPE vocab file (`bpe_simple_vocab_16e6.txt.gz`) from `open_clip`, available in the tt-metal Python env or via `SAM3_BPE_PATH`

## How to Run
Find the model instructions for each device below:

### Blackhole P150
[models/demos/vision/segmentation/sam3/blackhole](https://github.com/tenstorrent/tt-metal/tree/main/models/demos/vision/segmentation/sam3/blackhole)

#### Environment Variables
| Variable | Default | Description |
|---|---|---|
| `SAM3_VENV_PATH` | `~/.tenstorrent-venv/lib/python3.12/site-packages` | Path to site-packages containing the `sam3` package |
| `SAM3_BPE_PATH` | `$TT_METAL_HOME/python_env/lib/python3.12/site-packages/open_clip/bpe_simple_vocab_16e6.txt.gz` | Path to the BPE vocabulary file |

#### Run PCC tests
```bash
pytest models/demos/vision/segmentation/sam3/blackhole/tests/pcc/
```

#### Run performance tests
```bash
pytest models/demos/vision/segmentation/sam3/blackhole/tests/perf/test_perf_sam3.py
```

#### Run with custom paths
```bash
export SAM3_VENV_PATH=/path/to/venv/lib/python3.12/site-packages
export SAM3_BPE_PATH=/path/to/bpe_simple_vocab_16e6.txt.gz
pytest models/demos/vision/segmentation/sam3/blackhole/tests/
```

## Details

### Architecture
SAM3 is composed of four main components:

| Component | Class | Description |
|---|---|---|
| ViT Backbone | `ViT` | Vision Transformer image encoder (ViT-H/14 style with window attention) |
| FPN Neck | `Sam3DualViTDetNeck` | Feature Pyramid Network neck fusing multi-scale ViT features |
| Text Encoder | `VETextEncoder` | CLIP-based text encoder for open-vocabulary concept embeddings |
| Transformer | `transformer` | Encoder+decoder transformer for mask prediction |

### PCC Targets
| Module | PCC Target |
|---|---|
| ViT Backbone | 0.99 |
| FPN Neck | 0.99 |
| Text Encoder | 0.99 |
| Full model | 0.99 |

### Tests
Tests are organized under `blackhole/tests/`:
- `pcc/` — Per-component and end-to-end PCC correctness tests
- `perf/` — Throughput and device performance benchmarks
