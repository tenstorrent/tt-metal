# Facebook SAM 2 (Hiera-tiny) — Tenstorrent `ttnn` Native Port (WIP)

This directory contains a **work-in-progress** Tenstorrent native `ttnn` implementation of **Segment Anything Model 2 (SAM 2) (`facebook/sam2-hiera-tiny`)** for promptable visual segmentation in still images (`1024x1024` input mode).

> **Scope:** Still-image segmentation only (`Image Mode`). Video object tracking (`memory bank`, `memory encoder`, `temporal attention blocks`) is explicitly excluded per [Issue #48311](https://github.com/tenstorrent/tt-metal/issues/48311).

## 📁 Repository Structure

```text
models/demos/sam2/
├── reference/
│   ├── __init__.py
│   ├── sam2_reference.py         # Simplified PyTorch reference (debugging utility only)
│   ├── preprocess_sam2.py        # Parameter preprocessing for TTNN
│   └── summary.py                # Model summary
├── tt/
│   ├── __init__.py
│   ├── hiera_image_encoder.py    # 12-block Hiera encoder with windowed/global attention
│   ├── prompt_encoder.py         # Point, box, and mask prompt encoding
│   ├── mask_decoder.py           # Two-way transformer + upscaling + hypernetwork heads
│   └── sam2_model.py             # Orchestrator
├── tests/
│   ├── test_sam2_reference.py    # Reference model shape verification
│   └── test_sam2_accuracy.py     # PCC verification against HF Sam2Model reference
├── demo/
│   └── demo_segmentation.py      # End-to-end device demo
├── ENGINEERING_PLAYBOOK.md       # Engineering plan for bounty completion
├── ARCHITECTURE_GAP.md           # Honest gap analysis against HF architecture
└── README.md                     # This file
```

## Architecture Status

The implementation follows the HF `Sam2Model` architecture (transformers >= 4.56.0, revision `7c218be`):

- **12-block Hiera encoder** with windowed/global attention, LayerNorm, MLP, and residual connections
- **FPN neck** with 3 feature levels and positional encodings
- **Prompt encoder** supporting point, box, and mask prompts with correct label/corner encoding
- **Two-way transformer mask decoder** with self-attention, cross-attention (token→image, image→token), upscaling convolutions, hypernetwork MLPs, IoU prediction head, and object score head

## Current Limitations (Honest)

| Limitation | Detail |
|------------|--------|
| **Weights** | Random initialization — not the real SAM2 checkpoint |
| **Host fallbacks** | LayerNorm, GELU, pooling, and some convolutions use `torch.nn.functional` (need TTNN-native ops) |
| **Conv2d API** | Uses simplified signature — needs alignment with SDXL pattern for CI |
| **Upscaling** | Uses CPU fallback — waiting for `ttnn.upsample` + `ttnn.conv2d` pattern validation |
| **Sharding** | `L1_MEMORY_CONFIG` only — no actual sharding strategy |
| **Performance** | No performance test or report yet |
| **Hardware** | Not yet run on N150/N300 |

## Running Tests

```bash
# Reference model verification (CPU)
python3 models/demos/sam2/tests/test_sam2_reference.py

# TTNN tests (requires N150/N300 with built tt-metalium)
pytest models/demos/sam2/tests/test_sam2_accuracy.py -v --device_id 0
```

## References

- [Issue #48311 — SAM2 Bounty](https://github.com/tenstorrent/tt-metal/issues/48311)
- [Facebook SAM2 on HuggingFace](https://huggingface.co/facebook/sam2-hiera-tiny)
- [SAM2 Paper](https://arxiv.org/abs/2408.00714)
