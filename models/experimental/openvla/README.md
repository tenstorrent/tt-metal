# OpenVLA - Vision-Language-Action Model

OpenVLA (Open Vision-Language-Action) is a 7B parameter vision-language-action model for robot manipulation tasks.
This implementation supports multi-device configurations: N300 (2 devices) and T3K (8 devices).

## Model Architecture

OpenVLA combines:
- **DINOv2** (ViT-L/14): Self-supervised vision transformer for visual features
- **SigLIP** (ViT-SO400M/14): Contrastive vision-language model for semantic features
- **LLaMA-2-7B**: Language model backbone for action prediction
- **Projector**: Linear layers to project visual features to LLM space

## Supported Hardware

| Device | Configuration | Notes |
|--------|--------------|-------|
| N300 | 2 devices (1x2 mesh) | BF16 attention + BFP8 FFN |
| T3K | 8 devices | Full BF16 for best precision |

> **Note**: Single device (P150/N150) is not supported due to OOM errors with BF16 attention. N300 (2 devices) is the minimum requirement.

## Usage

### Prerequisites

1. Download OpenVLA weights from HuggingFace:
```bash
# Default: downloads to ~/openvla_weights/
bash models/experimental/openvla/references/setup_openvla_weights.sh

# Or specify custom path
bash models/experimental/openvla/references/setup_openvla_weights.sh /path/to/weights

# Set environment variable
export OPENVLA_WEIGHTS=<path_to_openvla_weights>
```

### Running Tests

```bash
# DINOv2 PCC tests (uses pretrained weights from TIMM)
pytest models/experimental/openvla/tests/test_dinov2_pcc.py -v

# SigLIP PCC tests (uses pretrained weights from TIMM)
pytest models/experimental/openvla/tests/test_siglip_pcc.py -v

# Vision backbone tests (uses random weights, no download needed)
pytest models/experimental/openvla/tests/test_openvla_vision_pcc.py -v
```

### Run Demo (TTNN Single Inference)

```bash
export OPENVLA_WEIGHTS=<path_to_openvla_weights>

# Run with synthetic image
python models/experimental/openvla/demo/demo.py

# Run with custom image
python models/experimental/openvla/demo/demo.py --image /path/to/image.jpg

# Run with custom prompt
python models/experimental/openvla/demo/demo.py --prompt "What action to pick up the block?"
```

### PyTorch Reference Benchmark

```bash
# Run FPS benchmark (CPU)
python models/experimental/openvla/references/run_pytorch_openvla.py --benchmark

# Run FPS benchmark (CUDA)
python models/experimental/openvla/references/run_pytorch_openvla.py --benchmark --device cuda
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENVLA_WEIGHTS` | Path to downloaded weights | Required |
| `HF_MODEL` | Must be `meta-llama/Llama-2-7b-hf` | Required |
| `HF_TOKEN` | HuggingFace token for Llama-2 access | Required |
| `MESH_DEVICE` | Device type: `N300`, `T3K`, `TG` | Auto-detect |

> **Note**: Llama-2 requires access permission from Meta. Request access at [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf), then create an HF token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Run Full Model Test

```bash
export OPENVLA_WEIGHTS=<path_to_openvla_weights>
export HF_MODEL=meta-llama/Llama-2-7b-hf
export HF_TOKEN=<your_huggingface_token>
pytest models/experimental/openvla/tt/open_vla.py::test_openvla_model -v
```

## Precision Configuration

The model automatically selects precision based on device:

- **N300 (2 devices)**: BF16 for attention (WQKV, WO, KV_CACHE, ACTIVATION), BFP8 for FFN (minimum supported)
- **T3K (8+ devices)**: Full BF16 for best instruction sensitivity
- **LM Head**: Always BF16 to prevent token logit corruption

> Single device (P150/N150) runs out of memory with BF16 attention, hence N300 is the minimum supported configuration.

## PCC Test Results

| Component | PCC |
|-----------|-----|
| Vision Encoder (DINOv2) | 0.99+ |
| Vision Encoder (SigLIP) | 0.99+ |
| Full Model | TBD |

## Known Issues

1. **Layer 0-2 Hidden State Divergence**: There's a numerical divergence in early LLM layers between PyTorch and TT implementations that needs investigation.

2. **Token 31872 Inflation**: When using BFP8 for LM head, token 31872 can have inflated logits. Fixed by using BF16 for LM head.

## References

- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [OpenVLA GitHub](https://github.com/openvla/openvla)
- [HuggingFace Model](https://huggingface.co/openvla/openvla-7b)
