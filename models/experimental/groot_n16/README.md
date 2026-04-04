# GR00T N1.6-3B on Tenstorrent Blackhole

NVIDIA GR00T N1.6-3B Vision-Language-Action model implementation for Tenstorrent Blackhole p150a.

## Architecture

```
[Images] -> SigLIP2 (27L, 1152d) -> Pixel Shuffle -> MLP Connector
                                                         |
[Text]   -> Tokenizer -----------------------------------+
                                                         v
                                              Qwen3-1.7B (28L, 2048d)
                                              (extract layer 16)
                                                         |
                                                    backbone_features
                                                         |
[State]  -> StateEncoder[embodiment] ----+               |
[Noise]  -> ActionEncoder[embodiment] ---+               |
                                         v               v
                                    AlternateVLDiT (32L, 1536d)
                                         |
                                    ActionDecoder[embodiment]
                                         |
                                    Flow Matching (4 Euler steps)
                                         |
                                    Predicted Actions [B, 16, 29]
```

## Key Specs

| Component | Detail |
|-----------|--------|
| Vision encoder | SigLIP2: 27 layers, 1152 hidden, 16 heads, 14x14 patches |
| Language model | Qwen3-1.7B: 28 layers, 2048 hidden, 16/8 GQA heads |
| Action head | AlternateVLDiT: 32 layers, 1536 inner dim, 32 heads x 48 head_dim |
| Flow matching | 4 Euler integration steps |
| Action horizon | 16 timesteps |
| Action dim | 29 (padded) |
| Embodiments | Up to 32 separate weight sets |
| Total params | ~3B |

## Quick Start

### Prerequisites

- Tenstorrent Blackhole p150a
- tt-metal built and activated

```bash
source python_env/bin/activate
export ARCH_NAME=blackhole
```

### Run PCC Tests

```bash
pytest -svv models/experimental/groot_n16/tests/test_groot_n16_pcc.py
```

### Docker

```bash
# Build
docker build -t groot-n16-bh -f models/experimental/groot_n16/Dockerfile .

# Run with device access
docker run --rm -it --device /dev/tenstorrent/2 -v /dev/hugepages:/dev/hugepages groot-n16-bh
```

## Files

```
groot_n16/
├── README.md
├── Dockerfile
├── conftest.py
├── common/
│   ├── configs.py              # All model configurations
│   └── weight_loader.py        # HuggingFace weight loading
├── reference/
│   └── torch_groot_n16.py      # PyTorch reference for PCC validation
├── tt/
│   ├── ttnn_common.py          # Shared TTNN utilities
│   ├── ttnn_siglip2.py         # SigLIP2 vision encoder
│   ├── ttnn_dit.py             # AlternateVLDiT action head
│   ├── ttnn_embodiment.py      # Per-embodiment MLPs
│   └── ttnn_groot_n16_model.py # Main model assembly
└── tests/
    └── test_groot_n16_pcc.py   # PCC verification tests
```

## References

- [GR00T N1.6 Model Card](https://huggingface.co/nvidia/GR00T-N1.6-3B)
- [Isaac-GR00T Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1 Paper](https://arxiv.org/abs/2503.14734)
