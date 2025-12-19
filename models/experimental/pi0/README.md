# PI0 Model for Tenstorrent

PI0 (Physical Intelligence Zero) is a vision-language-action model for robotics
that combines a vision encoder, language model, and action expert for end-to-end
robot control.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         PI0 Model                              │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐│
│  │   SigLIP        │  │   Gemma 2B      │  │   Gemma 300M    ││
│  │   Vision Tower  │  │   VLM Backbone  │  │   Action Expert ││
│  │   (27 blocks)   │  │   (18 blocks)   │  │   (18 blocks)   ││
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘│
│           │                    │                    │         │
│           └──────────┬─────────┴──────────┬────────┘         │
│                      │                    │                   │
│              ┌───────▼────────┐  ┌───────▼────────┐          │
│              │ Prefix Embed   │  │ Suffix Embed   │          │
│              │ (Images+Lang)  │  │ (State+Action) │          │
│              └───────┬────────┘  └───────┬────────┘          │
│                      │                    │                   │
│                      └──────────┬─────────┘                   │
│                                 │                             │
│                      ┌──────────▼──────────┐                  │
│                      │   Shared Attention   │                  │
│                      └──────────┬──────────┘                  │
│                                 │                             │
│                      ┌──────────▼──────────┐                  │
│                      │  Flow Matching      │                  │
│                      │  Denoiser           │                  │
│                      └──────────┬──────────┘                  │
│                                 │                             │
│                                 ▼                             │
│                         Action Output                         │
└────────────────────────────────────────────────────────────────┘
```

## Folder Structure

```
pi0/
├── common/                  # Shared configs and utilities
│   ├── configs.py          # All configuration dataclasses
│   ├── weight_loader.py    # Checkpoint loading utilities
│   └── utils.py            # Common helper functions
│
├── reference/              # Pure PyTorch implementations
│   ├── torch_gemma.py      # Gemma attention, MLP, block
│   ├── torch_siglip.py     # SigLIP vision tower
│   ├── torch_suffix.py     # Suffix embedding
│   ├── torch_prefix.py     # Prefix embedding
│   └── torch_paligemma.py  # PaliGemma backbone
│
├── tt/                     # TTNN implementations
│   ├── ttnn_gemma.py       # TtGemma attention, MLP, block
│   ├── ttnn_siglip.py      # TtSigLIP vision tower
│   ├── ttnn_suffix.py      # TtSuffix embedding
│   └── ttnn_prefix.py      # TtPrefix embedding
│
├── tests/
│   ├── pcc/               # PCC comparison tests
│   │   ├── test_gemma.py
│   │   ├── test_siglip.py
│   │   ├── test_suffix.py
│   │   └── test_pi0_model.py
│   └── perf/              # Performance benchmarks
│       └── test_perf_pi0.py
│
└── ttnn_pi0_reference/    # Original combined implementation (legacy)
    └── ...
```

## Usage

### PyTorch Reference

```python
from models.experimental.pi0.reference import GemmaBlock, SigLIPVisionTower
from models.experimental.pi0.common import GemmaConfig, SigLIPConfig

# Create config
config = GemmaConfig.gemma_2b()

# Create block with weights
block = GemmaBlock(config, weights, layer_idx=0)

# Forward pass
output, cache = block.forward(hidden_states, cos, sin)
```

### TTNN Implementation

```python
import ttnn
from models.experimental.pi0.tt import TtGemmaBlock, TtSigLIPVisionTower
from models.experimental.pi0.common import GemmaConfig

# Get device
device = ttnn.open_device(0)

# Create config
config = GemmaConfig.gemma_2b()

# Convert weights to TTNN
ttnn_weights = {...}  # See weight_loader.py

# Create block
block = TtGemmaBlock(config, ttnn_weights, layer_idx=0, device=device)

# Forward pass
output, cache = block.forward(hidden_states_ttnn, cos_ttnn, sin_ttnn)
```

## Running Tests

### PCC Tests

```bash
cd /home/ubuntu/work/sdawle_pi0/tt-metal
source python_env/bin/activate
export ARCH_NAME=wormhole_b0
export WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
export TT_METAL_HOME=$(pwd)
export PYTHONPATH=$(pwd)

# Run all PI0 PCC tests
pytest models/experimental/pi0/tests/pcc/ -v

# Run specific test
pytest models/experimental/pi0/tests/pcc/test_gemma.py -v
```

### Full Model Test

```bash
# With checkpoint
python models/experimental/pi0/ttnn_pi0_reference/test_full_model_inference_pcc.py \
    --checkpoint /path/to/pi0_base/model.safetensors
```

## PCC Results (Achieved)

| Module | PCC |
|--------|-----|
| Vision Tower | 0.9999 |
| Suffix Embedding | 0.9998 |
| Prefix Embedding | 0.9036 |
| Gemma Block | 0.96 |

## Key Features

- **Hybrid Implementation**: Uses PyTorch fallbacks for complex operations
- **Dynamic Position Embeddings**: Supports different image resolutions
- **Flow Matching Denoiser**: Iterative action generation
- **Multi-Query Attention**: Efficient KV-head sharing
