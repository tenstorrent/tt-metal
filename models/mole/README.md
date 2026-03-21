# MoLE-TTNN: Mixture-of-Linear-Experts for Tenstorrent

This is a TTNN (Tenstorrent Neural Network) implementation of **MoLE (Mixture-of-Linear-Experts)** for long-term time series forecasting on Tenstorrent Wormhole/Blackhole hardware.

## Overview

MoLE is a meta-architecture from Microsoft Research that enhances linear-centric time series forecasting models by:
- Training multiple expert models (4-8 experts)
- Using a router to dynamically weight expert outputs based on timestamp
- Enabling automatic specialization of experts for different temporal patterns

This implementation provides:
- 鉁?Full MoLE architecture with DLinear and RLinear experts
- 鉁?TTNN-accelerated layers for Tenstorrent hardware
- 鉁?End-to-end training pipeline
- 鉁?Benchmarking on standard datasets (ETT, Weather, Electricity, Traffic)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- TTNN (Tenstorrent SDK)
- Tenstorrent hardware (Wormhole or Blackhole)

### Install TTNN

```bash
# Follow Tenstorrent's official installation guide
# https://docs.tenstorrent.com/

# Install TT-Metal and TTNN
pip install tt-metal
```

### Install MoLE-TTNN

```bash
cd mole_ttnn
pip install -e .
```

## Quick Start

### 1. Basic Usage

```python
import torch
from mole_ttnn import create_mole_dlinear, TimestampEmbedding

# Create MoLE-DLinear model
model = create_mole_dlinear(
    seq_len=336,      # Input sequence length
    pred_len=96,      # Prediction length
    enc_in=7,         # Number of input channels
    num_experts=4     # Number of experts
)

# Create dummy input
batch_size = 8
x = torch.randn(batch_size, 336, 7)  # [batch, seq_len, channels]

# Create timestamp embedding (e.g., from pandas Timestamp)
import pandas as pd
ts = pd.Timestamp('2024-01-01 12:00:00')
ts_embed = TimestampEmbedding.embed_datetime(ts)
ts_embed = ts_embed.unsqueeze(0).repeat(batch_size, 1)  # [batch, 4]

# Forward pass
output = model(x, ts_embed)  # [batch, 96, 7]
print(f"Output shape: {output.shape}")
```

### 2. Training

```bash
python train.py \
    --dataset ETTh1 \
    --data_path ./dataset \
    --seq_len 336 \
    --pred_len 96 \
    --num_experts 4 \
    --expert_type dlinear \
    --batch_size 8 \
    --epochs 100 \
    --lr 0.005
```

### 3. Benchmarking

```bash
python benchmark.py \
    --data_path ./dataset \
    --expert_type dlinear \
    --num_experts 4 \
    --datasets ETTh1 ETTh2 ETTm1 ETTm2 \
    --pred_lengths 96 192 336 720 \
    --save_results results.json
```

## Architecture

### MoLE Framework

```
Input Time Series 鈹€鈹€鈹攢鈹€> Expert 1 鈹€鈹€鈹?                    鈹溾攢鈹€> Expert 2 鈹€鈹€鈹?                    鈹溾攢鈹€> Expert 3 鈹€鈹€鈹尖攢鈹€> Mixing Layer 鈹€鈹€> Output
                    鈹斺攢鈹€> Expert 4 鈹€鈹€鈹?                           鈫?Timestamp Embedding 鈹€鈹€> Router
```

### Components

1. **Experts**: Linear-centric models (DLinear, RLinear, RMLP)
2. **Router**: 2-layer MLP that outputs expert weights
3. **Mixing Layer**: Weighted combination of expert outputs
4. **Timestamp Embedding**: Encodes datetime into [-0.5, 0.5] range

## Supported Models

| Model | Description |
|-------|-------------|
| MoLE-DLinear | Decomposition Linear with MoE |
| MoLE-RLinear | RevIN Linear with MoE |
| MoLE-RMLP | MLP with RevIN and MoE |

## Supported Datasets

- ETTh1, ETTh2 (1-hour granularity)
- ETTm1, ETTm2 (15-minute granularity)
- Weather (10-minute granularity)
- Electricity (1-hour granularity)
- Traffic (1-hour granularity)
- Weather2K (1-hour granularity, multiple locations)

## Performance

Expected improvements over single-head baselines (from paper):

| Model | Improvement Rate |
|-------|-----------------|
| MoLE-DLinear | 73% of settings |
| MoLE-RLinear | 86% of settings |
| MoLE-RMLP | 75% of settings |

## TTNN Integration

To use TTNN acceleration on Tenstorrent hardware:

```python
from mole_ttnn import create_mole_dlinear
from mole_ttnn.ttnn_layers import convert_to_ttnn_model

# Create PyTorch model
pytorch_model = create_mole_dlinear(seq_len=336, pred_len=96, enc_in=7)

# Convert to TTNN model
with convert_to_ttnn_model(pytorch_model, device_id=0) as ttnn_model:
    output = ttnn_model(x, timestamp_embed)
```

## Project Structure

```
mole_ttnn/
鈹溾攢鈹€ __init__.py          # Package initialization
鈹溾攢鈹€ mole_ttnn.py         # Core MoLE implementation
鈹溾攢鈹€ ttnn_layers.py       # TTNN-accelerated layers
鈹溾攢鈹€ train.py             # Training script
鈹溾攢鈹€ benchmark.py         # Benchmarking script
鈹斺攢鈹€ README.md            # This file
```

## Citation

If you use this implementation, please cite the original MoLE paper:

```bibtex
@inproceedings{ni2024mixture,
  title={Mixture-of-Linear-Experts for Long-term Time Series Forecasting},
  author={Ni, Ronghao and Lin, Zinan and Wang, Shuaiqi and Fanti, Giulia},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4672--4680},
  year={2024},
  organization={PMLR}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- Original MoLE implementation: https://github.com/RogerNi/MoLE
- Tenstorrent TTNN: https://github.com/tenstorrent/tt-metal
