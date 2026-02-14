# Informer Time-Series Forecasting (TTNN)

## Platforms
- Wormhole (n150, n300)
- Blackhole

## Overview
Informer is a transformer-based model for long-sequence time-series forecasting with ProbSparse attention, distilling, and a generative decoder. This directory provides a TTNN implementation with a PyTorch reference and end-to-end demos.

## Directory Structure
```
informer/
├── README.md
├── __init__.py
├── demo/
│   ├── demo_informer_inference.py
│   ├── demo_informer_pytorch.py
│   └── ETTh1.csv                # Optional local dataset copy
├── reference/
│   ├── __init__.py
│   └── torch_informer.py
├── tests/
│   ├── __init__.py
│   ├── test_end_to_end.py
│   └── test_ttnn_informer.py
└── tt/
    ├── __init__.py
    └── ttnn_informer.py
```

## Setup
1) Build TT-Metal and create the repo venv:
```bash
./build_metal.sh
./create_venv.sh
source python_env/bin/activate
```

2) (Optional) Download ETTh1:
```bash
curl -L -o models/demos/informer/demo/ETTh1.csv \
    https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv
```

## Quick Start
### 1) TTNN Smoke Test
```bash
python -m models.demos.informer.demo.demo_informer_inference smoke \
    --batch 2 --seq-len 96 --label-len 48 --pred-len 24 --features 7
```

### 2) Benchmark (Latency/Throughput)
```bash
python -m models.demos.informer.demo.demo_informer_inference benchmark \
    --batch-sizes 1,2,4,8,16 --warmup 5 --iters 20
```

### 3) Train a Torch Checkpoint (ETTh1)
```bash
python -m models.demos.informer.demo.demo_informer_pytorch train \
    --dataset models/demos/informer/demo/ETTh1.csv \
    --output /tmp/etth1_torch.pt \
    --steps 200 --batch-size 8 --d-model 64 --n-heads 2 --d-ff 256
```

### 4) Evaluate TTNN vs Torch + Ground Truth
```bash
python -m models.demos.informer.demo.demo_informer_inference eval \
    --dataset models/demos/informer/demo/ETTh1.csv \
    --checkpoint /tmp/etth1_torch.pt \
    --time-features calendar --normalize
```

### 5) Torch-Only Ground Truth Evaluation
```bash
python -m models.demos.informer.demo.demo_informer_pytorch eval \
    --dataset models/demos/informer/demo/ETTh1.csv \
    --checkpoint /tmp/etth1_torch.pt \
    --time-features calendar --normalize
```

## Notes
- `d_model` and `head_dim` must be multiples of 32 for TTNN tile matmul.
- `demo_informer_inference eval` uses checkpoint normalization stats if present; `--normalize` computes stats from the training split when the checkpoint does not include them.
- ProbSparse top-k selection can introduce small divergence vs the torch reference for long sequences.

## References
- Informer paper (AAAI 2021): https://arxiv.org/abs/2012.07436
- HuggingFace Informer: https://huggingface.co/docs/transformers/en/model_doc/informer
- Original repo: https://github.com/zhouhaoyi/Informer2020
- ETT dataset: https://github.com/zhouhaoyi/ETDataset
