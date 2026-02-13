# MoLE (Mixture-of-Linear-Experts) for TT-NN

Implementation of MoLE (Mixture-of-Linear-Experts) for long-term time series forecasting using TT-NN APIs.

## Overview

MoLE is a meta-architecture that augments existing linear models (DLinear, RLinear, RMLP) with a Mixture-of-Experts framework, achieving SOTA performance on time series forecasting benchmarks.

## Architecture

- **Multiple Expert Models**: 4-8 experts, each a full linear-centric model
- **Router Model**: Small MLP that learns expert mixing weights
- **Expert Weighting**: Weighted combination of expert outputs
- **End-to-End Training**: Joint optimization of experts and router

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python scripts/train.py --dataset ETTh1 --model MoLE --num_experts 4
```

### Evaluation
```bash
python scripts/evaluate.py --dataset ETTh1 --model MoLE --checkpoint path/to/model.pth
```

## Project Structure

```
mole_ttnn/
├── models/
│   ├── __init__.py
│   ├── dlinear.py       # DLinear base model
│   ├── rlinear.py       # RLinear base model (optional)
│   ├── router.py        # Router model
│   └── mole.py          # MoLE framework
├── utils/
│   ├── __init__.py
│   ├── data_loader.py   # Dataset loaders
│   ├── metrics.py       # Evaluation metrics
│   └── trainer.py       # Training utilities
├── tests/
│   ├── __init__.py
│   ├── test_dlinear.py
│   ├── test_router.py
│   └── test_mole.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
└── requirements.txt
```

## References

- Ni et al., "Mixture-of-Linear-Experts for Long-term Time Series Forecasting", AISTATS 2024
- Zeng et al., "Are Transformers Effective for Time Series Forecasting?", AAAI 2023
