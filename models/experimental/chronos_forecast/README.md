# Chronos-2 Forecast

[Chronos](https://github.com/amazon-science/chronos-forecasting) pretrained time-series models. This tree vendors **Chronos-2** inference sources as a golden PyTorch reference for a later TTNN port.

## Setup

```bash
git submodule update --init models/experimental/chronos_forecast/third_party/chronos-forecasting
pip install -r models/experimental/chronos_forecast/requirements.txt
```

Pinned Chronos-2 copy: commit `10afa9ebe016e514f9d7dc1aa873f66af57e116b`. See [reference/PROVENANCE.md](reference/PROVENANCE.md).



## Weights

Real Chronos-2 weights are **not** committed, only testing for model shape right now will change when real infra

## Demo

Single `Chronos2Model.forward` (CPU):

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py
```

Chronos-1 tokenizer only:

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py --tokenizer-only --context-length 16
```

## References
- Paper: [Chronos](https://arxiv.org/abs/2403.07815)
- Chronos-2: [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- Upstream: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
