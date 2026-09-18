# Chronos-2 Forecast

[Chronos](https://github.com/amazon-science/chronos-forecasting) pretrained time-series models. This tree vendors **Chronos-2** inference sources as a golden PyTorch reference for a later TTNN port.

## Directory layout

```text
chronos_forecast/
├── common/                 submodule path helper + Chronos-1 demo config
├── reference/
│   ├── PROVENANCE.md       pin SHA / file mapping
│   ├── chronos2/           verbatim Amazon Chronos-2 (config, layers, model)
│   ├── chronos_bolt_ops.py Patch + InstanceNorm excerpt from chronos_bolt.py
│   └── pytorch_chronos.py  Chronos-1 tokenizer wrappers (submodule)
├── tt/                     TTNN stubs (not implemented)
├── tests/
│   ├── test_reference_vs_upstream.py   CPU golden vs submodule
│   └── test_submodule_import.py
├── demo/demo.py
└── third_party/chronos-forecasting/    git submodule
```

## Setup

```bash
git submodule update --init models/experimental/chronos_forecast/third_party/chronos-forecasting
pip install -r models/experimental/chronos_forecast/requirements.txt
```

Pinned Chronos-2 copy: commit `10afa9ebe016e514f9d7dc1aa873f66af57e116b`. See [reference/PROVENANCE.md](reference/PROVENANCE.md).

## Tests

CPU, no Tenstorrent device:

```bash
pytest models/experimental/chronos_forecast/tests/test_reference_vs_upstream.py -v
pytest models/experimental/chronos_forecast/tests/test_submodule_import.py -v
```

`test_reference_vs_upstream.py` locks each Chronos-2 layer against the submodule (dummy checkpoint). `test_model_forward_pretrained` runs a full forward on the downloaded `amazon/chronos-2` weights.

Accuracy test (needs `weights/chronos-2`):

```bash
PYTHONPATH=. python -m pytest --confcutdir=models/experimental/chronos_forecast \
  models/experimental/chronos_forecast/tests/test_reference_vs_upstream.py::test_model_forward_pretrained -v
```

## Weights

Real Chronos-2 weights are **not** committed. They live under `weights/` (gitignored). Download:

```bash
hf download amazon/chronos-2 --local-dir models/experimental/chronos_forecast/weights/chronos-2
```

## Demo

Single `Chronos2Model.forward` (CPU):

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py
```

Chronos-1 tokenizer only:

```bash
PYTHONPATH=. python models/experimental/chronos_forecast/demo/demo.py --tokenizer-only --context-length 16
```

TTNN forward is not implemented.

## References

- Paper: [Chronos](https://arxiv.org/abs/2403.07815)
- Chronos-2: [arXiv:2510.15821](https://arxiv.org/abs/2510.15821)
- Upstream: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
