# Chronos 2 Forecast 

[Chronos](https://github.com/amazon-science/chronos-forecasting), Amazon's pretrained time-series forecasting models

## Directory layout


## Setup

From the tt-metal repo root:

```bash
git submodule update --init models/experimental/chronos_forecast/third_party/chronos-forecasting
pip install -r models/experimental/chronos_forecast/requirements.txt
```

The Amazon package is also importable without a pip install: `common/chronos_src.py` puts `third_party/chronos-forecasting/src` on `sys.path`. Optionally install it editable:

```bash
pip install -e models/experimental/chronos_forecast/third_party/chronos-forecasting
```

## Tests

CPU smoke test (no Tenstorrent device required):

```bash
pytest models/experimental/chronos_forecast/tests/test_submodule_import.py -v
```

PCC tests against TTNN will live in `tests/pcc/` once the device implementation exists.

## Demo

```bash
python models/experimental/chronos_forecast/demo/demo.py
```

This runs the Amazon tokenizer on a synthetic series. Loading a pretrained checkpoint (for example `amazon/chronos-t5-tiny`) is not wired yet.

## References

- Paper: [Chronos: Learning the Language of Time Series](https://arxiv.org/abs/2403.07815)
- Chronos-2: [https://arxiv.org/abs/2510.15821](https://arxiv.org/abs/2510.15821)
- Upstream: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)
