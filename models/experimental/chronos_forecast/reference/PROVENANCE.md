# Chronos-2 reference provenance

Golden PyTorch inference sources under `reference/chronos2/` are a **verbatim copy** of Amazon Chronos-2, not a rewrite.

| Field | Value |
|---|---|
| Upstream | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| Submodule | [`third_party/chronos-forecasting`](../third_party/chronos-forecasting) |
| Commit | [`10afa9ebe016e514f9d7dc1aa873f66af57e116b`](https://github.com/amazon-science/chronos-forecasting/tree/10afa9ebe016e514f9d7dc1aa873f66af57e116b) (`v2.3.2-2-g10afa9e`) |
| Paper | [Chronos-2, arXiv:2510.15821](https://arxiv.org/abs/2510.15821) |
| License | Apache-2.0 (Amazon copyright headers kept on copied files) |

## Files

| This tree | Copied from (submodule) |
|---|---|
| [`chronos2/config.py`](chronos2/config.py) | `src/chronos/chronos2/config.py` |
| [`chronos2/layers.py`](chronos2/layers.py) | `src/chronos/chronos2/layers.py` |
| [`chronos2/model.py`](chronos2/model.py) | `src/chronos/chronos2/model.py` |
| [`chronos_bolt_ops.py`](chronos_bolt_ops.py) | `src/chronos/chronos_bolt.py` lines 74–139 (`Patch`, `InstanceNorm`) |

The only intentional edit in `chronos2/model.py` is the Bolt import:

```python
# was: from chronos.chronos_bolt import InstanceNorm, Patch
from models.experimental.chronos_forecast.reference.chronos_bolt_ops import InstanceNorm, Patch
```

Not copied (pipeline / data / training): `pipeline.py`, `preprocess.py`, `dataset.py`, `trainer.py`.

Dummy checkpoint used by golden tests (not vendored; root `.gitignore` excludes `*.bin` and we do not copy weights):

`third_party/chronos-forecasting/test/dummy-chronos2-model/`
