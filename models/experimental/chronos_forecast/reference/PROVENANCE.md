# Chronos-2 reference provenance

Golden PyTorch inference sources under `reference/chronos2/` are a **verbatim copy** of Amazon Chronos-2, not a rewrite.

| Field | Value |
|---|---|
| Upstream | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| Package | [`chronos-forecasting==2.3.2`](https://pypi.org/project/chronos-forecasting/2.3.2/) (PyPI, pinned in [`requirements.txt`](../requirements.txt)) |
| Commit | [`10afa9ebe016e514f9d7dc1aa873f66af57e116b`](https://github.com/amazon-science/chronos-forecasting/tree/10afa9ebe016e514f9d7dc1aa873f66af57e116b) (`v2.3.2-2-g10afa9e`) |
| Paper | [Chronos-2, arXiv:2510.15821](https://arxiv.org/abs/2510.15821) |
| License | Apache-2.0 (Amazon copyright notices kept on copied files) |

The files were copied at the commit above. The golden tests compare them against the 2.3.2 release, which differs from that commit only in `__about__.py` (version string) and `chronos2/dataset.py` (training data sampling); neither is copied or used here.

## Files

| This tree | Copied from (upstream) |
|---|---|
| [`chronos2/config.py`](chronos2/config.py) | `src/chronos/chronos2/config.py` |
| [`chronos2/layers.py`](chronos2/layers.py) | `src/chronos/chronos2/layers.py` |
| [`chronos2/model.py`](chronos2/model.py) | `src/chronos/chronos2/model.py` |
| [`chronos_bolt_ops.py`](chronos_bolt_ops.py) | `src/chronos/chronos_bolt.py` lines 74–139 (`Patch`, `InstanceNorm`) |

Intentional edits:

- Line 1 of each copied file: Amazon's `# Copyright Amazon.com, ...` notice is tagged as `# SPDX-FileCopyrightText: Copyright Amazon.com, ...` so the repo's SPDX check accepts it. The notice text is unchanged.
- The Bolt import in `chronos2/model.py`:

```python
# was: from chronos.chronos_bolt import InstanceNorm, Patch
from models.experimental.chronos_forecast.reference.chronos_bolt_ops import InstanceNorm, Patch
```

Not copied (pipeline / data / training): `pipeline.py`, `preprocess.py`, `dataset.py`, `trainer.py`.

## Dummy checkpoint

The golden tests load Amazon's tiny test checkpoint (`config.json` plus a 36 KB `model.safetensors`), copied byte-for-byte from `test/dummy-chronos2-model/` at the commit above (Apache-2.0):

[`../tests/fixtures/dummy-chronos2-model/`](../tests/fixtures/dummy-chronos2-model/)
