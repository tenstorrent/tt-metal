# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC on device against the reference goldens (each module fed the golden input of its reference module).

    embed       patch tokens -> patch_proj                         vs the patch_proj outputs
    time layer  golden layer i-1 output -> layer i                  vs the golden layer i output (all tokens)
    last time   golden layer L-3 output -> layer L-2, cut            vs the golden layer L-2 output at the read-out tokens
    variate     golden layer L-2 output at the read-out tokens       vs the golden layer L-1 output there (V = 1 and V > 1)
    head        golden layer L-1 output at the read-out tokens       vs the head output (out_norm + head)

Cases take every path: clean (causal), missing patches (the group mask), multivariate with two series ids per item
(the variate mask), and context 8192 (S = 288, 32 read-out tokens).
"""

from __future__ import annotations

import gc
import pytest
import torch

from models.experimental.toto2_forecast.tests.golden.goldens import have_golden, load_golden, pcc
from models.experimental.toto2_forecast.tt.preprocessing import pad_to_tile

# One device per module, so a loaded model (2.5B: about a minute) serves every case that uses it.
pytestmark = pytest.mark.use_module_device

MODULE_PCC = 0.999
CASES = [
    ("4m", "uni_4x2048"),
    ("4m", "missing_5x2048"),
    ("4m", "mv_2x4x2048"),
    ("4m", "long_2x8192"),
    ("22m", "uni_4x2048"),
    ("22m", "missing_5x2048"),
    ("22m", "mv_2x4x2048"),
    ("22m", "long_2x8192"),
]
PRECISIONS = ["accuracy", "performance"]
_MODELS: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _free_models_before_device_close(_device_module_impl):
    """The cached model's device tensors must go before the module's device closes."""
    yield
    _MODELS.clear()
    gc.collect()


def _model(device, ckpt: str, precision_name: str):
    from models.experimental.toto2_forecast.tt.common import Toto2Precision
    from models.experimental.toto2_forecast.tt.model import TtToto2

    key = (ckpt, precision_name)
    if key not in _MODELS:
        _MODELS.clear()
        gc.collect()
        _MODELS[key] = TtToto2.from_checkpoint(device, ckpt, Toto2Precision.named(precision_name))
    return _MODELS[key]


def _flat(device, x: torch.Tensor, padded_seq: int | None = None):
    """Host ``(R, S, d)`` -> device ``(1, 1, R * S_pad, d)`` bf16 tiles, zero pad tokens."""
    from models.experimental.toto2_forecast.tt.common import upload

    r, s, d = x.shape
    sp = padded_seq or s
    padded = torch.zeros(r, sp, d)
    padded[:, :s] = x
    return upload(device, padded.reshape(1, 1, r * sp, d))


def _back(t, rows: int, seq: int, padded_seq: int | None = None) -> torch.Tensor:
    import ttnn

    out = ttnn.to_torch(t).float().reshape(rows, padded_seq or seq, -1)
    return out[:, :seq]


def _rows_from_variate_layout(x: torch.Tensor, batch: int, variates: int) -> torch.Tensor:
    """Reference variate-layer layout ``(B * S, V, d)`` -> rows ``(B * V, S, d)``."""
    bs, v, d = x.shape
    return x.reshape(batch, bs // batch, v, d).transpose(1, 2).reshape(batch * v, bs // batch, d)


def _case(device, ckpt, case, precision_name):
    if not have_golden(ckpt, case):
        pytest.skip(f"golden {ckpt}/{case} not generated yet")
    g = load_golden(ckpt, case)
    model = _model(device, ckpt, precision_name)
    i = g["inputs"]
    host = model.prepare_inputs(i["target"], i["target_mask"], i["series_ids"], i["horizon"])
    return g, model, host


@pytest.mark.parametrize("precision_name", PRECISIONS)
@pytest.mark.parametrize("ckpt,case", CASES, ids=[f"{c}-{n}" for c, n in CASES])
def test_modules(device, ckpt, case, precision_name):
    import ttnn

    g, model, host = _case(device, ckpt, case, precision_name)
    cap = g["captures"]
    floor = MODULE_PCC if precision_name == "accuracy" else 0.995
    r, s = host.rows, host.seq_len
    sp = pad_to_tile(s)
    start, count = host.readout
    n_layers = model.config.num_layers
    kinds = model.layer_kinds
    inputs = model.upload_inputs(host)
    results = {}
    try:
        # Patch embedding.
        out = model.embed(inputs)
        expected = torch.cat([cap["embed_context"], cap["embed_horizon"]], dim=2).reshape(r, s, -1)
        results["embed"] = pcc(_back(out, r, s, sp), expected)
        ttnn.deallocate(out)

        # Time layers, each from the golden input.
        layer_in = expected
        for li, kind in enumerate(kinds):
            if kind == "variate":
                continue
            x = _flat(device, layer_in, sp)
            y = model.run_layer(li, x, inputs)
            golden = cap[f"layer_{li}"]
            if kind == "last_time":
                results[f"layer_{li}"] = pcc(_back(y, r, count), golden[:, start : start + count])
            else:
                results[f"layer_{li}"] = pcc(_back(y, r, s, sp), golden)
            ttnn.deallocate(y)
            layer_in = golden

        # Variate layer on the read-out tokens.
        vi = kinds.index("variate")
        x = _flat(device, cap[f"layer_{vi - 1}"][:, start : start + count])
        y = model.run_layer(vi, x, inputs)
        golden = _rows_from_variate_layout(cap[f"layer_{vi}"], host.batch, host.variates)[:, start : start + count]
        results[f"layer_{vi}"] = pcc(_back(y, r, count), golden)
        ttnn.deallocate(y)

        # out_norm + head.
        x = _flat(device, golden)
        out = model.head(x)
        head = model.normalized_output(ttnn.to_torch(out).float(), host)
        results["head"] = pcc(head, cap["head_out"])
        ttnn.deallocate(out)
    finally:
        model.deallocate_inputs(inputs)
    print(f"\n[MODULES] {ckpt}/{case}/{precision_name}: " + " ".join(f"{k}={v:.5f}" for k, v in results.items()))
    assert len([k for k in results if k.startswith("layer_")]) == n_layers
    bad = {k: v for k, v in results.items() if not v >= floor}
    assert not bad, f"below {floor}: {bad}"
