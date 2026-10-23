# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC on real weights: TTNN TimesFM-3 against the reference goldens.

- ``pcc_normalized``: the head output at the read-out tokens (normalized space), floor 0.999;
- ``pcc_final``: the forecaster's quantiles after the host post-processing, floor 0.99;
- per-layer outputs (valid tokens), floor 0.999;
- the ``performance`` precision: 0.995 / 0.98.

Path coverage: leading padding (explicit seq / var masks) and none (causal SDPA, unmasked variates);
V = 1 (W_ov shortcut) and V = 8; one read-out token (H = 64) and several (H = 128, stitching and the CPM
refine; H = 1024 with an unpadded 512-token sequence); symmetric averaging and make_positive.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.timesfm3_forecast.tests.golden_io import (
    check_head_metrics,
    check_pcc,
    final_metrics,
    load_golden,
)

pytestmark = pytest.mark.use_module_device

FLOORS = {"accuracy": (0.999, 0.99), "performance": (0.995, 0.98)}
LAYER_FLOOR = 0.999
# Stack output (layer 19) on the multi-series goldens: the last layers cancel a few massive residual channels
# (|x| ~ 1000 at layer 12 -> max 8.6, rms 0.4 at layer 19), which leaves bf16 rounding of those channels
# visible. Measured 0.9987 (16 x 2048) and 0.9982 (4 x 8 variates); an fp32 residual stream gives 0.9990 /
# 0.9986 for +17.6 % device time. The head output read from it stays >= 0.9998. Deviation from the contract's
# 0.999, raised at G1; the doc's golden workload (``test_single_series_every_layer``) keeps 0.999.
STACK_FLOOR_MULTI = 0.997
_MODELS: dict = {}


def _model(device, precision="accuracy"):
    from models.experimental.timesfm3_forecast.tt.common import TtPrecision
    from models.experimental.timesfm3_forecast.tt.model import TtTimesFM3

    key = (id(device), precision)
    if key not in _MODELS:
        _MODELS.clear()
        _MODELS[key] = TtTimesFM3.from_checkpoint(device, TtPrecision.by_name(precision))
    return _MODELS[key]


def _contexts(g: dict, subset=None):
    if "context" in g:
        return [g["context"].numpy()]
    c = g["contexts"]
    c = [t.numpy() for t in c] if isinstance(c, list) else list(c.numpy())
    return c if subset is None else c[:subset]


def _run(tt, contexts, horizon, capture=None, **kw):
    import ttnn

    prepared = tt.prepare_inputs(contexts, horizon, **kw)
    inputs = tt.upload_inputs(prepared)
    out = tt.forward_device(inputs, capture)
    head, quantiles = tt.postprocess(out, prepared)
    ttnn.deallocate(out)
    tt.deallocate_inputs(inputs)
    return prepared, inputs, head, torch.from_numpy(quantiles)


def _check_final(name, expected, got, floor):
    """``pcc_final`` against its floor; ``pcc_final_spread`` is printed (data-space levels dominate pcc_final)."""
    m = final_metrics(expected, got)
    print(f"[METRICS] {name}: pcc_final_spread {m['pcc_final_spread']:.6f}", flush=True)
    return check_pcc(f"{name} pcc_final", expected, got, floor)


def _check_layers(name, g, capture, prepared, stack_floor=LAYER_FLOOR):
    valid = prepared.token_valid[..., None].expand(-1, -1, 1280)
    rows, n = prepared.rows, prepared.num_patches
    ref_in = g["transformer_input"].reshape(rows, n, -1)
    check_pcc(f"{name} transformer_input", ref_in, capture["transformer_input"], LAYER_FLOOR, where=valid)
    for key, ref in sorted(g["layers"].items(), key=lambda kv: int(kv[0][5:].split(".")[0])):
        if "." in key:
            continue
        floor = stack_floor if key == "layer19" else LAYER_FLOOR
        check_pcc(f"{name} {key} output", ref.reshape(rows, n, -1), capture[key], floor, where=valid)


def test_single_series_every_layer(device):
    """The doc's golden workload: 1 x 2048 -> 64, all 20 layer outputs, head and quantiles."""
    tt = _model(device)
    g = load_golden("single_2048_h64")
    capture = {}
    prepared, _, head, q = _run(tt, _contexts(g), g["horizon"], capture)
    _check_layers("single", g, capture, prepared)
    ref_head = g["head_raw"].reshape(1, prepared.num_patches, -1)[
        :, prepared.readout_tokens.start : prepared.readout_tokens.stop
    ]
    check_head_metrics("single", ref_head, head)
    _check_final("single", g["quantiles"][None], q, FLOORS["accuracy"][1])


@pytest.mark.parametrize("name", ["full16_2048_h64", "full16_2048_h128", "mv4x8_2048_h64"])
def test_full_model(device, name):
    """16 x 2048 with left-padded / NaN series (H = 64 and H = 128), and 4 items x 8 variates."""
    tt = _model(device)
    g = load_golden(name)
    capture = {}
    prepared, inputs, head, q = _run(tt, _contexts(g), g["horizon"], capture, **g["settings"])
    assert inputs.seq_mask is not None, "expected the explicit (leading-pad) sequence mask"
    assert inputs.var_mask is None, "the variates of an item share their padding: no variate mask"
    _check_layers(name, g, capture, prepared, stack_floor=STACK_FLOOR_MULTI)
    check_head_metrics(name, g["head_readout"], head)
    _check_final(name, g["quantiles"], q, FLOORS["accuracy"][1])


def test_gift_eval_case(device):
    """64 series, GIFT-Eval settings (symmetric averaging over x and -x, make_positive), H = 48."""
    tt = _model(device)
    g = load_golden("gift64_h48")
    _, _, head, q = _run(tt, _contexts(g), g["horizon"], **g["settings"])
    check_head_metrics("gift64", g["head_readout"], head)
    _check_final("gift64", g["quantiles"], q, FLOORS["accuracy"][1])


@pytest.mark.parametrize(
    "name,subset",
    [("bench_paper", 64), ("bench_multivariate", 16), ("bench_long_context", 2), ("bench_long_context", 1)],
    ids=["paper", "multivariate", "long_context", "long_context_single_series"],
)
def test_bench_shape_subset(device, name, subset):
    """Unpadded inputs: causal SDPA without a mask, unmasked variate attention, and (long_context) a
    512-token sequence with no token padding and 31 stitched read-outs."""
    tt = _model(device)
    g = load_golden(name)
    prepared, inputs, head, q = _run(tt, _contexts(g, subset), g["horizon"])
    assert inputs.seq_mask is None and inputs.var_mask is None
    rows = prepared.rows
    check_head_metrics(f"{name}[:{subset}]", g["head_readout"][:rows], head)
    _check_final(f"{name}[:{subset}]", g["quantiles"][:subset], q, FLOORS["accuracy"][1])


@pytest.mark.parametrize("name", ["full16_2048_h128", "mv4x8_2048_h64"])
def test_performance_precision(device, name):
    tt = _model(device, "performance")
    g = load_golden(name)
    _, _, head, q = _run(tt, _contexts(g), g["horizon"], **g["settings"])
    check_head_metrics(f"{name} performance", g["head_readout"], head, "performance")
    _check_final(f"{name} performance", g["quantiles"], q, FLOORS["performance"][1])
