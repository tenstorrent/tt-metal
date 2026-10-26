# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Full-model PCC on real weights: TTNN TimesFM-3 against the reference goldens.

Every case asserts every gate of ``golden_io.GATES`` for its precision: on the head output at the read-out
tokens (normalized space) ``pcc_normalized``, ``pcc_spread``, the per-row PCC (all and active rows) and the
per-row ``spread_ratio`` (forecast band amplitude), and on the forecaster's quantiles ``pcc_final`` and
``pcc_final_spread``. Per-layer outputs (valid tokens): 0.999.

Path coverage: leading padding (explicit seq / var masks) and none (causal SDPA, unmasked variates);
V = 1 (W_ov shortcut) and V = 8; one read-out token (H = 64) and several (H = 128, stitching and the CPM
refine; H = 1024 with an unpadded 512-token sequence); symmetric averaging and make_positive.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.timesfm3_forecast.tests.golden_io import active_rows, check_gates, check_pcc, load_golden

pytestmark = pytest.mark.use_module_device

LAYER_FLOOR = 0.999
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


def _check_layers(name, g, capture, prepared, stack_floor=LAYER_FLOOR):
    rows, n, kept = prepared.rows, prepared.num_patches, prepared.num_kept_patches
    valid = prepared.token_valid[:, :kept, None].expand(-1, -1, 1280)
    ref_in = g["transformer_input"].reshape(rows, n, -1)[:, :kept]
    check_pcc(f"{name} transformer_input", ref_in, capture["transformer_input"], LAYER_FLOOR, where=valid)
    for key, ref in sorted(g["layers"].items(), key=lambda kv: int(kv[0][5:].split(".")[0])):
        if "." in key:
            continue
        floor = stack_floor if key == "layer19" else LAYER_FLOOR
        check_pcc(f"{name} {key} output", ref.reshape(rows, n, -1)[:, :kept], capture[key], floor, where=valid)


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
    check_gates("single", ref_head, head, g["quantiles"][None], q)


@pytest.mark.parametrize("name", ["full16_2048_h64", "full16_2048_h128", "mv4x8_2048_h64"])
def test_full_model(device, name):
    """16 x 2048 with left-padded / NaN series (H = 64 and H = 128), and 4 items x 8 variates."""
    tt = _model(device)
    g = load_golden(name)
    capture = {}
    prepared, inputs, head, q = _run(tt, _contexts(g), g["horizon"], capture, **g["settings"])
    assert inputs.seq_mask is not None, "expected the explicit (leading-pad) sequence mask"
    assert inputs.var_mask is None, "the variates of an item share their padding: no variate mask"
    _check_layers(name, g, capture, prepared)
    check_gates(name, g["head_readout"], head, g["quantiles"], q)


def test_gift_eval_case(device):
    """64 series, GIFT-Eval settings (symmetric averaging over x and -x, make_positive), H = 48."""
    tt = _model(device)
    g = load_golden("gift64_h48")
    _, _, head, q = _run(tt, _contexts(g), g["horizon"], **g["settings"])
    check_gates("gift64", g["head_readout"], head, g["quantiles"], q)


@pytest.mark.parametrize(
    "name,subset",
    [("bench_paper", None), ("bench_multivariate", None), ("bench_long_context", None), ("bench_long_context", 1)],
    ids=["paper", "multivariate", "long_context", "long_context_single_series"],
)
def test_bench_shape(device, name, subset):
    """The bench goldens (realistic mix, the gate metrics' home), unpadded: causal SDPA without a mask,
    unmasked variate attention, and (long_context) a 512-token sequence with no token padding and 31 stitched
    read-outs. The active rows come from the full golden."""
    tt = _model(device)
    g = load_golden(name)
    prepared, inputs, head, q = _run(tt, _contexts(g, subset), g["horizon"])
    assert inputs.seq_mask is None and inputs.var_mask is None
    rows = prepared.rows
    items = len(_contexts(g, subset))
    active = active_rows(g["head_readout"])[:rows]
    check_gates(f"{name}[:{subset}]", g["head_readout"][:rows], head, g["quantiles"][:items], q, active=active)


@pytest.mark.parametrize("name", ["bench_paper", "bench_long_context", "full16_2048_h128", "mv4x8_2048_h64"])
def test_performance_precision(device, name):
    tt = _model(device, "performance")
    g = load_golden(name)
    _, _, head, q = _run(tt, _contexts(g), g["horizon"], **g.get("settings", {}))
    check_gates(f"{name} performance", g["head_readout"], head, g["quantiles"], q, precision="performance")
