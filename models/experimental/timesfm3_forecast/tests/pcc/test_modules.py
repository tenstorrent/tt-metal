# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC: TTNN TimesFM-3 sublayers on real weights against the reference's module goldens.

The inputs are random residual-stream states with leading-pad masks (``modules`` golden): fully masked
query rows are exercised and must stay finite. Tokens of leading padding are excluded from the PCC where
the reference returns 0 for a fully masked attention row and the device returns a finite value no valid
token reads (proven harmless in ``tests/golden/check_dataflow.py``).
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.timesfm3_forecast.tests.golden_io import check_pcc, load_golden

pytestmark = pytest.mark.use_module_device

FLOOR = 0.999
CASES = ["layer0_uni", "layer0_mv", "layer0_mv32", "layer19_uni", "layer19_mv"]
_MODELS: dict = {}


def _model(device, precision="accuracy"):
    from models.experimental.timesfm3_forecast.tt.common import TtPrecision
    from models.experimental.timesfm3_forecast.tt.model import TtTimesFM3

    key = (id(device), precision)
    if key not in _MODELS:
        _MODELS.clear()
        _MODELS[key] = TtTimesFM3.from_checkpoint(device, TtPrecision.by_name(precision), layers=[0, 19])
    return _MODELS[key]


def _layer_index(case: str) -> int:
    return {"layer0": 0, "layer19": 1}[case.split("_")[0]]


def _inputs(tt, nfm: torch.Tensor, variates: int, n: int):
    """Device inputs (masks, RoPE tables) for module tests; no tokens."""
    from models.experimental.timesfm3_forecast.common.host import HostInputs, num_forecast_patches

    rows = nfm.shape[0]
    z = torch.zeros(rows, n)
    prepared = HostInputs(
        tokens=torch.zeros(rows, n, 192, dtype=torch.float64),
        num_front_masked=nfm,
        variates=variates,
        num_context_patches=n + 1 - num_forecast_patches(64),  # read-out ends at token n - 1: nothing dropped
        horizon=64,
        context_len=(n + 1 - num_forecast_patches(64)) * 32,
        running_n=z,
        running_mu=z,
        running_sigma=z,
        trend_m=z[:, 0],
        trend_c=z[:, 0],
        trend_apply=z[:, 0].bool(),
    )
    inputs = tt.upload_inputs(prepared)
    return inputs


def _upload_rows(tt, x: torch.Tensor, inputs):
    """(R, n, D) -> flat device (1, 1, R * S, D), zero token padding."""
    from models.experimental.timesfm3_forecast.tt.common import upload

    s = inputs.padded_seq
    flat = torch.nn.functional.pad(x, (0, 0, 0, s - x.shape[1])).reshape(1, 1, -1, x.shape[-1])
    return upload(tt.device, flat)


def _case(name: str):
    g = load_golden("modules")[name]
    b, v, n, d = g["x"].shape
    eff = torch.cumprod(g["patch_mask"].int(), dim=2).bool()
    nfm = eff.reshape(b * v, n).sum(-1)
    flat = lambda t: t.reshape(b * v, n, d)  # noqa: E731
    valid = (torch.arange(n)[None, :] >= nfm[:, None])[..., None].expand(-1, -1, d)
    return g, flat, nfm, v, n, valid


def test_tokenizer(device):
    import ttnn

    tt = _model(device)
    g = load_golden("modules")["tokenizer"]
    nfm = torch.zeros(g["x"].shape[0], dtype=torch.long)
    inputs = _inputs(tt, nfm, 1, g["x"].shape[1])
    inputs.tokens = _upload_rows(tt, g["x"].double(), inputs)
    out = tt.embed(inputs)
    got = tt.rows_to_torch(out, inputs)
    ttnn.deallocate(out)
    tt.deallocate_inputs(inputs)
    check_pcc("tokenizer", g["out"], got, FLOOR)


def test_head(device):
    import ttnn

    tt = _model(device)
    g = load_golden("modules")["head"]
    rows, n, _ = g["x"].shape
    inputs = _inputs(tt, torch.zeros(rows, dtype=torch.long), 1, n)
    inputs.readout_start, inputs.readout_count = 0, n
    x = _upload_rows(tt, g["x"], inputs)
    out = tt.head(x, inputs)
    got = tt.head_to_torch(out, inputs)
    for t in (x, out):
        ttnn.deallocate(t)
    tt.deallocate_inputs(inputs)
    check_pcc("head (all tokens)", g["out"], got, FLOOR)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("sublayer", ["seq_attention", "var_attention", "ffn", "layer"])
def test_sublayer(device, case, sublayer):
    """seq attention: RoPE, padded QK-norm, folded PerDimScale, SDPA with the causal + leading-pad mask;
    var attention: the V = 1 shortcut (``uni``), the general masked path (``mv``: V = 8, tile padding
    filled; ``mv32``: V = 32); ffn; the whole MixingTransformer layer."""
    import ttnn

    tt = _model(device)
    g, flat, nfm, v, n, valid = _case(case)
    i = _layer_index(case)
    inputs = _inputs(tt, nfm, v, n)
    if case.endswith("_mv"):
        assert inputs.var_mask is not None, "per-variate padding: expected the variate mask path"
    source, target = {
        "seq_attention": ("x", "h1"),
        "var_attention": ("h1", "h2"),
        "ffn": ("h2", "out"),
        "layer": ("x", "out"),
    }[sublayer]
    x = _upload_rows(tt, flat(g[source]), inputs)
    if sublayer == "layer":
        out = tt.layer(i, x, inputs)
    else:
        out = getattr(tt, sublayer)(i, x, inputs) if sublayer != "ffn" else tt.ffn(i, x)
        ttnn.deallocate(x)
    got = tt.rows_to_torch(out, inputs)
    ttnn.deallocate(out)
    tt.deallocate_inputs(inputs)
    where = None if sublayer == "ffn" else valid
    check_pcc(f"{case} {sublayer} (valid tokens)", flat(g[target]), got, FLOOR, where=where)


@pytest.mark.parametrize("case", ["layer0_uni", "layer19_mv"])
def test_layer_performance_precision(device, case):
    """The ``performance`` policy (bfloat8_b weights, LoFi) on a full layer: guard floor 0.995."""
    tt = _model(device, "performance")
    g, flat, nfm, v, n, valid = _case(case)
    inputs = _inputs(tt, nfm, v, n)
    x = _upload_rows(tt, flat(g["x"]), inputs)
    out = tt.layer(_layer_index(case), x, inputs)
    got = tt.rows_to_torch(out, inputs)
    import ttnn

    ttnn.deallocate(out)
    tt.deallocate_inputs(inputs)
    _MODELS.clear()
    check_pcc(f"{case} layer, performance precision", flat(g["out"]), got, 0.995, where=valid)


def test_sdpa_padded_keys(device):
    """SDPA over V = 8 keys (one tile, 24 pad rows): exact without a mask; with a (B, 1, V, V) user mask the
    pad keys are NOT masked (a ttnn gap, printed here), so the model pads the variate axis to 32 and masks the
    pad keys itself (``var_mask_tensor``), which this checks against torch."""
    import ttnn

    from models.experimental.timesfm3_forecast.tt.common import upload
    from models.experimental.timesfm3_forecast.tt.model import MASK_FILL

    torch.manual_seed(0)
    b, v, h, d = 4, 8, 16, 96
    q, k, val = (torch.randn(b, h, v, d) * s for s in (0.05, 0.05, 1.0))  # small logits: pad keys would matter
    ref = torch.softmax(q @ k.transpose(-1, -2), -1) @ val
    dev = [upload(device, t) for t in (q, k, val)]

    def run(qkv, mask):
        out = ttnn.transformer.scaled_dot_product_attention(*qkv, attn_mask=mask, is_causal=False, scale=1.0)
        return ttnn.to_torch(out).float()[:, :, :v]

    err_none = float((run(dev, None) - ref).abs().max())
    err_logical = float((run(dev, upload(device, torch.zeros(b, 1, v, v))) - ref).abs().max())
    pad = lambda t: torch.nn.functional.pad(t, (0, 0, 0, 32 - v))  # noqa: E731
    mask = torch.zeros(b, 1, 32, 32)
    mask[..., v:] = MASK_FILL
    err_padded = float((run([upload(device, pad(t)) for t in (q, k, val)], upload(device, mask)) - ref).abs().max())
    print(
        f"[SDPA] V=8 max error: no mask {err_none:.4f}, logical user mask {err_logical:.4f}, padded + masked {err_padded:.4f}"
    )
    assert err_none < 0.05 and err_padded < 0.05


@pytest.mark.parametrize("variates", [2, 8, 32])
def test_var_attention_packed(device, variates):
    """R2-2: the packed variate attention (32 / V items per tile, block-diagonal mask) against the unpacked
    path on the same input, shared leading padding (no variate mask); random h1 at realistic scale."""
    import ttnn

    tt = _model(device)
    items, n = 64 // variates, 40
    nfm = torch.full((items, 1), 3, dtype=torch.long).expand(-1, variates).reshape(-1)
    inputs = _inputs(tt, nfm, variates, n)
    assert inputs.var_mask is None
    h1 = torch.randn(items * variates, n, 1280, generator=torch.Generator().manual_seed(5)) * 4.0
    outs = []
    saved = tt.pack_variates
    try:
        for packed in (False, True):
            tt.pack_variates = packed
            x = _upload_rows(tt, h1, inputs)
            out = tt.var_attention(0, x, inputs)
            outs.append(tt.rows_to_torch(out, inputs))
            for t in (x, out):
                ttnn.deallocate(t)
    finally:
        tt.pack_variates = saved
        tt.deallocate_inputs(inputs)
    valid = (torch.arange(n)[None, :] >= nfm[:, None])[..., None].expand(-1, -1, 1280)
    check_pcc(f"packed vs unpacked V={variates}", outs[0], outs[1], 0.9999, where=valid)
    assert (outs[0][valid] - outs[1][valid]).abs().max() < 0.25


def test_var_attention_packed_mask_control(device):
    """Positive control for ``test_var_attention_packed``: with the block-diagonal mask dropped (items attend
    across each other), the packed output must differ clearly from the unpacked one."""
    import ttnn

    from models.experimental.timesfm3_forecast.tt.common import upload

    tt = _model(device)
    variates, items, n = 8, 8, 40
    nfm = torch.zeros(items * variates, dtype=torch.long)
    inputs = _inputs(tt, nfm, variates, n)
    h1 = torch.randn(items * variates, n, 1280, generator=torch.Generator().manual_seed(5)) * 4.0
    outs, saved, saved_mask = [], tt.pack_variates, tt._block_mask
    try:
        for packed in (False, True):
            tt.pack_variates = packed
            if packed:
                tt._block_mask = lambda v, tiles: upload(tt.device, torch.zeros(tiles, 1, 32, 32))
            x = _upload_rows(tt, h1, inputs)
            out = tt.var_attention(0, x, inputs)
            outs.append(tt.rows_to_torch(out, inputs))
            for t in (x, out):
                ttnn.deallocate(t)
    finally:
        tt.pack_variates, tt._block_mask = saved, saved_mask
        tt.deallocate_inputs(inputs)
    diff = (outs[0] - outs[1]).abs().max().item()
    print(f"[control] packed without mask: max |diff| {diff:.4f}")
    assert diff > 1.0
