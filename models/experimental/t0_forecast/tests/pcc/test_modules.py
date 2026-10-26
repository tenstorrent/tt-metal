# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Module PCC of TTNN t0 against the fp32 reference goldens (real weights).

    patch encoder     random tokens with PAD / MISSING cells and a historical row (type embedding path)
    time layer        random activations, causal and PAD-masked; q / k after QK-norm + RoPE (S = 66, 288)
    group layer       G = 1 (the V-only shortcut), G = 4, and G = 4 with PAD patches (masked path)
    MLP, head         random activations (the head also checks the raw decoder output before softplus)
    out_norm, decoder the single-series golden's layer-23 output: out_norm >= 0.999, decoder output >= 0.998
    every layer       the single-series golden: layer i on the reference output of layer i - 1

Floor: 0.999 for every module (the contract's module floor); PAD tokens are excluded (the reference returns
zero attention for them, the device lets them read themselves; no real token reads either).
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.t0_forecast.tests.golden.dataflow import Dataflow
from models.experimental.t0_forecast.tests.helpers import (
    clean_fields,
    download_rows,
    golden,
    module_inputs,
    pcc,
    upload_rows,
    weights,
)
from models.experimental.t0_forecast.tt import host
from models.experimental.t0_forecast.tt.common import TtT0Precision, pad_to_tile, upload
from models.experimental.t0_forecast.tt.model import TtT0

FLOOR = 0.999
DECODER_FLOOR = 0.998  # the contract's decoder output (pre-head) floor


def _model(device, layers, precision="accuracy"):
    return TtT0(device, weights().reduced(layers), TtT0Precision.named(precision))


def _run_layer(tt: TtT0, index: int, x: torch.Tensor, prepared: host.T0HostInputs):
    """Layer ``index`` of the reduced model on ``x`` ``[R, S, d]`` -> (after attention, after MLP), host fp32."""
    import ttnn

    inputs = tt.upload_inputs(prepared)
    attention, mlp = tt.layers[index]
    x_dev = upload_rows(tt.device, x)
    y = attention(x_dev, inputs.attention)
    z = mlp(y)
    r, s = x.shape[:2]
    out = download_rows(y, r, s), download_rows(z, r, s)
    for t in (x_dev, y, z):
        ttnn.deallocate(t)
    tt.deallocate_inputs(inputs)
    return out


def test_patch_encoder(device):
    import ttnn

    pe = golden("modules.pt")["patch_encoder"]
    ps = 32
    t_index = (torch.arange(ps, dtype=torch.float32) / ps).expand(*pe["values"].shape)
    tokens = torch.cat([pe["values"], t_index, (pe["mask"] == host.VALID).float()], dim=-1)
    groups, vtype, mask = clean_fields(4, 66)
    prepared = module_inputs(torch.zeros(4, 66, 1024), groups, vtype, mask)
    patch_type = host.reduce_patch_metadata(pe["variate_type"], pe["mask"]).clamp(min=0)
    from dataclasses import replace

    prepared = replace(prepared, tokens=tokens, patch_type=patch_type)
    assert bool((patch_type != 0).any()), "the historical row must take the type-embedding path"
    tt = _model(device, [])
    inputs = tt.upload_inputs(prepared)
    x = tt.embed(inputs)
    got = download_rows(x, 4, 66)
    ttnn.deallocate(x)
    tt.deallocate_inputs(inputs)
    value = pcc(pe["out"], got)
    print(f"\n[pcc] patch_encoder {value:.6f}")
    assert value >= FLOOR


@pytest.mark.parametrize("case", ["clean", "padded"])
def test_time_layer(device, case):
    m = golden("modules.pt")[f"time_layer_{case}"]
    prepared = module_inputs(m["x"], m["group_ids"], m["variate_type"], m["mask"])
    assert (prepared.time_mask is None) == (case == "clean")
    attn, out = _run_layer(_model(device, [0]), 0, m["x"], prepared)
    real = prepared.real_tokens
    a, o = pcc(m["attention"][real], attn[real]), pcc(m["out"][real], out[real])
    print(f"\n[pcc] time_layer_{case}: attention {a:.6f}, layer {o:.6f}")
    assert min(a, o) >= FLOOR


@pytest.mark.parametrize("patches", [256, 512, 768])
def test_time_layer_masked_long(device, patches):
    """D4 item 5 / D5 item 2: a PAD-masked time layer at a whole multiple of 256 patches (e.g. 8160 context + 32
    horizon). SDPA with a user mask must run at chunk 128 (chunk 256 was wrong, jobs 582, 584). Row 1 has its leading
    half PAD (trimmed history), so whole 128- and 256-key blocks are masked. Scored on the attention output alone
    (the device layer output minus its input; inputs are bf16-exact) against the fp32 dataflow, real tokens only.
    ``tests/perf/neg_chunk256.py`` reruns this with the cap forced to 256 as the negative control."""
    g = torch.Generator().manual_seed(256)
    x = torch.randn(2, patches, 1024, generator=g).bfloat16().float()
    groups, vtype, mask = clean_fields(2, patches)
    half = patches // 2
    mask[1, :half] = host.PAD
    groups[1, :half] = -1
    vtype[1, :half] = -1
    prepared = module_inputs(x, groups, vtype, mask)
    assert prepared.time_mask is not None
    ref = Dataflow(weights()).time_attention(weights().layers[0].attention, x, prepared)
    attn, _ = _run_layer(_model(device, [0]), 0, x, prepared)
    real = prepared.real_tokens
    value = pcc(ref[real], (attn - x)[real])
    print(f"\n[pcc] time_layer_masked_{patches} (attention alone vs fp32 dataflow): {value:.6f}")
    assert value >= FLOOR


@pytest.mark.parametrize("case", ["g1", "g4", "g4_padded"])
def test_group_layer(device, case):
    m = golden("modules.pt")[f"group_layer_{case}"]
    prepared = module_inputs(m["x"], m["group_ids"], m["variate_type"], m["mask"])
    assert prepared.group_size == (1 if case == "g1" else 4)
    assert (prepared.group_mask is not None) == (case == "g4_padded")
    attn, out = _run_layer(_model(device, [2]), 0, m["x"], prepared)
    real = prepared.real_tokens
    a, o = pcc(m["attention"][real], attn[real]), pcc(m["out"][real], out[real])
    print(f"\n[pcc] group_layer_{case}: attention {a:.6f}, layer {o:.6f}")
    assert min(a, o) >= FLOOR


def test_group_layer_g8_masked(device):
    """F1 (PLAN 9 v1.2): with a user mask and G = 8 keys (not a multiple of 32), the tile-padding keys must stay
    masked. Two series of 8 variates; rows 1 and 10 have three all-PAD patches (the group mask path). Device against
    the fp32 dataflow (which matches the reference at 1e-7 on the masked G = 4 golden), real tokens only."""
    g = torch.Generator().manual_seed(8)
    x = 4.0 * torch.randn(16, 66, 1024, generator=g)
    groups, vtype, mask = clean_fields(16, 66, group_size=8)
    for r in (1, 10):
        mask[r, :3] = host.PAD
        groups[r, :3] = -1
        vtype[r, :3] = -1
    prepared = module_inputs(x, groups, vtype, mask)
    assert prepared.group_size == 8 and prepared.group_mask is not None
    flow = Dataflow(weights())
    a = weights().layers[2].attention
    ref = x + flow.group_attention(a, x, prepared, prepared.group_mask)
    attn, _ = _run_layer(_model(device, [2]), 0, x, prepared)
    real = prepared.real_tokens
    value = pcc(ref[real], attn[real])
    print(f"\n[pcc] group_layer_g8_masked (vs fp32 dataflow): attention {value:.6f}")
    assert value >= FLOOR


def test_mlp(device):
    import ttnn

    m = golden("modules.pt")["mlp"]
    tt = _model(device, [0])
    x_dev = upload_rows(device, m["x"])
    y = tt.layers[0][1](x_dev)
    got = download_rows(y, 4, 66) - m["x"]  # the module returns x + mlp(x); the golden is mlp(x)
    ttnn.deallocate(y)
    value = pcc(m["out"], got)
    print(f"\n[pcc] mlp {value:.6f}")
    assert value >= FLOOR


def test_head(device):
    import ttnn

    m = golden("modules.pt")["head"]
    tt = _model(device, [])
    x = m["x"].reshape(1, 1, 4 * 66, -1)
    rows = torch.zeros(1, 1, pad_to_tile(x.shape[2]), x.shape[3])
    rows[:, :, : x.shape[2]] = x
    out = tt.head(upload(device, rows))
    got = ttnn.to_torch(out).float()[0, 0, : x.shape[2]].reshape(4, 66, 32, 21)
    ttnn.deallocate(out)
    value = pcc(m["out"], got)
    raw_dev = tt.decode(upload(device, rows))
    raw = ttnn.to_torch(raw_dev).float()[0, 0, : x.shape[2]].reshape(m["decoder"].shape)
    ttnn.deallocate(raw_dev)
    raw_value = pcc(m["decoder"], raw)
    print(f"\n[pcc] decoder raw (pre-softplus) {raw_value:.6f}, decoder + monotone head {value:.6f}")
    assert raw_value >= DECODER_FLOOR and value >= FLOOR
    assert bool((got[..., 1:] >= got[..., :-1]).all()), "quantiles must be monotone"


def test_out_norm_and_decoder(device):
    """out_norm and the decoder output (pre-head, the raw values before softplus) on the reference's layer-23
    output of the single-series golden: out_norm >= 0.999, decoder >= 0.998 (the contract's pre-head floor)."""
    import ttnn

    from models.experimental.t0_forecast.tt.common import rms_norm

    g = golden("single.pt")
    tt = _model(device, [])
    x = upload_rows(device, g["layer_23"])
    normed = rms_norm(x, tt._out_norm, 1e-8, tt.precision)
    p_norm = pcc(g["out_norm"], download_rows(normed, 1, 66))
    ttnn.deallocate(normed)
    raw_dev = tt.decode(x)
    raw = download_rows(raw_dev, 1, 66)
    ttnn.deallocate(raw_dev)
    p_dec = pcc(g["decoder"], raw)
    p_dec_ro = pcc(g["decoder"][:, 63:65], raw[:, 63:65])
    print(f"\n[pcc] out_norm {p_norm:.6f}; decoder output {p_dec:.6f} (read-out patches {p_dec_ro:.6f})")
    assert p_norm >= FLOOR and min(p_dec, p_dec_ro) >= DECODER_FLOOR


@pytest.mark.parametrize("name,layer", [("single.pt", 0), ("single.pt", 22), ("single_long.pt", 1)])
def test_qk_after_norm_and_rope(device, name, layer):
    """q and k of one time layer after QK-norm and RoPE + xPos, against the fp32 dataflow (which matches the
    reference at 1e-7) on the reference's attention input."""
    import ttnn

    g = golden(name)
    w = weights()
    xn = g[f"attention_{layer}.in"]  # [1, S, d], already RMS-normed by the reference block
    s = xn.shape[1]
    flow = Dataflow(w)
    a = w.layers[layer].attention
    q_ref, k_ref, _ = flow.qkv(a, xn)
    cos_q, sin_q, cos_k, sin_k = host.rope_tables(s, w.config)
    q_ref, k_ref = flow.rope(q_ref, cos_q, sin_q), flow.rope(k_ref, cos_k, sin_k)

    tt = _model(device, [layer])
    att = tt.layers[0][0]
    groups, vtype, mask = clean_fields(1, s)
    inputs = tt.upload_inputs(module_inputs(xn, groups, vtype, mask))
    sp = pad_to_tile(s)
    from models.experimental.t0_forecast.tt.common import linear

    qkv = linear(upload_rows(device, xn), att._wqkv, att.precision, role="qkv", bias=att._bqkv)
    q, k, v = att._heads(ttnn.reshape(qkv, (1, 1, sp, 3 * att.d)), inputs.attention.rope)
    got_q, got_k = (ttnn.to_torch(t).float()[:, :, :s] for t in (q, k))
    tt.deallocate_inputs(inputs)
    pq, pk = pcc(q_ref, got_q), pcc(k_ref, got_k)
    print(f"\n[pcc] {name} layer {layer} (S = {s}): q {pq:.6f}, k {pk:.6f}")
    assert min(pq, pk) >= FLOOR


@pytest.mark.parametrize("layer", list(range(24)))
def test_layer_on_reference_activations(device, layer):
    """Layer i on the reference output of layer i - 1 (patch encoder for i = 0), single series, S = 66."""
    g = golden("single.pt")
    x = g["patch_encoder"] if layer == 0 else g[f"layer_{layer - 1}"]
    groups, vtype, mask = clean_fields(1, x.shape[1])
    _, out = _run_layer(_model(device, [layer]), 0, x, module_inputs(x, groups, vtype, mask))
    value = pcc(g[f"layer_{layer}"], out)
    print(f"\n[pcc] layer {layer} ({weights().layers[layer].kind}) {value:.6f}")
    assert value >= FLOOR
