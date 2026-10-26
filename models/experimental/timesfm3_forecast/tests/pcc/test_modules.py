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


@pytest.mark.parametrize("causal", [False, True], ids=["keymask", "causal"])
@pytest.mark.parametrize("seq", [256, 512])
def test_sdpa_masked_chunk_cap(device, seq, causal):
    """PLAN F9 / v1.13: SDPA with a user mask at the model's chunk (capped at 128 when masked) against torch,
    with realistic logits (std ~3). keymask: masked leading keys and a masked key tail (no query row fully
    masked). causal (D6): causal + leading padding as the model builds seq_mask, so the leading query rows see
    only masked keys; those rows must be finite and the valid rows are compared. Prints the max |diff| between
    the capped run and a run forced to chunk 256; a positive control (the mask dropped on one key block) must
    fail the same check."""
    import ttnn

    from models.experimental.timesfm3_forecast.common.decomposition import HEAD_DIM_PAD, SDPA_SCALE
    from models.experimental.timesfm3_forecast.tt.common import upload
    from models.experimental.timesfm3_forecast.tt.model import MASK_FILL, sdpa_chunk

    torch.manual_seed(1)
    b, h, d = 2, 4, HEAD_DIM_PAD
    lead, tail = seq // 4 + 32, 32  # leading padding spans past one 128 key block; a padded key tail
    q, k, val = torch.randn(b, h, seq, d), torch.randn(b, h, seq, d), torch.randn(b, h, seq, d)
    q = q * (3.0 / (SDPA_SCALE * d**0.5))  # logit std ~3
    mask = torch.zeros(b, 1, seq, seq)
    mask[..., :lead] = MASK_FILL
    if causal:
        mask[..., torch.ones(seq, seq, dtype=torch.bool).triu(1)] = MASK_FILL
        rows = slice(lead, seq)  # rows < lead are fully masked (finite only); the rest are valid
    else:
        mask[..., seq - tail :] = MASK_FILL
        rows = slice(lead, seq - tail)  # valid query rows (masked rows are uniform over masked keys: not compared)
    ref = torch.softmax((q @ k.transpose(-1, -2)) * SDPA_SCALE + mask, -1) @ val
    dev = [upload(device, t) for t in (q, k, val)]

    def run(m, chunk):
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=chunk,
            k_chunk_size=chunk,
            exp_approx_mode=False,
        )
        out = ttnn.transformer.scaled_dot_product_attention(
            *dev, attn_mask=upload(device, m), is_causal=False, scale=SDPA_SCALE, program_config=cfg
        )
        got = ttnn.to_torch(out).float()
        assert torch.isfinite(got).all()
        return got[:, :, rows]

    chunk = sdpa_chunk(seq, masked=True)
    assert chunk == 128
    capped, forced = run(mask, chunk), run(mask, 256)
    broken_mask = mask.clone()
    broken_mask[..., 128:256] = 0.0  # positive control: one masked key block un-masked
    broken = run(broken_mask, chunk)
    want = ref[:, :, rows]
    err = lambda t: float((t - want).abs().max())  # noqa: E731
    print(
        f"[F9] S={seq}: max|err| capped {err(capped):.4f}, forced-256 {err(forced):.4f}, "
        f"max|capped - forced| {float((capped - forced).abs().max()):.4f}, positive control {err(broken):.4f}"
        f" ({'causal + leading pad' if causal else 'key mask'})"
    )
    tol = 0.5  # bf16 inputs at logit std 3 measure 0.12-0.20 (job 808); the positive control measures ~3.4
    assert err(capped) < tol
    assert err(broken) >= tol, "positive control passed: the check is too weak to judge the mask"


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


@pytest.mark.parametrize("variates", [2, 8])
def test_var_norm_fused(device, variates):
    """R4-2: the packed variate attention with the head split + QK norm fused (rope_qk_norm_qkv, identity RoPE)
    against the split + two rms_norm chain on the same input."""
    import ttnn

    tt = _model(device)
    items, n = 64 // variates, 40
    nfm = torch.full((items, 1), 3, dtype=torch.long).expand(-1, variates).reshape(-1)
    inputs = _inputs(tt, nfm, variates, n)
    h1 = torch.randn(items * variates, n, 1280, generator=torch.Generator().manual_seed(6)) * 4.0
    outs = []
    saved = tt.fuse_var_norm
    try:
        for fused in (False, True):
            tt.fuse_var_norm = fused
            x = _upload_rows(tt, h1, inputs)
            out = tt.var_attention(0, x, inputs)
            outs.append(tt.rows_to_torch(out, inputs))
            for t in (x, out):
                ttnn.deallocate(t)
    finally:
        tt.fuse_var_norm = saved
        tt.deallocate_inputs(inputs)
    from models.experimental.timesfm3_forecast.tests.golden import dataflow
    from models.experimental.timesfm3_forecast.tt.model import load_checkpoint_weights

    # D5: the tolerance is judged against the fp32 reference, which both device paths are measured against.
    masks = dataflow.Masks.build(nfm, variates, n)
    ref = dataflow.var_attention(h1, load_checkpoint_weights(layers=[0]).layers[0].var, masks, masks.token_valid)
    valid = (torch.arange(n)[None, :] >= nfm[:, None])[..., None].expand(-1, -1, 1280)
    err = lambda t: float((t[valid] - ref[valid]).abs().max())  # noqa: E731
    chain, fused = outs
    broken = fused.clone()
    broken[: items * variates // 2] = h1[: items * variates // 2]  # positive control: half the rows skip the sublayer
    diff = float((chain[valid] - fused[valid]).abs().max())
    print(
        f"[var norm fused] V={variates} max|err| vs fp32: chain {err(chain):.5f}, fused {err(fused):.5f}, "
        f"positive control {err(broken):.5f}; max|fused - chain| {diff:.5f}"
    )
    check_pcc(f"fused vs chain V={variates}", chain, fused, 0.99999, where=valid)
    # The unfused chain is itself ~0.40 off fp32 (job 811: bf16 at |h| >= 16, one ulp 0.125), so 0.25 fused vs
    # chain (2 ulp; 0.125 measured, jobs 777 and 811) is within the chain's own error; the fused path may add at
    # most one ulp of error over the chain against fp32.
    tol = 0.25
    assert diff < tol and err(fused) <= err(chain) + 0.125
    assert err(broken) > err(chain) + 0.125, "positive control passed: the check is too loose to judge the kernel"


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


@pytest.mark.parametrize("variates", [2, 8])
@pytest.mark.parametrize("control", [False, True])
def test_var_attention_packed_padded(device, variates, control):
    """R3-1: per-variate leading padding (a variate mask) on the packed path, against the unpacked path. The
    positive control drops the key padding from the packed mask (block-diagonal only) and must differ clearly."""
    import ttnn

    tt = _model(device)
    items, n = 64 // variates, 40
    nfm = (torch.arange(items * variates) * 7) % 23  # every item's variates differ in padding
    inputs = _inputs(tt, nfm, variates, n)
    assert inputs.var_mask is not None and inputs.packed_var_mask is not None
    h1 = torch.randn(items * variates, n, 1280, generator=torch.Generator().manual_seed(5)) * 4.0
    outs, saved = [], tt.pack_variates
    packed_mask = inputs.packed_var_mask
    if control:
        inputs.packed_var_mask = tt._block_mask(variates, inputs.padded_seq * items * variates // 32)
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
        inputs.packed_var_mask = packed_mask
        tt.deallocate_inputs(inputs)
    # A query whose own slot is padding is never read by a valid token; with all keys masked it is undefined.
    valid = (torch.arange(n)[None, :] >= nfm[:, None])[..., None].expand(-1, -1, 1280)
    diff = (outs[0][valid] - outs[1][valid]).abs().max().item()
    print(f"[packed padded] V={variates} control={control}: max |diff| {diff:.4f}")
    if control:
        assert diff > 1.0
    else:
        check_pcc(f"packed vs unpacked, padded V={variates}", outs[0], outs[1], 0.9999, where=valid)
        assert diff < 0.25


@pytest.mark.parametrize("rows, seq", [(4, 64), (3, 96)])
@pytest.mark.parametrize("control", [False, True])
def test_rope_qk_norm(device, rows, seq, control):
    """R3-2: the fused RoPE + QK-norm op alone against an fp32 torch reference, on real layer-0 gains, at
    realistic scale, with an uneven core split (3 x 16 x 3 x 2 units). The control swaps cos and sin, which
    must fail the same check."""
    import ttnn

    from models.experimental.timesfm3_forecast.common.decomposition import HEAD_DIM, HEAD_DIM_PAD, rope_tables
    from models.experimental.timesfm3_forecast.tt.common import upload
    from models.experimental.timesfm3_forecast.tt.model import NUM_HEADS, QK_EPS
    from models.experimental.timesfm3_forecast.tt.ops.rope_qk_norm.op import rope_qk_norm

    tt = _model(device)
    w = tt.layers[0]["seq"]
    gen = torch.Generator().manual_seed(11)
    cos, sin = rope_tables(seq, torch.float32)
    xs = []
    for scale in (3.0, 6.0):
        x = torch.randn(rows, NUM_HEADS, seq, HEAD_DIM_PAD, generator=gen) * scale
        x[..., HEAD_DIM:] = 0
        xs.append(x.bfloat16().float())
    gains = [ttnn.to_torch(w[n]).float()[:1] for n in ("q_gain", "k_gain")]
    dev_cs = [upload(device, t.reshape(1, 1, seq, HEAD_DIM_PAD)) for t in ((sin, cos) if control else (cos, sin))]
    q, k = (upload(device, x) for x in xs)
    outs = rope_qk_norm(q, k, *dev_cs, tt.rope_trans, w["q_gain"], w["k_gain"], epsilon=QK_EPS)
    got = [ttnn.to_torch(t).float() for t in outs]
    for t in (q, k, *dev_cs, *outs):
        ttnn.deallocate(t)
    ok = True
    for name, x, gain, out in zip("qk", xs, gains, got):
        rot = torch.zeros_like(x)
        rot[..., 0::2], rot[..., 1::2] = -x[..., 1::2], x[..., 0::2]
        y = x * cos + rot * sin
        ref = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + QK_EPS) * gain
        diff = (out - ref).abs().max().item()
        print(f"[rope_qk_norm] {name} R={rows} S={seq} control={control}: max |diff| {diff:.4f}")
        pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
        print(f"[rope_qk_norm] {name}: pcc {pcc:.6f}, max |ref| {ref.abs().max().item():.2f}")
        ok &= diff < 0.02 * ref.abs().max().item() and pcc >= 0.99999
    assert ok != control


@pytest.mark.parametrize("rows, seq", [(4, 64), (3, 96)])
@pytest.mark.parametrize("control", [False, True])
def test_rope_qk_norm_qkv(device, rows, seq, control):
    """R3-5: the head split folded into the fused op, alone: (qn, kn, v) from the QKV output must equal
    split_query_key_value_and_split_heads followed by ``rope_qk_norm`` (same kernel math, so bitwise). The control
    compares against the split of a QKV whose q heads are rolled by one, which must differ."""
    import ttnn

    from models.experimental.timesfm3_forecast.common.decomposition import HEAD_DIM_PAD, rope_tables
    from models.experimental.timesfm3_forecast.tt.common import upload
    from models.experimental.timesfm3_forecast.tt.model import NUM_HEADS, QK_EPS
    from models.experimental.timesfm3_forecast.tt.ops.rope_qk_norm.op import rope_qk_norm, rope_qk_norm_qkv

    tt = _model(device)
    w = tt.layers[0]["seq"]
    gen = torch.Generator().manual_seed(12)
    qkv = torch.randn(1, 1, rows * seq, 3 * NUM_HEADS * HEAD_DIM_PAD, generator=gen) * 4.0
    ref_qkv = qkv.clone()
    if control:
        q_part = ref_qkv[..., : NUM_HEADS * HEAD_DIM_PAD].reshape(1, 1, rows * seq, NUM_HEADS, HEAD_DIM_PAD)
        ref_qkv[..., : NUM_HEADS * HEAD_DIM_PAD] = q_part.roll(1, dims=3).reshape(1, 1, rows * seq, -1)
    cos, sin = (upload(device, t.reshape(1, 1, seq, HEAD_DIM_PAD)) for t in rope_tables(seq, torch.float32))
    dev_qkv, dev_ref = upload(device, qkv), upload(device, ref_qkv)
    got = rope_qk_norm_qkv(dev_qkv, rows, NUM_HEADS, cos, sin, tt.rope_trans, w["q_gain"], w["k_gain"], epsilon=QK_EPS)
    q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
        ttnn.reshape(dev_ref, (rows, seq, dev_ref.shape[-1])), num_heads=NUM_HEADS, transpose_key=False
    )
    qn, kn = rope_qk_norm(q, k, cos, sin, tt.rope_trans, w["q_gain"], w["k_gain"], epsilon=QK_EPS)
    pairs = [(ttnn.to_torch(a).float(), ttnn.to_torch(b).float()) for a, b in zip(got, (qn, kn, v))]
    for t in (*got, q, k, v, qn, kn, cos, sin, dev_qkv, dev_ref):
        ttnn.deallocate(t)
    diffs = [(a - b).abs().max().item() for a, b in pairs]
    print(f"[rope_qk_norm_qkv] R={rows} S={seq} control={control}: max |diff| q, k, v {diffs}")
    if control:
        assert diffs[0] > 0.5 and diffs[1] == 0 and diffs[2] == 0
    else:
        assert diffs == [0.0, 0.0, 0.0]


@pytest.mark.parametrize("rows", [16384, 65536 + 32 * 3])
def test_post_add_norm(device, rows):
    """R2-4: the fused post-norm + residual + pre-norm kernel against the 3-op chain it replaces (the op output
    alone), on real layer-0 post gains, at the fused threshold and at a row count that splits unevenly."""
    import ttnn

    from models.experimental.timesfm3_forecast.tt.common import rms_norm, upload
    from models.experimental.timesfm3_forecast.tt.model import EPS
    from models.experimental.timesfm3_forecast.tt.ops.post_add_norm.op import post_add_norm

    tt = _model(device)
    gain = tt.layers[0]["seq"]["post"]
    gen = torch.Generator().manual_seed(11)
    o = upload(tt.device, torch.randn(1, 1, rows, 1280, generator=gen) * 3)
    r = upload(tt.device, torch.randn(1, 1, rows, 1280, generator=gen) * 20)
    y = rms_norm(o, EPS, tt.precision, gain)
    h_ref = ttnn.add(r, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    xn_ref = rms_norm(h_ref, EPS, tt.precision)
    h, xn = post_add_norm(o, r, gain, epsilon=EPS)
    got = [ttnn.to_torch(t).float() for t in (h, xn)]
    ref = [ttnn.to_torch(t).float() for t in (h_ref, xn_ref)]
    o_t, r_t = ttnn.to_torch(o).float(), ttnn.to_torch(r).float()
    scale = r_t.abs() + ttnn.to_torch(y).float().abs()
    for t in (o, r, y, h_ref, xn_ref, h, xn):
        ttnn.deallocate(t)
    for name, a, b in zip(("h", "xn"), ref, got):
        check_pcc(f"post_add_norm {name} @ {rows}", a, b, 0.99999)
    # h: within one bf16 rounding of its operands (|res| + |y|; h itself can cancel to ~0); xn: a few ulp.
    within = (
        lambda h, x: ((h - ref[0]).abs() <= scale * 2**-7 + 1e-3).all() and (x - ref[1]).abs().max() <= 0.0625
    )  # noqa: E731
    assert within(*got)
    # Positive control: the post norm skipped on one 32-row tile (h = r + o there) must fail the same check.
    bad_h, bad_x = got[0].clone(), got[1].clone()
    bad_h[..., :32, :] = r_t[..., :32, :] + o_t[..., :32, :]
    excess = float(((bad_h - ref[0]).abs() / (scale * 2**-7 + 1e-3)).max())
    print(f"[post_add_norm] R={rows} positive control: max h error / allowed {excess:.2f} (must exceed 1)")
    assert not within(bad_h, bad_x), "positive control passed: the tolerance is too loose to judge the kernel"
