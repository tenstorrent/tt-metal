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
import math
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


def _stream_scale(model, layer: int | None) -> float:
    """``c`` of the folded residual stream at the input of ``layer`` (None: the head); 1 when unfolded."""
    w = model.weights
    eps = w.head_eps if layer is None else w.layers[layer].eps_attn
    return 1.0 if eps is None else math.sqrt(w.config.norm_eps / eps)


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

        # Time layers, each from the golden input, scaled to the folded residual stream y = x / c (R2-1).
        layer_in = expected
        for li, kind in enumerate(kinds):
            if kind == "variate":
                continue
            x = _flat(device, layer_in / _stream_scale(model, li), sp)
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
        x = _flat(device, cap[f"layer_{vi - 1}"][:, start : start + count] / _stream_scale(model, vi))
        y = model.run_layer(vi, x, inputs)
        golden = _rows_from_variate_layout(cap[f"layer_{vi}"], host.batch, host.variates)[:, start : start + count]
        results[f"layer_{vi}"] = pcc(_back(y, r, count), golden)
        ttnn.deallocate(y)

        # out_norm + head.
        x = _flat(device, golden / _stream_scale(model, None))
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


@pytest.mark.parametrize("variates,masked", [(8, True), (8, False), (4, True)], ids=["v8-mask", "v8-nomask", "v4-mask"])
def test_variate_attention(device, variates, masked):
    """The variate attention core (``_across_variates``) against torch SDPA, at V not tile-aligned. A user mask
    must hide the key axis's tile padding (PLAN section 9, F1): the device pads V to 32 and masks the pad keys."""
    import ttnn

    from models.experimental.toto2_forecast.tests.golden.folded import FoldedToto2
    from models.experimental.toto2_forecast.tt.attention import VariateInputs
    from models.experimental.toto2_forecast.tt.common import upload
    from models.experimental.toto2_forecast.tt.model import _additive, _pad_variate_mask

    model = _model(device, "22m", "accuracy")
    attn = model._layers[-1][1]
    lw = model.weights.layers[-1]
    d, h = model.config.d_model, model.config.num_heads
    b, n = 4, 2
    torch.manual_seed(0)
    xn = torch.randn(b * variates * n, d)
    mask = None
    if masked:
        ids = torch.zeros(b, variates, dtype=torch.long)
        ids[1::2, variates // 2 :] = 1
        ids[2, 0] = -1  # a fully missing patch's group
        slots = ids[:, None, :].expand(b, n, variates)
        mask = (slots[..., :, None] == slots[..., None, :]).reshape(b * n, 1, variates, variates)
    inputs = VariateInputs(
        batch=b,
        variates=variates,
        slots=n,
        mask=None if mask is None else upload(device, _additive(_pad_variate_mask(mask))),
    )
    x_dev = upload(device, xn.reshape(1, 1, -1, d))
    got = ttnn.to_torch(attn._across_variates(x_dev, inputs)).float().reshape(b, variates, n, d)

    qkv = torch.nn.functional.linear(xn, lw.w_qkv, lw.b_qkv).reshape(b, variates, n, 3 * d).transpose(1, 2)
    qkv = qkv.reshape(b * n, variates, 3 * d)
    q, k, v = (FoldedToto2._heads(FoldedToto2(model.weights), qkv[..., i * d : (i + 1) * d]) for i in range(3))
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / 64)
    ref = ref.transpose(1, 2).reshape(b, n, variates, d).transpose(1, 2)
    # PCC is blind to a uniform shrink (pad keys in the softmax scale every output down), so gate the relative error.
    rel = ((got - ref).norm() / ref.norm()).item()
    p = pcc(got, ref)
    print(f"\n[VARIATE] V={variates} mask={masked}: pcc {p:.6f} rel err {rel:.2e} |ref| max {ref.abs().max():.1f}")
    assert p >= 0.9995 and rel < 0.02, (p, rel)


@pytest.mark.parametrize("concat", [False, True], ids=["heads", "concat"])
@pytest.mark.parametrize("s", [256, 768], ids=["s256", "s768"])
def test_masked_time_sdpa_256(device, s, concat):
    """F9 (PLAN section 9 v1.10/v1.13, D2 condition 2, D3 condition 2): the time-attention SDPA call with a user mask
    at a padded length of 256 (and t0-gamma's repro length 768), against torch, on the attention output alone (no
    residual). Logits have a std of about 2.5. The mask is built like the model's: ``_pad_time_mask`` over
    ``s - 40`` real tokens (a masked pad-key tail), a two-group packing that hides whole 128-key blocks, and real
    query rows with every key masked (excluded from the comparison: torch gives NaN there; the model drops them).

    Three runs: the capped chunk (must pass), a negative control forced to chunk 256 (reported: max |diff| against
    the capped run and whether it fails), and a positive control with the mask dropped on one 128-key block (must
    fail, so the comparison can fail). ``concat`` (D6 condition 3): the model's call, ``output_concat_heads=True``
    (R1-6), compared against torch with the heads concatenated."""
    import ttnn

    from models.experimental.toto2_forecast.tt.attention import sdpa_chunk
    from models.experimental.toto2_forecast.tt.common import Toto2Precision, upload
    from models.experimental.toto2_forecast.tt.model import _additive, _pad_time_mask

    h, dh, real = 4, 64, s - 40
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, h, s, dh) for _ in range(3))
    q = q * 20.0  # logits q.k / dh: std 20 * 8 / 64 = 2.5 (D3: >= 2)
    group = torch.arange(real) // 128
    causal = torch.ones(real, real, dtype=torch.bool).tril()
    packed = (group[:, None] == group[None, :]) & causal
    dead = torch.zeros(real, dtype=torch.bool)
    dead[5:9] = True  # real query rows that see no key at all
    packed[dead] = False
    mask = _pad_time_mask(torch.stack([causal, packed]), s)[:, None]  # (2, 1, s, s)
    live = torch.ones(2, 1, s, 1, dtype=torch.bool)
    live[1, 0, :real][dead] = False
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0 / dh)
    if concat:
        ref = ref.transpose(1, 2).reshape(2, 1, s, h * dh)
    p = Toto2Precision.named("accuracy")
    dev = [upload(device, t) for t in (q, k, v)]

    def run(chunk, m):
        out = ttnn.transformer.scaled_dot_product_attention(
            *dev,
            attn_mask=upload(device, _additive(m)),
            is_causal=False,
            scale=1.0 / dh,
            program_config=ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
                q_chunk_size=chunk,
                k_chunk_size=chunk,
                exp_approx_mode=False,
            ),
            compute_kernel_config=p.sdpa_config(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_concat_heads=concat,
        )
        return ttnn.to_torch(out).float()

    def score(got):
        g, r = got.masked_select(live), ref.masked_select(live)
        rel = ((g - r).norm() / r.norm()).item()
        pc = pcc(g, r)
        return pc, rel, pc >= 0.9995 and rel < 0.02

    chunk = sdpa_chunk(s, masked=True)
    capped = run(chunk, mask)
    forced = run(256, mask)
    leaky = mask.clone()
    leaky[1, 0, :real, 128:256] = True  # positive control: one 128-key block unmasked for the real queries
    leak = run(chunk, leaky)
    res = {name: score(t) for name, t in (("capped", capped), ("forced256", forced), ("positive", leak))}
    diff = (capped - forced).masked_select(live).abs().max().item()
    dead_finite = bool(torch.isfinite(capped[1, :, :real][:, dead]).all())
    tag = f"s {s} {'concat' if concat else 'heads'}"
    for name, (pc, rel, ok) in res.items():
        print(f"\n[F9] {tag} {name}: pcc {pc:.6f} rel err {rel:.2e} pass {ok}")
    print(f"[F9] {tag}: max |capped - forced256| {diff:.3e}; dead query rows finite on device: {dead_finite}")
    assert res["capped"][2], res["capped"]
    assert not res["positive"][2], f"positive control passed: the comparison cannot fail {res['positive']}"
