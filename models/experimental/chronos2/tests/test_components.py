# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
#
# Component checks for the host-side parts of the Chronos-2 port (CPU only).
# Checkpoint-dependent tests need CHRONOS2_CHECKPOINT; tests that compare against
# the official implementation also need the chronos-forecasting package
# (pip install chronos-forecasting) and skip without it.

import os
from pathlib import Path

import numpy as np
import pytest

from models.experimental.chronos2.tt.model_config import Chronos2Config
from models.experimental.chronos2.tt.preprocess import Preprocessor
from models.experimental.chronos2.tt.weights import fuse_group_attention, load_weights_fp32

WEIGHTS = os.environ.get("CHRONOS2_CHECKPOINT")

needs_weights = pytest.mark.skipif(
    not WEIGHTS or not Path(WEIGHTS).exists(), reason="set CHRONOS2_CHECKPOINT to a Chronos-2 checkpoint directory"
)

try:
    import chronos.chronos2  # noqa: F401

    HAS_CHRONOS = True
    CHRONOS_UNAVAILABLE = ""
except ImportError as e:  # pragma: no cover - environment-dependent
    HAS_CHRONOS = False
    CHRONOS_UNAVAILABLE = f"chronos-forecasting is not installed ({e})"

needs_chronos = pytest.mark.skipif(not HAS_CHRONOS, reason=CHRONOS_UNAVAILABLE or "chronos unavailable")


def _reference_model():
    """Load the official Chronos2Model in FP32 on CPU (reference implementation)."""
    import torch
    from chronos.chronos2 import Chronos2Model

    try:  # transformers v5 renamed torch_dtype -> dtype
        model = Chronos2Model.from_pretrained(WEIGHTS, dtype=torch.float32)
    except TypeError:
        model = Chronos2Model.from_pretrained(WEIGHTS, torch_dtype=torch.float32)
    model.eval()
    return model


def _nrmse(a, b):
    d = np.sqrt(np.mean((a - b) ** 2))
    r = np.sqrt(np.mean(b**2))
    return float(d / r) if r > 0 else float(d)


# --------------------------------------------------------------------------- #
# 1. Scaling edge cases (no fixture required)                                 #
# --------------------------------------------------------------------------- #


def _cfg(**over):
    base = dict(
        d_model=8,
        d_ff=16,
        d_kv=4,
        num_heads=2,
        num_layers=1,
        layer_norm_epsilon=1e-6,
        dense_act_fn="relu",
        dropout_rate=0.0,
        initializer_factor=1.0,
        rope_theta=10000.0,
        context_length=64,
        input_patch_size=16,
        input_patch_stride=16,
        output_patch_size=16,
        max_output_patches=4,
        use_reg_token=True,
        use_arcsinh=True,
        quantiles=tuple(0.05 * i for i in range(21)),
        time_encoding_scale=64,
    )
    base.update(over)
    return Chronos2Config(**base)


def test_scale_constant_series_uses_eps():
    prep = Preprocessor(_cfg())
    x = np.full((1, 32), 7.0, np.float32)
    loc, scale = prep.fit_scale(x)
    assert loc[0, 0] == 7.0 and scale[0, 0] == np.float32(1e-5)
    z = prep.apply_scale(x, loc, scale)
    assert np.all(np.isfinite(z))
    y = prep.unscale(z, loc, scale)
    np.testing.assert_allclose(y, x, rtol=1e-3, atol=1e-3)


def test_scale_all_masked_row_falls_back():
    prep = Preprocessor(_cfg())
    x = np.full((2, 32), np.nan, np.float32)
    x[1] = np.linspace(-1, 1, 32)
    loc, scale = prep.fit_scale(x)
    assert loc[0, 0] == 0.0 and scale[0, 0] == 1.0
    assert scale[1, 0] > 0


def test_prepare_shapes_and_masks():
    cfg = _cfg(context_length=128)  # > 65 so no truncation of the boundary input
    prep = Preprocessor(cfg)
    values = np.arange(65, dtype=np.float32).reshape(1, 65)  # uneven -> 5 patches
    mask = np.ones((1, 65), np.float32)
    mask[:, :15] = 0.0  # only real point of patch 0 (original index 0) is masked
    out = prep.prepare(values, mask, num_output_patches=4)
    assert out["ctx_features"].shape == (1, 5, 48)  # ceil(65/16) = 5
    assert out["ctx_features"].dtype == np.float32
    assert out["attn_mask"].shape == (1, 5 + 1 + 4)  # ctx + REG + future
    assert out["attn_mask"][0, :5].tolist() == [0, 1, 1, 1, 1]  # patch0 fully masked
    # masked cells must be zeroed in the value channel of patch 0
    assert out["ctx_features"][0, 0, 16:32].sum() == 0.0
    assert out["ctx_features"][0, 0, 32:].sum() == 0.0  # mask channel all zero too
    assert out["fut_features"].shape == (1, 4, 48)
    # future time encoding goes 0..63 / scale; ctx ends at -1/scale
    assert out["fut_features"][0, 0, 0] == 0.0
    assert out["ctx_features"][0, -1, 15] == np.float32(-1.0 / 64.0)


# --------------------------------------------------------------------------- #
# 2. Checkpoint-dependent parity vs chronos-forecasting                       #
# --------------------------------------------------------------------------- #


@needs_weights
def test_config_loads_from_checkpoint():
    cfg = Chronos2Config.from_json(Path(WEIGHTS) / "config.json")
    assert cfg.input_patch_size == cfg.output_patch_size
    assert cfg.num_quantiles == len(cfg.quantiles) == 21
    assert cfg.input_feature_dim == 3 * cfg.input_patch_size
    assert 0.5 in cfg.quantiles


@needs_weights
@needs_chronos
def test_preprocess_parity_vs_reference():
    import torch

    rng = np.random.default_rng(7)
    cfg = Chronos2Config.from_json(Path(WEIGHTS) / "config.json")
    prep = Preprocessor(cfg)
    model = _reference_model()

    B, T, n_out = 3, 100, 4  # T=100 -> left-pad to 112 -> 7 patches
    values = rng.normal(size=(B, T)).astype(np.float32) * 3.0
    mask = np.ones((B, T), np.float32)
    mask[:, -16:] = 0.0  # masked tail -> fully-masked last patch
    mask[1, :5] = 0.0  # masked head
    values = np.where(mask > 0, values, 0.0)  # masked cells carry placeholder zeros

    mine = prep.prepare(values, mask, n_out)

    x = torch.from_numpy(np.where(mask > 0, values, np.nan).astype(np.float32))
    with torch.no_grad():
        ref_feats, ref_attn, ref_loc_scale = model._prepare_patched_context(x)

    assert mine["ctx_features"].shape == tuple(ref_feats.shape)
    assert mine["ctx_features"].dtype == np.float32 == ref_feats.numpy().dtype
    np.testing.assert_allclose(mine["loc"], ref_loc_scale[0].numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(mine["scale"], ref_loc_scale[1].numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        mine["ctx_features"],
        ref_feats.numpy(),
        rtol=1e-4,
        atol=1e-5,
        err_msg="patched context features diverge from reference",
    )
    np.testing.assert_array_equal(mine["attn_mask"][:, : ref_attn.shape[-1]], ref_attn.numpy().astype(np.float32))
    assert mine["n_ctx_patches"] == ref_attn.shape[-1]


# --------------------------------------------------------------------------- #
# 3. Encoder-block math parity (independent numpy reimplementation)           #
# --------------------------------------------------------------------------- #


def _rms_norm(x, w, eps):
    var = (x.astype(np.float32) ** 2).mean(axis=-1, keepdims=True)
    return (w * (x / np.sqrt(var + eps))).astype(np.float32)


def _rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _rope(q, k, theta, d_kv):
    inv = 1.0 / (theta ** (np.arange(0, d_kv, 2, dtype=np.float32) / d_kv))
    pos = np.arange(q.shape[-2], dtype=np.float32)
    freqs = pos[:, None] * inv[None, :]  # [S, d/2]
    emb = np.concatenate([freqs, freqs], axis=-1)  # [S, d]
    cos, sin = np.cos(emb), np.sin(emb)
    return (q * cos + _rotate_half(q) * sin).astype(np.float32), (k * cos + _rotate_half(k) * sin).astype(np.float32)


def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


def _block_forward_np(weights, prefix_enc, cfg, x, attn_mask):
    """Independent numpy implementation of one Chronos2 encoder block (FP32)."""
    eps = cfg.layer_norm_epsilon
    H, dk = cfg.num_heads, cfg.d_kv

    def lin(w, x):
        return x @ w.T

    # --- time self-attention (pre-norm, RoPE, NO 1/sqrt(d) score scaling) ---
    p = f"{prefix_enc}.layer.0"
    h = _rms_norm(x, weights[f"{p}.layer_norm.weight"], eps)
    q = lin(weights[f"{p}.self_attention.q.weight"], h).reshape(*x.shape[:2], H, dk).transpose(0, 2, 1, 3)
    k = lin(weights[f"{p}.self_attention.k.weight"], h).reshape(*x.shape[:2], H, dk).transpose(0, 2, 1, 3)
    v = lin(weights[f"{p}.self_attention.v.weight"], h).reshape(*x.shape[:2], H, dk).transpose(0, 2, 1, 3)
    q, k = _rope(q, k, cfg.rope_theta, dk)
    scores = q @ k.transpose(0, 1, 3, 2)
    add_mask = ((1.0 - attn_mask)[:, None, None, :] * np.finfo(np.float32).min).astype(np.float32)
    probs = _softmax(scores + add_mask)
    att = (probs @ v).transpose(0, 2, 1, 3).reshape(*x.shape[:2], H * dk)
    x = x + lin(weights[f"{p}.self_attention.o.weight"], att)

    # --- group self-attention: independent rows => o(v(rms(x))) exactly ---
    p = f"{prefix_enc}.layer.1"
    h = _rms_norm(x, weights[f"{p}.layer_norm.weight"], eps)
    x = x + lin(fuse_group_attention(weights, p), h)

    # --- feed-forward (pre-norm, ReLU, no bias) ---
    p = f"{prefix_enc}.layer.2"
    h = _rms_norm(x, weights[f"{p}.layer_norm.weight"], eps)
    h = np.maximum(lin(weights[f"{p}.mlp.wi.weight"], h), 0.0)
    x = x + lin(weights[f"{p}.mlp.wo.weight"], h)
    return x.astype(np.float32)


@needs_weights
@needs_chronos
def test_encoder_block_parity_and_group_fusion():
    import torch

    rng = np.random.default_rng(11)
    cfg = Chronos2Config.from_json(Path(WEIGHTS) / "config.json")
    model = _reference_model()
    weights = load_weights_fp32(Path(WEIGHTS))

    B, S = 2, 9
    x = rng.normal(size=(B, S, cfg.d_model)).astype(np.float32) * 0.5
    attn = np.ones((B, S), np.float32)
    attn[:, :2] = 0.0  # some masked keys

    block = model.encoder.block[0].eval()
    pos = torch.arange(S).unsqueeze(0)
    attn_t = torch.from_numpy(attn)
    ext = model.encoder._expand_and_invert_time_attention_mask(attn_t, torch.float32)
    group_ids = torch.arange(B)  # default: independent rows
    gtm = model.encoder._construct_and_invert_group_time_mask(group_ids, attn_t, torch.float32)
    with torch.no_grad():
        ref = block(
            torch.from_numpy(x), position_ids=pos, attention_mask=ext, group_time_mask=gtm
        ).hidden_states.numpy()

    mine = _block_forward_np(weights, "encoder.block.0", cfg, x, attn)
    assert mine.shape == ref.shape and mine.dtype == np.float32
    # Fully masked time steps attend to no group member in the official block,
    # while the fused W_o @ W_v path always attends to itself. Those steps are
    # masked out as keys in every time-attention layer and never reach the
    # quantile head, so only attended positions must match.
    attended = attn > 0
    err = _nrmse(mine[attended], ref[attended])
    assert err < 1e-6, f"encoder block NRMSE vs reference on attended positions: {err}"

    # direct equivalence: fused o(v(.)) vs the reference GroupSelfAttention module
    p = "encoder.block.0.layer.1"
    h = _rms_norm(x, weights[f"{p}.layer_norm.weight"], cfg.layer_norm_epsilon)
    fused = h @ fuse_group_attention(weights, p).T
    ga = block.layer[1]
    h_t = torch.from_numpy(h).permute(1, 0, 2)  # [S, B, D]: time/batch flipped
    ind_mask = ((1.0 - torch.eye(B)) * torch.finfo(torch.float32).min).reshape(1, 1, B, B).expand(S, 1, B, B)
    with torch.no_grad():
        out = ga.self_attention(h_t, mask=ind_mask)[0].permute(1, 0, 2).numpy()
    np.testing.assert_allclose(
        fused, out, rtol=1e-4, atol=1e-5, err_msg="fused o(v(.)) != reference GroupSelfAttention for independent rows"
    )


# --------------------------------------------------------------------------- #
# 4. End-to-end: FP32 reference vs chronos-forecasting's public pipeline       #
# --------------------------------------------------------------------------- #


@needs_weights
@needs_chronos
def test_reference_matches_chronos_pipeline():
    """The test oracle agrees with Chronos2Pipeline.predict (FP32, CPU)."""
    import torch
    from chronos import Chronos2Pipeline

    from models.experimental.chronos2.reference.chronos2_reference import Chronos2ReferenceFP32

    try:  # transformers v5 renamed torch_dtype -> dtype
        pipeline = Chronos2Pipeline.from_pretrained(WEIGHTS, dtype=torch.float32, device_map="cpu")
    except TypeError:
        pipeline = Chronos2Pipeline.from_pretrained(WEIGHTS, torch_dtype=torch.float32, device_map="cpu")
    reference = Chronos2ReferenceFP32(WEIGHTS)
    rng = np.random.default_rng(3)
    t = np.arange(512, dtype=np.float32)
    rows = [
        10 + 3 * np.sin(2 * np.pi * t / 24) + 0.3 * rng.standard_normal(512),
        np.cumsum(rng.standard_normal(512)) + 5.0,
        np.sinh(4 * np.sin(2 * np.pi * t / 40)),
    ]
    for context, horizon in ((512, 64), (64, 64), (100, 16)):
        values = np.stack(rows)[:, -context:].astype(np.float32)
        mask = np.ones_like(values)
        mine = reference.predict(values, mask, horizon)["quantiles"]  # [B, H, Q]
        with torch.no_grad():
            preds = pipeline.predict(
                torch.from_numpy(values).unsqueeze(1),  # [B, 1 variate, T]
                prediction_length=horizon,
                batch_size=len(values),
                limit_prediction_length=True,
            )
        official = np.stack([p[0].float().numpy().T for p in preds])  # [1, Q, H] per series -> [B, H, Q]
        for row in range(len(values)):
            err = _nrmse(mine[row], official[row])
            assert err < 1e-3, (context, horizon, row, err)
