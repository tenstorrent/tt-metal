# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the opt-in denoise trims: LTX_V2A_SKIP_PAD_MUL, LTX_BATCH_ADALN_ADDS, LTX_AGMM_K2048.

No device: ttnn ops used by the batched AdaLN path are emulated with torch so the test checks the
indexing, broadcast and fallback logic. The device A/B checks the real ops for bit identity.
"""

from types import SimpleNamespace

import pytest
import torch

import models.tt_dit.models.transformers.ltx.attention_ltx as attn_mod
import models.tt_dit.models.transformers.ltx.transformer_ltx as tl
import ttnn


def _sdpa_logical_n(q, k, v, logical_n):
    """Reference SDPA that drops keys >= logical_n, as the ring joint SDPA does."""
    scores = (q @ k.transpose(-1, -2)) / q.shape[-1] ** 0.5
    scores[..., logical_n:] = float("-inf")
    return torch.softmax(scores, dim=-1) @ v


@pytest.mark.parametrize(("n", "n_real"), [(128, 100), (96, 95), (64, 33)])
def test_v2a_pad_rows_do_not_reach_output(n, n_real):
    """Padded K/V rows are masked out, so zeroing them first cannot change the output."""
    gen = torch.Generator().manual_seed(n + n_real)
    q = torch.randn(1, 4, 40, 64, generator=gen)
    kv = torch.randn(1, 1, n, 128, generator=gen)
    w_k = torch.randn(128, 4 * 64, generator=gen) / 11
    w_v = torch.randn(128, 4 * 64, generator=gen) / 11
    # Unmasked padded rows equal the AdaLN shift, which is nonzero.
    kv[..., n_real:, :] = torch.randn(128, generator=gen) * 3
    mask = torch.ones(1, 1, n, 1)
    mask[..., n_real:, :] = 0

    def run(x):
        k = (x @ w_k).reshape(1, n, 4, 64).transpose(1, 2)
        v = (x @ w_v).reshape(1, n, 4, 64).transpose(1, 2)
        return _sdpa_logical_n(q, k, v, n_real)

    assert kv[..., n_real:, :].abs().sum() > 0
    assert torch.equal(run(kv), run(kv * mask))


def test_v2a_skip_gating(monkeypatch):
    monkeypatch.setattr(tl, "LTX_V2A_SKIP_PAD_MUL", False)
    assert not tl._v2a_skip_pad_mul(100, 8)
    monkeypatch.setattr(tl, "LTX_V2A_SKIP_PAD_MUL", True)
    assert tl._v2a_skip_pad_mul(100, 8)
    # The non-distilled pipeline passes no kv_logical_n; SP=1 takes the non-ring path.
    assert not tl._v2a_skip_pad_mul(None, 8)
    assert not tl._v2a_skip_pad_mul(100, 1)


def _emulate_ttnn(monkeypatch):
    def slice_(t, starts, ends):
        return t[tuple(slice(s, e) for s, e in zip(starts, ends))]

    fake = SimpleNamespace(
        add=lambda a, b: a + b,
        reshape=lambda t, shape: t.reshape(shape),
        concat=lambda ts, dim: torch.cat(ts, dim=dim),
        slice=slice_,
        chunk=lambda t, n, dim: list(torch.chunk(t, n, dim=dim)),
    )
    monkeypatch.setattr(tl, "ttnn", fake)


def _fake_model(nb, d_v, d_a, gen):
    def table(coeff, d):
        return SimpleNamespace(data=torch.randn(coeff, 1, 1, d, generator=gen).to(torch.bfloat16))

    blocks = [
        SimpleNamespace(
            scale_shift_table=table(9, d_v),
            prompt_scale_shift_table=table(2, 2 * d_v),
            audio_scale_shift_table=table(9, d_a),
            audio_prompt_scale_shift_table=table(2, 2 * d_a),
            scale_shift_table_a2v_ca_video=table(5, d_v),
            scale_shift_table_a2v_ca_audio=table(5, d_a),
        )
        for _ in range(nb)
    ]
    return SimpleNamespace(
        transformer_blocks=blocks,
        has_audio=True,
        cross_attention_adaln=True,
        _ADALN_TABLES=tl.LTXTransformerModel._ADALN_TABLES,
    )


def test_batched_adaln_matches_per_block(monkeypatch):
    _emulate_ttnn(monkeypatch)
    monkeypatch.setattr(tl, "LTX_BATCH_ADALN_ADDS", True)
    gen = torch.Generator().manual_seed(0)
    d_v, d_a, nb = 96, 64, 5
    model = _fake_model(nb, d_v, d_a, gen)
    widths = {"v": (9, d_v), "pv": (2, 2 * d_v), "a": (9, d_a), "pa": (2, 2 * d_a), "av": (5, d_v), "ava": (5, d_a)}
    tembs = {k: torch.randn(c, 1, 1, d, generator=gen).to(torch.bfloat16) for k, (c, d) in widths.items()}

    stacks = tl.LTXTransformerModel._adaln_pre_stacks(model, tembs)
    for block_idx, block in enumerate(model.transformer_blocks):
        for key, attr in tl.LTXTransformerModel._ADALN_TABLES:
            ref = list(torch.chunk(getattr(block, attr).data + tembs[key], widths[key][0], dim=0))
            got = tl._slice_block_row(stacks[key], block_idx)
            assert len(got) == len(ref)
            for g, r in zip(got, ref):
                assert g.shape == r.shape
                assert torch.equal(g, r)


def test_batched_adaln_fallbacks(monkeypatch):
    _emulate_ttnn(monkeypatch)
    gen = torch.Generator().manual_seed(1)
    model = _fake_model(2, 32, 32, gen)
    tembs = {k: torch.zeros(c, 1, 1, 32) for k, c in (("v", 9), ("pv", 2), ("a", 9), ("pa", 2), ("av", 5), ("ava", 5))}
    tembs["pv"] = torch.zeros(2, 1, 1, 64)
    tembs["pa"] = torch.zeros(2, 1, 1, 64)

    monkeypatch.setattr(tl, "LTX_BATCH_ADALN_ADDS", False)
    assert tl.LTXTransformerModel._adaln_pre_stacks(model, tembs) is None

    monkeypatch.setattr(tl, "LTX_BATCH_ADALN_ADDS", True)
    per_token = dict(tembs, v=torch.zeros(9, 1, 7, 32))
    assert tl.LTXTransformerModel._adaln_pre_stacks(model, per_token) is None
    assert tl.LTXTransformerModel._adaln_pre_stacks(model, dict(tembs, pv=None)) is None


def test_agmm_k2048_lookup(monkeypatch):
    grid = ttnn.CoreCoord(12, 10)
    monkeypatch.setattr(attn_mod, "LTX_AGMM_K2048", False)
    assert attn_mod._to_out_fabric_agmm_config(4480, 2048, 1024, grid) is None
    monkeypatch.setattr(attn_mod, "LTX_AGMM_K2048", True)
    cfg = attn_mod._to_out_fabric_agmm_config(4480, 2048, 1024, grid)
    assert cfg == attn_mod._to_out_fabric_agmm_config(4480, 4096, 1024, grid)
    assert cfg is not None
    # Only the A2V to_out shape is redirected.
    assert attn_mod._to_out_fabric_agmm_config(4480, 2048, 512, grid) is None


@pytest.mark.parametrize("env, expected", [(None, "True"), ("0", "False")])
def test_v2a_skip_default_on(env, expected):
    import os
    import pathlib
    import subprocess
    import sys

    e = {k: v for k, v in os.environ.items() if k != "LTX_V2A_SKIP_PAD_MUL"}
    if env is not None:
        e["LTX_V2A_SKIP_PAD_MUL"] = env
    # Import from this tree, not whatever checkout the venv's .pth points at.
    root = str(pathlib.Path(tl.__file__).resolve().parents[5])
    e["PYTHONPATH"] = os.pathsep.join(p for p in (root, e.get("PYTHONPATH")) if p)
    out = subprocess.run(
        [sys.executable, "-c", f"import {tl.__name__} as m; print(m.LTX_V2A_SKIP_PAD_MUL)"],
        env=e,
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert out[-1] == expected
