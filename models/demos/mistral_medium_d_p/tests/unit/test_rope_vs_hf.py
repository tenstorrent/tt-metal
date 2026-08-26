# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HOST-ONLY: our YaRN table generator vs HuggingFace, bit-for-bit. **No TT hardware needed.**

This is the highest-value cheap test in the bring-up. The YaRN knobs that differ from every other
model already in the repo are exactly the ones that fail silently:

  * ``truncate`` (Mistral omits it -> HF default True; gpt-oss sets it False)
  * ``mscale`` / ``mscale_all_dim`` (1.0 / 0.0 -> HF's ``m(factor)`` branch, not the DeepSeek ratio)
  * ``beta_fast=4.0`` (not the YaRN paper's 32)

A wrong choice here is invisible at short sequence length and destroys K PCC at long context, so
compare inv_freq to 0 ULP and the assembled cos/sin at every position out to the full 262144 window.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_rope_vs_hf.py
"""

import json
import os

import pytest
import torch

from models.demos.mistral_medium_d_p.tt.rope_tables import (
    build_hf_cos_sin,
    build_yarn_cos_sin,
    yarn_inv_freq,
    yarn_params_from_config,
)

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "Mistral-Medium-3.5-128B")
HEAD_DIM = 128


def _hf_config():
    transformers = pytest.importorskip("transformers")
    with open(os.path.join(CONFIG_DIR, "config.json")) as f:
        raw = json.load(f)
    raw.pop("quantization_config", None)  # AutoConfig would try to build a quantizer
    return transformers.Ministral3Config(**{k: v for k, v in raw.items() if not k.startswith("_")})


def test_yarn_inv_freq_matches_hf_exactly():
    """inv_freq and attention_factor vs transformers ``_compute_yarn_parameters``."""
    rope_utils = pytest.importorskip("transformers.modeling_rope_utils")
    cfg = _hf_config()

    hf_inv_freq, hf_attn_factor = rope_utils.ROPE_INIT_FUNCTIONS["yarn"](cfg, device=torch.device("cpu"))
    ours_inv_freq, ours_attn_factor = yarn_inv_freq(HEAD_DIM, **_kw(cfg))

    assert ours_attn_factor == pytest.approx(
        hf_attn_factor, rel=0, abs=0
    ), f"attention_factor {ours_attn_factor} != HF {hf_attn_factor}"
    # 0.1*ln(64)+1 -- pinned so a future config change is visible, not just self-consistent.
    assert ours_attn_factor == pytest.approx(1.4158883083359672, abs=1e-12)
    torch.testing.assert_close(ours_inv_freq, hf_inv_freq, rtol=0, atol=0)


def _kw(cfg):
    p = yarn_params_from_config(cfg)
    return dict(
        base=p["rope_theta"],
        factor=p["yarn_factor"],
        orig_max_pos=p["yarn_orig_max_pos"],
        beta_fast=p["yarn_beta_fast"],
        beta_slow=p["yarn_beta_slow"],
        mscale=p["yarn_mscale"],
        mscale_all_dim=p["yarn_mscale_all_dim"],
        truncate=p["yarn_truncate"],
    )


@pytest.mark.parametrize("seq_len", [512, 8192, 262144], ids=["s512", "s8k", "s256k"])
def test_hf_convention_cos_sin_matches_hf_rotary_embedding(seq_len):
    """Our HF-convention table vs the real ``Ministral3RotaryEmbedding`` forward, at full context."""
    transformers = pytest.importorskip("transformers")
    from transformers.models.ministral3.modeling_ministral3 import Ministral3RotaryEmbedding

    cfg = _hf_config()
    rot = Ministral3RotaryEmbedding(cfg)
    pos_ids = torch.arange(seq_len)[None]
    hf_cos, hf_sin = rot(torch.zeros(1, seq_len, cfg.hidden_size), pos_ids)

    ours_cos, ours_sin = build_hf_cos_sin(seq_len, HEAD_DIM, **yarn_params_from_config(cfg))
    torch.testing.assert_close(ours_cos, hf_cos[0].float(), rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(ours_sin, hf_sin[0].float(), rtol=1e-6, atol=1e-6)


def test_meta_interleaved_table_is_the_hf_table_reordered():
    """The device table is the HF table in Meta interleaved order: [c0,c0,c1,c1,...]."""
    cfg = _hf_config()
    seq_len = 1024
    hf_cos, _ = build_hf_cos_sin(seq_len, HEAD_DIM, **yarn_params_from_config(cfg))
    meta_cos, _ = build_yarn_cos_sin(seq_len, HEAD_DIM, **yarn_params_from_config(cfg))
    half = hf_cos[:, : HEAD_DIM // 2]  # HF stacks [half, half]
    expected = torch.stack([half, half], dim=-1).flatten(-2)[None, None]
    torch.testing.assert_close(meta_cos, expected, rtol=0, atol=0)


def test_truncate_actually_changes_the_table():
    """Guard the knob that silently rots long context: truncate=False must NOT be equivalent."""
    cfg = _hf_config()
    kw = _kw(cfg)
    inv_true, _ = yarn_inv_freq(HEAD_DIM, **{**kw, "truncate": True})
    inv_false, _ = yarn_inv_freq(HEAD_DIM, **{**kw, "truncate": False})
    assert not torch.allclose(
        inv_true, inv_false
    ), "truncate has no effect on this config — the test can no longer catch the gpt-oss/Mistral mismatch"
    # Phase drift the wrong choice would inject by the end of the 256K window.
    drift = ((inv_true - inv_false).abs() * cfg.max_position_embeddings).max().item()
    assert drift > 1.0, f"expected a large late-position phase drift, got {drift} rad"


def test_rejects_non_yarn_rope_type():
    cfg = _hf_config()
    cfg.rope_parameters = {**cfg.rope_parameters, "rope_type": "llama3"}
    with pytest.raises(ValueError, match="rope_type='yarn'"):  # allow-pytest.raises: host-only, no root conftest
        yarn_params_from_config(cfg)
