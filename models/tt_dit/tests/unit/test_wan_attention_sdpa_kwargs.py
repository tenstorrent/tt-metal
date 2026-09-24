# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: WanAttention must read its SDPA recipe (or legacy compute config) at call time.

apply_quant_config (pipelines/wan/quant_config.py) replaces attn1.sdpa_precision on Blackhole, and
attn1.sdpa_compute_kernel_config elsewhere, after construction; kwargs captured in __init__ would
silently keep the old setting.
"""

from types import SimpleNamespace

import ttnn

from models.tt_dit.models.transformers.wan2_2.attention_wan import WanAttention
from models.tt_dit.pipelines.wan.quant_config import QuantConfig, _apply_sdpa_config


def _bare_attention(precision=None):
    # Bypass __init__, which needs a mesh device; only the kwargs helper is under test.
    attention = object.__new__(WanAttention)
    attention.sdpa_precision = precision
    attention.sdpa_kv_dtype = ttnn.bfloat16
    attention.sdpa_compute_kernel_config = None if precision is not None else "initial"
    return attention


def test_legacy_sdpa_kwargs_follow_reassigned_compute_config():
    attention = _bare_attention()
    assert attention._self_sdpa_kwargs() == {"compute_kernel_config": "initial"}
    attention.sdpa_compute_kernel_config = "quantized"
    assert attention._self_sdpa_kwargs() == {"compute_kernel_config": "quantized"}


def test_recipe_sdpa_kwargs_follow_reassigned_precision():
    attention = _bare_attention(ttnn.SDPAPrecision.BALANCED)
    assert attention._self_sdpa_kwargs() == {"precision": ttnn.SDPAPrecision.BALANCED, "inputs_prepared": False}
    attention.sdpa_precision = ttnn.SDPAPrecision.LOW_PRECISION
    assert attention._self_sdpa_kwargs() == {"precision": ttnn.SDPAPrecision.LOW_PRECISION, "inputs_prepared": True}


def test_quant_config_sets_the_recipe_on_blackhole():
    attention = _bare_attention(ttnn.SDPAPrecision.BALANCED)
    _apply_sdpa_config(attention, QuantConfig.default().ring_sdpa, arch=None)
    assert attention.sdpa_precision == ttnn.SDPAPrecision.BALANCED  # default keeps WanAttention's recipe
    _apply_sdpa_config(attention, QuantConfig.all_bf8_lofi().ring_sdpa, arch=None)
    assert (attention.sdpa_precision, attention.sdpa_kv_dtype) == (ttnn.SDPAPrecision.FAST, ttnn.bfloat16)
    assert attention.sdpa_compute_kernel_config is None and not hasattr(attention, "_sdpa_input_dtype")


def test_quant_config_sets_the_legacy_config_off_blackhole(monkeypatch):
    from models.tt_dit.pipelines.wan import quant_config

    monkeypatch.setattr(quant_config, "_make_sdpa_compute_config", lambda arch, sc: SimpleNamespace(sc=sc))
    attention = _bare_attention()
    sc = QuantConfig.all_bf8_lofi().ring_sdpa
    _apply_sdpa_config(attention, sc, arch=None)
    assert attention.sdpa_precision is None
    assert attention.sdpa_compute_kernel_config.sc is sc and attention._sdpa_input_dtype == ttnn.bfloat8_b


def test_recipe_chunks_are_op_selected():
    # WanAttention no longer maps its tuned chunks for recipes; SDPA chooses them.
    assert not hasattr(WanAttention, "_recipe_q_chunk")
    assert all(not key[0] for key in WanAttention.sdpa_chunk_size_map)
