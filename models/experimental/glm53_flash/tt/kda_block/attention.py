# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``Glm5NextTextLinearAttention`` (KDA) built from HF-named weights onto :class:`~models.experimental.glm53_flash.tt.kda.KDA`.

``weights`` keys (the HF module's state_dict, ``self_attn.`` stripped):
    q_proj / k_proj / v_proj.weight [W, D], conv1d.weight [3W, 1, 4], b_proj.weight [H, D],
    forget_gate.{f_a_proj.weight [K, D], f_b_proj.weight [W, K], dt_bias [W], A_log [H]},
    g_a_proj.weight [K, D], g_b_proj.weight [W, K], o_norm.weight [K], o_proj.weight [D, W]
with H = linear_num_heads, K = linear_head_dim, W = H * K. Checkpoint names map onto these with
``reference/hf_layer_reference.py::convert_to_hf_module_keys``.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn

from models.experimental.glm53_flash.tt.ds_v4.weight_cache import WeightCache, _as_cache, _load_weight, _materialize
from models.experimental.glm53_flash.tt.kda import KDA


def _host(weight) -> torch.Tensor:
    return (weight() if callable(weight) else weight).detach().float()


def build_kda(
    config,
    weights: dict,
    device: ttnn.MeshDevice,
    cache: Optional[WeightCache] = None,
    weight_dtype: ttnn.DataType = ttnn.bfloat8_b,
) -> KDA:
    """``weight_dtype`` is that of the two wide projections (q|k|v and o); the low-rank gates, conv taps and norm stay bf16."""
    cache = _as_cache(cache)
    w = lambda name: _host(weights[name])

    def load(name, thunk, dtype):
        file = cache.file(name)
        return _load_weight(_materialize(thunk, file, dtype), device, cache_file_name=file, dtype=dtype)

    return KDA(
        device,
        w_qkv=load(
            "qkv_proj",
            lambda: torch.cat([w("q_proj.weight"), w("k_proj.weight"), w("v_proj.weight")], 0).t().contiguous(),
            weight_dtype,
        ),
        w_lowrank=load(
            "fa_ga_b_proj",
            lambda: torch.cat([w("forget_gate.f_a_proj.weight"), w("g_a_proj.weight"), w("b_proj.weight")], 0)
            .t()
            .contiguous(),
            ttnn.bfloat16,
        ),
        w_fb=load("f_b_proj", lambda: w("forget_gate.f_b_proj.weight").t().contiguous(), ttnn.bfloat16),
        w_gb=load("g_b_proj", lambda: w("g_b_proj.weight").t().contiguous(), ttnn.bfloat16),
        w_o=load("o_proj", lambda: w("o_proj.weight").t().contiguous(), weight_dtype),
        taps=load("conv_taps", lambda: w("conv1d.weight").squeeze(1).t().contiguous(), ttnn.bfloat16),
        o_norm=load("o_norm", lambda: w("o_norm.weight"), ttnn.bfloat16),
        dt_bias=w("forget_gate.dt_bias"),
        a_log=w("forget_gate.A_log"),
        num_heads=config.linear_num_heads,
        head_dim=config.linear_head_dim,
        hidden=config.hidden_size,
        lower_bound=config.linear_lower_bound,
        eps=config.rms_norm_eps,
    )
