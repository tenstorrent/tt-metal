# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet weight loading.

Two projections need a load-time permutation, for the same reason attention's fused QKV does: a
column-parallel shard hands TP rank ``r`` the ``r``-th contiguous slice of the output, and the
natural HF layout ``[all q | all k | all v]`` does not cut on a head boundary there. Laying
``in_proj_qkv`` out **rank-block-major** —

    rank r: [ its key heads' q | its key heads' k | its value heads' v ]

— makes an ordinary column-parallel mapper produce per-chip tensors the conv and the scan can use
directly. ``conv1d`` is depthwise, so its per-channel taps must be permuted the *same* way; a
mismatched tap permutation is a channel-wise scramble that still yields a smooth, plausible tensor.

The other four projections need nothing: ``in_proj_z`` (value heads, contiguous), ``in_proj_b`` and
``in_proj_a`` (one column per value head, contiguous), and ``out_proj`` (row-parallel on the value
dim, which is the same contiguous value-head split).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

import ttnn

from ...config import MeshConfig
from ...reference.config import Qwen35TextConfig
from ...utils.general_utils import get_cache_file_name
from ...utils.substate import substate


def rank_block_channel_order(cfg: Qwen35TextConfig, tp: int) -> torch.Tensor:
    """Index vector re-ordering the ``conv_dim`` channels of ``in_proj_qkv`` to rank-block-major.

    Returned as channel indices so the same permutation can be applied to the projection's output
    rows AND to the depthwise conv's per-channel taps — the only way to keep them in step.
    """
    n_k, n_v = cfg.linear_num_key_heads, cfg.linear_num_value_heads
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    assert n_k % tp == 0 and n_v % tp == 0, f"tp={tp} must divide {n_k} key heads and {n_v} value heads"
    k_local, v_local = n_k // tp, n_v // tp

    q_base, k_base, v_base = 0, cfg.gdn_key_dim, 2 * cfg.gdn_key_dim
    order: list[torch.Tensor] = []
    for r in range(tp):
        q_start, k_start = r * k_local * dk, r * k_local * dk
        v_start = r * v_local * dv
        order.extend(
            [
                torch.arange(q_base + q_start, q_base + q_start + k_local * dk),
                torch.arange(k_base + k_start, k_base + k_start + k_local * dk),
                torch.arange(v_base + v_start, v_base + v_start + v_local * dv),
            ]
        )
    perm = torch.cat(order)
    assert perm.numel() == cfg.gdn_conv_dim and perm.unique().numel() == cfg.gdn_conv_dim
    return perm


def device_conv_state_to_hf(state: torch.Tensor, cfg: Qwen35TextConfig, tp: int) -> torch.Tensor:
    """Device conv history ``[1, kernel-1, conv_dim]`` -> reference layout ``[1, conv_dim, kernel-1]``.

    Two changes, and skipping either gives a comparison that is wrong without looking wrong: the
    device keeps the history token-major (the layout ``qkv_causal_conv1d_silu`` takes) while the
    reference keeps it channel-major, and the device's channels are in the rank-block-major order
    of :func:`rank_block_channel_order`, not the checkpoint's. This is the inverse of both.
    """
    assert state.shape[-1] == cfg.gdn_conv_dim, f"expected the full conv_dim, got {tuple(state.shape)}"
    perm = rank_block_channel_order(cfg, tp)
    hf_order = torch.empty_like(state)
    hf_order[..., perm] = state
    return hf_order.transpose(-1, -2).contiguous()


@dataclass
class GdnWeights:
    in_proj_qkv: ttnn.Tensor  # [hidden, conv_dim] column-parallel, rank-block-major
    in_proj_z: ttnn.Tensor  # [hidden, value_dim] column-parallel
    in_proj_b: ttnn.Tensor  # [hidden, num_v_heads] column-parallel
    in_proj_a: ttnn.Tensor  # [hidden, num_v_heads] column-parallel
    out_proj: ttnn.Tensor  # [value_dim, hidden] row-parallel
    conv_taps: list[ttnn.Tensor]  # 4 x [1, 1, 1, conv_dim] column-parallel, permuted to match qkv
    neg_a_exp: ttnn.Tensor  # [1, 1, 1, num_v_heads] fp32, = -exp(A_log)
    dt_bias: ttnn.Tensor  # [1, 1, 1, num_v_heads] fp32
    norm_weight: ttnn.Tensor  # [head_v_dim] PLAIN gain (no Gemma fold) for the gated norm


def load_gdn_weights(
    mesh_device,
    cfg: Qwen35TextConfig,
    state_dict: dict[str, torch.Tensor],
    *,
    mesh_config: MeshConfig,
    weight_dtype=ttnn.bfloat8_b,
    tensor_cache_path: Optional[str] = None,
) -> GdnWeights:
    tp = mesh_config.tp
    col = mesh_config.column_parallel(mesh_device)
    row = mesh_config.row_parallel(mesh_device)
    have = bool(state_dict)
    perm = rank_block_channel_order(cfg, tp)

    def _linear(name: str, mapper, *, dtype=weight_dtype, channel_perm: Optional[torch.Tensor] = None):
        w = None
        if have:
            w = substate(state_dict, name)["weight"]  # HF [out, in]
            if channel_perm is not None:
                w = w[channel_perm]
            w = w.transpose(0, 1).unsqueeze(0).unsqueeze(0).contiguous()
        return ttnn.as_tensor(
            w,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            mesh_mapper=mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    in_proj_qkv = _linear("in_proj_qkv", col, channel_perm=perm)
    in_proj_z = _linear("in_proj_z", col)
    # b/a are one column per value head (48 of them). bf16 rather than the weight default: the
    # decay and beta gates feed exp/softplus, where a bf8 block exponent across 48 channels of
    # very different scale is a real accuracy loss for a negligible memory saving.
    in_proj_b = _linear("in_proj_b", col, dtype=ttnn.bfloat16)
    in_proj_a = _linear("in_proj_a", col, dtype=ttnn.bfloat16)
    out_proj = _linear("out_proj", row)

    conv_taps = []
    for j in range(cfg.linear_conv_kernel_dim):
        tap = None
        if have:
            # HF Conv1d weight is [channels, 1, kernel]; tap j multiplies x[t - (kernel-1-j)].
            tap = state_dict["conv1d.weight"][perm, 0, j].reshape(1, 1, 1, -1).contiguous()
        conv_taps.append(
            ttnn.as_tensor(
                tap,
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=col,
                cache_file_name=get_cache_file_name(tensor_cache_path, f"conv_tap{j}"),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        )

    def _head_vector(name: str, values: Optional[torch.Tensor]) -> ttnn.Tensor:
        return ttnn.as_tensor(
            None if values is None else values.reshape(1, 1, 1, -1).contiguous(),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.float32,
            mesh_mapper=col,
            cache_file_name=get_cache_file_name(tensor_cache_path, name),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    # -exp(A_log) is folded on the host in fp32. Upstream's comment is the reason: A_log held and
    # exponentiated in a narrow dtype can reach -inf, which poisons the whole scan.
    neg_a = -state_dict["A_log"].float().exp() if have else None
    dt_bias = state_dict["dt_bias"].float() if have else None

    norm_weight = ttnn.as_tensor(
        # PLAIN gain: Qwen35RMSNormGated has no (1 + w) fold, unlike every other norm in the model.
        # Rank 1 exactly: sigmoid_gated_rms_norm asserts ``weight must be [V]``.
        None if not have else substate(state_dict, "norm")["weight"].reshape(-1).contiguous(),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=mesh_config.replicate(mesh_device),
        cache_file_name=get_cache_file_name(tensor_cache_path, "norm"),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return GdnWeights(
        in_proj_qkv=in_proj_qkv,
        in_proj_z=in_proj_z,
        in_proj_b=in_proj_b,
        in_proj_a=in_proj_a,
        out_proj=out_proj,
        conv_taps=conv_taps,
        neg_a_exp=_head_vector("neg_a_exp", neg_a),
        dt_bias=_head_vector("dt_bias", dt_bias),
        norm_weight=norm_weight,
    )
