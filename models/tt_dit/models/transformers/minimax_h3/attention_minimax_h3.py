# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 self-attention.

H3 has no cross-attention anywhere: one packed sequence holding text, keyframe
conditioning, audio and video attends to itself in full. With SP > 1 the sequence
is sharded and ring attention gathers K/V, so every device keeps all of its
``num_heads / TP`` heads and sees the whole sequence.

``inner_dim`` (56 x 128 = 7168) is *larger* than ``hidden_size`` (5376), which is
unusual and means ``qkv_proj`` expands rather than preserves width.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.linear import ColParallelLinear, prepare_chunked_linear_output
from ....layers.module import Module
from ....layers.normalization import RMSNorm
from ....parallel.config import DiTParallelConfig
from ....parallel.manager import CCLManager


def reorder_interleaved_qkv(weight: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Raw per-head-interleaved fused QKV to ``[q_all; k_all; v_all]``.

    The checkpoint stores rows as ``[head0_q, head0_k, head0_v, head1_q, ...]``,
    i.e. ``num_heads * 3 * head_dim`` rows. Every consumer expects the three
    logical projections stacked instead. Confirmed against the diffusers
    conversion script, which applies exactly this at load time.
    """
    expected_rows = num_heads * 3 * head_dim
    if weight.shape[0] != expected_rows:
        raise ValueError(
            f"fused qkv has {weight.shape[0]} rows, expected {expected_rows} " f"= {num_heads} heads * 3 * {head_dim}"
        )
    grouped = weight.reshape(num_heads, 3 * head_dim, *weight.shape[1:])
    query, key, value = grouped.split(head_dim, dim=1)
    return torch.cat(
        [part.reshape(num_heads * head_dim, *weight.shape[1:]) for part in (query, key, value)],
        dim=0,
    )


class MiniMaxH3Attention(Module):
    """Fused-QKV self-attention with per-head q/k RMSNorm and 3D RoPE."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        qk_norm_eps: float,
        mesh_device,
        ccl_manager: CCLManager,
        parallel_config: DiTParallelConfig,
        k_chunk_size: int = 512,
        q_chunk_size: int = 128,
    ) -> None:
        super().__init__()
        tp_factor = parallel_config.tensor_parallel.factor
        if num_heads % tp_factor:
            raise ValueError(f"num_heads {num_heads} must be divisible by TP {tp_factor}")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.num_local_heads = num_heads // tp_factor
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.tp_axis = parallel_config.tensor_parallel.mesh_axis
        self.sp_axis = parallel_config.sequence_parallel.mesh_axis
        self.k_chunk_size = k_chunk_size
        self.q_chunk_size = q_chunk_size

        self.qkv_proj = ColParallelLinear(
            hidden_size,
            3 * self.inner_dim,
            bias=False,
            chunks=3,
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )
        # Per-head norms act on head_dim, which TP does not shard, so these are
        # plain rather than distributed norms.
        self.q_norm = RMSNorm(head_dim, norm_eps=qk_norm_eps, bias=False, mesh_device=mesh_device)
        self.k_norm = RMSNorm(head_dim, norm_eps=qk_norm_eps, bias=False, mesh_device=mesh_device)
        # Consumes the TP-gathered inner_dim and re-fractures hidden, so the
        # block's residual stream stays column-sharded.
        self.out_proj = ColParallelLinear(
            self.inner_dim,
            hidden_size,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=self.tp_axis,
            ccl_manager=ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.get("qkv_proj.weight")
        if weight is not None:
            state["qkv_proj.weight"] = reorder_interleaved_qkv(weight, self.num_heads, self.head_dim)
            # A plain column shard would hand device 0 the first quarter of q
            # rather than a slice of each of q/k/v; this interleaves per device.
            prepare_chunked_linear_output(
                state,
                prefix="qkv_proj",
                device_count=self.parallel_config.tensor_parallel.factor,
                chunks=3,
            )

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        rope_cos: ttnn.Tensor | None = None,
        rope_sin: ttnn.Tensor | None = None,
        trans_mat: ttnn.Tensor | None = None,
        logical_n: int | None = None,
    ) -> ttnn.Tensor:
        """``x`` is ``[1, 1, S_local, hidden]`` replicated over TP.

        Returns the same layout column-fractured, ready for the gated residual.
        """
        query, key, value = self.qkv_proj(x)
        query = _to_heads(query, self.num_local_heads, self.head_dim)
        key = _to_heads(key, self.num_local_heads, self.head_dim)
        value = _to_heads(value, self.num_local_heads, self.head_dim)

        query = self.q_norm(query)
        key = self.k_norm(key)
        if rope_cos is not None:
            query = _apply_rope(query, rope_cos, rope_sin, trans_mat)
            key = _apply_rope(key, rope_cos, rope_sin, trans_mat)

        if self.parallel_config.sequence_parallel.factor > 1:
            # A zero-length joint stream: H3 is one homogeneous document, so the
            # joint spatial/prompt split this op offers is unused. `logical_n`
            # keeps live rows from attending the padding tail, which reproduces
            # the reference's two-document cu_seqlens on every row that survives.
            empty = ttnn.zeros(
                [1, self.num_local_heads, 0, self.head_dim],
                device=self.mesh_device,
                layout=query.layout,
                dtype=query.dtype,
            )
            attended, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                query,
                key,
                value,
                empty,
                empty,
                empty,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(key.shape, 2, self.sp_axis),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(value.shape, 2, self.sp_axis),
                joint_strategy="rear",
                logical_n=logical_n if logical_n is not None else query.shape[2],
                program_config=None,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(self.sp_axis),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.sp_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
            )
        else:
            attended = ttnn.transformer.scaled_dot_product_attention(query, key, value, is_causal=False)

        attended = ttnn.transformer.concatenate_heads(attended)
        attended = self.ccl_manager.all_gather_persistent_buffer(attended, dim=3, mesh_axis=self.tp_axis)
        return self.out_proj(attended)


def _to_heads(x: ttnn.Tensor, num_heads: int, head_dim: int) -> ttnn.Tensor:
    sequence = x.shape[-2]
    x = ttnn.reshape(x, (1, sequence, num_heads, head_dim))
    return ttnn.permute(x, (0, 2, 1, 3))


def _apply_rope(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, trans_mat: ttnn.Tensor | None) -> ttnn.Tensor:
    """NeoX-style rotation of the leading ``rot_dim`` head dims, rest passed through.

    H3 rotates 96 of 128 dims (16 frequencies per axis over t/h/w, doubled), so
    the tail is not rotated and must survive untouched.
    """
    rot_dim = cos.shape[-1]
    if rot_dim == x.shape[-1]:
        return (
            ttnn.experimental.rotary_embedding_llama(x, cos, sin, trans_mat)
            if trans_mat is not None
            else (x * cos + ttnn.alt_complex_rotate90(x) * sin)
        )
    rotated = x[..., :rot_dim]
    passthrough = x[..., rot_dim:]
    rotated = rotated * cos + ttnn.alt_complex_rotate90(rotated) * sin
    return ttnn.concat([rotated, passthrough], dim=-1)
