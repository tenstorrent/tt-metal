# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
V2-VISION-V2 (4/N): qwen3.6 vision attention with TP=8 across mesh row axis.

Mirrors tt_dit's `Qwen25VlAttention` pattern (fused QKV + ColParallelLinear)
adapted for qwen3.6's vision specifics:
  - hidden=1152, num_heads=16, head_dim=72 (NOT tile-aligned!)
  - padded_head_dim = ceil(72/32)*32 = 96 (next tile multiple)
  - Non-causal SDPA (no causal mask, no cu_seqlens)
  - 2D vision RoPE from `qwen3_vision_transformer_preprocess`

Topology on BH GLX (8, 4):
  - TP=8 across cluster_axis=0 (rows): heads split 16 → 2 per chip; the
    qkv proj `out_features = num_heads * 3 * padded_head_dim = 4608`
    sharded into per-chip 576 = (2 heads × 3 × 96).
  - DP/replicate across cluster_axis=1 (cols): no change inside the block.

Weight reorganization (`_prepare_attention_state`):
  - HF qkv weight `[3*hidden=3456, hidden=1152]` decomposes into separate
    Q, K, V `[hidden, hidden]` slabs. Each gets reshaped to
    `[tp_factor, num_local_heads, padded_head_dim, hidden_in]` with
    head_dim padded zeros at the head_dim tail. Concat [Q, K, V] along
    the head dim, flatten to `[tp_factor * 3 * num_local_heads *
    padded_head_dim, hidden_in] = [4608, 1152]`. ColParallelLinear's
    out-dim shard now puts a coherent (Q-heads + K-heads + V-heads) on
    each chip ready for `nlp_create_qkv_heads`.
  - HF o_proj weight `[hidden=1152, hidden=1152]` reshapes to
    `[hidden_out, num_heads, head_dim]` with head_dim padded → flatten
    to `[hidden_out, num_heads * padded_head_dim] = [1152, 1536]`.
    The padded columns are zeros so garbage in those columns from
    concat_heads is masked out.

Forward pass:
  1. qkv_proj(x) → per-chip [B, S, 576]
  2. nlp_create_qkv_heads → q, k, v each [B, num_local_heads=2, S, padded_head_dim=96]
  3. apply RoPE to q, k via cos, sin (padded to [S, 96] with cos=1 / sin=0 in tail)
  4. ttnn.transformer.scaled_dot_product_attention(is_causal=False)
  5. concatenate_heads → per-chip [B, S, 192]
  6. all_gather on cluster_axis=0 → per-chip [B, S, 1536] (replicated)
  7. o_proj → per-chip [B, S, 144]
  8. all_gather → [B, S, 1152] (replicated)
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.tt_dit.layers.linear import ColParallelLinear
from models.tt_dit.layers.module import Module
from models.tt_dit.parallel.manager import CCLManager
from models.tt_transformers.tt.common import get_rot_transformation_mat


def _pad_head_dim(t: torch.Tensor, *, head_dim_axis: int, head_dim: int, padded_head_dim: int) -> torch.Tensor:
    """Pad a tensor's head-dim axis from head_dim to padded_head_dim with zeros.

    For weights: zeros in the padded tail of each head ensures matmul ignores
    garbage values that may end up in the padded positions of activations.
    """
    if padded_head_dim == head_dim:
        return t
    shape = list(t.shape)
    shape[head_dim_axis] = padded_head_dim - head_dim
    pad = torch.zeros(*shape, dtype=t.dtype)
    return torch.cat([t, pad], dim=head_dim_axis)


def reorg_fused_qkv_weight(
    qkv_w: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
    tp_factor: int,
    hidden_in: int,
) -> torch.Tensor:
    """Reorganize HF fused QKV weight for TP-friendly ColParallelLinear layout.

    Input: `[3 * hidden, hidden_in]` HF convention with order [Q, K, V].
    Output: `[tp_factor * 3 * num_local_heads * padded_head_dim, hidden_in]`
            with each per-chip slice containing (Q-heads, K-heads, V-heads).
    """
    hidden = num_heads * head_dim
    num_local_heads = num_heads // tp_factor
    assert qkv_w.shape == (3 * hidden, hidden_in), f"unexpected qkv weight shape {qkv_w.shape}"

    Q = qkv_w[0:hidden, :]
    K = qkv_w[hidden : 2 * hidden, :]
    V = qkv_w[2 * hidden : 3 * hidden, :]

    def reshape_for_tp(W: torch.Tensor) -> torch.Tensor:
        # [hidden, hidden_in] → [num_heads, head_dim, hidden_in] → pad head_dim → [num_heads, padded_head_dim, hidden_in]
        W = W.view(num_heads, head_dim, hidden_in)
        W = _pad_head_dim(W, head_dim_axis=1, head_dim=head_dim, padded_head_dim=padded_head_dim)
        # [num_heads, padded_head_dim, hidden_in] → [tp_factor, num_local_heads, padded_head_dim, hidden_in]
        W = W.view(tp_factor, num_local_heads, padded_head_dim, hidden_in)
        return W

    Q = reshape_for_tp(Q)
    K = reshape_for_tp(K)
    V = reshape_for_tp(V)

    # Concat along dim=1 (the local-heads axis) so per-chip slice = (Q-heads, K-heads, V-heads).
    qkv = torch.cat([Q, K, V], dim=1)  # [tp_factor, 3 * num_local_heads, padded_head_dim, hidden_in]
    # Flatten dims 0-2: [tp_factor * 3 * num_local_heads * padded_head_dim, hidden_in]
    qkv = qkv.reshape(tp_factor * 3 * num_local_heads * padded_head_dim, hidden_in)
    return qkv


def reorg_fused_qkv_bias(
    qkv_b: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
    tp_factor: int,
) -> torch.Tensor:
    """Same reorganization as the weight but for the bias (1-D)."""
    hidden = num_heads * head_dim
    num_local_heads = num_heads // tp_factor
    assert qkv_b.shape == (3 * hidden,), f"unexpected qkv bias shape {qkv_b.shape}"

    Q = qkv_b[0:hidden]
    K = qkv_b[hidden : 2 * hidden]
    V = qkv_b[2 * hidden : 3 * hidden]

    def reshape_for_tp(b: torch.Tensor) -> torch.Tensor:
        b = b.view(num_heads, head_dim)
        b = _pad_head_dim(b, head_dim_axis=1, head_dim=head_dim, padded_head_dim=padded_head_dim)
        b = b.view(tp_factor, num_local_heads, padded_head_dim)
        return b

    Q = reshape_for_tp(Q)
    K = reshape_for_tp(K)
    V = reshape_for_tp(V)

    qkv = torch.cat([Q, K, V], dim=1)  # [tp_factor, 3 * num_local_heads, padded_head_dim]
    qkv = qkv.reshape(tp_factor * 3 * num_local_heads * padded_head_dim)
    return qkv


def reorg_o_proj_weight(
    proj_w: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
    hidden_out: int,
) -> torch.Tensor:
    """Pad o_proj input dim from hidden = num_heads*head_dim to num_heads*padded_head_dim with zeros.

    Padded columns are zeros so the matmul ignores garbage values in the
    padded positions of the concat_heads output activation.
    """
    if padded_head_dim == head_dim:
        return proj_w
    assert proj_w.shape == (hidden_out, num_heads * head_dim), f"unexpected proj weight shape {proj_w.shape}"
    proj_w = proj_w.view(hidden_out, num_heads, head_dim)
    proj_w = _pad_head_dim(proj_w, head_dim_axis=2, head_dim=head_dim, padded_head_dim=padded_head_dim)
    return proj_w.reshape(hidden_out, num_heads * padded_head_dim)


def build_vision_rope_tensors(
    *,
    seq_len: int,
    grid_thw: torch.Tensor,
    head_dim: int,
    padded_head_dim: int,
    spatial_merge_size: int,
    mesh_device: ttnn.MeshDevice,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Construct cos, sin TTNN tensors for vision 2D RoPE.

    Uses qwen3_vl/reference/functional.py:qwen3_vision_transformer_preprocess
    (which cat's the raw [seq, head_dim/2] rotary embeddings with themselves
    to reach [seq, head_dim]) and then pads to padded_head_dim with cos=1,
    sin=0 in the tail (so RoPE on padded positions is a no-op).
    """
    from models.demos.qwen3_vl.reference.functional import qwen3_vision_transformer_preprocess

    _cu_seqlens, (cos, sin) = qwen3_vision_transformer_preprocess(
        seq_len=seq_len,
        grid_thw=grid_thw,
        head_dim=head_dim,
        spatial_merge_size=spatial_merge_size,
    )
    assert cos.shape == (seq_len, head_dim), f"unexpected cos shape {cos.shape}"

    if padded_head_dim != head_dim:
        cos_pad = torch.ones(seq_len, padded_head_dim - head_dim, dtype=cos.dtype)
        sin_pad = torch.zeros(seq_len, padded_head_dim - head_dim, dtype=sin.dtype)
        cos = torch.cat([cos, cos_pad], dim=-1)
        sin = torch.cat([sin, sin_pad], dim=-1)

    # Upload as [1, 1, S, padded_head_dim] replicated across the full mesh.
    def upload(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.unsqueeze(0).unsqueeze(0),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    return upload(cos), upload(sin)


class Qwen36VisionAttentionTP(Module):
    """qwen3.6 vision attention with TP=8 across cluster_axis=0.

    The forward signature matches what `Qwen36VisionBlockTP` needs:
      - x: replicated input `[B=1, 1, S, hidden=1152]`
      - cos, sin: replicated rotary tables `[1, 1, S, padded_head_dim=96]`
      - returns: replicated output `[B, 1, S, hidden=1152]`
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        ccl_manager: CCLManager,
        state_dict: dict[str, torch.Tensor],
        *,
        hidden_size: int = 1152,
        num_heads: int = 16,
        head_dim: int = 72,
        tile_size: int = 32,
        tp_mesh_axis: int = 0,
        dtype: ttnn.DataType = ttnn.bfloat16,
        state_dict_prefix: str = "",
    ) -> None:
        super().__init__()
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.tp_mesh_axis = tp_mesh_axis
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.padded_head_dim = math.ceil(head_dim / tile_size) * tile_size

        tp_factor = mesh_device.shape[tp_mesh_axis]
        assert num_heads % tp_factor == 0, f"num_heads={num_heads} must be divisible by tp_factor={tp_factor}"
        self.tp_factor = tp_factor
        self.num_local_heads = num_heads // tp_factor

        # qkv: in=hidden, out = num_heads * 3 * padded_head_dim (head-major layout)
        qkv_out = num_heads * 3 * self.padded_head_dim
        self.qkv_proj = ColParallelLinear(
            in_features=hidden_size,
            out_features=qkv_out,
            bias=True,
            dtype=dtype,
            mesh_device=mesh_device,
            mesh_axis=tp_mesh_axis,
        )

        # o_proj: in = num_heads * padded_head_dim (head-padded), out = hidden
        o_in = num_heads * self.padded_head_dim
        self.o_proj = ColParallelLinear(
            in_features=o_in,
            out_features=hidden_size,
            bias=True,
            dtype=dtype,
            mesh_device=mesh_device,
            mesh_axis=tp_mesh_axis,
        )

        # SDPA compute config (mirrors qwen25vl pattern)
        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

        # Rotary-embedding transformation matrix (single 32x32 tile). Used by
        # `ttnn.experimental.rotary_embedding_llama` to handle the padded
        # head_dim correctly (rotate_half pairing for the REAL head_dim, not
        # the padded one). Mirrors qwen3_vl/tt/vision_attention.py:433.
        self._rope_transformation_mat = ttnn.as_tensor(
            get_rot_transformation_mat(head_dim),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # Load + transform weights
        clean = self._prepare_attention_state(state_dict, state_dict_prefix)
        self.load_state_dict(clean)

    def _prepare_attention_state(
        self, state_dict: dict[str, torch.Tensor], state_dict_prefix: str
    ) -> dict[str, torch.Tensor]:
        """Reorganize HF attn.{qkv,proj}.{weight,bias} into TP-friendly layout."""
        out: dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            k2 = k[len(state_dict_prefix) :] if state_dict_prefix and k.startswith(state_dict_prefix) else k
            out[k2] = v

        if "qkv.weight" in out:
            out["qkv_proj.weight"] = reorg_fused_qkv_weight(
                out.pop("qkv.weight"),
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                padded_head_dim=self.padded_head_dim,
                tp_factor=self.tp_factor,
                hidden_in=self.hidden_size,
            )
        if "qkv.bias" in out:
            out["qkv_proj.bias"] = reorg_fused_qkv_bias(
                out.pop("qkv.bias"),
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                padded_head_dim=self.padded_head_dim,
                tp_factor=self.tp_factor,
            )
        if "proj.weight" in out:
            out["o_proj.weight"] = reorg_o_proj_weight(
                out.pop("proj.weight"),
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                padded_head_dim=self.padded_head_dim,
                hidden_out=self.hidden_size,
            )
        if "proj.bias" in out:
            out["o_proj.bias"] = out.pop("proj.bias")
        return out

    def _sdpa_program_config(self, seq_len: int) -> ttnn.SDPAProgramConfig:
        grid_size = self.mesh_device.compute_with_storage_grid_size()
        seq_len = -(-seq_len // 32) * 32  # round up to tile
        chunk_size = min(seq_len, 128)
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=chunk_size,
            k_chunk_size=chunk_size,
            exp_approx_mode=False,
        )

    def forward(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Run attention. Input + output are replicated across the TP axis."""
        # qkv projection → per-chip [B, 1, S, 3 * num_local_heads * padded_head_dim]
        # = [B, 1, S, 3 * 2 * 96] = [B, 1, S, 576]
        xqkv = self.qkv_proj.forward(x)

        # If xqkv came out 3D, unsqueeze to 4D for nlp_create_qkv_heads.
        if len(xqkv.shape) == 3:
            xqkv = ttnn.unsqueeze(xqkv, 1)

        # Split into q, k, v: each [B, num_local_heads, S, padded_head_dim]
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_local_heads,
            num_kv_heads=self.num_local_heads,  # vision has no GQA
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        # Apply 2D vision RoPE to q, k via the TTNN op that handles padded
        # head_dim correctly (the manual `cos*x + sin*rotate_half(x)` would
        # pair k↔k+padded/2 instead of k↔k+head_dim/2, causing PCC loss).
        q_rot = ttnn.experimental.rotary_embedding_llama(
            q, cos, sin, self._rope_transformation_mat, is_decode_mode=False
        )
        ttnn.deallocate(q)
        k_rot = ttnn.experimental.rotary_embedding_llama(
            k, cos, sin, self._rope_transformation_mat, is_decode_mode=False
        )
        ttnn.deallocate(k)

        # Non-causal SDPA with dynamic grid. Pass scale = 1/sqrt(head_dim) explicitly
        # using the UNPADDED head_dim so SDPA's internal scaling matches the reference
        # (otherwise it would divide by sqrt(padded_head_dim=96) instead of sqrt(72)).
        x = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            attn_mask=None,
            is_causal=False,
            scale=self.head_dim**-0.5,
            program_config=self._sdpa_program_config(q_rot.shape[2]),
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )
        ttnn.deallocate(q_rot)
        ttnn.deallocate(k_rot)
        ttnn.deallocate(v)

        # concat heads: [B, num_local_heads, S, padded_head_dim] → [B, S, num_local_heads * padded_head_dim]
        # = [B, S, 192]
        x = ttnn.transformer.concatenate_heads(x)

        # all_gather on cluster_axis=tp_mesh_axis to recombine heads across chips → [B, S, 1536]
        needs_unsqueeze = len(x.shape) <= 3
        if needs_unsqueeze:
            x = ttnn.unsqueeze(x, 0)
        x = self.ccl_manager.all_gather_persistent_buffer(x, dim=3, mesh_axis=self.tp_mesh_axis)
        if needs_unsqueeze:
            x = ttnn.squeeze(x, 0)

        # o_proj (ColParallel, input replicated [B, S, 1536]) → per-chip [B, S, hidden/tp = 144]
        x = self.o_proj.forward(x)

        # all_gather to restore replicated output [B, S, 1152]
        needs_unsqueeze = len(x.shape) <= 3
        if needs_unsqueeze:
            x = ttnn.unsqueeze(x, 0)
        x = self.ccl_manager.all_gather_persistent_buffer(x, dim=3, mesh_axis=self.tp_mesh_axis)
        if needs_unsqueeze:
            x = ttnn.squeeze(x, 0)
        return x
