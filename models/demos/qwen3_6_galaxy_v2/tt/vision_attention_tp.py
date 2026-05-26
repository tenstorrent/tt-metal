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
import os

import torch

import ttnn
from models.tt_dit.layers.linear import ColParallelLinear
from models.tt_dit.layers.module import Module
from models.tt_dit.parallel.manager import CCLManager
from models.tt_transformers.tt.common import get_rot_transformation_mat


def _cpu_apply_rope_fp32_per_chip(
    q_tt: ttnn.Tensor,
    cos_hf_per_chip: torch.Tensor,
    sin_hf_per_chip: torch.Tensor,
    mesh_device: ttnn.MeshDevice,
    head_dim: int,
    padded_head_dim: int,
    num_local_heads: int,
    cluster_shape,  # tuple (rows, cols)
) -> ttnn.Tensor:
    """CPU-fallback RoPE: gather all heads to CPU, apply HF rotate_half in
    fp32 on unpadded head_dim, re-pad to padded_head_dim, scatter back.

    Different chips hold different heads (TP=8 along cluster_axis=0). After
    gather we have all `num_heads` heads on CPU; after rotation we scatter
    each chip's 2-head slice back via ShardTensor2dMesh(dims=(1, None)).
    """
    import ttnn  # local

    # Gather all 32 chips' data: per-chip [1, num_local_heads, S, padded] →
    # ConcatMeshToTensor(dim=0) gives [32, num_local_heads, S, padded] with
    # row-major chip ordering (chip 0..3 = row 0 cols 0..3; chip 4..7 = row 1...).
    rows, cols = cluster_shape[0], cluster_shape[1]
    t_all = ttnn.to_torch(q_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    # shape: [num_devices=32, num_local_heads=2, S, padded]
    # Slice to ONE col (cols replicate the same heads, so col 0 has all 8 rows'
    # heads — reshape to [8, num_local_heads, S, padded]).
    # In row-major: indices 0, cols, 2*cols, ... = col 0 of each row.
    # For cluster_shape=(8, 4): col-0 chips are 0, 4, 8, 12, 16, 20, 24, 28.
    row_chips = t_all[::cols]  # [rows=8, num_local_heads=2, S, padded]
    # Flatten heads: chip 0 has heads 0..1, chip 1 has heads 2..3, etc.
    all_heads = row_chips.reshape(rows * num_local_heads, *row_chips.shape[2:])
    # shape: [num_heads=16, S, padded]
    all_real = all_heads[..., :head_dim].to(torch.float32)  # [num_heads, S, head_dim]

    # HF rotate_half on unpadded head_dim
    cos = cos_hf_per_chip.to(torch.float32).unsqueeze(0)  # [1, S, head_dim]
    sin = sin_hf_per_chip.to(torch.float32).unsqueeze(0)
    half = head_dim // 2
    x1 = all_real[..., :half]
    x2 = all_real[..., half:]
    rotated_half = torch.cat([-x2, x1], dim=-1)
    q_rot_real = all_real * cos + rotated_half * sin  # [num_heads, S, head_dim]
    # Re-pad to padded_head_dim
    q_rot_padded = torch.nn.functional.pad(q_rot_real, (0, padded_head_dim - head_dim))
    # Reshape back to per-frame layout for scatter:
    # We need to scatter heads across cluster_axis=0 (rows) and replicate across
    # cluster_axis=1 (cols). Use ShardTensor2dMesh(dims=(0, None)).
    # Input to scatter: shape [num_heads, S, padded]
    # Unsqueeze to 4D [1, num_heads, S, padded] for the from_torch.
    q_rot_4d = q_rot_padded.unsqueeze(0).to(torch.bfloat16)
    return ttnn.from_torch(
        q_rot_4d,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(1, None), mesh_shape=cluster_shape),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


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


def _interleaved_rope_perm_indices(head_dim: int, padded_head_dim: int) -> torch.Tensor:
    """Build index tensor that permutes a `padded_head_dim`-wide axis from HF's
    rotate-half convention to the interleaved RoPE convention.

    HF rotate-half:  positions 0..H/2-1 are first half (cos_freq[0..H/2-1]),
                     positions H/2..H-1 are second half (same freqs)
    TTNN interleaved: pair (2k, 2k+1) both use cos_freq[k]

    Mapping: new[2k] = orig[k], new[2k+1] = orig[k + H/2] for k in [0, H/2).
    Pad positions [H..PH) stay at the end (still zeros).

    For qwen3.6: H=72, PH=96. Returns a `[96]` index tensor with the first 72
    indices being the interleaved permutation and positions 72..95 unchanged.
    """
    assert head_dim % 2 == 0, f"head_dim {head_dim} must be even for rotate-half"
    half = head_dim // 2
    interleaved: list[int] = []
    for k in range(half):
        interleaved.append(k)
        interleaved.append(k + half)
    # Pad positions stay at their original indices.
    interleaved.extend(range(head_dim, padded_head_dim))
    return torch.tensor(interleaved, dtype=torch.long)


def _apply_interleaved_rope_perm(
    t: torch.Tensor, *, head_dim_axis: int, head_dim: int, padded_head_dim: int
) -> torch.Tensor:
    """Permute the head_dim axis from rotate-half to interleaved convention.

    See `_interleaved_rope_perm_indices` for the mapping.
    """
    idx = _interleaved_rope_perm_indices(head_dim, padded_head_dim).to(t.device)
    return t.index_select(head_dim_axis, idx)


def _hf_to_meta_qk_head_perm(W: torch.Tensor, *, head_dim_axis: int, head_dim: int) -> torch.Tensor:
    """Permute Q/K weight (or bias) head_dim axis from HF rotate-half to Meta interleaved.

    HF layout per head: [r_0, r_1, ..., r_{H/2-1}, i_0, i_1, ..., i_{H/2-1}]
    Meta layout per head: [r_0, i_0, r_1, i_1, ..., r_{H/2-1}, i_{H/2-1}]

    Identical to `reverse_permute` in tt_transformers/load_checkpoints.py used by
    qwen3_vl, but parameterised on which axis of an arbitrary-rank tensor.
    """
    assert head_dim % 2 == 0, f"head_dim {head_dim} must be even"
    half = head_dim // 2
    if head_dim_axis < 0:
        head_dim_axis = t_dim_normalize(W, head_dim_axis)
    assert W.shape[head_dim_axis] == head_dim, f"axis size {W.shape[head_dim_axis]} != head_dim {head_dim}"
    # Split head_dim axis into (real|imag, k) then transpose to (k, real|imag).
    pre = W.shape[:head_dim_axis]
    post = W.shape[head_dim_axis + 1 :]
    W = W.reshape(*pre, 2, half, *post)
    W = W.transpose(head_dim_axis, head_dim_axis + 1)
    W = W.reshape(*pre, head_dim, *post)
    return W


def t_dim_normalize(t: torch.Tensor, dim: int) -> int:
    return dim if dim >= 0 else t.dim() + dim


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

    def reshape_for_tp(W: torch.Tensor, *, qk_permute: bool) -> torch.Tensor:
        # [hidden, hidden_in] → [num_heads, head_dim, hidden_in]
        W = W.view(num_heads, head_dim, hidden_in)
        # Q and K: permute head_dim from HF [r_0..r_{H/2-1}, i_0..i_{H/2-1}] to
        # Meta interleaved [r_0, i_0, r_1, i_1, ...] so the on-device
        # `rotary_embedding_llama` (interleaved-pair convention) rotates
        # mathematically equivalent positions vs HF rotate_half. V is unchanged.
        if qk_permute:
            W = _hf_to_meta_qk_head_perm(W, head_dim_axis=1, head_dim=head_dim)
        # pad head_dim with zeros: [num_heads, padded_head_dim, hidden_in]
        W = _pad_head_dim(W, head_dim_axis=1, head_dim=head_dim, padded_head_dim=padded_head_dim)
        # [num_heads, padded_head_dim, hidden_in] → [tp_factor, num_local_heads, padded_head_dim, hidden_in]
        W = W.view(tp_factor, num_local_heads, padded_head_dim, hidden_in)
        return W

    Q = reshape_for_tp(Q, qk_permute=True)
    K = reshape_for_tp(K, qk_permute=True)
    V = reshape_for_tp(V, qk_permute=False)

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

    def reshape_for_tp(b: torch.Tensor, *, qk_permute: bool) -> torch.Tensor:
        b = b.view(num_heads, head_dim)
        # Q/K bias: same HF→Meta head_dim permutation as the weight (so the
        # bias contribution lands on the same Meta channel as the weight matmul).
        if qk_permute:
            b = _hf_to_meta_qk_head_perm(b, head_dim_axis=1, head_dim=head_dim)
        b = _pad_head_dim(b, head_dim_axis=1, head_dim=head_dim, padded_head_dim=padded_head_dim)
        b = b.view(tp_factor, num_local_heads, padded_head_dim)
        return b

    Q = reshape_for_tp(Q, qk_permute=True)
    K = reshape_for_tp(K, qk_permute=True)
    V = reshape_for_tp(V, qk_permute=False)

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

    # Convert cos/sin from HF rotate-half layout [c0..c_{H/2-1}, c0..c_{H/2-1}]
    # to Meta interleaved layout [c0, c0, c1, c1, ..., c_{H/2-1}, c_{H/2-1}] so
    # they pair with the Meta-permuted Q/K weights (see reorg_fused_qkv_weight).
    # Identical to qwen3_vl's `convert_rope_style_hf_to_meta`.
    cos = _apply_interleaved_rope_perm(cos, head_dim_axis=-1, head_dim=head_dim, padded_head_dim=head_dim)
    sin = _apply_interleaved_rope_perm(sin, head_dim_axis=-1, head_dim=head_dim, padded_head_dim=head_dim)
    if padded_head_dim != head_dim:
        # Pad to padded_head_dim with cos=1, sin=0 → identity rotation on padded
        # positions, leaves the zero-padded weight columns unaffected.
        cos_pad = torch.ones(seq_len, padded_head_dim - head_dim, dtype=cos.dtype)
        sin_pad = torch.zeros(seq_len, padded_head_dim - head_dim, dtype=sin.dtype)
        cos = torch.cat([cos, cos_pad], dim=-1)
        sin = torch.cat([sin, sin_pad], dim=-1)

    # Upload as [1, 1, S, padded_head_dim] replicated across the full mesh.
    def upload(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
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

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
        *,
        cos_hf_cpu: torch.Tensor | None = None,
        sin_hf_cpu: torch.Tensor | None = None,
    ) -> ttnn.Tensor:
        """Run attention. Input + output are replicated across the TP axis.

        If `QWEN36_VISION_CPU_ROPE=1` and cos_hf_cpu/sin_hf_cpu are provided,
        applies RoPE on CPU in fp32 using HF's exact rotate_half — recovers
        the bf16 RoPE precision floor at the cost of a per-layer D2H+H2D
        round-trip. Used to validate the precision hypothesis.
        """
        # qkv projection → per-chip [B, 1, S, 3 * num_local_heads * padded_head_dim]
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

        # Apply 2D vision RoPE — two paths:
        #  (a) on-device via `rotary_embedding_llama` (bf16-only; ~0.987 single-block PCC)
        #  (b) CPU fp32 fallback via HF's exact rotate_half (slower; precision-control experiment)
        use_cpu_rope = (
            os.environ.get("QWEN36_VISION_CPU_ROPE", "0") == "1" and cos_hf_cpu is not None and sin_hf_cpu is not None
        )
        if use_cpu_rope:
            cs = tuple(self.mesh_device.shape)
            q_rot = _cpu_apply_rope_fp32_per_chip(
                q,
                cos_hf_cpu,
                sin_hf_cpu,
                self.mesh_device,
                self.head_dim,
                self.padded_head_dim,
                self.num_local_heads,
                cs,
            )
            ttnn.deallocate(q)
            k_rot = _cpu_apply_rope_fp32_per_chip(
                k,
                cos_hf_cpu,
                sin_hf_cpu,
                self.mesh_device,
                self.head_dim,
                self.padded_head_dim,
                self.num_local_heads,
                cs,
            )
            ttnn.deallocate(k)
        else:
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
