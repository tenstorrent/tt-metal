# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B Gated Attention block for BH GLX 8×4 mesh.

Forked from models/demos/llama3_70b_galaxy/tt/llama_attention.py and adapted for:
1. Output gate: q_proj output is [Q|gate] fused per-head (2×head_dim per head).
   After reshape: query, gate split → apply sigmoid(gate) * attn_out before WO.
2. Partial RoPE: rotary_dim=64 of head_dim=256 via Qwen36RopeSetup.apply_partial_rope.
   (NOT the fused full-head-dim kernel.)
3. Zero-centered QK-norm: q_norm/k_norm weight baked with +1.0 shift.
4. Pad-and-slice for head count: native n_q=24, n_kv=4 padded to 64/8 to fit
   existing Qwen3-32B program configs.
5. head_dim=256 vs 128: reshape constants re-derived for head_dim=256.

Weight layout (per-head interleaved Q+gate in q_proj):
    q_proj.weight: [n_q * head_dim * 2, H] = [12288, 5120]
    Per-head interleave: rows [Q_h0(256) | gate_h0(256) | Q_h1(256) | gate_h1(256) | ...]

Pad-and-slice strategy (matching OLMo pattern):
    - n_q native=24 → padded=64 heads (zero-pad extra head rows in wq and wgate)
    - n_kv native=4 → padded=8 heads (zero-pad extra rows in wk, wv)
    - WO input dim: native 6144 → padded 16384 (zero-pad WO input cols beyond 6144)
    - Padded heads produce zero SDPA output → correct final output

Data flow (prefill):
    x [B, T, H] (replicated or sharded)
    → xqkvg [B, T, (64+64+8+8)*256 = 36864] via fused wqkvg matmul
    → split: q[B,T,64,256], gate[B,T,64,256], k[B,T,8,256], v[B,T,8,256]
    → QK-norm per-head (zero-centered weight = 1+w baked in)
    → transpose to [B, n_heads, T, hd]
    → partial RoPE on Q and K (first 64 dims rotated)
    → KV cache fill
    → GQA expand k,v: 8 → 64 heads
    → SDPA with causal mask → [B, 64, T, 256]
    → transpose + flatten → [B, T, 64*256=16384]
    → sigmoid(gate) * attn_out  (gate: [B, T, 16384])
    → WO [16384, 5120] → [B, T, 5120]
    → all_reduce across mesh rows (cluster_axis=0)

Standalone test usage (replicated input):
    attn = TtQwen36GatedAttention(mesh_device, args, state_dict, layer_num=3)
    out_tt = attn.forward_prefill(x_tt, rot_mats=(cos_tt, sin_tt), kv_cache=None)
"""
from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen3_6_galaxy.tt.qwen36_model_config import TtQwen36ModelArgs

TILE = 32


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _pad_rows(t: torch.Tensor, target_rows: int, value: float = 0.0) -> torch.Tensor:
    """Pad tensor along dim=0 (rows) to target_rows."""
    current = t.shape[0]
    if current >= target_rows:
        return t
    pad_size = target_rows - current
    return torch.nn.functional.pad(t, (0, 0, 0, pad_size), value=value)


def _pad_cols(t: torch.Tensor, target_cols: int, value: float = 0.0) -> torch.Tensor:
    """Pad tensor along last dim (cols) to target_cols."""
    current = t.shape[-1]
    if current >= target_cols:
        return t
    return torch.nn.functional.pad(t, (0, target_cols - current), value=value)


def _make_qknorm_weight_tt(weight_torch: torch.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Create zero-centered QK-norm weight on device.

    Applies (1+w) and stores as [1, head_dim//TILE, TILE] (3-D) row-major,
    replicated across all mesh devices.

    TTNN rms_norm constraint: gamma.padded_shape[-1] == tile_width (32) AND
    gamma.physical_volume / tile_width == input.padded_shape[-1] / tile_width.
    A 3-D weight [1, hd//32, 32] satisfies this for any input with last padded
    dim == hd (including 2-D input [rows, hd] from _apply_qk_norm).

    3-D weight is preferred over 4-D [1, 1, hd//32, 32] because the 4-D format
    can produce wrong results when the input's n_kv_pc dim is 1 due to tile
    padding interactions.
    """
    w = 1.0 + weight_torch.float()
    dim = w.numel()
    assert dim % TILE == 0, f"head_dim={dim} must be tile-aligned"
    w_3d = w.reshape(1, dim // TILE, TILE)  # [1, hd//32, 32]
    return ttnn.from_torch(
        w_3d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _qknorm_flat_to_heads(
    x_flat: "ttnn.Tensor",
    weight: "ttnn.Tensor",
    eps: float,
    B: int,
    n_heads: int,
    T: int,
    hd: int,
    compute_kernel_config,
) -> "ttnn.Tensor":
    """Apply per-head RMSNorm and return [B, n_heads, T, hd].

    Processes each head independently via slice, rms_norm, then concatenates.

    Why not reshape [B, T, n_heads*hd] → [B, n_heads, T, hd] then rms_norm?
    -----------------------------------------------------------------------
    A row-major reshape [B, T, N*hd] → [B, N, T, hd] reinterprets the data
    as N contiguous blocks of T*hd elements each, NOT as N interleaved slots
    of hd within each time step. To get the correct per-head per-time-step
    view, one would need reshape to [B, T, N, hd] + transpose(1,2). But
    n_heads (N=6 or N=1) is < 32, causing tile-padding issues in TILE_LAYOUT
    when that dimension is second-to-last.

    Slice approach:
    -----------------------------------------------------------------------
    For each head h (0..n_heads-1), slice [B, T, hd] from x_flat starting at
    column h*hd. Each such slice has T and hd both tile-aligned → no padding
    issues. Apply rms_norm to [B, T, hd] with weight [1, hd//32, 32]:
      - input.padded_shape[-1] = hd
      - gamma.physical_volume / 32 = hd/32 ✓
      - Normalizes each of B*T rows of hd elements independently.

    Args:
        x_flat: [B, T, n_heads*hd] flat projection from wqkvg split
        weight: [1, hd//32, 32] in ROW_MAJOR_LAYOUT (zero-centered, 1+w baked in)
        eps: RMSNorm epsilon
        B, n_heads, T, hd: logical dimensions
        compute_kernel_config: TTNN compute kernel config

    Returns:
        [B, n_heads, T, hd] per-head normalized tensor.
    """
    if n_heads == 1:
        # Fast path: x_flat is already [B, T, hd] (single head)
        # Apply rms_norm directly, then reshape to [B, 1, T, hd].
        x_normed_3d = ttnn.rms_norm(
            x_flat,
            weight=weight,
            epsilon=eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        # ttnn.reshape creates a VIEW (shares buffer with x_normed_3d).
        # Clone immediately to get an independent buffer before x_normed_3d is freed.
        view_4d = ttnn.reshape(x_normed_3d, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_4d.deallocate(True)
        x_normed_3d.deallocate(True)
        return out  # [B, 1, T, hd]

    # Multi-head path: slice each head, norm, then concat along new head dim.
    # IMPORTANT: ttnn.reshape creates a view (alias) — do NOT deallocate any
    # intermediate tensor while a reshape view of it is still live.  We keep
    # both the normed 3D tensor AND its 4D reshape view alive until after
    # ttnn.concat (which copies data into its own new buffer).
    head_normed_list = []  # [B, T, hd] normed tensors (keep alive until after concat)
    head_tensors = []  # [B, 1, T, hd] reshape views (aliases of head_normed_list)
    for h in range(n_heads):
        # Slice head h: [B, T, hd] (all tile-aligned)
        head_h = ttnn.slice(x_flat, [0, 0, h * hd], [B, T, (h + 1) * hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Per-head rms_norm creates a new independent buffer:
        head_normed = ttnn.rms_norm(
            head_h,
            weight=weight,
            epsilon=eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=compute_kernel_config,
        )
        head_h.deallocate(True)
        # Reshape [B, T, hd] → [B, 1, T, hd]: this is a VIEW of head_normed.
        # Keep head_normed alive (in head_normed_list) until after concat.
        head_normed_4d = ttnn.reshape(head_normed, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        head_normed_list.append(head_normed)
        head_tensors.append(head_normed_4d)

    # Concatenate: concat creates its own new buffer — safe to deallocate views.
    out = ttnn.concat(head_tensors, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # Now safe to release all intermediate tensors
    for t in head_tensors:
        t.deallocate(True)
    for t in head_normed_list:
        t.deallocate(True)
    return out  # [B, n_heads, T, hd]


def _flat_to_heads(
    x_flat: "ttnn.Tensor",
    B: int,
    n_heads: int,
    T: int,
    hd: int,
) -> "ttnn.Tensor":
    """Convert [B, T, n_heads*hd] → [B, n_heads, T, hd] via slice+reshape+concat.

    Avoids tile-padding problems of reshaping n_heads into the second-to-last
    tile dimension (which gets padded to 32 when n_heads < 32).

    For n_heads=1: simple reshape [B, T, hd] → [B, 1, T, hd].
    """
    if n_heads == 1:
        # ttnn.reshape creates a VIEW — clone immediately to get an independent buffer.
        # (Caller typically deallocates x_flat after this call, which would otherwise
        # invalidate the reshape view.)
        view_4d = ttnn.reshape(x_flat, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_4d.deallocate(True)
        return out

    # Keep slices alive until after concat (reshape creates a view, not a copy).
    slice_list = []  # [B, T, hd] slice tensors (keep alive until after concat)
    head_tensors = []  # [B, 1, T, hd] reshape views (aliases of slice_list)
    for h in range(n_heads):
        head_h = ttnn.slice(x_flat, [0, 0, h * hd], [B, T, (h + 1) * hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        head_h_4d = ttnn.reshape(head_h, [B, 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        slice_list.append(head_h)
        head_tensors.append(head_h_4d)

    out = ttnn.concat(head_tensors, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in head_tensors:
        t.deallocate(True)
    for s in slice_list:
        s.deallocate(True)
    return out  # [B, n_heads, T, hd]


def _heads_to_flat(
    x_heads: "ttnn.Tensor",
    B: int,
    n_heads: int,
    T: int,
    hd: int,
) -> "ttnn.Tensor":
    """Convert [B, n_heads, T, hd] → [B, T, n_heads*hd] via slice+reshape+concat.

    Reverses _flat_to_heads. Avoids tile-padding problems.

    For n_heads=1: simple reshape [B, 1, T, hd] → [B, T, hd].
    """
    if n_heads == 1:
        # ttnn.reshape creates a VIEW — clone to get an independent buffer before
        # caller deallocates x_heads.
        view_3d = ttnn.reshape(x_heads, [B, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.clone(view_3d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        view_3d.deallocate(True)
        return out

    # Slice each head [B, 1, T, hd] and reshape to [B, T, hd].
    # Keep slices alive until after concat (reshape creates a view, not a copy).
    slice_list = []  # [B, 1, T, hd] slice tensors (keep alive until concat)
    time_slice_tensors = []  # [B, T, hd] reshape views (aliases of slice_list)
    for h in range(n_heads):
        head_h = ttnn.slice(
            x_heads, [0, h, 0, 0], [B, h + 1, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )  # [B, 1, T, hd]
        head_h_3d = ttnn.reshape(head_h, [B, T, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        slice_list.append(head_h)
        time_slice_tensors.append(head_h_3d)

    out = ttnn.concat(time_slice_tensors, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in time_slice_tensors:
        t.deallocate(True)
    for s in slice_list:
        s.deallocate(True)
    return out  # [B, T, n_heads*hd]


# ---------------------------------------------------------------------------
# Main attention class
# ---------------------------------------------------------------------------


class TtQwen36GatedAttention(LightweightModule):
    """Qwen3.6-27B Gated Multi-Head Attention for BH GLX 8×4 mesh.

    Key differences from TtLlamaAttention (Qwen3-32B):
    - Output gate fused with Q projection (per-head interleaved layout)
    - Partial RoPE (rotary_dim=64, not full head_dim=256)
    - Zero-centered QK-norm weights (baked +1 shift)
    - Native head counts: n_q=24, n_kv=4 (no padding in forward path)

    Weight layout (fused QKVG):
        wqkvg: [Q_native | gate_native | K_native | V_native].T
               = [14336, 5120] → stored as [5120, 14336] for ttnn.linear
    WO weight:
        wo: [H, n_q * head_dim] = [5120, 6144] → stored as [6144, 5120]

    Parameters
    ----------
    mesh_device : ttnn.MeshDevice
        Full 8×4 BH GLX mesh.
    args : TtQwen36ModelArgs
        Model configuration.
    state_dict : dict
        State dict for THIS layer's self_attn block. Expected keys:
        q_proj.weight [12288, 5120], k_proj.weight [1024, 5120],
        v_proj.weight [1024, 5120], o_proj.weight [5120, 6144],
        q_norm.weight [256], k_norm.weight [256]
    layer_num : int
        Layer index (for logging/debug only).
    rope_setup : Qwen36RopeSetup | None
        Optional rope setup for partial RoPE application.
    dtype : ttnn.DataType
        Activation dtype. Default bfloat16.
    kv_cache_max_seq_len : int
        Maximum KV cache sequence length.
    """

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        args: TtQwen36ModelArgs,
        state_dict: dict,
        layer_num: int = 0,
        rope_setup=None,
        dtype=ttnn.bfloat16,
        kv_cache_max_seq_len: int = 512,
        use_paged_kv_cache: bool = False,
        block_size: int = 64,
        max_num_blocks: int = None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype
        self.rope_setup = rope_setup
        self.cluster_shape = args.cluster_shape  # [8, 4]

        # Native dimensions
        self.n_q = args.n_heads  # 24
        self.n_kv = args.n_kv_heads  # 4
        self.head_dim = args.head_dim  # 256
        self.hidden_size = args.dim  # 5120
        self.rotary_dim = args.rope_dim  # 64
        self.eps = args.norm_eps  # 1e-6

        # GQA ratio (native)
        self.gqa_ratio = self.n_q // self.n_kv  # 6

        # Scale
        self.scale = self.head_dim**-0.5

        # KV cache params
        self.kv_cache_max_seq_len = kv_cache_max_seq_len
        self.max_batch_size = args.max_batch_size

        # Paged KV cache params
        self.use_paged_kv_cache = use_paged_kv_cache
        self.block_size = block_size
        if max_num_blocks is None:
            import math as _math

            self.max_num_blocks = _math.ceil(kv_cache_max_seq_len / block_size) * args.max_batch_size
        else:
            self.max_num_blocks = max_num_blocks

        # Compute kernels
        self.compute_kernel_hifi2 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.compute_kernel_hifi4 = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Build weights and KV cache
        self._build_weights(state_dict)
        self._build_kv_cache()

    # ------------------------------------------------------------------
    # Weight construction
    # ------------------------------------------------------------------

    def _build_weights(self, sd: dict):
        """Upload weights with column-parallel sharding across the 4 mesh cols.

        Parallelization plan (cluster_axis=1, 4 cols):
            Each col handles n_q/4=6 Q-heads, 6 gate-heads, n_kv/4=1 K-head, 1 V-head.
            wqkvg sharded by output dim across cols → per col [H, (6+6+1+1)*hd = 3584].
            wo sharded by INPUT dim across cols → per col [6*hd=1536, H].
            Forward: each col runs its own attention independently; wo all-reduces
            partial sums across cols (all_gather + fast_reduce_nc on cluster_axis=1).

        HF q_proj layout: [Q_h0(hd) | gate_h0(hd) | Q_h1(hd) | gate_h1(hd) | ...].
        We de-interleave Q vs gate first, then concat [Q | gate | K | V] by head
        groups so that contiguous slices on the output dim correspond to whole heads.

        Per-chip weight DRAM (BF16, sharded /4 cols):
            wqkvg: 5120 × 3584 × 2 = 36.7 MB    (vs 147 MB replicated)
            wo:    1536 × 5120 × 2 = 15.7 MB    (vs 31 MB replicated bfloat8_b)
            Total per layer per chip: ~52 MB    (vs 178 MB replicated)
        """
        hd = self.head_dim  # 256
        n_q = self.n_q  # 24
        n_kv = self.n_kv  # 4
        H = self.hidden_size  # 5120
        n_cols = self.cluster_shape[1]  # 4
        assert n_q % n_cols == 0, f"n_q={n_q} not divisible by n_cols={n_cols}"
        assert n_kv % n_cols == 0, f"n_kv={n_kv} not divisible by n_cols={n_cols}"
        n_q_per_col = n_q // n_cols  # 6
        n_kv_per_col = n_kv // n_cols  # 1
        self.n_q_per_col = n_q_per_col
        self.n_kv_per_col = n_kv_per_col
        self.q_dim_per_col = n_q_per_col * hd  # 1536
        self.gate_dim_per_col = n_q_per_col * hd  # 1536
        self.k_dim_per_col = n_kv_per_col * hd  # 256
        self.v_dim_per_col = n_kv_per_col * hd  # 256
        # total_per_col = (n_q + n_q + n_kv + n_kv)/n_cols * hd = 3584
        self.total_per_col = self.q_dim_per_col + self.gate_dim_per_col + self.k_dim_per_col + self.v_dim_per_col

        # 1. De-interleave Q and gate from q_proj.weight
        q_proj_w = sd["q_proj.weight"]  # [12288, 5120]
        expected_q = (n_q * 2 * hd, H)
        assert q_proj_w.shape == expected_q, f"q_proj.weight: expected {expected_q}, got {q_proj_w.shape}"
        q_2hd = q_proj_w.reshape(n_q, 2, hd, H)
        wq_native = q_2hd[:, 0, :, :].reshape(n_q * hd, H)  # [6144, 5120]
        wgate_native = q_2hd[:, 1, :, :].reshape(n_q * hd, H)  # [6144, 5120]
        wk_native = sd["k_proj.weight"]  # [1024, 5120]
        wv_native = sd["v_proj.weight"]  # [1024, 5120]

        # 2. Build the col-sharded wqkvg by interleaving HEAD GROUPS per col.
        #    For col c (0..n_cols-1):
        #      [ Q_heads (c*n_q_per_col .. (c+1)*n_q_per_col),
        #        gate_heads same range,
        #        K_heads (c*n_kv_per_col .. (c+1)*n_kv_per_col),
        #        V_heads same range ]
        #    Concatenated along the OUTPUT dim, the i-th col's slice occupies columns
        #    [c*total_per_col, (c+1)*total_per_col]. Then ShardTensor2dMesh on cols
        #    splits cleanly so each col owns its own head group.
        col_blocks = []
        for c in range(n_cols):
            qs = c * n_q_per_col * hd
            qe = (c + 1) * n_q_per_col * hd
            ks = c * n_kv_per_col * hd
            ke = (c + 1) * n_kv_per_col * hd
            col_blocks.append(
                torch.cat(
                    [
                        wq_native[qs:qe],  # 6 Q-heads for this col
                        wgate_native[qs:qe],  # 6 gate-heads for this col
                        wk_native[ks:ke],  # 1 K-head for this col
                        wv_native[ks:ke],  # 1 V-head for this col
                    ],
                    dim=0,
                )
            )  # [total_per_col=3584, H]
        wqkvg = torch.cat(col_blocks, dim=0)  # [total_per_col * n_cols = 14336, H]
        wqkvg_T = wqkvg.T.contiguous()  # [H, total_per_col * n_cols]

        col_parallel = ttnn.ShardTensor2dMesh(
            self.mesh_device, dims=(None, 1), mesh_shape=self.cluster_shape
        )  # split along TENSOR dim=1 across MESH cols (cluster_axis=1)
        self.wqkvg = ttnn.from_torch(
            wqkvg_T,
            device=self.mesh_device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=col_parallel,
        )  # per chip: [H, total_per_col=3584]

        # 3. WO weight: row-parallel — shard the INPUT dim (n_q*hd=6144) across cols.
        #    For col c, the rows that match col c's Q heads: rows [c*n_q_per_col*hd, (c+1)*n_q_per_col*hd]
        wo_native = sd["o_proj.weight"]  # [H=5120, n_q*hd=6144]
        expected_wo = (H, n_q * hd)
        assert wo_native.shape == expected_wo, f"o_proj.weight: expected {expected_wo}, got {wo_native.shape}"
        # Transpose to [n_q*hd, H] = [6144, 5120]; split dim 0 across cols
        wo_T = wo_native.T.contiguous()  # [6144, 5120]
        row_parallel_cols = ttnn.ShardTensor2dMesh(
            self.mesh_device, dims=(None, 0), mesh_shape=self.cluster_shape
        )  # split tensor dim=0 across mesh cols
        self.wo = ttnn.from_torch(
            wo_T,
            device=self.mesh_device,
            dtype=self.dtype,  # BF16 (no quantization per user constraint #2)
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_parallel_cols,
        )  # per chip: [1536, H]

        # 4. QK-norm weights (zero-centered: bake +1 shift) — small, replicate
        self.q_norm_w = _make_qknorm_weight_tt(sd["q_norm.weight"], self.mesh_device)
        self.k_norm_w = _make_qknorm_weight_tt(sd["k_norm.weight"], self.mesh_device)

    # ------------------------------------------------------------------
    # KV cache construction
    # ------------------------------------------------------------------

    def _build_kv_cache(self):
        """Allocate on-device KV cache, sharded across cols.

        Non-paged (use_paged_kv_cache=False):
            Shape: [max_batch_size, n_kv=4, max_seq_len, head_dim]
            Sharded on dim=1 (n_kv) across 4 mesh cols → per chip:
                [max_batch_size, n_kv_per_col=1, max_seq_len, head_dim]

        Paged (use_paged_kv_cache=True):
            Shape: [max_num_blocks, n_kv=4, block_size, head_dim]
            Sharded on dim=1 (n_kv) across 4 mesh cols → per chip:
                [max_num_blocks, n_kv_per_col=1, block_size, head_dim]
            Physical blocks are indexed via page_table at runtime.
        """
        col_shard_kv = ttnn.ShardTensor2dMesh(
            self.mesh_device, dims=(None, 1), mesh_shape=self.cluster_shape
        )  # split tensor dim=1 (n_kv) across mesh cols

        if self.use_paged_kv_cache:
            # Paged cache: [max_num_blocks, n_kv, block_size, head_dim]
            cache_k = torch.zeros(
                self.max_num_blocks,
                self.n_kv,
                self.block_size,
                self.head_dim,
            )
            cache_v = torch.zeros(
                self.max_num_blocks,
                self.n_kv,
                self.block_size,
                self.head_dim,
            )
        else:
            # Non-paged cache: [max_batch_size, n_kv, max_seq_len, head_dim]
            cache_k = torch.zeros(
                self.max_batch_size,
                self.n_kv,
                self.kv_cache_max_seq_len,
                self.head_dim,
            )
            cache_v = torch.zeros(
                self.max_batch_size,
                self.n_kv,
                self.kv_cache_max_seq_len,
                self.head_dim,
            )

        self.layer_past = [
            ttnn.from_torch(
                kv,
                device=self.mesh_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=col_shard_kv,
            )
            for kv in [cache_k, cache_v]
        ]

    # ------------------------------------------------------------------
    # Partial RoPE application
    # ------------------------------------------------------------------

    def _apply_partial_rope(
        self,
        x_tt: ttnn.Tensor,
        cos_tt: ttnn.Tensor,
        sin_tt: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """Apply partial RoPE (first rotary_dim=64 dims) to x_tt.

        Delegates to rope_setup if provided, otherwise implements inline.
        x_tt: [..., head_dim=256]
        cos_tt, sin_tt: [..., rotary_dim=64]
        Returns: same shape as x_tt.
        """
        if self.rope_setup is not None:
            return self.rope_setup.apply_partial_rope(x_tt, cos_tt, sin_tt)

        # Inline implementation (slice → rotate → concat)
        rd = self.rotary_dim  # 64
        hd = self.head_dim  # 256
        shape = list(x_tt.shape)
        ndim = len(shape)

        # Slice rotated and pass-through parts
        b_rot = [0] * ndim
        e_rot = shape[:]
        e_rot[-1] = rd

        b_pass = [0] * ndim
        e_pass = shape[:]
        b_pass[-1] = rd

        x_rot = ttnn.slice(x_tt, b_rot, e_rot)
        x_pass = ttnn.slice(x_tt, b_pass, e_pass)

        # rotate_half: cat([-x2, x1])
        half = rd // 2
        sr = list(x_rot.shape)
        b_x1 = [0] * ndim
        e_x1 = sr[:]
        e_x1[-1] = half
        b_x2 = [0] * ndim
        e_x2 = sr[:]
        b_x2[-1] = half

        x1 = ttnn.slice(x_rot, b_x1, e_x1)
        x2 = ttnn.slice(x_rot, b_x2, e_x2)
        neg_x2 = ttnn.neg(x2)
        rh = ttnn.concat([neg_x2, x1], dim=-1)
        x1.deallocate(True)
        x2.deallocate(True)
        neg_x2.deallocate(True)

        # x_rot * cos + rh * sin
        rc = ttnn.multiply(x_rot, cos_tt)
        rs = ttnn.multiply(rh, sin_tt)
        x_rotated = ttnn.add(rc, rs)
        x_rot.deallocate(True)
        rh.deallocate(True)
        rc.deallocate(True)
        rs.deallocate(True)

        # Concat [rotated | pass-through]
        out = ttnn.concat([x_rotated, x_pass], dim=-1)
        x_rotated.deallocate(True)
        x_pass.deallocate(True)
        return out

    # ------------------------------------------------------------------
    # Forward: prefill
    # ------------------------------------------------------------------

    def forward_prefill(
        self,
        x: ttnn.Tensor,
        rot_mats: tuple,
        user_id: int = 0,
        kv_cache=None,
        page_table=None,
        batch_size: int = 1,
        current_pos=None,
    ) -> ttnn.Tensor:
        """Prefill forward pass.

        Args:
            x: Input tensor. Acceptable shapes:
               - [B, 1, T, H] (standard 4D sharded)
               - [B, T, H]    (3D)
               All chips receive the same tensor (replicated or full).
            rot_mats: (cos_tt, sin_tt) from Qwen36RopeSetup.get_cos_sin_for_prefill.
                      cos_tt shape: [1, 1, T, 64], replicated.
            user_id: Batch slot for KV cache fill (0..max_batch_size-1).
            kv_cache: External (k, v) tuple or None to use internal layer_past.
            page_table: Paged attention table (not used in standalone test path).
            batch_size: Number of users (typically 1 for test path).
            current_pos: Not used for prefill.

        Returns:
            Output tensor, same leading shape as input, last dim H=5120.
        """
        cos_tt, sin_tt = rot_mats

        # Normalize to 3D [B, T, H]
        orig_shape = list(x.shape)
        is_4d = len(orig_shape) == 4
        if is_4d:
            B, _, T, H = orig_shape
            x_3d = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape
            x_3d = x

        hd = self.head_dim  # 256
        # Col-sharded sizes: each chip in col c sees only its own 6 Q-heads + 1 KV-head
        n_q_pc = self.n_q_per_col  # 6
        n_kv_pc = self.n_kv_per_col  # 1
        q_dim_pc = self.q_dim_per_col  # 6*256 = 1536
        g_dim_pc = self.gate_dim_per_col  # 1536
        k_dim_pc = self.k_dim_per_col  # 256
        v_dim_pc = self.v_dim_per_col  # 256
        total_pc = self.total_per_col  # 3584

        # ------------------------------------------------------------------
        # 1. QKVgate projection: x [B, T, H] @ wqkvg_per_col [H, 3584] → [B, T, 3584] per col
        #    (each col holds different Q/K/V data; rows in same col are replicated)
        # ------------------------------------------------------------------
        xqkvg = ttnn.linear(
            x_3d,
            self.wqkvg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_hifi4,
        )

        # ------------------------------------------------------------------
        # 2. Split Q, gate, K, V (per-col sizes)
        # ------------------------------------------------------------------
        q_flat = ttnn.slice(xqkvg, [0, 0, 0], [B, T, q_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim_pc], [B, T, q_dim_pc + g_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k_flat = ttnn.slice(
            xqkvg,
            [0, 0, q_dim_pc + g_dim_pc],
            [B, T, q_dim_pc + g_dim_pc + k_dim_pc],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim_pc + g_dim_pc + k_dim_pc], [B, T, total_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        xqkvg.deallocate(True)

        # ------------------------------------------------------------------
        # 3. QK-norm (per head_dim, zero-centered weight = 1+w baked in)
        # Reshape flat [B, T, n*hd] → [B, n, T, hd] and apply rms_norm in one shot.
        # This avoids the [B, T, n, hd] intermediate that has n < 32 (tile-padding
        # in the n_heads dim would make a subsequent reshape to [B*T*n, 1, hd]
        # physically incompatible).  See _qknorm_flat_to_heads docstring.
        # ------------------------------------------------------------------
        # Keep gate_flat as [B, T, g_dim_pc=1536] for now (apply after SDPA)
        q_rot_pre = _qknorm_flat_to_heads(
            q_flat, self.q_norm_w, self.eps, B, n_q_pc, T, hd, self.compute_kernel_hifi4
        )  # [B, n_q_pc, T, hd]
        k_rot_pre = _qknorm_flat_to_heads(
            k_flat, self.k_norm_w, self.eps, B, n_kv_pc, T, hd, self.compute_kernel_hifi4
        )  # [B, n_kv_pc, T, hd]
        q_flat.deallocate(True)
        k_flat.deallocate(True)

        # V: [B, T, n_kv_pc*hd] → [B, n_kv_pc, T, hd] (no norm for V)
        v_t = _flat_to_heads(v_flat, B, n_kv_pc, T, hd)
        v_flat.deallocate(True)

        # ------------------------------------------------------------------
        # 4-5. Partial RoPE on Q and K (already in [B, n_heads, T, hd] format)
        # ------------------------------------------------------------------
        q_rot = self._apply_partial_rope(q_rot_pre, cos_tt, sin_tt)
        k_rot = self._apply_partial_rope(k_rot_pre, cos_tt, sin_tt)
        q_rot_pre.deallocate(True)
        k_rot_pre.deallocate(True)

        # ------------------------------------------------------------------
        # 6. KV cache fill
        # ------------------------------------------------------------------
        if kv_cache is not None:
            keys_cache, values_cache = kv_cache[0], kv_cache[1]
        else:
            keys_cache, values_cache = self.layer_past[0], self.layer_past[1]

        # KV-cache fill: each col stores its OWN single KV head (per-chip [B, 1, max_seq, hd])
        # k_rot, v_t are per-col [B, n_kv_pc=1, T, hd] — matches the sharded cache layout.
        if page_table is not None and self.use_paged_kv_cache:
            # Paged fill: paged_fill_cache(cache [max_num_blocks, n_kv_pc, block_size, hd],
            #                              input  [1, n_kv_pc, T, hd],  page_table, batch_idx=user_id)
            # k_rot/v_t are [B=1, n_kv_pc, T, hd] — matches the required input shape.
            ttnn.experimental.paged_fill_cache(keys_cache, k_rot, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_cache, v_t, page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(keys_cache, k_rot, user_id % max(self.max_batch_size, 1))
            ttnn.fill_cache(values_cache, v_t, user_id % max(self.max_batch_size, 1))

        # ------------------------------------------------------------------
        # 7. GQA expand K, V (per col): n_kv_pc=1 → n_q_pc=6 heads
        # ------------------------------------------------------------------
        gqa_pc = self.n_q_per_col // self.n_kv_per_col  # 6
        k_exp = ttnn.repeat_interleave(k_rot, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_exp = ttnn.repeat_interleave(v_t, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k_rot.deallocate(True)
        v_t.deallocate(True)

        # ------------------------------------------------------------------
        # 8. SDPA with causal mask (per col — each col attends to its own KV head)
        # ------------------------------------------------------------------
        # Build causal mask on CPU and move to device (replicated — same mask on all chips)
        causal = torch.zeros(B, 1, T, T, dtype=torch.bfloat16)
        causal = causal.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1), float("-inf"))
        mask_tt = ttnn.from_torch(
            causal,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        # attn_out per col: [B, n_q_pc=6, T, hd]
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_exp,
            v_exp,
            is_causal=False,
            attn_mask=mask_tt,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_hifi4,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        mask_tt.deallocate(True)
        k_exp.deallocate(True)
        v_exp.deallocate(True)
        q_rot.deallocate(True)

        # ------------------------------------------------------------------
        # 9. Output gate (per col): sigmoid(gate_flat) * attn_out
        # ------------------------------------------------------------------
        # attn_out: [B, n_q_pc, T, hd] (heads-first)
        # gate_flat: [B, T, q_dim_pc=1536] (time-first, heads-interleaved)
        # Convert attn_out → [B, T, q_dim_pc] via _heads_to_flat for gate multiply.
        attn_flat = _heads_to_flat(attn_out, B, n_q_pc, T, hd)
        attn_out.deallocate(True)

        gate_sig = ttnn.sigmoid(gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat.deallocate(True)
        gated = ttnn.multiply(attn_flat, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_flat.deallocate(True)
        gate_sig.deallocate(True)

        # ------------------------------------------------------------------
        # 10. WO projection: per col [B,T,1536] @ [1536, H] → [B,T,H] PARTIAL per col
        #     Then all-gather + fast_reduce_nc across cols (cluster_axis=1) to sum.
        # ------------------------------------------------------------------
        dense_partial = ttnn.linear(
            gated,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_hifi4,
        )  # [B, T, H] partial per col
        gated.deallocate(True)

        # All-reduce across the 4 cols: gather + sum (same pattern as MLP/DeltaNet)
        gathered = ttnn.all_gather(
            dense_partial,
            dim=0,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # stacks 4 col partials on dim=0
        dense_partial.deallocate(True)

        dense_out = ttnn.experimental.fast_reduce_nc(
            gathered,
            dims=[0],
            output=None,
            compute_kernel_config=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [B, T, H] replicated
        gathered.deallocate(True)

        return dense_out

    # ------------------------------------------------------------------
    # Forward: decode
    # ------------------------------------------------------------------

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats: tuple,
        page_table=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        """Decode step forward pass.

        Args:
            x: [B, 1, 1, H] or [B, 1, H] (T=1 decode token).
            current_pos: ttnn tensor [max_batch_size] int32 with current positions.
            rot_mats: (cos_tt, sin_tt) from Qwen36RopeSetup.get_cos_sin_for_decode.
                      cos_tt: [1, 1, 1, 64], replicated.
            page_table: Paged attention table or None.
            kv_cache: External (k, v) or None to use internal layer_past.

        Returns:
            [B, 1, H] bfloat16.
        """
        cos_tt, sin_tt = rot_mats

        orig_shape = list(x.shape)
        is_4d = len(orig_shape) == 4
        if is_4d:
            B, _, T, H = orig_shape
            x_3d = ttnn.reshape(x, [B, T, H])
        else:
            B, T, H = orig_shape
            x_3d = x

        # TTNN tile layout pads T to the nearest multiple of 32 (minimum 32).
        # For decode mode the true sequence length is always 1, but the on-device tensor
        # may report T=32 due to tile padding.  Slice back to T=1 so that all downstream
        # matmuls, SDPA, and reshapes operate on 1 real token (not 32 with 31 zero-padded
        # rows that would corrupt SDPA outputs and shape-mismatch on return).
        T_logical = 1
        if T > T_logical:
            x_3d = ttnn.slice(x_3d, [0, 0, 0], [B, T_logical, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            T = T_logical
        assert T == 1, f"Decode expects T=1, got T={T}"

        hd = self.head_dim  # 256
        # Per-col sizes (col-sharded attention)
        n_q_pc = self.n_q_per_col  # 6
        n_kv_pc = self.n_kv_per_col  # 1
        q_dim_pc = self.q_dim_per_col  # 1536
        g_dim_pc = self.gate_dim_per_col  # 1536
        k_dim_pc = self.k_dim_per_col  # 256
        v_dim_pc = self.v_dim_per_col  # 256
        total_pc = self.total_per_col  # 3584

        # QKVgate projection (col-sharded weights)
        xqkvg = ttnn.linear(
            x_3d,
            self.wqkvg,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_hifi4,
        )

        # Split per-col
        q_flat = ttnn.slice(xqkvg, [0, 0, 0], [B, T, q_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim_pc], [B, T, q_dim_pc + g_dim_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k_flat = ttnn.slice(
            xqkvg,
            [0, 0, q_dim_pc + g_dim_pc],
            [B, T, q_dim_pc + g_dim_pc + k_dim_pc],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v_flat = ttnn.slice(
            xqkvg, [0, 0, q_dim_pc + g_dim_pc + k_dim_pc], [B, T, total_pc], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        xqkvg.deallocate(True)

        # QK-norm: reshape flat [B, T, n*hd] → [B, n, T, hd] and apply rms_norm.
        # Skips the [B, T, n, hd] intermediate (n < 32 → tile-padding issues).
        # See _qknorm_flat_to_heads docstring for tile-compatibility analysis.
        q_normed = _qknorm_flat_to_heads(
            q_flat, self.q_norm_w, self.eps, B, n_q_pc, T, hd, self.compute_kernel_hifi4
        )  # [B, n_q_pc, T, hd]
        k_normed = _qknorm_flat_to_heads(
            k_flat, self.k_norm_w, self.eps, B, n_kv_pc, T, hd, self.compute_kernel_hifi4
        )  # [B, n_kv_pc, T, hd]
        q_flat.deallocate(True)
        k_flat.deallocate(True)

        # V: [B, T, n_kv_pc*hd] → [B, n_kv_pc, T, hd] (no norm for V)
        v_t = _flat_to_heads(v_flat, B, n_kv_pc, T, hd)
        v_flat.deallocate(True)

        # Partial RoPE (already in [B, n_heads, T, hd] format — no transpose needed)
        q_rot = self._apply_partial_rope(q_normed, cos_tt, sin_tt)
        k_rot = self._apply_partial_rope(k_normed, cos_tt, sin_tt)
        q_normed.deallocate(True)
        k_normed.deallocate(True)

        # KV cache update
        if kv_cache is not None:
            keys_cache, values_cache = kv_cache[0], kv_cache[1]
        else:
            keys_cache, values_cache = self.layer_past[0], self.layer_past[1]

        # current_pos is either an int or a TTNN tensor [max_batch_size] int32.
        # ttnn.update_cache accepts an integer update_index.
        if isinstance(current_pos, int):
            cur_pos_int = current_pos
        else:
            # Gather from all devices (replicated) then take first element
            cur_pos_cpu = ttnn.to_torch(
                current_pos,
                mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0),
            )
            cur_pos_int = int(cur_pos_cpu[0].item())

        # Update KV cache at position cur_pos_int for batch 0.  Each col stores
        # its own 1 KV head (cache per-chip shape varies by paged/non-paged).
        if page_table is not None and self.use_paged_kv_cache:
            # Paged update via paged_update_cache (single-tensor version).
            # The op requires:
            #   - input_tensor to be HEIGHT_SHARDED (not WIDTH_SHARDED)
            #   - cache_tensor to be INTERLEAVED
            #   - page_table int32, ROW_MAJOR
            #   - update_idxs_tensor int32, DRAM INTERLEAVED, shape [batch_size]
            #   - input_tensor.padded_shape()[0] == 1
            #   - input_tensor.padded_shape()[1] == batch_size
            #
            # Our k_rot/v_t are [B=1, n_kv_pc=1, T=1, hd=256] in DRAM (interleaved).
            # Tile-padded: [1, 1, 32, 256]. Physical volume = 1*1*32*256 = 8192.
            # Required shard: HEIGHT_SHARDED, 1 core, shard_shape = [32, 256].
            #
            # We reshape to [1, 1, 32, 256] and shard on 1 core HEIGHT_SHARDED.
            # cur_pos tensor: [B=1] int32, DRAM INTERLEAVED.

            # Build update_idxs_tensor [B=1] int32 DRAM INTERLEAVED
            if isinstance(cur_pos_int, int):
                upd_idxs = ttnn.from_torch(
                    torch.tensor([cur_pos_int], dtype=torch.int32),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            else:
                # current_pos is already a TTNN tensor [max_batch_size]
                # Slice to get [B=1] element
                upd_idxs = current_pos

            # Convert k_rot [B=1, n_kv_pc=1, T=1, hd=256] →
            # physically [1, 1, 32, 256] → height-shard on 1 core → shard_shape [32, 256].
            # paged_update_cache requires padded_shape[0]=1 (dim 0 is 1 ✓).
            # It also requires padded_shape[1] == num_indices == 1 ✓ (n_kv_pc=1 padded=1).
            tile_rows = ((T + TILE - 1) // TILE) * TILE  # 32

            shard_grid = ttnn.CoreGrid(y=1, x=1)
            height_shard_cfg = ttnn.create_sharded_memory_config(
                shape=[tile_rows, hd],
                core_grid=shard_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            k_rot_sharded = ttnn.to_memory_config(k_rot, memory_config=height_shard_cfg)
            v_t_sharded = ttnn.to_memory_config(v_t, memory_config=height_shard_cfg)

            ttnn.experimental.paged_update_cache(
                keys_cache, k_rot_sharded, update_idxs_tensor=upd_idxs, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                values_cache, v_t_sharded, update_idxs_tensor=upd_idxs, page_table=page_table
            )
            k_rot_sharded.deallocate(True)
            v_t_sharded.deallocate(True)
        else:
            ttnn.update_cache(keys_cache, k_rot, cur_pos_int, batch_offset=0)
            ttnn.update_cache(values_cache, v_t, cur_pos_int, batch_offset=0)
        k_rot.deallocate(True)
        v_t.deallocate(True)

        if page_table is not None and self.use_paged_kv_cache:
            # Paged SDPA decode: uses page_table to gather scattered KV blocks.
            # Input shapes:
            #   q: [1, B, n_q_pc, hd]  (needs to be [1, B, n_q_pc, hd])
            #   k: [max_num_blocks, n_kv_pc, block_size, hd]
            #   v: [max_num_blocks, n_kv_pc, block_size, hd]
            #   page_table_tensor: [B, max_blocks_per_seq]
            #   cur_pos_tensor: [B] int32
            #
            # Our q_rot is [B, n_q_pc, T, hd] = [1, 6, 1, 256].
            # paged_scaled_dot_product_attention_decode expects q: [1, B, n_q_pc, hd].
            # Transpose: [B, n_q_pc, 1, hd] → [1, B, n_q_pc, hd]
            q_1bnd = ttnn.permute(q_rot, (2, 0, 1, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            q_rot.deallocate(True)

            if isinstance(current_pos, int):
                cur_pos_tensor_sdpa = ttnn.from_torch(
                    torch.tensor([current_pos] * B, dtype=torch.int32),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            else:
                cur_pos_tensor_sdpa = current_pos

            # Program config for paged SDPA decode.
            # Tree reduction constraint: num_cores_per_head ≤ 2^6 = 64.
            # With batch=1 and n_q_pc=6, use a small grid so (cores / n_q_pc_heads) ≤ 64.
            # (1,1) is the safest: 1 core total → 1 core per batch-head group.
            paged_sdpa_prog_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(1, 1),
                exp_approx_mode=False,
                q_chunk_size=0,
                k_chunk_size=0,
            )
            attn_out_1bnd = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_1bnd,
                keys_cache,
                values_cache,
                page_table,
                cur_pos_tensor=cur_pos_tensor_sdpa,
                scale=self.scale,
                program_config=paged_sdpa_prog_cfg,
                compute_kernel_config=self.compute_kernel_hifi4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            q_1bnd.deallocate(True)
            # attn_out_1bnd: [1, B, n_q_pc, hd] → back to [B, n_q_pc, T=1, hd]
            attn_out = ttnn.permute(attn_out_1bnd, (1, 2, 0, 3), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            attn_out_1bnd.deallocate(True)
        else:
            # Slice KV cache up to cur_pos_int (per-col, n_kv_pc=1 dim).
            # T_kv is the number of VALID tokens; T_kv_pad is tile-aligned (≥ T_kv).
            # Important: the tile-padded region beyond T_kv contains zeros — we must
            # mask those positions with -inf in SDPA so they don't corrupt attention.
            T_kv = cur_pos_int + 1
            T_kv_pad = ((T_kv + TILE - 1) // TILE) * TILE
            k_cached = ttnn.slice(
                keys_cache, [0, 0, 0, 0], [B, n_kv_pc, T_kv_pad, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            v_cached = ttnn.slice(
                values_cache, [0, 0, 0, 0], [B, n_kv_pc, T_kv_pad, hd], memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

            # GQA expand (per col): n_kv_pc=1 → n_q_pc=6
            gqa_pc = n_q_pc // n_kv_pc  # 6
            k_exp = ttnn.repeat_interleave(k_cached, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v_exp = ttnn.repeat_interleave(v_cached, gqa_pc, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k_cached.deallocate(True)
            v_cached.deallocate(True)

            # Build attention mask: mask positions T_kv..T_kv_pad with -inf so the
            # zero-padded KV slots don't corrupt the attention distribution.
            # TTNN SDPA requires mask_shape[2] == q_shape[2] (T_q=1).
            if T_kv_pad > T_kv:
                decode_mask_cpu = torch.zeros(B, 1, T_logical, T_kv_pad, dtype=torch.bfloat16)
                decode_mask_cpu[:, :, :, T_kv:] = float("-inf")  # mask pad slots for q row 0
                decode_mask_tt = ttnn.from_torch(
                    decode_mask_cpu,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            else:
                decode_mask_tt = None

            # Decode SDPA per col: q [B, 6, 1, hd] attends to k/v [B, 6, T_kv_pad, hd]
            attn_out = ttnn.transformer.scaled_dot_product_attention(
                q_rot,
                k_exp,
                v_exp,
                is_causal=False,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_hifi4,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                attn_mask=decode_mask_tt,
            )
            if decode_mask_tt is not None:
                decode_mask_tt.deallocate(True)
            k_exp.deallocate(True)
            v_exp.deallocate(True)
            q_rot.deallocate(True)

        # Output gate (per col)
        # attn_out: [B, n_q_pc, T, hd] (heads-first) → [B, T, q_dim_pc] (time-first)
        attn_flat = _heads_to_flat(attn_out, B, n_q_pc, T, hd)
        attn_out.deallocate(True)

        gate_sig = ttnn.sigmoid(gate_flat, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        gate_flat.deallocate(True)
        gated = ttnn.multiply(attn_flat, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn_flat.deallocate(True)
        gate_sig.deallocate(True)

        # WO projection (row-parallel by input dim across cols) → partial → all-reduce
        dense_partial = ttnn.linear(
            gated,
            self.wo,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_hifi4,
        )
        gated.deallocate(True)

        gathered = ttnn.all_gather(
            dense_partial,
            dim=0,
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        dense_partial.deallocate(True)

        dense_out_full = ttnn.experimental.fast_reduce_nc(
            gathered,
            dims=[0],
            output=None,
            compute_kernel_config=None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered.deallocate(True)

        # fast_reduce_nc tile-pads T to 32; slice back to T_logical=1 for decode.
        out_T = list(dense_out_full.shape)[-2]
        if out_T != T_logical:
            dense_out = ttnn.slice(dense_out_full, [0, 0, 0], [B, T_logical, H], memory_config=ttnn.DRAM_MEMORY_CONFIG)
            dense_out_full.deallocate(True)
        else:
            dense_out = dense_out_full

        return dense_out

    # ------------------------------------------------------------------
    # Unified forward dispatch
    # ------------------------------------------------------------------

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos=None,
        rot_mats=None,
        user_id: int = 0,
        mode: str = "decode",
        page_table=None,
        kv_cache=None,
        batch_size: int = 1,
    ) -> ttnn.Tensor:
        """Dispatch to forward_prefill or forward_decode."""
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                user_id=user_id,
                kv_cache=kv_cache,
                page_table=page_table,
                batch_size=batch_size,
            )
        else:
            return self.forward_decode(
                x,
                current_pos,
                rot_mats,
                page_table=page_table,
                kv_cache=kv_cache,
            )
