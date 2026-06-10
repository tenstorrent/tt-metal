# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN vision attention for dots.ocr.

DotsVisionAttention (modeling_dots_vision.VisionAttention): eager MHA with a
fused QKV projection (no bias), 12 heads x head_dim 128, 2D rotary position
embedding, and a block-diagonal attention mask defined by ``cu_seqlens``
(patches of one image attend only to each other).

TTNN mapping (cf. reference_impl models/demos/qwen25_vl/tt/vision_attention.py):

- fused QKV ``ttnn.linear`` — the HF fused weight already lays the output
  columns out as ``q | k | v`` with contiguous heads, exactly what
  ``ttnn.experimental.nlp_create_qkv_heads`` expects;
- head split/merge via the fused ops ``nlp_create_qkv_heads`` /
  ``nlp_concat_heads`` (mandatory TTNN idiom, no torch equivalent);
- rope via ``ttnn.experimental.rotary_embedding_llama`` — the kernel computes
  the META (pairwise-interleaved) rope convention, so the HF rotate-half
  reference maps onto it by ``reverse_permute``-ing the q/k weight rows and
  converting the cos/sin tables HF->meta (``convert_rope_style_hf_to_meta``),
  the identical recipe qwen25_vl uses for its vision tower; tables are
  computed on host (per ARCHITECTURE.md hybrid_notes: rot_pos_emb tables and
  cu_seqlens stay on host) and passed in as device tensors;
- block-diagonal mask via
  ``ttnn.transformer.windowed_scaled_dot_product_attention`` which takes
  ``cu_seqlens`` directly and generates the mask in-kernel (O(seq) memory).

Sequence padding: callers pad seq to a tile multiple (e.g. 784 -> 800/896)
and pass the UNPADDED boundaries in ``cu_seqlens``; rows past the final
boundary are never attended to and are sliced off by the caller — the same
convention qwen25_vl uses for its padded prefill.

Parallelism plan (ARCHITECTURE.md): vision tower placement=replicate — both
weights are ``ReplicateTensorToMesh`` on the 1x4 mesh, activations stay
replicated, no CCL. On a single device the mesh_mapper degenerates gracefully.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import get_rot_transformation_mat
from models.tt_transformers.tt.load_checkpoints import convert_rope_style_hf_to_meta, reverse_permute


class TtVisionAttention(LightweightModule):
    """dots.ocr vision MHA: fused QKV -> heads -> rope -> windowed SDPA -> proj.

    Args:
        mesh_device: ttnn mesh device handle (weights replicated).
        state_dict: {"qkv.weight": [3*dim, dim], "proj.weight": [dim, dim]}
            torch tensors (HF keys vision_tower.blocks.N.attn.*).
        num_heads: attention heads (dots.ocr vision: 12, head_dim 128).
        dtype: on-device weight/activation dtype.
    """

    def __init__(self, mesh_device, state_dict, num_heads=12, dtype=ttnn.bfloat16):
        super().__init__()
        self.mesh_device = mesh_device
        self.num_heads = num_heads
        # fp32 weights select the high-precision path: fp32 fused QKV +
        # explicit fp32 HF-convention rope, with bf16 only at the (bf16-only)
        # windowed-SDPA kernel boundary. Used by the 42-layer full tower,
        # where per-layer bf16 rope/qkv rounding compounds below the PCC bar.
        self.high_precision = dtype == ttnn.float32

        qkv_w = state_dict["qkv.weight"]  # [3*dim, dim], rows = q | k | v
        proj_w = state_dict["proj.weight"]  # [dim, dim]
        dim = qkv_w.shape[-1]
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        # rotary_embedding_llama computes META (pairwise-interleaved) rope, so
        # reorder the q/k weight rows HF->meta; v is rope-free and QK^T is
        # invariant to the (common) head_dim permutation, so attn output and
        # o_proj are unchanged. The high-precision path applies HF-convention
        # rope explicitly instead, so it keeps the HF row order.
        if not self.high_precision:
            q_w, k_w, v_w = qkv_w[:dim], qkv_w[dim : 2 * dim], qkv_w[2 * dim :]
            q_w = reverse_permute(q_w, num_heads, dim, dim)
            k_w = reverse_permute(k_w, num_heads, dim, dim)
            qkv_w = torch.cat([q_w, k_w, v_w], dim=0)
        # Transpose for x @ W^T; fused columns stay q|k|v with contiguous
        # heads — exactly the nlp_create_qkv_heads input contract.
        self.wqkv = ttnn.from_torch(
            qkv_w.transpose(-2, -1).reshape(1, 1, dim, 3 * dim).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.wo = ttnn.from_torch(
            proj_w.transpose(-2, -1).reshape(1, 1, dim, dim).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        # Single-tile transformation matrix for rotary_embedding_llama prefill.
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(dhead=ttnn.TILE_SIZE),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------
    # Host-side input preparation (rope tables / cu_seqlens stay on host —
    # ARCHITECTURE.md hybrid_notes). Not part of the device forward path.
    # ------------------------------------------------------------------
    def prepare_rope(self, rotary_pos_emb: torch.Tensor, padded_seq: int):
        """Build replicated cos/sin device tensors.

        rotary_pos_emb: [seq, head_dim//2] table from vision_rot_pos_emb.
        Pad rows use cos=1 / sin=0 (identity rotation).

        Default path: bf16 META-convention tables [1, 1, padded_seq, head_dim]
        for rotary_embedding_llama. High-precision path: fp32 HF-convention
        tables pre-expanded to [1, num_heads, padded_seq, head_dim] for the
        explicit rope in forward.
        """
        seq = rotary_pos_emb.shape[0]
        freqs = torch.cat([rotary_pos_emb, rotary_pos_emb], dim=-1).float()  # [seq, head_dim] HF layout
        cos, sin = freqs.cos(), freqs.sin()
        if not self.high_precision:
            # HF half-dim-duplicated -> meta pairwise-interleaved, matching
            # the reverse_permute applied to the q/k weights above.
            cos, sin = convert_rope_style_hf_to_meta(cos, sin)
        if padded_seq > seq:
            cos = torch.cat([cos, torch.ones(padded_seq - seq, cos.shape[-1])], dim=0)
            sin = torch.cat([sin, torch.zeros(padded_seq - seq, sin.shape[-1])], dim=0)

        dtype = ttnn.float32 if self.high_precision else ttnn.bfloat16

        def _to_dev(t):
            t = t.reshape(1, 1, padded_seq, -1)
            if self.high_precision:
                t = t.expand(1, self.num_heads, padded_seq, t.shape[-1]).contiguous()
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return _to_dev(cos), _to_dev(sin)

    def prepare_cu_seqlens(self, cu_seqlens: torch.Tensor):
        """Replicated uint32 ROW_MAJOR cu_seqlens tensor for windowed SDPA."""
        return ttnn.from_torch(
            cu_seqlens.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    # ------------------------------------------------------------------
    # Device forward
    # ------------------------------------------------------------------
    def _apply_rope_fp32(self, t: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Explicit fp32 HF-convention rope: t*cos + rotate_half(t)*sin.

        t / cos / sin: [1, num_heads, padded_seq, head_dim] fp32 TILE.
        rotate_half(t) = concat(-t[..., d/2:], t[..., :d/2]).

        Every intermediate is pinned to L1 interleaved ([1,12,896,128] fp32 is
        ~5.5 MB — comfortably interleaved across the 110-core grid): tracy
        showed the DRAM-default chain at ~512 us/block, L1 pinning is the
        single biggest lever after the bf16-only SDPA kernel.
        """
        L1 = ttnn.L1_MEMORY_CONFIG
        shape = t.shape
        half = self.head_dim // 2
        t1 = ttnn.slice(t, [0, 0, 0, 0], [shape[0], shape[1], shape[2], half], memory_config=L1)
        t2 = ttnn.slice(t, [0, 0, 0, half], [shape[0], shape[1], shape[2], self.head_dim], memory_config=L1)
        neg_t2 = ttnn.neg(t2, memory_config=L1)
        ttnn.deallocate(t2)
        rot = ttnn.concat([neg_t2, t1], dim=-1, memory_config=L1)
        ttnn.deallocate(neg_t2)
        ttnn.deallocate(t1)
        out = ttnn.add(
            ttnn.multiply(t, cos, memory_config=L1),
            ttnn.multiply(rot, sin, memory_config=L1),
            memory_config=L1,
        )
        ttnn.deallocate(rot)
        ttnn.deallocate(t)
        return out

    def forward(self, x_11SH: ttnn.Tensor, rot_mats, cu_seqlens: ttnn.Tensor) -> ttnn.Tensor:
        """x_11SH: [1, 1, padded_seq, dim] TILE_LAYOUT, replicated.

        rot_mats: (cos, sin) from prepare_rope (layout depends on precision
        path — see prepare_rope).
        cu_seqlens: uint32 device tensor of unpadded window boundaries.
        Returns: [1, 1, padded_seq, dim], replicated (rows past the last
        cu_seqlens boundary are padding garbage — caller slices them off).
        """
        act_dtype = ttnn.float32 if self.high_precision else ttnn.bfloat16
        xqkv = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=act_dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        cos, sin = rot_mats
        if self.high_precision:
            # fp32 explicit rope (L1-pinned chain), then drop to bf16 only for
            # the bf16-only windowed-SDPA kernel. The boundary typecasts MUST
            # output DRAM: windowed SDPA TT_FATALs on non-DRAM operands.
            q = ttnn.typecast(self._apply_rope_fp32(q, cos, sin), ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            k = ttnn.typecast(self._apply_rope_fp32(k, cos, sin), ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v = ttnn.typecast(v, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            q = ttnn.experimental.rotary_embedding_llama(q, cos, sin, self.trans_mat, is_decode_mode=False)
            k = ttnn.experimental.rotary_embedding_llama(k, cos, sin, self.trans_mat, is_decode_mode=False)

        attn = ttnn.transformer.windowed_scaled_dot_product_attention(
            q,
            k,
            v,
            cu_seqlens,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.high_precision:
            # Match o_proj input dtype to the fp32 weight (the SDPA output is
            # already bf16-rounded; this costs no precision).
            attn = ttnn.typecast(attn, ttnn.float32)
        # o_proj output follows the residual-stream dtype (fp32 callers keep
        # an fp32 skip connection; bf16 callers are unchanged). The SDPA core
        # itself stays bf16: the windowed SDPA kernel is bf16-only.
        out = ttnn.linear(
            attn,
            self.wo,
            dtype=x_11SH.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)
        return out
