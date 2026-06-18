# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# TTNN implementation of HunyuanImage3SDPAAttention (prefill, single device).
#
# Mapping to PyTorch reference (ref/attention/attention.py)
# ------------------------------------------------
#  Step | PyTorch                                  | TTNN
#  -----+------------------------------------------+----------------------------------
#   1   | qkv_proj linear                          | ttnn.linear
#   2   | reshape + split Q/K/V                    | ttnn.experimental.nlp_create_qkv_heads
#   3   | apply_rotary_pos_emb(q, k, cos, sin)     | HunyuanTtRoPE2D.forward (split-half RoPE)
#   4   | query_layernorm / key_layernorm (per-head)| ttnn.rms_norm per head via reshape
#   5   | repeat_kv  (GQA expansion)               | slice + concat (interleaved KV repeat)
#   6   | F.scaled_dot_product_attention           | ttnn.transformer.scaled_dot_product_attention
#   7   | concat heads                             | ttnn.experimental.nlp_concat_heads
#   8   | o_proj linear                            | ttnn.linear
#
# References
# ----------
#   models/demos/gpt_oss/tt/attention/prefill.py   — full prefill pipeline
#   models/demos/gpt_oss/tt/attention/operations.py — split/concat/rope helpers
#   models/common/rmsnorm.py                        — HunyuanTtRMSNorm wrapper
#   models/tt_transformers/tt/attention.py          — qkv weight loading pattern (not used directly)

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule

from .rms_norm import HunyuanTtRMSNorm
from .rope_2d import HunyuanTtRoPE2D

_TILE = 32


class HunyuanTtAttention(LightweightModule):
    """
    TTNN prefill attention for HunyuanImage-3.0, single device.

    Args:
        device:           TTNN device.
        state_dict:       Model state_dict (plain torch tensors).
        layer_num:        Layer index (0-based), used to build weight keys.
        hidden_size:      Model hidden dimension (default 4096).
        num_heads:        Number of Q heads (default 32).
        num_kv_heads:     Number of KV heads for GQA (default 8).
        head_dim:         Per-head dimension (default 128).
        use_qk_norm:      Whether to apply per-head RMSNorm on Q/K (default True).
        eps:              RMSNorm epsilon (default 1e-5).
        weight_dtype:     TTNN dtype for weight tensors (default bfloat16).
        weight_cache_path: Optional pathlib.Path for weight caching.
    """

    def __init__(
        self,
        device,
        state_dict: dict,
        layer_num: int = 0,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_kv_heads: int = 8,
        head_dim: int = 128,
        use_qk_norm: bool = True,
        eps: float = 1e-5,
        weight_dtype=ttnn.bfloat16,
        weight_cache_path=None,
        ccl_manager=None,
        tp_axis: int = 1,
        tp_factor: int = 1,
        sp_axis: int = 0,
        sp_factor: int = 1,
    ):
        super().__init__()
        self.device = device
        self.head_dim = head_dim
        self.use_qk_norm = use_qk_norm

        # Sequence parallel (SP) over `sp_axis`: Q stays sequence-sharded ([B,*,S/sp])
        # while K/V are all-gathered to the full sequence inside forward(), so each
        # device produces attention for its own query rows. RoPE is applied to the
        # LOCAL slice (global positions, via sharded cos/sin) BEFORE the gather, so
        # the gathered K/V carry correct per-position rotation.
        self.sp_axis = sp_axis
        self.sp_factor = sp_factor

        # --- Tensor parallel (TP) over `tp_axis` --------------------------------
        # qkv_proj is column-parallel (heads split across TP), o_proj is row-parallel
        # (input/head dim split across TP) followed by an all-reduce over tp_axis.
        # Each device therefore holds num_heads/tp_factor q-heads and
        # num_kv_heads/tp_factor kv-heads; SDPA runs locally on that head shard.
        # tp_factor==1 reproduces the single-device weights byte-for-byte.
        self.ccl = ccl_manager
        self.tp_axis = tp_axis
        self.tp_factor = tp_factor
        assert num_heads % tp_factor == 0, f"{num_heads} q-heads not divisible by TP={tp_factor}"
        assert num_kv_heads % tp_factor == 0, f"{num_kv_heads} kv-heads not divisible by TP={tp_factor}"
        # Per-device (local) head counts — used everywhere in forward().
        self.num_heads = num_heads // tp_factor
        self.num_kv_heads = num_kv_heads // tp_factor

        prefix = f"model.layers.{layer_num}.self_attn"

        # ------------------------------------------------------------------
        # QKV projection weight (column-parallel for TP).
        # Hunyuan ships an interleaved GQA layout; we reorder into per-TP-shard
        # contiguous [Q_local | K_local | V_local] blocks so that a plain contiguous
        # column split across tp_axis lands a balanced head set on each device and
        # nlp_create_qkv_heads(num_heads=local, num_kv_heads=local) pairs them right.
        # For tp_factor==1 this is exactly [Q_all | K_all | V_all] as before.
        # ------------------------------------------------------------------
        grp = num_heads // num_kv_heads
        kv_per = num_kv_heads // tp_factor
        w = state_dict[f"{prefix}.qkv_proj.weight"].to(torch.float32)
        w = w.reshape(num_kv_heads, grp + 2, head_dim, hidden_size)  # [n_kv, grp+2, hd, H]
        q_all = w[:, :grp, :, :]  # [n_kv, grp, hd, H]
        k_all = w[:, grp, :, :]  # [n_kv, hd, H]
        v_all = w[:, grp + 1, :, :]  # [n_kv, hd, H]
        blocks = []
        for s in range(tp_factor):
            ksl = slice(s * kv_per, (s + 1) * kv_per)
            q_s = q_all[ksl].reshape(kv_per * grp * head_dim, hidden_size)
            k_s = k_all[ksl].reshape(kv_per * head_dim, hidden_size)
            v_s = v_all[ksl].reshape(kv_per * head_dim, hidden_size)
            blocks.append(torch.cat([q_s, k_s, v_s], dim=0))  # [(grp+2)*kv_per*hd, H]
        qkv_w = torch.cat(blocks, dim=0).T.contiguous()  # [H, Q+2*KV], shard dim=1 in tp parts

        o_w = state_dict[f"{prefix}.o_proj.weight"].to(torch.float32).T.contiguous()  # [Q_dim, hidden]

        if tp_factor > 1:
            # Replicate on the non-TP axis, shard on tp_axis: qkv on the column
            # (output) dim, o_proj on the row (input) dim. ShardTensor2dMesh dims[ax]
            # is the tensor dim sharded on mesh axis ax (None == replicate).
            mesh_shape = tuple(device.shape)
            qkv_dims = [None, None]
            qkv_dims[tp_axis] = 1  # shard output columns
            o_dims = [None, None]
            o_dims[tp_axis] = 0  # shard input rows
            self.qkv_proj = ttnn.from_torch(
                qkv_w,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=mesh_shape, dims=qkv_dims),
            )
            self.o_proj = ttnn.from_torch(
                o_w,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, mesh_shape=mesh_shape, dims=o_dims),
            )
        else:
            self.qkv_proj = ttnn.as_tensor(
                qkv_w,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=None if weight_cache_path is None else weight_cache_path / f"{prefix}.qkv_proj.weight",
            )
            self.o_proj = ttnn.as_tensor(
                o_w,
                device=device,
                dtype=weight_dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_file_name=None if weight_cache_path is None else weight_cache_path / f"{prefix}.o_proj.weight",
            )

        # ------------------------------------------------------------------
        # Per-head QK norms (weight shape [head_dim] → [1,1,head_dim//TILE,TILE])
        # ------------------------------------------------------------------
        if use_qk_norm:
            self.query_norm = HunyuanTtRMSNorm(
                device=device,
                dim=head_dim,
                state_dict=state_dict,
                weight_key=f"{prefix}.query_layernorm.weight",
                eps=eps,
                weight_dtype=weight_dtype,
                weight_cache_path=weight_cache_path,
            )
            self.key_norm = HunyuanTtRMSNorm(
                device=device,
                dim=head_dim,
                state_dict=state_dict,
                weight_key=f"{prefix}.key_layernorm.weight",
                eps=eps,
                weight_dtype=weight_dtype,
                weight_cache_path=weight_cache_path,
            )

        # ------------------------------------------------------------------
        # RoPE module (CPU cos/sin build + on-device split-half apply)
        # ------------------------------------------------------------------
        self.rope = HunyuanTtRoPE2D(device=device, head_dim=head_dim)

        # Compute kernel config: HiFi2 + fp32 acc for attention matmuls
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def forward(
        self,
        x: ttnn.Tensor,
        cos_tt: ttnn.Tensor,
        sin_tt: ttnn.Tensor,
        attention_mask=None,
    ) -> ttnn.Tensor:
        """
        Prefill forward pass.

        Args:
            x:             Input [B, S, hidden_size] TILE_LAYOUT on device.
            cos_tt:        [1, 1, S, head_dim] from HunyuanTtRoPE2D.prepare_cos_sin().
            sin_tt:        [1, 1, S, head_dim] from HunyuanTtRoPE2D.prepare_cos_sin().
            attention_mask: Optional [B, 1, S, S] TTNN tensor (0=attend, -inf=mask).
                            Pass None to use is_causal=True (pure causal text).

        Returns:
            Output [B, S, hidden_size] on device.
        """
        # Under SP the queries are sequence-sharded, so a plain causal SDPA (which
        # assumes query i attends keys 0..i) would be wrong for non-first shards.
        # The image-gen path always supplies an explicit (sharded) mask; require it.
        if self.sp_factor > 1:
            assert attention_mask is not None, "SP attention requires an explicit (query-sharded) mask"

        # ---- 1. Fused QKV projection ----------------------------------------
        # x: [B, S, H]  →  xqkv: [B, S, Q_dim + 2*KV_dim]
        xqkv = ttnn.linear(
            x,
            self.qkv_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # ---- 2. Split into Q / K / V heads (GQA-aware) ---------------------
        # nlp_create_qkv_heads expects [B, 1, S, QKV_dim] (4D).
        # xqkv from linear is [B, S, QKV_dim] (3D), so unsqueeze dim 1.
        # ttnn.reshape returns a view (same buffer); deallocate xqkv_4d only after
        # nlp_create_qkv_heads — never deallocate xqkv before xqkv_4d is consumed.
        xqkv_4d = ttnn.reshape(xqkv, [xqkv.shape[0], 1, xqkv.shape[1], xqkv.shape[2]])
        # Output shapes:
        #   q: [B, num_heads,    S, head_dim]
        #   k: [B, num_kv_heads, S, head_dim]
        #   v: [B, num_kv_heads, S, head_dim]
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_4d,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        xqkv_4d.deallocate(True)  # frees xqkv's buffer too (reshape is a view)

        # ---- 3. 2D RoPE (fused rotary_embedding_llama) ----------------------
        q_rot, k_rot = self.rope.forward(q, k, cos_tt, sin_tt)
        q.deallocate(True)
        k.deallocate(True)

        # ---- 4. Per-head QK normalisation (after RoPE) ----------------------
        if self.use_qk_norm:
            # q_rot/k_rot are [B, heads, S, head_dim].
            # ttnn.rms_norm normalises the last dim — applies over head_dim dimension,
            # which is exactly what Hunyuan's per-head layernorm does.
            q_rot = self.query_norm(q_rot)
            k_rot = self.key_norm(k_rot)

        # ---- 4b. SP K/V all-gather -----------------------------------------
        # Q stays sequence-sharded; gather K/V along the sequence dim (2) over the
        # SP axis so every device sees the FULL sequence of keys/values. RoPE+QK-norm
        # were applied to the local slice (global positions), so the gathered K/V are
        # already correctly rotated. Done on the kv-head tensors (pre-GQA-expansion)
        # to gather the smaller num_kv_heads payload.
        if self.sp_factor > 1:
            k_rot = self.ccl.all_gather(k_rot, dim=2, mesh_axis=self.sp_axis, use_hyperparams=False)
            v = self.ccl.all_gather(v, dim=2, mesh_axis=self.sp_axis, use_hyperparams=False)

        # ---- 5. GQA expansion: interleaved repeat K/V heads to match Q --------
        # PyTorch GQA: Q head i uses KV head i//grp, so expansion is interleaved:
        #   [K0,K0,K0,K0, K1,K1,K1,K1, ..., K7,K7,K7,K7]
        # ttnn.repeat([1,grp,1,1]) gives [K0..K7, K0..K7, ...] — wrong ordering.
        # Use slice+concat to get the correct interleaved layout.
        if self.num_kv_heads < self.num_heads:
            grp = self.num_heads // self.num_kv_heads
            # K/V seq length is the (possibly SP-gathered) full sequence, which may
            # differ from Q's sharded length — slice each by its OWN seq extent.
            k_seq = k_rot.shape[2]
            v_seq = v.shape[2]
            k_chunks, v_chunks = [], []
            for h in range(self.num_kv_heads):
                kh = ttnn.slice(
                    k_rot, [0, h, 0, 0], [1, h + 1, k_seq, self.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                vh = ttnn.slice(
                    v, [0, h, 0, 0], [1, h + 1, v_seq, self.head_dim], memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                for _ in range(grp):
                    k_chunks.append(kh)
                    v_chunks.append(vh)
            k_rot = ttnn.concat(k_chunks, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            v = ttnn.concat(v_chunks, dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # ---- 6. SDPA -------------------------------------------------------
        is_causal = attention_mask is None
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=is_causal,
            attn_mask=attention_mask,
            compute_kernel_config=self.compute_kernel_config,
        )
        q_rot.deallocate(True)
        k_rot.deallocate(True)
        v.deallocate(True)

        # ---- 7. Merge heads -------------------------------------------------
        # [B, num_heads, S, head_dim] → [B, S, hidden_size]
        attn_out = ttnn.experimental.nlp_concat_heads(
            attn_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ---- 8. Output projection -------------------------------------------
        # Row-parallel under TP: each device multiplies its head-shard's concat
        # output [B,S,Q_dim/tp] by its o_proj rows [Q_dim/tp, H] to get a PARTIAL
        # [B,S,H], then we all-reduce the partials over tp_axis for the full output.
        out = ttnn.linear(
            attn_out,
            self.o_proj,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        attn_out.deallocate(True)

        if self.tp_factor > 1:
            out = self._tp_all_reduce(out)

        return out

    def _tp_all_reduce(self, partial: ttnn.Tensor) -> ttnn.Tensor:
        """All-reduce a [B,1,S,H] o_proj partial over the TP axis (all-gather+sum)."""
        n = self.tp_factor
        gathered = self.ccl.all_gather(partial, dim=0, mesh_axis=self.tp_axis, use_hyperparams=False)
        ttnn.deallocate(partial)
        shape = list(gathered.shape)
        B = shape[0] // n
        gathered = ttnn.reshape(gathered, (n, B, *shape[1:]))
        out = ttnn.sum(gathered, dim=0)
        ttnn.deallocate(gathered)
        return out
