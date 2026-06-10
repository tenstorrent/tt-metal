# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TTNN text attention for dots.ocr.

Qwen2Attention (the dots.ocr text decoder): causal GQA with 12 Q heads /
2 KV heads, head_dim 128, separate q/k/v projections WITH bias, o_proj
without bias, default rope theta=1e6.

TTNN mapping (cf. reference_impl models/tt_transformers/tt/attention.py):

- per-device FUSED QKV ``ttnn.linear`` (+ fused bias) — the host loader
  re-packs HF q/k/v into the ``q | k | v`` column layout that
  ``ttnn.experimental.nlp_create_qkv_heads`` expects;
- head split/merge via the fused ops ``nlp_create_qkv_heads`` /
  ``nlp_concat_heads`` (mandatory TTNN idiom, no torch equivalent);
- explicit fp32 HF-convention rope (slice/neg/concat/mul/add), the same
  high-precision recipe as this model's vision tower;
- explicit fp32 attention core: ``ttnn.matmul`` QK^T -> scale -> causal
  mask add -> ``ttnn.softmax`` -> ``ttnn.matmul`` PV.

Why not ``ttnn.transformer.scaled_dot_product_attention``: the dots.ocr
text layer-0 attention logits reach ±3122 (std 664 — Qwen2 attention-sink
behaviour), so bf16 rounding of q/k perturbs logits by O(10) and scrambles
the softmax; the SDPA kernel is bf16/bf8b-only and measures PCC ~0.92 on
this block (per-stage isolation; all surrounding stages 0.9999+). The
whole attention path therefore runs fp32 weights + fp32 activations with
HiFi4 + fp32 accumulation. Tiny per-chip core (3 heads, hd 128) so the
fp32 cost is negligible at this phase.

Parallelism plan (ARCHITECTURE.md / inventory notes): placement=shard 4-way
— 3 Q heads per chip; kv_replication=2, i.e. each of the 2 KV heads is
REPLICATED onto the 2 chips serving its Q-head group so attention stays
chip-local (do NOT copy tt_transformers' kv-head divisibility assert). The
host loader builds one per-chip fused QKV slice (KV rows replicated up to
the chip's Q-head count, q|k|v) per device and shards the concatenation
with ``ShardTensorToMesh(dim=-1)``; o_proj is row-parallel (``dim=-2``)
and its per-chip PARTIAL sums are combined with a sync all-reduce
(``ttnn.reduce_scatter`` + ``ttnn.all_gather``, fp32 fabric accumulation;
swapped from all_gather + local adds in the optimization phase per
tp-guidance). On a single device the sharding degenerates to the
replicated full computation and the CCL is skipped.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtTextAttention(LightweightModule):
    """dots.ocr text GQA: fused QKV(+bias) -> heads -> fp32 rope -> fp32 causal attention -> o_proj -> all-reduce.

    Args:
        mesh_device: ttnn mesh device handle (1xN line; weights TP-sharded).
        state_dict: {"q_proj.weight": [1536, 1536], "q_proj.bias": [1536],
            "k_proj.weight": [256, 1536], "k_proj.bias": [256],
            "v_proj.weight": [256, 1536], "v_proj.bias": [256],
            "o_proj.weight": [1536, 1536]} torch tensors
            (HF keys model.layers.N.self_attn.*).
        num_heads: Q heads (dots.ocr text: 12, head_dim 128).
        num_kv_heads: KV heads (dots.ocr text: 2; kv_replication onto the
            chips of each Q-head group, then up to heads_per_device).
        dtype: on-device weight/activation dtype (fp32 default — see module
            docstring; bf16 loses the softmax to logit quantization).
    """

    def __init__(self, mesh_device, state_dict, num_heads=12, num_kv_heads=2, dtype=ttnn.float32):
        super().__init__()
        self.mesh_device = mesh_device
        num_devices = mesh_device.get_num_devices()
        self.num_devices = num_devices

        q_w = state_dict["q_proj.weight"]  # [nh*hd, hidden]
        k_w = state_dict["k_proj.weight"]  # [nkv*hd, hidden]
        v_w = state_dict["v_proj.weight"]  # [nkv*hd, hidden]
        q_b = state_dict["q_proj.bias"]
        k_b = state_dict["k_proj.bias"]
        v_b = state_dict["v_proj.bias"]
        o_w = state_dict["o_proj.weight"]  # [hidden, nh*hd], no bias

        hidden = q_w.shape[-1]
        head_dim = q_w.shape[0] // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5

        assert num_heads % num_devices == 0, f"{num_heads} Q heads on {num_devices} devices"
        hpd = num_heads // num_devices  # Q heads per device (qb 1x4: 3)
        self.heads_per_device = hpd

        # kv_replication: each chip serves ONE KV-head group; replicate that
        # head's K/V rows up to hpd so the fp32 attention core is plain MHA
        # (q/k/v all hpd heads, no GQA broadcast).
        per_dev_w, per_dev_b = [], []
        for d in range(num_devices):
            q_rows = slice(d * hpd * head_dim, (d + 1) * hpd * head_dim)
            g = (d * num_kv_heads) // num_devices  # this chip's KV head
            kv_rows = slice(g * head_dim, (g + 1) * head_dim)
            wk = k_w[kv_rows].repeat(hpd, 1)
            wv = v_w[kv_rows].repeat(hpd, 1)
            bk = k_b[kv_rows].repeat(hpd)
            bv = v_b[kv_rows].repeat(hpd)
            per_dev_w.append(torch.cat([q_w[q_rows], wk, wv], dim=0))
            per_dev_b.append(torch.cat([q_b[q_rows], bk, bv], dim=0))
        fused_w = torch.cat(per_dev_w, dim=0)  # [N*3*hpd*hd, hidden]
        fused_b = torch.cat(per_dev_b, dim=0)

        shard_cols = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        # Transpose for x @ W^T; ShardTensorToMesh(dim=-1) hands each device
        # exactly its pre-packed q|k|v slice (column-parallel QKV).
        self.wqkv = ttnn.from_torch(
            fused_w.transpose(-2, -1).reshape(1, 1, hidden, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard_cols,
        )
        self.bqkv = ttnn.from_torch(
            fused_b.reshape(1, 1, 1, -1),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=shard_cols,
        )
        # o_proj row-parallel: input rows follow the Q-head order, so a plain
        # dim=-2 shard of W^T gives chip d the rows of its own 3 heads; the
        # matmul yields a PARTIAL [.., hidden] sum per chip (all-reduced in
        # forward).
        self.wo = ttnn.from_torch(
            o_w.transpose(-2, -1).reshape(1, 1, num_heads * head_dim, hidden).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-2) if num_devices > 1 else replicate,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    # ------------------------------------------------------------------
    # Host-side input preparation (rope tables stay on host —
    # ARCHITECTURE.md hybrid_notes). Not part of the device forward path.
    # ------------------------------------------------------------------
    def prepare_rope(self, cos: torch.Tensor, sin: torch.Tensor):
        """Replicated fp32 HF-convention cos/sin device tensors.

        cos/sin: [batch, seq, head_dim] HF-convention tables from
        reference text_rope_cos_sin (theta=1e6). Pre-expanded to
        [1, heads_per_device, seq, head_dim] for the explicit fp32 rope.
        """

        def _to_dev(t):
            t = t.float().reshape(1, 1, t.shape[-2], t.shape[-1])
            t = t.expand(1, self.heads_per_device, t.shape[-2], t.shape[-1]).contiguous()
            return ttnn.from_torch(
                t,
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        return _to_dev(cos.squeeze(0)), _to_dev(sin.squeeze(0))

    def prepare_causal_mask(self, seq: int):
        """Replicated fp32 additive causal mask [1, heads_per_device, seq, seq]."""
        mask = torch.triu(torch.full((seq, seq), torch.finfo(torch.float32).min / 2), diagonal=1)
        mask = mask.reshape(1, 1, seq, seq).expand(1, self.heads_per_device, seq, seq).contiguous()
        return ttnn.from_torch(
            mask,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

    # ------------------------------------------------------------------
    # Device forward (causal prefill over the full sequence)
    # ------------------------------------------------------------------
    def _apply_rope_fp32(self, t: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Explicit fp32 HF-convention rope: t*cos + rotate_half(t)*sin."""
        shape = t.shape
        half = self.head_dim // 2
        t1 = ttnn.slice(t, [0, 0, 0, 0], [shape[0], shape[1], shape[2], half])
        t2 = ttnn.slice(t, [0, 0, 0, half], [shape[0], shape[1], shape[2], self.head_dim])
        neg_t2 = ttnn.neg(t2)
        ttnn.deallocate(t2)
        rot = ttnn.concat([neg_t2, t1], dim=-1)
        ttnn.deallocate(neg_t2)
        ttnn.deallocate(t1)
        out = ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(rot, sin))
        ttnn.deallocate(rot)
        ttnn.deallocate(t)
        return out

    def forward(self, x_11SH: ttnn.Tensor, rot_mats, causal_mask: ttnn.Tensor) -> ttnn.Tensor:
        """x_11SH: [1, 1, seq, hidden] fp32 TILE_LAYOUT, replicated on the mesh.

        rot_mats: (cos, sin) from prepare_rope, [1, hpd, seq, head_dim] fp32.
        causal_mask: from prepare_causal_mask, [1, hpd, seq, seq] fp32.
        Returns: [1, 1, seq, hidden] fp32, replicated (all-reduced o_proj output).
        """
        xqkv = ttnn.linear(
            x_11SH,
            self.wqkv,
            bias=self.bqkv,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Chip-local heads: 3 Q + the chip's KV head replicated to 3 (MHA core).
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.heads_per_device,
            num_kv_heads=self.heads_per_device,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        cos, sin = rot_mats
        q = self._apply_rope_fp32(q, cos, sin)
        k = self._apply_rope_fp32(k, cos, sin)

        # Explicit fp32 causal attention (bf16 SDPA kernel scrambles the
        # ±3000-magnitude Qwen2 layer-0 logits — see module docstring).
        k_t = ttnn.transpose(k, -2, -1)
        ttnn.deallocate(k)
        scores = ttnn.matmul(
            q,
            k_t,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k_t)
        scores = ttnn.multiply(scores, self.scale)
        scores = ttnn.add(scores, causal_mask)
        probs = ttnn.softmax(scores, dim=-1, numeric_stable=True)
        ttnn.deallocate(scores)
        attn = ttnn.matmul(
            probs,
            v,
            dtype=ttnn.float32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(probs)
        ttnn.deallocate(v)

        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Row-parallel o_proj: per-chip PARTIAL sum over its 3 heads' rows.
        out = ttnn.linear(
            attn,
            self.wo,
            dtype=x_11SH.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)

        if self.num_devices > 1:
            # All-reduce of the per-chip partials: reduce_scatter (each chip
            # sums its hidden/N shard, fp32 fabric accumulation) + all_gather
            # to re-replicate. Replaces the original all_gather + N slices +
            # N-1 local adds (full 4*hidden payload gathered then summed on
            # 110-core BinaryNg) — tracy tick-28 A/B: 364.8 -> 281.3 us/iter
            # (-23%), PCC unchanged.
            reduced = ttnn.reduce_scatter(out, dim=3, topology=ttnn.Topology.Linear)
            ttnn.deallocate(out)
            out = ttnn.all_gather(reduced, dim=3, topology=ttnn.Topology.Linear)
            ttnn.deallocate(reduced)
        return out
