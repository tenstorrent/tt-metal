# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 GQA attention for dots.ocr prefill (replicated-hidden Megatron design).

Head sharding with GQA (heads=12, kv_heads=2, head_dim=128) across nd=4 chips:
  - 3 Q-heads per chip.
  - Each chip holds the single KV head its Q-heads attend to (KV head 0 is
    replicated on chips 0-1, KV head 1 on chips 2-3). So per-chip the layout is
    pure MHA with num_heads=3, num_kv_heads=1.
  - qkv_proj column-parallel: per-chip fused weight [Q3 | K1 | V1] (width 640),
    sharded across chips along the output dim.
  - o_proj row-parallel: input is the chip's 3-head attention output (384), the
    weight is sharded along the contraction dim, partial sums all-reduced to the
    full replicated hidden.

Input  x   : replicated [B, S, H]
Output out : replicated [B, S, H]
"""

import torch
import ttnn

from models.experimental.dots_ocr_tp4.tt.common import all_reduce, mesh_num_devices, shard_to_mesh
from models.experimental.tt_symbiote.modules.rope import BailingRotarySetup


class DotsOCRAttentionTP4:
    def __init__(self, mesh_device, config, layer_idx=0, weight_dtype=ttnn.bfloat16):
        self.mesh_device = mesh_device
        self.config = config
        self.layer_idx = layer_idx
        self.weight_dtype = weight_dtype
        self.num_devices = max(1, mesh_num_devices(mesh_device))

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5

        nd = self.num_devices
        assert self.num_heads % nd == 0, f"heads {self.num_heads} not divisible by {nd}"
        self.q_heads_per_chip = self.num_heads // nd
        self.kv_groups = self.num_heads // self.num_kv_heads
        assert (
            self.kv_groups % self.q_heads_per_chip == 0
        ), "a chip's Q-heads must stay within one KV group for single-KV-head sharding"
        self.kv_heads_per_chip = 1

        self.qkv_w = None
        self.qkv_bias = None
        self.o_w = None
        self.o_bias = None
        self._rotary_setup = None

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.sdpa_program_config = None
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(cls, mesh_device, config, torch_attn, layer_idx=0, weight_dtype=ttnn.bfloat16):
        m = cls(mesh_device, config, layer_idx=layer_idx, weight_dtype=weight_dtype)
        m.load_weights(torch_attn)
        return m

    def _chip_kv_index(self, chip: int) -> int:
        """Which KV head this chip's Q-heads attend to."""
        first_q_head = chip * self.q_heads_per_chip
        return first_q_head // self.kv_groups

    def load_weights(self, torch_attn):
        nd = self.num_devices
        hd = self.head_dim
        qhpc = self.q_heads_per_chip

        q_w = torch_attn.q_proj.weight.data  # [q_size, H]
        k_w = torch_attn.k_proj.weight.data  # [kv_size, H]
        v_w = torch_attn.v_proj.weight.data  # [kv_size, H]
        has_bias = getattr(torch_attn.q_proj, "bias", None) is not None
        q_b = torch_attn.q_proj.bias.data if has_bias else None
        k_b = torch_attn.k_proj.bias.data if has_bias else None
        v_b = torch_attn.v_proj.bias.data if has_bias else None

        # Build the stacked per-chip fused QKV weight: rows ordered
        # [chip0(Q3|K1|V1) ; chip1(...) ; ...], shape [nd*640, H].
        w_blocks = []
        b_blocks = []
        for c in range(nd):
            kv = self._chip_kv_index(c)
            q_rows = q_w[c * qhpc * hd : (c + 1) * qhpc * hd]  # [qhpc*hd, H]
            k_rows = k_w[kv * hd : (kv + 1) * hd]  # [hd, H]
            v_rows = v_w[kv * hd : (kv + 1) * hd]  # [hd, H]
            w_blocks.append(torch.cat([q_rows, k_rows, v_rows], dim=0))
            if has_bias:
                qb = q_b[c * qhpc * hd : (c + 1) * qhpc * hd]
                kb = k_b[kv * hd : (kv + 1) * hd]
                vb = v_b[kv * hd : (kv + 1) * hd]
                b_blocks.append(torch.cat([qb, kb, vb], dim=0))

        fused_w = torch.cat(w_blocks, dim=0)  # [nd*640, H]
        # ttnn.linear wants [K=H, N]; transpose then shard the N dim across chips.
        self.qkv_w = shard_to_mesh(fused_w.t().contiguous(), self.mesh_device, dim=-1, dtype=self.weight_dtype)
        if has_bias:
            fused_b = torch.cat(b_blocks, dim=0).reshape(1, -1)  # [1, nd*640]
            self.qkv_bias = shard_to_mesh(fused_b, self.mesh_device, dim=-1, dtype=ttnn.bfloat16)

        # o_proj row-parallel: weight [H, q_size]; shard the contraction dim
        # (q_size) which lines up 1:1 with the per-chip Q-head columns.
        o_w = torch_attn.o_proj.weight.data  # [H, q_size]
        self.o_w = shard_to_mesh(o_w.t().contiguous(), self.mesh_device, dim=0, dtype=self.weight_dtype)
        o_has_bias = getattr(torch_attn.o_proj, "bias", None) is not None
        if o_has_bias:
            # o_proj bias is added once to the full (reduced) output -> replicate.
            from models.experimental.dots_ocr_tp4.tt.common import to_replicated

            self.o_bias = to_replicated(
                torch_attn.o_proj.bias.data.reshape(1, -1), self.mesh_device, dtype=ttnn.bfloat16
            )

        self._rotary_setup = BailingRotarySetup(
            device=self.mesh_device,
            head_dim=self.head_dim,
            max_seq_len=self.config.max_position_embeddings,
            rope_theta=self.config.rope_theta,
            partial_rotary_factor=1.0,
            rope_convention="half_half",
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        batch_size = int(x.shape[0])
        seq_len = int(x.shape[-2])

        qkv = ttnn.linear(
            x,
            self.qkv_w,
            bias=self.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        qkv = ttnn.reshape(qkv, (batch_size, 1, seq_len, -1))

        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.q_heads_per_chip,
            num_kv_heads=self.kv_heads_per_chip,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        cos, sin = self._rotary_setup.get_cos_sin_for_prefill(seq_len)
        q = ttnn.experimental.rotary_embedding(q, cos, sin)
        k = ttnn.experimental.rotary_embedding(k, cos, sin)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            scale=self.scaling,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        attn = ttnn.squeeze(attn, 1)  # [B, S, q_heads_per_chip*head_dim]

        out = ttnn.linear(
            attn,
            self.o_w,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(attn)

        out = all_reduce(out, self.mesh_device)
        if self.o_bias is not None:
            out = ttnn.add(out, self.o_bias)
        return out
