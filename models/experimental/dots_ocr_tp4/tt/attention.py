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

from models.experimental.dots_ocr_tp4.tt.common import (
    all_reduce,
    matmul_m_dim,
    mesh_num_devices,
    prefill_matmul_2d_config,
    shard_to_mesh,
)
from models.experimental.tt_symbiote.core.module import TTNNModule
from models.experimental.tt_symbiote.modules.rope import BailingRotarySetup


class DotsOCRAttentionTP4(TTNNModule):
    def __init__(self, mesh_device, config, layer_idx=0, weight_dtype=ttnn.bfloat16):
        super().__init__()
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

        # Precision recipe matched to the production dots.ocr prefill profile:
        #   qkv : BF16 x BFP8 -> BF16 @ HiFi2
        #   o   : BF16 x BFP4 -> BFP8 @ LoFi
        #   sdpa: BF16 @ LoFi (math_approx)
        self.qkv_weight_dtype = ttnn.bfloat8_b
        self.o_weight_dtype = ttnn.bfloat4_b
        self.qkv_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.o_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self.sdpa_program_config = None
        # SDPA is BF16 in/out (matches the profile); fidelity is internal and
        # not constrained by the dtype recipe, so keep it at HiFi2 for accuracy.
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def from_torch(cls, mesh_device, config, torch_attn, layer_idx=0, weight_dtype=ttnn.bfloat16):
        m = cls(mesh_device, config, layer_idx=layer_idx, weight_dtype=weight_dtype)
        m.load_weights(torch_attn)
        m.to_device(mesh_device)
        # Weights are sharded to device in load_weights (construction-time), so the
        # TTNNModule weight lifecycle is already satisfied -- mark it ready.
        m._preprocessed_weight = True
        m._weights_on_device = True
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
        self.qkv_w = shard_to_mesh(fused_w.t().contiguous(), self.mesh_device, dim=-1, dtype=self.qkv_weight_dtype)
        if has_bias:
            fused_b = torch.cat(b_blocks, dim=0).reshape(1, -1)  # [1, nd*640]
            self.qkv_bias = shard_to_mesh(fused_b, self.mesh_device, dim=-1, dtype=ttnn.bfloat16)

        # o_proj row-parallel: weight [H, q_size]; shard the contraction dim
        # (q_size) which lines up 1:1 with the per-chip Q-head columns.
        o_w = torch_attn.o_proj.weight.data  # [H, q_size]
        self.o_w = shard_to_mesh(o_w.t().contiguous(), self.mesh_device, dim=0, dtype=self.o_weight_dtype)
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

        # Decode SDPA program config (q/k chunk 0 -> kernel picks per cur_pos).
        try:
            grid = self.mesh_device.compute_with_storage_grid_size()
            self._sdpa_decode_program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                q_chunk_size=0,
                k_chunk_size=0,
                exp_approx_mode=False,
            )
            # Prefill SDPA config: hardware-swept on BH P150x4 for the per-chip
            # q[1,3,S,128]/kv[1,1,S,128] causal shape -> grid 8x10, q_chunk=128,
            # k_chunk=256 (~210 us, faster than the auto-config). exp_approx left
            # off to preserve PCC (it was speed-neutral, ~0.3%). Needs a >=8x10
            # grid; otherwise leave the auto-config (None).
            if int(grid.x) >= 8 and int(grid.y) >= 10:
                self.sdpa_program_config = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 10),
                    q_chunk_size=128,
                    k_chunk_size=256,
                    exp_approx_mode=False,
                )
        except Exception:
            self._sdpa_decode_program_config = None

    def _project_qkv_heads(self, x, batch_size, seq_len, heads_mem_config):
        """qkv linear (+bias) -> per-chip [Q3 | K1 | V1] -> create heads.

        Kept DRAM-interleaved: q/k/v feed the paged KV kernels, which require a
        DRAM (or sharded) Q for ``paged_sdpa_decode`` and sharded K/V for
        ``paged_update`` -- L1-interleaved q/k/v is rejected. The decode L1 win is
        in the residual stream (RMSNorm, residual adds, o_proj/MLP + all-reduce),
        not these tiny immediately-consumed head tensors.
        """
        qkv_pc = prefill_matmul_2d_config(
            self.mesh_device, matmul_m_dim(x), int(x.shape[-1]), int(self.qkv_w.shape[-1]), fp32_dest=False
        )
        qkv = ttnn.linear(
            x,
            self.qkv_w,
            bias=self.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.qkv_compute,
            program_config=qkv_pc,
        )
        qkv = ttnn.reshape(qkv, (batch_size, 1, seq_len, -1))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.q_heads_per_chip,
            num_kv_heads=self.kv_heads_per_chip,
            transpose_k_heads=False,
            memory_config=heads_mem_config,
        )
        ttnn.deallocate(qkv)
        return q, k, v

    def _o_proj_all_reduce(self, attn):
        # Decode (seq==1) keeps the o_proj matmul + all-reduce L1-resident.
        is_decode = int(attn.shape[-2]) == 1
        out_mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        o_pc = prefill_matmul_2d_config(
            self.mesh_device, matmul_m_dim(attn), int(attn.shape[-1]), int(self.o_w.shape[-1]), fp32_dest=False
        )
        out = ttnn.linear(
            attn,
            self.o_w,
            dtype=ttnn.bfloat8_b,
            memory_config=out_mc,
            compute_kernel_config=self.o_compute,
            program_config=o_pc,
        )
        ttnn.deallocate(attn)
        out = all_reduce(out, self.mesh_device, output_memory_config=out_mc)
        if self.o_bias is not None:
            out = ttnn.add(out, self.o_bias, memory_config=out_mc)
        return out

    def forward(self, x: ttnn.Tensor, past_key_value=None, cache_position=None) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        seq_len = int(x.shape[-2])
        if past_key_value is not None and seq_len == 1:
            return self._forward_decode(x, past_key_value, cache_position)
        return self._forward_prefill(x, past_key_value)

    def _forward_prefill(self, x: ttnn.Tensor, past_key_value=None) -> ttnn.Tensor:
        batch_size = int(x.shape[0])
        seq_len = int(x.shape[-2])

        q, k, v = self._project_qkv_heads(x, batch_size, seq_len, ttnn.DRAM_MEMORY_CONFIG)

        cos, sin = self._rotary_setup.get_cos_sin_for_prefill(seq_len)
        q = ttnn.experimental.rotary_embedding(q, cos, sin)
        k = ttnn.experimental.rotary_embedding(k, cos, sin)

        # ``rotary_embedding`` materializes the tile-padded seq dim, so for a
        # non-tile-multiple ``seq_len`` (e.g. 2814) Q/K come back padded to 2816
        # while V (no rotary) stays at ``seq_len`` -- SDPA then rejects the K/V
        # length mismatch. Slice Q/K back to the real ``seq_len``. No-op when the
        # input was already tile-aligned (the standalone TP4 model pads on host).
        if int(q.shape[2]) != seq_len:
            q = q[:, :, :seq_len, :]
        if int(k.shape[2]) != seq_len:
            k = k[:, :, :seq_len, :]

        # Populate the paged KV cache (per chip: this chip's 1 KV head, rotated K
        # + raw V, bf16 as paged_fill_cache requires) so decode can read it.
        if past_key_value is not None:
            past_key_value.paged_fill_on_device(k, v, layer_idx=self.layer_idx, batch_idx=0)

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
        return self._o_proj_all_reduce(attn)

    def _forward_decode(self, x: ttnn.Tensor, past_key_value, cache_position) -> ttnn.Tensor:
        """Single-token decode reading/writing the paged KV cache (per chip)."""
        batch_size = int(x.shape[0])
        cur_pos = cache_position  # ttnn int32 [batch] = position of the new token

        q, k, v = self._project_qkv_heads(x, batch_size, 1, ttnn.DRAM_MEMORY_CONFIG)

        cos, sin = self._rotary_setup.get_cos_sin_for_decode(cur_pos)
        q = ttnn.experimental.rotary_embedding(q, cos, sin)
        k = ttnn.experimental.rotary_embedding(k, cos, sin)

        # rotary_embedding materializes the tile-padded seq dim (S=1 -> 32);
        # slice it back to the single decode token before the paged kernels,
        # otherwise the permute leaves a 32-tall seq that over-shards K/V.
        if int(q.shape[2]) != 1:
            q = q[:, :, :1, :]
        if int(k.shape[2]) != 1:
            k = k[:, :, :1, :]

        # [B, H, S=1, D] -> [S=1, B, H, D] for the paged kernels.
        q = ttnn.permute(q, (2, 0, 1, 3))
        kv_key = ttnn.permute(k, (2, 0, 1, 3))
        kv_value = ttnn.permute(v, (2, 0, 1, 3))
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Height-shard K/V for paged_update (it requires sharded input). One
        # batch per core; the per-core shard is [TILE, head_dim] holding this
        # chip's single KV head (padded to a tile). Use the explicit shard-shape
        # form so 1 KV head -> exactly 1 shard (the auto-divide form mis-counts
        # shards when num_kv_heads/chip == 1).
        shard_cfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=ttnn.CoreGrid(y=1, x=batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        kv_key = ttnn.to_memory_config(kv_key, shard_cfg)
        kv_value = ttnn.to_memory_config(kv_value, shard_cfg)
        past_key_value.paged_update_on_device(kv_key, kv_value, layer_idx=self.layer_idx, current_pos=cur_pos)
        ttnn.deallocate(kv_key)
        ttnn.deallocate(kv_value)

        attn = past_key_value.paged_sdpa_decode(
            q,
            self.layer_idx,
            current_pos=cur_pos,
            scale=self.scaling,
            program_config=self._sdpa_decode_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        ttnn.deallocate(q)

        # [S=1, B, H, D] -> concat heads -> [1, 1, B, H*D]
        sdpa_out_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, self.head_dim),
            core_grid=ttnn.CoreGrid(y=1, x=batch_size),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        attn = ttnn.to_memory_config(attn, sdpa_out_memcfg)
        attn = ttnn.experimental.nlp_concat_heads_decode(attn, num_heads=self.q_heads_per_chip)
        attn = ttnn.to_memory_config(attn, ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.slice(attn, [0, 0, 0, 0], [1, 1, batch_size, int(attn.shape[-1])])
        attn = ttnn.squeeze(attn, 1)  # [1, B, H*D]
        return self._o_proj_all_reduce(attn)
