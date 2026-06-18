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
  - o_proj REPLICATED: the chip's 3-head attention output (384) is all-gathered
    to the full ctx (H), then a full (replicated) o_proj weight produces the full
    replicated hidden directly -- one ctx all_gather instead of a reduce_scatter +
    all_gather of the full-H partial.

Input  x   : replicated [B, S, H]
Output out : replicated [B, S, H]
"""

import torch
import ttnn

from models.experimental.dots_ocr_tp4.tt.common import (
    all_gather_last_dim,
    matmul_m_dim,
    prefill_matmul_2d_config,
    qkv_prefill_matmul_config,
    shard_to_mesh,
    to_l1,
    to_replicated,
    tp_degree,
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
        self.tp_degree = max(1, tp_degree(mesh_device))

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.scaling = self.head_dim**-0.5

        nd = self.tp_degree
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
        from models.experimental.dots_ocr_tp4.tt.common import tp4_lossy_matmul_dtype

        self.qkv_weight_dtype = ttnn.bfloat8_b
        self.o_weight_dtype = tp4_lossy_matmul_dtype()
        self.qkv_compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
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
        # SDPA is BF16 in/out (matches the profile); fidelity is internal. LoFi +
        # math_approx (the production tt_symbiote prefill-SDPA recipe) is the fastest
        # and still passes the block/attention PCC -- ~391 us with the q256/k256
        # config below vs ~480 us at HiFi2+fp32acc and ~1335 us auto (~3.4x).
        self.sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
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
        nd = self.tp_degree
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

        # o_proj REPLICATED (vision-attention pattern): every chip holds the full
        # [ctx=H, H] weight and computes the full hidden from the all-gathered ctx,
        # so there is NO partial-sum reduce. This replaces the row-parallel o_proj +
        # reduce_scatter + all_gather with a single all_gather of the (narrower)
        # head-group ctx -- the gathered o_proj output is already replicated.
        o_w = torch_attn.o_proj.weight.data  # [H, ctx=H]
        self.o_w = to_replicated(o_w.t().contiguous(), self.mesh_device, dtype=self.o_weight_dtype)
        o_has_bias = getattr(torch_attn.o_proj, "bias", None) is not None
        if o_has_bias:
            # o_proj bias is added once to the full output -> replicate.
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
            # Prefill SDPA config for the per-chip q[1,3,S,128]/kv[1,1,S,128] causal
            # shape. The auto-config is terrible here (~1335 us) -- it picks a bad
            # chunk schedule for the tall-skinny 3-head GQA shape. Both grids
            # hardware-swept (BH: P150x4 11x10; WH: T3K 8x8); see
            # test_prefill_sdpa_sweep. exp_approx_mode left off (speed-neutral, keeps
            # PCC); the LoFi+math_approx compute kernel below is the accuracy lever.
            if int(grid.x) >= 8 and int(grid.y) >= 10:
                self.sdpa_program_config = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 10),
                    q_chunk_size=128,
                    k_chunk_size=256,
                    exp_approx_mode=False,
                )
            elif int(grid.x) >= 8 and int(grid.y) >= 8:
                # Wormhole 8x8: q256/k256 -> ~422 us vs ~1335 us auto (~3.2x). q256/k512
                # is ~30 us faster still but its larger score CB clashes in L1 with the
                # L1 input_layernorm output (decoder_block) inside the full block.
                self.sdpa_program_config = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
                    q_chunk_size=256,
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
        # Prefill QKV is activation-read-bound: with the input_layernorm output fed
        # from L1 (see decoder_block) the tuned full-grid config runs the
        # M=2848/K=1536/N=640 op at ~148 us vs ~206 us auto (~1.4x). Decode (M<=1
        # tile) gets None here and keeps the auto path.
        m = matmul_m_dim(x)
        qkv_pc = qkv_prefill_matmul_config(self.mesh_device, m, int(x.shape[-1]), int(self.qkv_w.shape[-1]))
        if qkv_pc is None:
            qkv_pc = prefill_matmul_2d_config(
                self.mesh_device, m, int(x.shape[-1]), int(self.qkv_w.shape[-1]), fp32_dest=False
            )
        x_mm = x
        if qkv_pc is not None and m > 32:
            # This op is activation-read-bound: an L1-interleaved in0 runs the tuned
            # 2D-mcast config (8x8 / per_core_M=12 / per_core_N=3 / in0_block_w=4 /
            # subblock 2x3) at ~151 us vs ~204 us with a DRAM in0 (~1.35x). The
            # tuned grid is already at the sweep optimum -- 7x8 and 8x8 are tied at a
            # 150.9 us median over 12 reps, so geometry has no headroom left; L1 in0
            # is the lever. The decoder block already hands us the input_layernorm
            # output in L1 (decoder_block.py), so to_l1 is a no-op there; this only
            # does real work (and still wins) if a caller feeds a DRAM hidden.
            # We always pass a program_config here, so the L1-in0 -> HiFi4 trap in
            # [[matmul-l1-in0-forces-hifi4]] (which needs program_config=None) does
            # not apply. See test_qkv_prefill_matmul_sweep / _confirm.
            x_mm = to_l1(x)
        qkv = ttnn.linear(
            x_mm,
            self.qkv_w,
            bias=self.qkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            compute_kernel_config=self.qkv_compute,
            program_config=qkv_pc,
        )
        if x_mm is not x:
            ttnn.deallocate(x_mm)
        qkv = ttnn.reshape(qkv, (batch_size, 1, seq_len, -1))
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.q_heads_per_chip,
            num_kv_heads=self.kv_heads_per_chip,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)
        return q, k, v

    def _o_proj_replicated(self, attn):
        """All-gather this chip's head-group ctx -> full ctx, then a REPLICATED
        o_proj (full weight) -> full hidden on every chip.

        Replaces the row-parallel o_proj's reduce_scatter + all_gather (an
        all-reduce of the full-H partial) with a single all_gather of the narrower
        head-group ctx (q_heads_per_chip*head_dim) -- ~half the CCL bytes and the
        result is already replicated, so no post-matmul collective. Gathering ctx
        to DRAM also keeps the matmul in0 off L1, avoiding the L1-in0->HiFi4 trap
        that pinned the old row-parallel o_proj at HiFi4.
        """
        is_decode = int(attn.shape[-2]) == 1
        out_mc = ttnn.L1_MEMORY_CONFIG if is_decode else ttnn.DRAM_MEMORY_CONFIG
        ctx_full = all_gather_last_dim(attn, self.mesh_device)  # [.., ctx/TP] -> [.., ctx=H]
        ttnn.deallocate(attn)
        o_pc = prefill_matmul_2d_config(
            self.mesh_device, matmul_m_dim(ctx_full), int(ctx_full.shape[-1]), int(self.o_w.shape[-1]), fp32_dest=False
        )
        out = ttnn.linear(
            ctx_full,
            self.o_w,
            dtype=ttnn.bfloat16,
            memory_config=out_mc,
            compute_kernel_config=self.o_compute,
            program_config=o_pc,
        )
        ttnn.deallocate(ctx_full)
        if self.o_bias is not None:
            out = ttnn.add(out, self.o_bias, memory_config=out_mc)
        return out

    def forward(self, x: ttnn.Tensor, past_key_value=None, cache_position=None) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
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
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn = ttnn.squeeze(attn, 1)  # [B, S, q_heads_per_chip*head_dim]
        return self._o_proj_replicated(attn)

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
        return self._o_proj_replicated(attn)
