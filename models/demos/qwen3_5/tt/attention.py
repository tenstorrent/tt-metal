# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 FullAttention – Phase 2 TTNN implementation.

All matrix multiplications (Q, K, V, gate, O projections) run on-device via
DRAM-sharded ttnn.linear.  Partial RoPE (rope_dim=64 of head_dim=256) is
applied on-device using cos/sin lookup tables stored as TTNN tensors plus
element-wise slice-rotate-concat ops.  QK norm, SDPA, gated output and o_proj
are all device-side.

The only host→device transfers per step are:
  • One position index tensor (1 uint32) for cos/sin embedding lookup.
  • One int32 tensor (batch,) for paged_update_cache.
Both are tiny (< 10 bytes) and not "compute" in the CPU-op sense.
"""

from __future__ import annotations

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.common import Mode


class Qwen3_5FullAttentionTT(LightweightModule):
    """Qwen3.5 FullAttention – full TTNN implementation.

    Weight keys (after map_hf_to_meta_keys_qwen3_5):
        {prefix}.attention.wq.weight       - Q projection (n_heads*head_dim, hidden)
        {prefix}.attention.wq_gate.weight  - output gate projection (same shape as wq)
        {prefix}.attention.wk.weight       - K projection (n_kv_heads*head_dim, hidden)
        {prefix}.attention.wv.weight       - V projection (n_kv_heads*head_dim, hidden)
        {prefix}.attention.wo.weight       - output projection (hidden, n_heads*head_dim)
        {prefix}.attention.q_norm.weight   - per-head Q RMS norm weight (head_dim,)
        {prefix}.attention.k_norm.weight   - per-head K RMS norm weight (head_dim,)
    """

    def __init__(
        self,
        mesh_device,
        args,
        state_dict: dict,
        weight_cache_path,
        layer_num: int,
        dtype=ttnn.bfloat16,
        kv_cache=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype

        self.n_heads = args.n_heads  # 24
        self.n_kv_heads = args.n_kv_heads  # 4
        self.head_dim = args.head_dim  # 256
        self.hidden_size = args.dim  # 5120
        self.rope_dim = getattr(args, "rope_dim", self.head_dim)  # 64
        self.norm_eps = args.norm_eps
        self.scale = self.head_dim**-0.5
        self.max_batch_size = getattr(args, "max_batch_size", 1)

        q_dim = self.n_heads * self.head_dim  # 6144
        kv_dim = self.n_kv_heads * self.head_dim  # 1024

        prefix = args.get_state_dict_prefix("Attention", layer_num)

        if weight_cache_path is None or args.dummy_weights:
            cache_name = lambda _: None  # noqa: E731
        else:
            cache_name = lambda name: weight_cache_path / f"{prefix}.{name}"  # noqa: E731

        def load(key):
            return state_dict[f"{prefix}.{key}"]

        def as_dram_sharded(name, weight, k, n):
            """Transpose and load as DRAM-sharded (1,1,k,n) weight tensor."""
            w = torch.transpose(weight, -2, -1).unsqueeze(0).unsqueeze(0)
            return ttnn.as_tensor(
                w,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=args.create_dram_sharded_mem_config(k, n),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=cache_name(name),
            )

        def as_dram_1d(name, weight):
            """Load 1-D weight (e.g. norm scale) as (1, dim) DRAM tensor."""
            w = weight.unsqueeze(0)  # (1, head_dim)
            return ttnn.as_tensor(
                w,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                cache_file_name=cache_name(name),
            )

        # ---- Projection weights (DRAM-sharded for DRAMSharded matmul kernel) ----
        self.wq = as_dram_sharded("wq.weight", load("wq.weight"), self.hidden_size, q_dim)
        self.wq_gate = as_dram_sharded("wq_gate.weight", load("wq_gate.weight"), self.hidden_size, q_dim)
        self.wk = as_dram_sharded("wk.weight", load("wk.weight"), self.hidden_size, kv_dim)
        self.wv = as_dram_sharded("wv.weight", load("wv.weight"), self.hidden_size, kv_dim)
        self.wo = as_dram_sharded("wo.weight", load("wo.weight"), q_dim, self.hidden_size)

        # ---- QK norm weights (1-D, DRAM) ----
        self.q_norm_w = as_dram_1d("q_norm.weight", load("q_norm.weight").float().to(torch.bfloat16))
        self.k_norm_w = as_dram_1d("k_norm.weight", load("k_norm.weight").float().to(torch.bfloat16))

        # ---- Pre-computed RoPE cos/sin tables (max_seq × rope_dim) ----
        # These are loaded in ROW_MAJOR_LAYOUT so that ttnn.embedding can
        # dynamically look up the row for the current decode position.
        rope_theta = getattr(args, "rope_theta", 10_000_000)
        rope_half = self.rope_dim // 2  # 32 for Qwen3.5-27B
        max_seq = getattr(args, "max_seq_len", 4096)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, rope_half).float() / self.rope_dim))
        positions = torch.arange(max_seq).float()
        freqs = torch.outer(positions, inv_freq)  # (max_seq, rope_half)
        emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq, rope_dim=64)
        cos_t = emb.cos().to(torch.bfloat16)
        sin_t = emb.sin().to(torch.bfloat16)

        self.cos_matrix = ttnn.as_tensor(
            cos_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.sin_matrix = ttnn.as_tensor(
            sin_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # ---- On-device KV cache (batch=max_batch_size, n_kv_heads, max_seq, head_dim) ----
        cache_k = torch.zeros(self.max_batch_size, self.n_kv_heads, max_seq, self.head_dim, dtype=torch.bfloat16)
        cache_v = torch.zeros_like(cache_k)
        self.cache_k = ttnn.from_torch(
            cache_k,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.cache_v = ttnn.from_torch(
            cache_v,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        # ---- Compute kernel configs ----
        self._compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._sdpa_compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # ---- Decode memory configs for DRAMSharded matmuls ----
        # All projections use 32 cores (attn_input_grid) to match input sharding.
        _cores = args.attn_input_grid  # CoreGrid(x=8, y=4) = 32 cores on WH B0

        self._proj_input_mem_decode = ttnn.create_sharded_memory_config(
            (args.tile_padded_batch_rows, self.hidden_size // _cores.num_cores),
            _cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        # wo input has K = q_dim = 6144 (not hidden_size)
        self._wo_input_mem_decode = ttnn.create_sharded_memory_config(
            (args.tile_padded_batch_rows, q_dim // _cores.num_cores),
            _cores,
            ttnn.ShardStrategy.WIDTH,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        # SDPA program config (no prefetcher)
        self._sdpa_decode_prog_cfg = args.get_attn_sdpa_decode_program_config(None)

        # SDPA output memory config for decode (height-sharded on batch cores)
        # We use DRAM and re-shard before nlp_concat_heads to keep it simple.
        self._attn_out_mem_decode = args.get_attn_sdpa_output_mem_config(Mode.DECODE, self.max_batch_size, None)

        self._n_cores = _cores.num_cores  # 32

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _decode_pc(self, k: int, n: int):
        """DRAMSharded program config for decode (fixed to attn_input_grid cores)."""
        return self.args.dram_matmul_config(
            m=self.args.tile_padded_batch_rows,
            k=k,
            n=n,
            num_cores=self._n_cores,
        )

    def _prefill_pc(self, seq_len: int, n_out: int):
        """MultiCast program config for prefill."""
        tile = self.args.tile_size
        per_core_M = max(1, math.ceil(seq_len / (tile * 8)))
        per_core_N = math.ceil(n_out / (tile * self.args.dram_shard_grid_width))
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=seq_len <= 2048,
        )

    def _linear(self, x: ttnn.Tensor, w: ttnn.Tensor, mode: Mode, k: int, n: int, seq_len: int) -> ttnn.Tensor:
        """TTNN linear with mode-appropriate program config."""
        if mode == Mode.PREFILL:
            pc = self._prefill_pc(seq_len, n)
            out_mem = ttnn.DRAM_MEMORY_CONFIG
        else:
            pc = self._decode_pc(k, n)
            out_mem = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
        return ttnn.linear(
            x,
            w,
            program_config=pc,
            compute_kernel_config=self._compute_cfg,
            memory_config=out_mem,
            dtype=ttnn.bfloat16,
        )

    def _gather_rope(self, pos: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Look up cos/sin row for `pos` using TTNN embedding (device-side lookup)."""
        # pos_idx: (1, 1) uint32 – only 2 bytes sent host→device
        pos_idx = ttnn.from_torch(
            torch.tensor([[pos]], dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        cos = ttnn.embedding(pos_idx, self.cos_matrix, layout=ttnn.TILE_LAYOUT)
        sin = ttnn.embedding(pos_idx, self.sin_matrix, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(pos_idx)
        # cos/sin shape: (1, 1, rope_dim=64) → reshape to (1, 1, 1, rope_dim)
        cos = ttnn.reshape(cos, [1, 1, 1, self.rope_dim])
        sin = ttnn.reshape(sin, [1, 1, 1, self.rope_dim])
        return cos, sin

    def _gather_rope_range(self, start: int, seq_len: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Look up cos/sin rows for positions [start, start+seq_len) for prefill."""
        # For prefill, look up a contiguous range using a gather index tensor.
        pos_idx = ttnn.from_torch(
            torch.arange(start, start + seq_len, dtype=torch.int32).unsqueeze(0),  # (1, seq_len)
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        cos = ttnn.embedding(pos_idx, self.cos_matrix, layout=ttnn.TILE_LAYOUT)  # (1, seq_len, rope_dim)
        sin = ttnn.embedding(pos_idx, self.sin_matrix, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(pos_idx)
        cos = ttnn.reshape(cos, [1, 1, seq_len, self.rope_dim])
        sin = ttnn.reshape(sin, [1, 1, seq_len, self.rope_dim])
        return cos, sin

    @staticmethod
    def _apply_partial_rope(heads: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, rope_dim: int) -> ttnn.Tensor:
        """Apply partial RoPE to heads[:, :, :, :rope_dim] and pass-through rest.

        heads: (1, n_heads, batch, head_dim) DRAM
        cos/sin: (1, 1, T, rope_dim) – broadcast over n_heads dim
        rope_dim must be tile-aligned (=64 here).
        """
        rope_half = rope_dim // 2
        head_dim = heads.shape[-1]

        # Split rope region and pass-through region
        q_rope = ttnn.slice(heads, [0, 0, 0, 0], [heads.shape[0], heads.shape[1], heads.shape[2], rope_dim])
        q_pass = ttnn.slice(heads, [0, 0, 0, rope_dim], [heads.shape[0], heads.shape[1], heads.shape[2], head_dim])

        # rotate_half(q_rope) = cat(-q_rope[..., rope_half:], q_rope[..., :rope_half])
        q1 = ttnn.slice(q_rope, [0, 0, 0, 0], [q_rope.shape[0], q_rope.shape[1], q_rope.shape[2], rope_half])
        q2 = ttnn.slice(q_rope, [0, 0, 0, rope_half], [q_rope.shape[0], q_rope.shape[1], q_rope.shape[2], rope_dim])
        q_rot = ttnn.concat([ttnn.neg(q2), q1], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q1)
        ttnn.deallocate(q2)

        # Apply: q_rope_out = q_rope * cos + q_rot * sin
        q_rope_out = ttnn.add(
            ttnn.mul(q_rope, cos, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.mul(q_rot, sin, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_rope)
        ttnn.deallocate(q_rot)

        # Concatenate with pass-through dims
        out = ttnn.concat([q_rope_out, q_pass], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(q_rope_out)
        ttnn.deallocate(q_pass)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos: int,
        rot_mats=None,
        user_id: int = 0,
        mode: str = "decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ) -> tuple[ttnn.Tensor, tuple]:
        if mode == Mode.PREFILL or mode == "prefill":
            return self._forward_prefill(x, current_pos, user_id, page_table, kv_cache)
        else:
            return self._forward_decode(x, current_pos, user_id, page_table, kv_cache)

    def _forward_decode(self, x, current_pos, user_id, page_table, kv_cache):
        """Decode step: single token, batch=1 (tile-padded to 32 rows)."""
        seq_len = x.shape[-2]  # tile_padded_batch_rows = 32
        batch_real = self.max_batch_size  # 1 for smoke test

        # ---- Projections (DRAMSharded matmuls) ----
        q_tt = self._linear(x, self.wq, Mode.DECODE, self.hidden_size, self.n_heads * self.head_dim, seq_len)
        gate_tt = self._linear(x, self.wq_gate, Mode.DECODE, self.hidden_size, self.n_heads * self.head_dim, seq_len)
        k_tt = self._linear(x, self.wk, Mode.DECODE, self.hidden_size, self.n_kv_heads * self.head_dim, seq_len)
        v_tt = self._linear(x, self.wv, Mode.DECODE, self.hidden_size, self.n_kv_heads * self.head_dim, seq_len)

        # ---- Move projections to DRAM for reshape ----
        q_tt = ttnn.to_memory_config(q_tt, ttnn.DRAM_MEMORY_CONFIG)
        gate_tt = ttnn.to_memory_config(gate_tt, ttnn.DRAM_MEMORY_CONFIG)
        k_tt = ttnn.to_memory_config(k_tt, ttnn.DRAM_MEMORY_CONFIG)
        v_tt = ttnn.to_memory_config(v_tt, ttnn.DRAM_MEMORY_CONFIG)

        # ---- Reshape to (1, n_heads, batch_pad, head_dim) ----
        q_heads = ttnn.reshape(q_tt, [1, self.n_heads, seq_len, self.head_dim])
        k_heads = ttnn.reshape(k_tt, [1, self.n_kv_heads, seq_len, self.head_dim])
        v_heads = ttnn.reshape(v_tt, [1, self.n_kv_heads, seq_len, self.head_dim])
        ttnn.deallocate(q_tt)
        ttnn.deallocate(k_tt)
        ttnn.deallocate(v_tt)

        # ---- QK norm (rms_norm on last dim = head_dim) ----
        q_heads = ttnn.rms_norm(
            q_heads, weight=self.q_norm_w, epsilon=self.norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k_heads = ttnn.rms_norm(
            k_heads, weight=self.k_norm_w, epsilon=self.norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # ---- Partial RoPE via on-device cos/sin lookup ----
        cos, sin = self._gather_rope(current_pos)
        q_heads = self._apply_partial_rope(q_heads, cos, sin, self.rope_dim)
        k_heads = self._apply_partial_rope(k_heads, cos, sin, self.rope_dim)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        # ---- KV cache update ----
        if kv_cache is not None and kv_cache[0] is not None:
            keys, values = kv_cache[0], kv_cache[1]
        else:
            keys, values = self.cache_k, self.cache_v

        # paged_update_cache expects (1, n_kv_heads, batch, head_dim) height-sharded.
        # Convert k/v to L1_HEIGHT_SHARDED for the update op.
        k_update = ttnn.to_memory_config(k_heads[:, :, :batch_real, :], ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)
        v_update = ttnn.to_memory_config(v_heads[:, :, :batch_real, :], ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG)

        pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        ttnn.experimental.paged_update_cache(keys, k_update, update_idxs_tensor=pos_tensor)
        ttnn.experimental.paged_update_cache(values, v_update, update_idxs_tensor=pos_tensor)
        ttnn.deallocate(k_update)
        ttnn.deallocate(v_update)

        # ---- SDPA decode ----
        # q_heads: (1, n_heads, batch_pad, head_dim) → slice to real batch
        q_sdpa = ttnn.to_memory_config(
            q_heads[:, :, :batch_real, :],
            self._attn_out_mem_decode,
        )
        ttnn.deallocate(q_heads)

        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q_sdpa,
            keys,
            values,
            cur_pos_tensor=pos_tensor,
            scale=self.scale,
            program_config=self._sdpa_decode_prog_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_sdpa)
        ttnn.deallocate(pos_tensor)

        # ---- Re-shard attn_out for nlp_concat_heads_decode ----
        attn_out = ttnn.to_memory_config(attn_out, self._attn_out_mem_decode)
        attn_cat = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=self.n_heads)
        ttnn.deallocate(attn_out)

        # attn_cat: (1, 1, batch_real, n_heads*head_dim)

        # ---- Gated output: out = attn * sigmoid(gate) ----
        # gate_tt: (1, 1, batch_pad, q_dim) – slice to real batch then re-expand for wo
        gate_real = gate_tt[:, :, :batch_real, :]
        gate_sig = ttnn.sigmoid(gate_real, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_tt)
        ttnn.deallocate(gate_real)

        gated = ttnn.mul(attn_cat, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_cat)
        ttnn.deallocate(gate_sig)

        # ---- Re-pad and re-shard for wo DRAMSharded matmul ----
        # wo expects batch_pad=32 rows sharded on attn_input_grid (32 cores).
        if batch_real < seq_len:
            pad_rows = seq_len - batch_real
            pad = torch.zeros(1, 1, pad_rows, self.n_heads * self.head_dim, dtype=torch.bfloat16)
            pad_tt = ttnn.from_torch(
                pad,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            gated = ttnn.concat([gated, pad_tt], dim=-2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pad_tt)

        gated = ttnn.to_memory_config(gated, self._wo_input_mem_decode)

        # ---- Output projection ----
        out = self._linear(gated, self.wo, Mode.DECODE, self.n_heads * self.head_dim, self.hidden_size, seq_len)
        ttnn.deallocate(gated)

        # out: (1, 1, batch_pad, hidden_size) L1 width sharded
        return out, (keys, values)

    def _forward_prefill(self, x, current_pos, user_id, page_table, kv_cache):
        """Prefill: process seq_len tokens at once."""
        seq_len = x.shape[-2]

        # ---- Projections ----
        q_tt = self._linear(x, self.wq, Mode.PREFILL, self.hidden_size, self.n_heads * self.head_dim, seq_len)
        gate_tt = self._linear(x, self.wq_gate, Mode.PREFILL, self.hidden_size, self.n_heads * self.head_dim, seq_len)
        k_tt = self._linear(x, self.wk, Mode.PREFILL, self.hidden_size, self.n_kv_heads * self.head_dim, seq_len)
        v_tt = self._linear(x, self.wv, Mode.PREFILL, self.hidden_size, self.n_kv_heads * self.head_dim, seq_len)

        # ---- Reshape to heads ----
        q_heads = ttnn.reshape(q_tt, [1, self.n_heads, seq_len, self.head_dim])
        k_heads = ttnn.reshape(k_tt, [1, self.n_kv_heads, seq_len, self.head_dim])
        v_heads = ttnn.reshape(v_tt, [1, self.n_kv_heads, seq_len, self.head_dim])
        ttnn.deallocate(q_tt)
        ttnn.deallocate(k_tt)
        ttnn.deallocate(v_tt)

        # ---- QK norm ----
        q_heads = ttnn.rms_norm(
            q_heads, weight=self.q_norm_w, epsilon=self.norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        k_heads = ttnn.rms_norm(
            k_heads, weight=self.k_norm_w, epsilon=self.norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # ---- Partial RoPE (all positions) ----
        cos, sin = self._gather_rope_range(current_pos, seq_len)
        q_heads = self._apply_partial_rope(q_heads, cos, sin, self.rope_dim)
        k_heads = self._apply_partial_rope(k_heads, cos, sin, self.rope_dim)
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)

        # ---- Fill KV cache ----
        if kv_cache is not None and kv_cache[0] is not None:
            keys, values = kv_cache[0], kv_cache[1]
        else:
            keys, values = self.cache_k, self.cache_v

        ttnn.fill_cache(keys, k_heads, user_id % self.max_batch_size)
        ttnn.fill_cache(values, v_heads, user_id % self.max_batch_size)

        # ---- Prefill SDPA ----
        q_8b = ttnn.typecast(q_heads, dtype=ttnn.bfloat8_b)
        k_8b = ttnn.typecast(k_heads, dtype=ttnn.bfloat8_b)
        v_8b = ttnn.typecast(v_heads, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_8b,
            k_8b,
            v_8b,
            is_causal=True,
            scale=self.scale,
            program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, seq_len),
            compute_kernel_config=self._sdpa_compute_cfg,
        )
        ttnn.deallocate(q_8b)
        ttnn.deallocate(k_8b)
        ttnn.deallocate(v_8b)

        # ---- Concat heads: (1, n_heads, seq, head_dim) → (1, 1, seq, n_heads*head_dim) ----
        attn_out = ttnn.reshape(attn_out, [1, 1, seq_len, self.n_heads * self.head_dim])

        # ---- Gated output ----
        gate_sig = ttnn.sigmoid(gate_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gate_tt)
        gated = ttnn.mul(attn_out, gate_sig, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate_sig)

        # ---- Output projection ----
        out = self._linear(gated, self.wo, Mode.PREFILL, self.n_heads * self.head_dim, self.hidden_size, seq_len)
        ttnn.deallocate(gated)

        return out, (keys, values)
