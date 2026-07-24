# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Pixtral vision attention with Devstral-compatible vision RoPE.

import os

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.experimental.devstral2_small.devstral_utils.vision_ccl import vision_sum_all_reduce
from models.experimental.devstral2_small.devstral_utils.pixtral_seq_chunk import (
    pad_seq_to_chunk_multiple,
    pixtral_effective_mm_seq_len,
    trim_seq_dim2,
    vision_nlp_concat_input_memcfg,
    vision_rms_norm_block_shard_eligible,
    vision_rms_norm_block_shard_memcfg,
    vision_rms_norm_prepare_block_shard_input,
    vision_rope_memcfg,
    vision_seq_memcfg,
    vision_use_sharded_nlp_concat,
)


def _pixtral_sdpa_grid_size(configuration) -> tuple[int, int]:
    grid = configuration.max_grid_size
    if hasattr(grid, "x") and hasattr(grid, "y"):
        return (int(grid.x), int(grid.y))
    if isinstance(grid, (tuple, list)) and len(grid) >= 2:
        return (int(grid[0]), int(grid[1]))
    return (8, 8)


def _pixtral_sdpa_program_config(
    seq_len: int, max_mm_seq_len: int, grid_size: tuple[int, int]
) -> ttnn.SDPAProgramConfig:
    """SDPA tiles scale with matmul seq chunks (same policy as ``llama_image_attention``)."""
    force_q = os.environ.get("PIXTRAL_SDPA_Q_CHUNK")
    if force_q is not None and str(force_q).strip() != "":
        q_chunk = max(32, min(256, nearest_32(int(force_q))))
        force_k = os.environ.get("PIXTRAL_SDPA_K_CHUNK")
        k_chunk = max(32, min(256, nearest_32(int(force_k or force_q))))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    if seq_len < 2048:
        # Sweep winner (tests/matmul/test_sdpa_vision_sweep.py) for the [1, 4, 1024, 64]
        # vision SDPA: q_chunk=64, k_chunk=256 on an 8x8 grid. The small q_chunk gives
        # 4 heads * (1024/64) = 64 q work units to fill the 8x8 = 64-core grid (vs only
        # 32 units at q_chunk=128, which left ~2/3 of the cores idle).
        gx, gy = int(grid_size[0]), int(grid_size[1])
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(min(8, gx), min(8, gy)),
            q_chunk_size=64,
            k_chunk_size=256,
            exp_approx_mode=False,
        )

    num_chunks = max(1, (seq_len + max_mm_seq_len - 1) // max_mm_seq_len)
    chunk = min(256, max(64, nearest_32(32 * num_chunks)))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
    )


def apply_rotary_pos_emb_vision_tt(q, k, cos, sin):
    seq_len = int(q.shape[2])
    head_dim = int(q.shape[-1])
    rope_mem_cfg = vision_rope_memcfg(seq_len, head_dim)
    cos = ttnn.unsqueeze(cos, 0)
    sin = ttnn.unsqueeze(sin, 0)

    def _rope_mem(t: ttnn.Tensor) -> ttnn.Tensor:
        if t.memory_config().buffer_type != rope_mem_cfg.buffer_type:
            return ttnn.to_memory_config(t, rope_mem_cfg)
        return t

    q = _rope_mem(q)
    k = _rope_mem(k)
    cos = _rope_mem(cos)
    sin = _rope_mem(sin)

    q_embed = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=rope_mem_cfg)
    k_embed = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=rope_mem_cfg)

    if q_embed.shape[2] != seq_len:
        q_embed = q_embed[:, :, :seq_len, :]
    if k_embed.shape[2] != seq_len:
        k_embed = k_embed[:, :, :seq_len, :]
    return q_embed, k_embed


class TtMistralImageAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices

        self.hidden_size = configuration.vision_dim
        self.n_heads = configuration.vision_attn_n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.n_kv_heads = self.n_heads

        self.n_local_heads = self.n_heads // configuration.num_devices
        self.n_local_kv_heads = self.n_kv_heads // configuration.num_devices

        self.dtype = dtype

        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_sdpa = configuration.compute_kernel_config_sdpa

        # Optimised fused-QKV matmul kernel config (recommendation B). HiFi2 matches the
        # BF16 activation precision; fp32_dest_acc_en=False keeps the full 8-tile DST so the
        # output subblock can be maximised; packer_l1_acc accumulates the K-blocks in L1.
        self.qkv_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )
        # Output ``wo`` matmul: same DST/subblock policy as QKV (fp32 dest-acc off).
        self.wo_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
            dst_full_sync_en=False,
        )
        self.configuration = configuration

        self.model_config = configuration.get_model_config()

        wq_str = f"{state_dict_prefix}wq.weight"
        wk_str = f"{state_dict_prefix}wk.weight"
        wv_str = f"{state_dict_prefix}wv.weight"
        wo_str = f"{state_dict_prefix}wo.weight"

        if self.n_heads % configuration.num_devices != 0:
            raise ValueError(f"n_heads {self.n_heads} must divide num_devices {configuration.num_devices}")
        if self.n_kv_heads % configuration.num_devices != 0:
            raise ValueError(f"n_kv_heads {self.n_kv_heads} must divide num_devices {configuration.num_devices}")

        def pad_head_dim(weight, heads_out=True):
            dim = weight.shape[1]
            assert weight.shape[0] == dim
            padded_head_dim = nearest_32(self.head_dim)
            padding_size = padded_head_dim - self.head_dim
            if padding_size > 0:
                if heads_out:
                    weight = weight.transpose(-1, -2)
                weight = weight.reshape(dim, self.n_heads, self.head_dim)
                padded = weight.new_zeros((dim, self.n_heads, padded_head_dim))
                padded[:, :, : self.head_dim] = weight
                weight = padded
                weight = weight.reshape(dim, self.n_heads * padded_head_dim)
                if heads_out:
                    weight = weight.transpose(-1, -2)
            return weight

        wq_padded = pad_head_dim(state_dict[wq_str])
        wk_padded = pad_head_dim(state_dict[wk_str])
        wv_padded = pad_head_dim(state_dict[wv_str])
        wo_padded = pad_head_dim(state_dict[wo_str], heads_out=False)

        def pack_qkv_for_sharding(wq, wk, wv):
            local_width = wq.shape[0] // configuration.num_devices
            packed = wq.new_empty((configuration.num_devices, self.hidden_size, local_width * 3))
            for index, weight in enumerate((wq, wk, wv)):
                start = index * local_width
                packed[:, :, start : start + local_width] = weight.reshape(
                    configuration.num_devices, local_width, self.hidden_size
                ).transpose(-1, -2)
            return packed.transpose(0, 1).reshape(self.hidden_size, -1)

        # ".bfp8" tag keeps the BFP8 cache distinct from any pre-existing BF16 weight cache (avoids a stale reload).
        wqkv_cache = None if weight_cache_path is None else weight_cache_path / f"{state_dict_prefix}wqkv.weight.bfp8"
        wo_cache = None if weight_cache_path is None else weight_cache_path / f"{state_dict_prefix}wo.weight.bfp8"

        self.wqkv = ttnn.as_tensor(
            pack_qkv_for_sharding(wq_padded, wk_padded, wv_padded),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=ttnn.bfloat8_b,  # recommendation C: BFP8 weights halve the DRAM-bound QKV weight stream
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=wqkv_cache,
        )

        self.wo = ttnn.as_tensor(
            wo_padded.transpose(-1, -2),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-2),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=wo_cache,
        )

        self.scale = self.head_dim**-0.5
        self._padded_head_dim = nearest_32(self.head_dim)

    def _nlp_create_qkv_heads(
        self, xqkv_fused: ttnn.Tensor, seq_len: int
    ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        rope_mem_cfg = vision_rope_memcfg(seq_len, self._padded_head_dim)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=rope_mem_cfg,
        )
        ttnn.deallocate(xqkv_fused)
        return q, k, v

    def _nlp_concat_heads(
        self, attn_output_1QSD: ttnn.Tensor, seq_len: int, block_shard_out: bool = False
    ) -> ttnn.Tensor:
        sdpa_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        if vision_use_sharded_nlp_concat(seq_len, self.n_local_heads, self._padded_head_dim, self.configuration):
            concat_in_mem = vision_nlp_concat_input_memcfg(
                seq_len, self._padded_head_dim, self.n_local_heads, self.configuration
            )
            attn_sharded = ttnn.interleaved_to_sharded(attn_output_1QSD, concat_in_mem)
            ttnn.deallocate(attn_output_1QSD)
            out = ttnn.experimental.nlp_concat_heads(attn_sharded, memory_config=sdpa_mem_cfg)
            ttnn.deallocate(attn_sharded)
        else:
            out = ttnn.experimental.nlp_concat_heads(attn_output_1QSD, memory_config=sdpa_mem_cfg)
            ttnn.deallocate(attn_output_1QSD)
        if block_shard_out:
            # nlp_concat_heads can only emit a heads-grouped shard (its output shard is derived from the
            # input shard), not the 2D [seq/8, width/8] block layout the wo bs/dram/bs matmul wants, so
            # reshard the interleaved concat output to that block layout here. This feeds the wo a
            # block-sharded in0 and lets the wo output stay block-sharded into the reduce-scatter,
            # removing the post-wo L1->DRAM CopyDeviceOp.
            bs_mem = vision_rms_norm_block_shard_memcfg(seq_len, self.n_local_heads * self._padded_head_dim, 8, 8)
            sharded = ttnn.interleaved_to_sharded(out, bs_mem)
            ttnn.deallocate(out)
            return sharded
        return out

    @staticmethod
    def _best_out_subblock(per_core_M: int, per_core_N: int, max_tiles: int = 8) -> tuple[int, int]:
        """Largest (h, w) dividing the per-core block with h*w <= DST half (8 tiles, fp32_acc off)."""
        best_h, best_w = 1, 1
        for h in range(1, per_core_M + 1):
            if per_core_M % h:
                continue
            for w in range(1, per_core_N + 1):
                if per_core_N % w:
                    continue
                if h * w <= max_tiles and h * w > best_h * best_w:
                    best_h, best_w = h, w
        return best_h, best_w

    def _qkv_program_config(
        self, seq_len: int, max_seq: int, n: int
    ) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
        m = min(seq_len, max_seq)
        m_tiles = (m + 31) // 32
        n_tiles = (n + 31) // 32
        k_tiles = self.hidden_size // 32
        dev_x, dev_y = int(self.grid_size.x), int(self.grid_size.y)
        # Pick the largest grid dim that evenly divides the tile count (no wasted/padded cores).
        grid_x = max(d for d in range(1, min(n_tiles, dev_x) + 1) if n_tiles % d == 0)
        grid_y = max(d for d in range(1, min(m_tiles, dev_y) + 1) if m_tiles % d == 0)
        per_core_M = m_tiles // grid_y
        per_core_N = n_tiles // grid_x
        in0_block_w = next(d for d in (8, 4, 2, 1) if k_tiles % d == 0)
        out_subblock_h, out_subblock_w = self._best_out_subblock(per_core_M, per_core_N)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=seq_len <= max_seq,
        )

    def _wo_program_config(self, seq_len: int, max_seq: int) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
        """Output ``wo`` matmul: M=seq, K=n_local_heads*padded_head_dim (256), N=hidden (1024).

        Sweep winner for the 1024x256x1024 vision-tower wo
        (tests/matmul/test_matmul_1024x256x1024_sweep.py): 8x8 grid, in0_block_w=4
        (vs the IMAGE_ATTN_OUT_PROGCFG default of 1 -> 2 inner-K passes over Kt=8
        instead of 8). BF16 activations in, BF8 weights; wo_compute_kernel_config uses fp32 dest-acc off
        (out_subblock h*w<=8).
        """
        m = min(seq_len, max_seq)
        m_tiles = (m + 31) // 32
        n_tiles = self.hidden_size // 32
        k_tiles = (self.n_local_heads * self._padded_head_dim) // 32
        dev_x, dev_y = int(self.grid_size.x), int(self.grid_size.y)
        grid_x = max(d for d in range(1, min(n_tiles, dev_x) + 1) if n_tiles % d == 0)
        grid_y = max(d for d in range(1, min(m_tiles, dev_y) + 1) if m_tiles % d == 0)
        per_core_M = m_tiles // grid_y
        per_core_N = n_tiles // grid_x
        in0_block_w = next(d for d in (4, 2, 1) if k_tiles % d == 0)
        out_subblock_h, out_subblock_w = self._best_out_subblock(per_core_M, per_core_N, max_tiles=8)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            in0_block_w=in0_block_w,
            out_subblock_h=out_subblock_h,
            out_subblock_w=out_subblock_w,
            per_core_M=per_core_M,
            per_core_N=per_core_N,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=seq_len <= max_seq,
        )

    def _qkv_block_shard_progcfg(
        self, seq_len: int, n: int, grid_x: int = 8, grid_y: int = 8
    ) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
        """Recommendation D block-sharded QKV progcfg. out_subblock_w is the largest divisor of per_core_N
        (<=4, DST half) -> (1,3) for N=768, i.e. 3x fewer subblock passes than a (1,1) subblock."""
        per_core_m = (seq_len // 32) // grid_y
        per_core_n = (n // 32) // grid_x
        out_subblock_w = max(w for w in range(1, per_core_n + 1) if per_core_n % w == 0 and w <= 4)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=per_core_m,
            out_block_w=per_core_n,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )

    def _wo_block_shard_eligible(self, seq_len: int, max_mm_seq_len: int) -> bool:
        """True when the wo runs as the bs/dram/bs sweep winner (8x8 block-sharded, single chunk).

        Needs num_devices>1 (so the reduce-scatter — and thus the L1->DRAM copy — exists), a single
        matmul chunk, and Mt/Nt divisible by the 8-row grid with Kt divisible by the 8-col grid
        (Kt=8 -> Kt/gx=1 -> in0_block_w=1).
        """
        if self.num_devices <= 1 or seq_len > max_mm_seq_len or seq_len % 32 != 0:
            return False
        gx, gy = 8, 8
        mt = seq_len // 32
        nt = self.hidden_size // 32
        kt = (self.n_local_heads * self._padded_head_dim) // 32
        return (
            mt % gy == 0
            and nt % gx == 0
            and kt % gx == 0
            and int(self.grid_size.x) >= gx
            and int(self.grid_size.y) >= gy
        )

    def _wo_block_shard_progcfg(
        self, seq_len: int, grid_x: int = 8, grid_y: int = 8
    ) -> ttnn.MatmulMultiCoreReuseMultiCastProgramConfig:
        """Block-sharded wo (sweep winner ``2D bs/dram/bs 8x8 w1``): in0 block-sharded (K=256 split
        across gx -> Kt/gx=1 -> in0_block_w=1), out block-sharded. out_subblock_w is the largest
        divisor of per_core_N (<=4, within the DST budget)."""
        per_core_m = (seq_len // 32) // grid_y
        per_core_n = (self.hidden_size // 32) // grid_x
        out_subblock_w = max(w for w in range(1, per_core_n + 1) if per_core_n % w == 0 and w <= 4)
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(grid_x, grid_y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=out_subblock_w,
            out_block_h=per_core_m,
            out_block_w=per_core_n,
            per_core_M=per_core_m,
            per_core_N=per_core_n,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=True,
        )

    def _linear_qkv_seq_chunked(self, x_11SH, seq_len: int, max_mm_seq_len: int) -> ttnn.Tensor:
        """Fused QKV ``ttnn.linear`` over the sequence axis; chunk so matmul ``m`` fits L1 CB budget."""
        x_11SH, seq_len, original_seq_len = pad_seq_to_chunk_multiple(x_11SH, seq_len, max_mm_seq_len)
        qkv_width = (self.n_local_heads + 2 * self.n_local_kv_heads) * self._padded_head_dim
        qkv_mem_cfg = vision_seq_memcfg(seq_len, qkv_width)
        if seq_len <= max_mm_seq_len:
            mt, nt, kt = seq_len // 32, qkv_width // 32, self.hidden_size // 32
            block_shard = (
                seq_len % 32 == 0
                and mt % 8 == 0
                and nt % 8 == 0
                and kt % 8 == 0
                and int(self.grid_size.x) >= 8
                and int(self.grid_size.y) >= 8
            )
            if block_shard:
                # Block-shard in0 on 8x8 (same layout as block-sharded RMSNorm); reuse when already sharded.
                x_bs = vision_rms_norm_prepare_block_shard_input(x_11SH, seq_len, self.hidden_size)
                out = ttnn.linear(
                    x_bs,
                    self.wqkv,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1),
                    compute_kernel_config=self.qkv_compute_kernel_config,
                    program_config=self._qkv_block_shard_progcfg(seq_len, qkv_width),
                )
                out = ttnn.to_memory_config(out, qkv_mem_cfg)
            else:
                if x_11SH.is_sharded():
                    x_11SH = ttnn.sharded_to_interleaved(x_11SH, qkv_mem_cfg)
                out = ttnn.linear(
                    x_11SH,
                    self.wqkv,
                    dtype=ttnn.bfloat16,
                    memory_config=qkv_mem_cfg,
                    compute_kernel_config=self.qkv_compute_kernel_config,
                    program_config=self._qkv_program_config(seq_len, seq_len, qkv_width),
                )
            return trim_seq_dim2(out, original_seq_len)

        x_batched = ttnn.reshape(x_11SH, [1, seq_len // max_mm_seq_len, max_mm_seq_len, -1])
        out = ttnn.linear(
            x_batched,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=qkv_mem_cfg,
            compute_kernel_config=self.qkv_compute_kernel_config,
            program_config=self._qkv_program_config(seq_len, max_mm_seq_len, qkv_width),
        )
        out = ttnn.reshape(out, [1, 1, seq_len, -1])
        return trim_seq_dim2(out, original_seq_len)

    def _linear_wo_seq_chunked(
        self,
        attn_output_11SH,
        seq_len: int,
        max_mm_seq_len: int,
        output_memory_config=None,
    ) -> ttnn.Tensor:
        """Output ``wo`` linear with the same chunking (BF16 in from concat_heads, BF8 weights)."""
        attn_output_11SH, seq_len, original_seq_len = pad_seq_to_chunk_multiple(
            attn_output_11SH, seq_len, max_mm_seq_len
        )
        wo_mem_cfg = (
            output_memory_config if output_memory_config is not None else vision_seq_memcfg(seq_len, self.hidden_size)
        )
        if seq_len <= max_mm_seq_len:
            # Block-sharded in0 (from concat_heads) -> bs/dram/bs sweep winner; output stays
            # block-sharded for the reduce-scatter (no post-wo L1->DRAM copy).
            program_config = (
                self._wo_block_shard_progcfg(seq_len)
                if attn_output_11SH.is_sharded()
                else self._wo_program_config(seq_len, seq_len)
            )
            out = ttnn.linear(
                attn_output_11SH,
                self.wo,
                compute_kernel_config=self.wo_compute_kernel_config,
                dtype=ttnn.bfloat16,
                memory_config=wo_mem_cfg,
                program_config=program_config,
            )
            return trim_seq_dim2(out, original_seq_len)

        x_batched = ttnn.reshape(attn_output_11SH, [1, seq_len // max_mm_seq_len, max_mm_seq_len, -1])
        out = ttnn.linear(
            x_batched,
            self.wo,
            compute_kernel_config=self.wo_compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=wo_mem_cfg,
            program_config=self._wo_program_config(seq_len, max_mm_seq_len),
        )
        out = ttnn.reshape(out, [1, 1, seq_len, -1])
        return trim_seq_dim2(out, original_seq_len)

    def forward(self, x_11SH, position_embeddings=None):
        seq_len = int(x_11SH.shape[-2])
        act_mem_cfg = vision_seq_memcfg(seq_len, self.hidden_size)
        if not x_11SH.is_sharded() and x_11SH.memory_config().buffer_type != act_mem_cfg.buffer_type:
            x_11SH = ttnn.to_memory_config(x_11SH, act_mem_cfg)

        max_mm_seq_len = pixtral_effective_mm_seq_len(self.configuration, seq_len)
        # Block-shard the wo (bs/dram/bs sweep winner) so its output feeds the reduce-scatter directly,
        # removing the L1->DRAM copy. wo_out_mem_cfg becomes block-sharded and concat_heads emits the
        # matching block-sharded in0.
        wo_block_shard = self._wo_block_shard_eligible(seq_len, max_mm_seq_len)
        wo_out_mem_cfg = (
            vision_rms_norm_block_shard_memcfg(seq_len, self.hidden_size, 8, 8) if wo_block_shard else act_mem_cfg
        )

        xqkv_fused = self._linear_qkv_seq_chunked(x_11SH, seq_len, max_mm_seq_len)
        if seq_len > max_mm_seq_len and seq_len % max_mm_seq_len == 0:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        q_heads_1QSD, k_heads_1KSD, v_heads_1VSD = self._nlp_create_qkv_heads(xqkv_fused, seq_len)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_heads_1QSD, k_heads_1KSD = apply_rotary_pos_emb_vision_tt(q_heads_1QSD, k_heads_1KSD, cos, sin)

        sdpa_cfg = _pixtral_sdpa_program_config(seq_len, max_mm_seq_len, _pixtral_sdpa_grid_size(self.configuration))
        attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_1QSD,
            k_heads_1KSD,
            v_heads_1VSD,
            is_causal=False,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config_sdpa,
        )
        ttnn.deallocate(q_heads_1QSD)
        ttnn.deallocate(k_heads_1KSD)
        ttnn.deallocate(v_heads_1VSD)

        attn_output_11SH = self._nlp_concat_heads(attn_output_1QSD, seq_len, block_shard_out=wo_block_shard)

        output_11SH = self._linear_wo_seq_chunked(
            attn_output_11SH,
            seq_len,
            max_mm_seq_len,
            output_memory_config=wo_out_mem_cfg,
        )
        if seq_len > max_mm_seq_len and seq_len % max_mm_seq_len == 0:
            if not (len(output_11SH.shape) == 4 and int(output_11SH.shape[0]) == 1 and int(output_11SH.shape[1]) == 1):
                output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        if self.num_devices > 1:
            if not (len(output_11SH.shape) == 4 and int(output_11SH.shape[0]) == 1 and int(output_11SH.shape[1]) == 1):
                output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
            ag_out_mem = None
            if vision_rms_norm_block_shard_eligible(seq_len, self.hidden_size, 8, 8):
                ag_out_mem = vision_rms_norm_block_shard_memcfg(seq_len, self.hidden_size, 8, 8)
            output_11SH = vision_sum_all_reduce(
                output_11SH,
                self.mesh_device,
                self.tt_ccl,
                seq_len,
                self.hidden_size,
                self.configuration,
                ag_out_mem=ag_out_mem,
            )
        return output_11SH


__all__ = ["TtMistralImageAttention", "apply_rotary_pos_emb_vision_tt"]
