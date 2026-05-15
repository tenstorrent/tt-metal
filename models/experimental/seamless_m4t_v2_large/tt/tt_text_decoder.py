# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Decoder`] with prefill and KV-cache decode."""

from __future__ import annotations

import math
from typing import Optional

import torch
import ttnn

from models.common.utility_functions import nearest_32


def _core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


def init_text_decoder_kv_cache(
    device: ttnn.Device,
    *,
    num_hidden_layers: int,
    num_attention_heads: int,
    hidden_size: int,
    max_batch_size: int,
    max_seq_len: int,
    encoder_seq_len: int,
) -> tuple[list[list[ttnn.Tensor]], list[list[ttnn.Tensor]]]:
    """
    Allocate per-layer self-attention and cross-attention KV caches.

    Follows the SpeechT5 / Whisper pattern in ``ttnn_speecht5_decoder.init_kv_cache``.

    Returns:
        ``(kv_cache, cross_attn_cache)`` where each is a list of length ``num_hidden_layers``
        containing ``[K, V]`` device tensors.
    """
    head_dim = hidden_size // num_attention_heads
    chunk_size = 256
    padded_max_seq_len = ((max_seq_len + chunk_size - 1) // chunk_size) * chunk_size

    kv_cache: list[list[ttnn.Tensor]] = []
    cross_attn_cache: list[list[ttnn.Tensor]] = []

    for _ in range(num_hidden_layers):
        k_cache = torch.zeros((max_batch_size, num_attention_heads, padded_max_seq_len, head_dim))
        v_cache = torch.zeros((max_batch_size, num_attention_heads, padded_max_seq_len, head_dim))
        kv_cache.append(
            [
                ttnn.from_torch(
                    k_cache,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.from_torch(
                    v_cache,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            ]
        )

        cross_k = torch.zeros((max_batch_size, num_attention_heads, encoder_seq_len, head_dim))
        cross_v = torch.zeros((max_batch_size, num_attention_heads, encoder_seq_len, head_dim))
        cross_attn_cache.append(
            [
                ttnn.from_torch(
                    cross_k,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
                ttnn.from_torch(
                    cross_v,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                ),
            ]
        )

    return kv_cache, cross_attn_cache


def make_current_decode_pos_tensor(device: ttnn.Device, position: int, batch_size: int = 1) -> ttnn.Tensor:
    """Build ``int32`` ``[batch]`` index tensor for ``paged_update_cache`` / decode SDPA."""
    return ttnn.from_torch(
        torch.full((batch_size,), position, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _get_decode_sdpa_configs(
    device: ttnn.Device,
    *,
    num_attention_heads: int,
    hidden_size: int,
    max_batch_size: int,
    max_seq_len: int,
) -> tuple[ttnn.MemoryConfig, ttnn.SDPAProgramConfig, ttnn.DeviceComputeKernelConfig]:
    """Sharded memory + program config for ``scaled_dot_product_attention_decode``."""
    head_dim = hidden_size // num_attention_heads
    padded_num_heads = nearest_32(num_attention_heads)

    grid_size = device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(max_batch_size, grid_size, row_wise=True)
    sdpa_batch_sharded_memcfg = ttnn.create_sharded_memory_config(
        shape=(padded_num_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    chunk_size = 256
    padded_max_seq_len = ((max_seq_len + chunk_size - 1) // chunk_size) * chunk_size

    def _next_power_of_2(n: int) -> int:
        if n >= 256:
            return 256
        power = 1
        while power * 2 <= n:
            power *= 2
        return power

    k_chunk_size = _next_power_of_2(padded_max_seq_len)
    q_chunk_size = _next_power_of_2(padded_max_seq_len)
    compute_grid_size = device.compute_with_storage_grid_size()
    sdpa_decode_progcfg = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=(compute_grid_size.x, compute_grid_size.y),
        exp_approx_mode=False,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )
    sdpa_decode_compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    return sdpa_batch_sharded_memcfg, sdpa_decode_progcfg, sdpa_decode_compute_cfg


class TTSeamlessM4Tv2Decoder:
    """
    Device port of Hugging Face ``SeamlessM4Tv2Decoder``.

    Prefill: ``forward`` with full sequence (no cache arguments).
    Decode: pass ``kv_cache``, ``cross_attn_cache``, and ``current_decode_pos`` with ``seq_len=1``.

    Use ``create_text_decoder_parameters`` to build ``parameters`` from the PyTorch decoder.
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters,
        *,
        layer_norm_eps: float,
        num_hidden_layers: int,
        num_attention_heads: int,
        hidden_size: int,
        max_batch_size: int = 1,
        max_seq_len: int = 256,
    ):
        self.device = device
        self.parameters = parameters
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._ffn_fc1_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._ffn_fc2_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        self._ln_sharded_cache: dict = {}
        self._projection_pc_cache: dict = {}
        self._decode_sdpa_cache: dict = {}

    def _decode_sdpa_configs(self) -> tuple[ttnn.MemoryConfig, ttnn.SDPAProgramConfig, ttnn.DeviceComputeKernelConfig]:
        key = (self.max_batch_size, self.max_seq_len)
        cached = self._decode_sdpa_cache.get(key)
        if cached is None:
            cached = _get_decode_sdpa_configs(
                self.device,
                num_attention_heads=self.num_attention_heads,
                hidden_size=self.hidden_size,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
            )
            self._decode_sdpa_cache[key] = cached
        return cached

    def _sdpa_program_config(self, seq_q: int, seq_k: int, *, large_chunks: bool = True) -> ttnn.SDPAProgramConfig:
        if large_chunks:
            q_chunk = max(64, min(256, nearest_32(seq_q)))
            k_chunk = max(64, min(256, nearest_32(seq_k)))
        else:
            q_chunk = max(32, min(256, nearest_32(seq_q)))
            k_chunk = max(32, min(256, nearest_32(seq_k)))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    _PROJECTION_1D_SEQ_THRESHOLD = 384

    def _prefill_projection_program_config(self, token_rows: int, in_dim: int, out_dim: int):
        key = (token_rows, in_dim, out_dim)
        cached = self._projection_pc_cache.get(key)
        if cached is not None:
            return cached

        cg = self.device.compute_with_storage_grid_size()
        k_tiles = max(1, in_dim // 32)
        in0_block_w = min(4, k_tiles)
        if token_rows <= self._PROJECTION_1D_SEQ_THRESHOLD:
            per_core_n = (out_dim + cg.x * 32 - 1) // (cg.x * 32)
            per_core_m = (token_rows + 31) // 32
            out_subblock_w = min(4, max(1, per_core_n))
            while out_subblock_w > 1 and per_core_n % out_subblock_w != 0:
                out_subblock_w -= 1
            result = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(cg.x, 1),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=max(1, per_core_m),
                per_core_N=max(1, per_core_n),
                fuse_batch=True,
                mcast_in0=True,
            )
        else:
            grid_y = min(cg.y, (token_rows + 31) // 32)
            per_core_m = (token_rows + grid_y * 32 - 1) // (grid_y * 32)
            per_core_n = (out_dim + cg.x * 32 - 1) // (cg.x * 32)
            out_subblock_w = min(4, max(1, per_core_n))
            while out_subblock_w > 1 and per_core_n % out_subblock_w != 0:
                out_subblock_w -= 1
            result = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(cg.x, grid_y),
                in0_block_w=in0_block_w,
                out_subblock_h=1,
                out_subblock_w=out_subblock_w,
                per_core_M=max(1, per_core_m),
                per_core_N=max(1, per_core_n),
                transpose_mcast=False,
                fused_activation=None,
            )
        self._projection_pc_cache[key] = result
        return result

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        compute_cfg: Optional[ttnn.DeviceComputeKernelConfig] = None,
        memory_config: ttnn.MemoryConfig = ttnn.DRAM_MEMORY_CONFIG,
        program_config=None,
        activation: Optional[str] = None,
    ) -> ttnn.Tensor:
        ck = compute_cfg if compute_cfg is not None else self._linear_ln_compute_cfg
        if program_config is not None:
            return ttnn.linear(
                x,
                weight,
                bias=bias,
                program_config=program_config,
                memory_config=memory_config,
                compute_kernel_config=ck,
                activation=activation,
            )
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            core_grid=_core_grid(self.device),
            memory_config=memory_config,
            compute_kernel_config=ck,
            activation=activation,
        )

    def _build_ln_sharded_config(self, m_tiles: int, n_tiles: int):
        key = (m_tiles, n_tiles)
        cached = self._ln_sharded_cache.get(key)
        if cached is not None:
            return cached

        device_grid = self.device.compute_with_storage_grid_size()
        grid_x = device_grid.x
        while grid_x > 1 and n_tiles % grid_x != 0:
            grid_x -= 1
        block_w = n_tiles // grid_x

        grid_y = min(device_grid.y, m_tiles)
        while grid_y > 1 and m_tiles % grid_y != 0:
            grid_y -= 1
        block_h = m_tiles // grid_y

        subblock_w = min(block_w, 4)
        while block_w % subblock_w != 0:
            subblock_w -= 1

        program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(grid_x, grid_y),
            subblock_w=subblock_w,
            block_h=block_h,
            block_w=block_w,
            inplace=False,
        )

        shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
            [block_h * 32, block_w * 32],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        memory_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED if grid_y == 1 else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            ttnn.BufferType.L1,
            shard_spec,
        )

        cached = (memory_config, program_config)
        self._ln_sharded_cache[key] = cached
        return cached

    def _layer_norm_sharded(
        self,
        x: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        m_tiles: int,
        n_tiles: int,
    ) -> ttnn.Tensor:
        sharded_mem_config, sharded_pc = self._build_ln_sharded_config(m_tiles, n_tiles)
        x_sharded = ttnn.to_memory_config(x, sharded_mem_config)
        normed_sharded = ttnn.layer_norm(
            x_sharded,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=sharded_mem_config,
            program_config=sharded_pc,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )
        ttnn.deallocate(x_sharded)
        normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        ttnn.deallocate(normed_sharded)
        return normed

    @staticmethod
    def _heads(x: ttnn.Tensor, batch: int, seq: int, num_heads: int, head_dim: int) -> ttnn.Tensor:
        x = ttnn.reshape(x, (batch, seq, num_heads, head_dim))
        return ttnn.permute(x, (0, 2, 1, 3))

    def _self_attention_decode(
        self,
        hidden_states: ttnn.Tensor,
        attn_module,
        kv_cache: list[ttnn.Tensor],
        current_decode_pos: ttnn.Tensor,
        *,
        batch: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
    ) -> ttnn.Tensor:
        seq_q = 1
        pc_qkv = self._prefill_projection_program_config(batch * seq_q, hidden_size, 3 * hidden_size)
        qkv = self._linear(
            hidden_states,
            attn_module.qkv.weight,
            attn_module.qkv.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=pc_qkv,
        )
        qkv_4d = ttnn.reshape(qkv, (batch, 1, seq_q, 3 * hidden_size))
        ttnn.deallocate(qkv)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_4d,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv_4d)

        sdpa_batch_sharded_memcfg, sdpa_decode_progcfg, sdpa_decode_compute_cfg = self._decode_sdpa_configs()
        k_cache, v_cache = kv_cache

        query = ttnn.transpose(q, 0, 2)
        query = ttnn.transpose(query, 1, 2)
        key = ttnn.transpose(k, 0, 2)
        key = ttnn.transpose(key, 1, 2)
        value = ttnn.transpose(v, 0, 2)
        value = ttnn.transpose(value, 1, 2)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        query = ttnn.multiply(query, 1.0 / math.sqrt(head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
        query = ttnn.interleaved_to_sharded(query, sdpa_batch_sharded_memcfg)
        key = ttnn.interleaved_to_sharded(key, sdpa_batch_sharded_memcfg)
        value = ttnn.interleaved_to_sharded(value, sdpa_batch_sharded_memcfg)

        ttnn.experimental.paged_update_cache(k_cache, key, update_idxs_tensor=current_decode_pos, page_table=None)
        ttnn.experimental.paged_update_cache(v_cache, value, update_idxs_tensor=current_decode_pos, page_table=None)
        ttnn.deallocate(key)
        ttnn.deallocate(value)

        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            query,
            k_cache,
            v_cache,
            cur_pos_tensor=current_decode_pos,
            scale=1.0,
            program_config=sdpa_decode_progcfg,
            compute_kernel_config=sdpa_decode_compute_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_out = ttnn.transpose(attn_out, 1, 2)
        attn_out = ttnn.transpose(attn_out, 0, 2)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        merged = ttnn.reshape(merged_4d, (batch, seq_q, hidden_size))
        ttnn.deallocate(merged_4d)
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=self._prefill_projection_program_config(batch * seq_q, hidden_size, hidden_size),
        )
        ttnn.deallocate(merged)
        return proj

    def _cross_attention_decode(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        attn_module,
        cross_attn_cache: Optional[list[ttnn.Tensor]],
        cross_attn_cache_valid: bool,
        cross_attention_mask: Optional[ttnn.Tensor],
        sdpa_cfg: ttnn.SDPAProgramConfig,
        *,
        batch: int,
        enc_seq: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
    ) -> ttnn.Tensor:
        seq_q = 1
        pc_q = self._prefill_projection_program_config(batch * seq_q, hidden_size, hidden_size)

        if cross_attn_cache is not None and cross_attn_cache_valid:
            q = self._linear(
                hidden_states,
                attn_module.q_proj.weight,
                attn_module.q_proj.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_q,
            )
            qh = self._heads(q, batch, seq_q, num_heads, head_dim)
            ttnn.deallocate(q)
            kh, vh = cross_attn_cache[0], cross_attn_cache[1]
        else:
            q = self._linear(
                hidden_states,
                attn_module.q_proj.weight,
                attn_module.q_proj.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_q,
            )
            pc_kv2 = self._prefill_projection_program_config(batch * enc_seq, hidden_size, 2 * hidden_size)
            kv_packed = self._linear(
                encoder_hidden_states,
                attn_module.kv.weight,
                attn_module.kv.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_kv2,
            )
            k = ttnn.slice(kv_packed, [0, 0, 0], [batch, enc_seq, hidden_size], [1, 1, 1])
            v = ttnn.slice(kv_packed, [0, 0, hidden_size], [batch, enc_seq, 2 * hidden_size], [1, 1, 1])
            ttnn.deallocate(kv_packed)
            qh = self._heads(q, batch, seq_q, num_heads, head_dim)
            kh = self._heads(k, batch, enc_seq, num_heads, head_dim)
            vh = self._heads(v, batch, enc_seq, num_heads, head_dim)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            if cross_attn_cache is not None:
                ttnn.copy(kh, cross_attn_cache[0])
                ttnn.copy(vh, cross_attn_cache[1])
                kh, vh = cross_attn_cache[0], cross_attn_cache[1]

        qh = ttnn.multiply(qh, 1.0 / math.sqrt(head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=cross_attention_mask,
            is_causal=False,
            scale=1.0,
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qh)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        merged = ttnn.reshape(merged_4d, (batch, seq_q, hidden_size))
        ttnn.deallocate(merged_4d)
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=self._prefill_projection_program_config(batch * seq_q, hidden_size, hidden_size),
        )
        ttnn.deallocate(merged)
        return proj

    def _attention(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: Optional[ttnn.Tensor],
        attn_module,
        attn_mask: Optional[ttnn.Tensor],
        *,
        batch: int,
        seq_q: int,
        seq_k: int,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        sdpa_cfg: ttnn.SDPAProgramConfig,
        kv_cache: Optional[list[ttnn.Tensor]] = None,
        cross_attn_cache: Optional[list[ttnn.Tensor]] = None,
        cross_attn_cache_valid: bool = False,
        current_decode_pos: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        is_decode = kv_cache is not None and current_decode_pos is not None
        if is_decode and encoder_hidden_states is None:
            return self._self_attention_decode(
                hidden_states,
                attn_module,
                kv_cache,
                current_decode_pos,
                batch=batch,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
            )
        if is_decode and encoder_hidden_states is not None:
            return self._cross_attention_decode(
                hidden_states,
                encoder_hidden_states,
                attn_module,
                cross_attn_cache,
                cross_attn_cache_valid,
                attn_mask,
                sdpa_cfg,
                batch=batch,
                enc_seq=seq_k,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
            )

        q_src = hidden_states
        kv_src = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        if encoder_hidden_states is None and hasattr(attn_module, "qkv"):
            pc_qkv = self._prefill_projection_program_config(batch * seq_q, hidden_size, 3 * hidden_size)
            qkv = self._linear(
                q_src,
                attn_module.qkv.weight,
                attn_module.qkv.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_qkv,
            )
            qkv_4d = ttnn.reshape(qkv, (batch, 1, seq_q, 3 * hidden_size))
            ttnn.deallocate(qkv)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv_4d,
                num_heads=num_heads,
                num_kv_heads=num_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv_4d)
            qh = ttnn.multiply(q, 1.0 / math.sqrt(head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q)
            kh, vh = k, v
        else:
            pc_q = self._prefill_projection_program_config(batch * seq_q, hidden_size, hidden_size)
            pc_kv_single = (
                pc_q
                if seq_k == seq_q and kv_src is q_src
                else self._prefill_projection_program_config(batch * seq_k, hidden_size, hidden_size)
            )

            q = self._linear(
                q_src,
                attn_module.q_proj.weight,
                attn_module.q_proj.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_q,
            )
            kv_packed = None
            if hasattr(attn_module, "kv"):
                pc_kv2 = self._prefill_projection_program_config(batch * seq_k, hidden_size, 2 * hidden_size)
                kv_packed = self._linear(
                    kv_src,
                    attn_module.kv.weight,
                    attn_module.kv.bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    program_config=pc_kv2,
                )
                k = ttnn.slice(kv_packed, [0, 0, 0], [batch, seq_k, hidden_size], [1, 1, 1])
                v = ttnn.slice(kv_packed, [0, 0, hidden_size], [batch, seq_k, 2 * hidden_size], [1, 1, 1])
            else:
                k = self._linear(
                    kv_src,
                    attn_module.k_proj.weight,
                    attn_module.k_proj.bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    program_config=pc_kv_single,
                )
                v = self._linear(
                    kv_src,
                    attn_module.v_proj.weight,
                    attn_module.v_proj.bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    program_config=pc_kv_single,
                )

            qh = self._heads(q, batch, seq_q, num_heads, head_dim)
            kh = self._heads(k, batch, seq_k, num_heads, head_dim)
            vh = self._heads(v, batch, seq_k, num_heads, head_dim)

            ttnn.deallocate(q)
            if kv_packed is not None:
                ttnn.deallocate(kv_packed)
            else:
                ttnn.deallocate(k)
                ttnn.deallocate(v)

            qh = ttnn.multiply(qh, 1.0 / math.sqrt(head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)

        is_causal = encoder_hidden_states is None and attn_mask is None

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=1.0,
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qh)
        ttnn.deallocate(kh)
        ttnn.deallocate(vh)

        ls = attn_out.shape
        ps = attn_out.padded_shape
        if (
            len(ls) == 4
            and int(ls[0]) == batch
            and int(ls[1]) == num_heads
            and int(ls[2]) == seq_q
            and int(ls[3]) == head_dim
            and len(ps) >= 4
            and int(ps[3]) == head_dim
        ):
            attn_for_concat = attn_out
        else:
            attn_for_concat = ttnn.slice(
                attn_out,
                [0, 0, 0, 0],
                [batch, num_heads, seq_q, head_dim],
                [1, 1, 1, 1],
            )
            ttnn.deallocate(attn_out)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_for_concat, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_for_concat)
        merged = ttnn.reshape(merged_4d, (batch, seq_q, hidden_size))
        ttnn.deallocate(merged_4d)
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=self._prefill_projection_program_config(batch * seq_q, hidden_size, hidden_size),
        )
        ttnn.deallocate(merged)
        return proj

    def _decoder_layers(
        self,
        hidden: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        *,
        batch: int,
        seq: int,
        enc_seq: int,
        causal_attention_mask: Optional[ttnn.Tensor],
        cross_attention_mask: Optional[ttnn.Tensor],
        kv_cache: Optional[list[list[ttnn.Tensor]]] = None,
        cross_attn_cache: Optional[list[list[ttnn.Tensor]]] = None,
        cross_attn_cache_valid: bool = False,
        current_decode_pos: Optional[ttnn.Tensor] = None,
        trace_no_profiler: bool = False,
    ) -> ttnn.Tensor:
        parameters = self.parameters
        num_heads = self.num_attention_heads
        hidden_size = self.hidden_size
        head_dim = hidden_size // num_heads
        num_layers = self.num_hidden_layers
        is_decode = kv_cache is not None and current_decode_pos is not None

        sdpa_self = self._sdpa_program_config(seq, seq, large_chunks=True)
        sdpa_cross = self._sdpa_program_config(seq, enc_seq, large_chunks=(enc_seq >= 32))

        ffn_intermediate = int(parameters.layers[0].ffn.fc1.weight.shape[1])
        token_m = batch * seq
        pc_ffn_fc1 = self._prefill_projection_program_config(token_m, hidden_size, ffn_intermediate)
        pc_ffn_fc2 = self._prefill_projection_program_config(token_m, ffn_intermediate, hidden_size)

        m_tiles = (batch * seq + 31) // 32
        n_tiles = hidden_size // 32

        for i in range(num_layers):
            layer = parameters.layers[i]
            layer_kv = kv_cache[i] if kv_cache is not None else None
            layer_cross = cross_attn_cache[i] if cross_attn_cache is not None else None

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.self_attn_layer_norm.weight,
                bias=layer.self_attn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
            )
            self_mask = None if is_decode and seq == 1 else causal_attention_mask
            attn_out = self._attention(
                normed,
                None,
                layer.self_attn,
                self_mask,
                batch=batch,
                seq_q=seq,
                seq_k=seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_cfg=sdpa_self,
                kv_cache=layer_kv,
                current_decode_pos=current_decode_pos,
            )
            ttnn.deallocate(normed)
            residual = hidden
            hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.cross_attention_layer_norm.weight,
                bias=layer.cross_attention_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
            )
            attn_out = self._attention(
                normed,
                encoder_hidden_states,
                layer.cross_attention,
                cross_attention_mask,
                batch=batch,
                seq_q=seq,
                seq_k=enc_seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_cfg=sdpa_cross,
                kv_cache=layer_kv,
                cross_attn_cache=layer_cross,
                cross_attn_cache_valid=cross_attn_cache_valid,
                current_decode_pos=current_decode_pos,
            )
            ttnn.deallocate(normed)
            residual = hidden
            hidden = ttnn.add(residual, attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.ffn_layer_norm.weight,
                bias=layer.ffn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
            )
            ff = self._linear(
                normed,
                layer.ffn.fc1.weight,
                layer.ffn.fc1.bias,
                compute_cfg=self._ffn_fc1_compute_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_ffn_fc1,
                activation="relu",
            )
            ttnn.deallocate(normed)
            ff_in = ff
            ff = self._linear(
                ff_in,
                layer.ffn.fc2.weight,
                layer.ffn.fc2.bias,
                compute_cfg=self._ffn_fc2_compute_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_ffn_fc2,
            )
            ttnn.deallocate(ff_in)
            residual = hidden
            hidden = ttnn.add(residual, ff, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual)
            ttnn.deallocate(ff)

            # Per-layer drain: decode forwards exceed the ~12k per-core profiler buffer
            # before ``forward`` returns (see Metal "markers were dropped" warnings).
            if not trace_no_profiler:
                ttnn.ReadDeviceProfiler(self.device)

        return hidden

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor],
        position_ids: ttnn.Tensor,
        encoder_hidden_states: ttnn.Tensor,
        causal_attention_mask: Optional[ttnn.Tensor],
        cross_attention_mask: Optional[ttnn.Tensor] = None,
        *,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        kv_cache: Optional[list[list[ttnn.Tensor]]] = None,
        cross_attn_cache: Optional[list[list[ttnn.Tensor]]] = None,
        cross_attn_cache_valid: bool = False,
        current_decode_pos: Optional[ttnn.Tensor] = None,
        trace_no_profiler: bool = False,
    ) -> ttnn.Tensor:
        """
        Prefill when ``kv_cache`` is ``None``; decode when ``kv_cache`` and ``current_decode_pos`` are set.

        Decode expects ``seq_len=1``, no causal mask, and position ids with the correct
        ``past_key_values_length``.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify only one of input_ids or inputs_embeds.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("One of input_ids or inputs_embeds is required.")

        parameters = self.parameters
        hidden_size = self.hidden_size
        enc_seq = int(encoder_hidden_states.shape[1])

        if inputs_embeds is not None:
            batch = int(inputs_embeds.shape[0])
            seq = int(inputs_embeds.shape[1])
            pos = ttnn.embedding(
                position_ids,
                weight=parameters.embed_positions.weight,
                layout=ttnn.TILE_LAYOUT,
            )
            hidden = ttnn.add(inputs_embeds, pos, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(pos)
        else:
            batch = int(input_ids.shape[0])  # type: ignore[union-attr]
            seq = int(input_ids.shape[1])
            tok = ttnn.embedding(
                input_ids,
                weight=parameters.embed_tokens.weight,
                layout=ttnn.TILE_LAYOUT,
            )
            pos = ttnn.embedding(
                position_ids,
                weight=parameters.embed_positions.weight,
                layout=ttnn.TILE_LAYOUT,
            )
            hidden = ttnn.add(tok, pos, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(tok)
            ttnn.deallocate(pos)

        hidden = self._decoder_layers(
            hidden,
            encoder_hidden_states,
            batch=batch,
            seq=seq,
            enc_seq=enc_seq,
            causal_attention_mask=causal_attention_mask,
            cross_attention_mask=cross_attention_mask,
            kv_cache=kv_cache,
            cross_attn_cache=cross_attn_cache,
            cross_attn_cache_valid=cross_attn_cache_valid,
            current_decode_pos=current_decode_pos,
            trace_no_profiler=trace_no_profiler,
        )

        m_tiles = (batch * seq + 31) // 32
        n_tiles = hidden_size // 32
        out = self._layer_norm_sharded(
            hidden,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
            m_tiles=m_tiles,
            n_tiles=n_tiles,
        )
        ttnn.deallocate(hidden)

        # Drain on-device profiler markers so Tracy can match host ops to
        # ``cpp_device_perf_report.csv``. No-op in non-profiler builds. Skip during trace capture.
        if not trace_no_profiler:
            ttnn.ReadDeviceProfiler(self.device)

        return out
