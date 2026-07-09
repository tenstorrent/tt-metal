# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Encoder`] (prefill / inference)."""

from __future__ import annotations

import math
import os
from typing import Optional

import ttnn

from models.experimental.seamless_m4t_v2_large.tt.common import (
    TILE,
    build_ln_sharded_config,
    encoder_all_reduce_sum_replicate,
    encoder_tp_activation_memory_config,
    dram_linear_input_mem_config,
    dram_matmul_program_config,
    encoder_tp_block_sharded_matmul,
    encoder_tp_interleaved_matmul_program_config,
    encoder_tp_matmul_in0_ln_config,
    ensure_interleaved_bsh,
    ensure_l1_width_sharded_activation,
    sdpa_program_config,
    width_sharded_to_l1_interleaved,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import mesh_cluster_axis, get_tp


class TTSeamlessM4Tv2Encoder:
    """
    Device port of Hugging Face ``SeamlessM4Tv2Encoder``.

    ``forward`` takes tensors already placed on the device. Use
    ``create_text_encoder_parameters`` to build ``parameters`` from the PyTorch encoder.

    Prefill matmuls use L1 width-sharded activations + DRAM width-sharded BFP8 weights
    (``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig``).
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
    ):
        self.device = device
        self.parameters = parameters
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self._tp = get_tp(device)
        self._cluster_axis = mesh_cluster_axis(device)
        # Long-seq TP: block-sharded LN when enabled (``SEAMLESS_TP_BS_LN=0`` disables).
        self._tp_bs_ln = os.environ.get("SEAMLESS_TP_BS_LN", "1") == "1"
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._linear_ln_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self._layernorm_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        # Sharded-LN program config + shard spec are shape-dependent; we build
        # them once per (M_tiles, N_tiles) shape and cache the result.  Default
        # ``LayerNormDefaultProgramConfig`` runs on a single core (~44 us per
        # call x 49 calls = 22 % of device time); the sharded variant spreads
        # the reduction across grid_x cores -- typically 8 cores at seq=32 --
        # for a ~4-5x per-op speedup.
        self._ln_sharded_cache: dict = {}
        self._tp_ln_fusion_cache: dict = {}
        self._sdpa_pc_cache: dict = {}
        self._dram_matmul_pc_cache: dict = {}
        self._width_shard_mem_cache: dict = {}
        self._chunked_tp_compute_cfg = None

    @staticmethod
    def _tp_bs_chunk_rows() -> int:
        """Row chunk size for long-seq TP block-sharded matmul (``SEAMLESS_TP_BS_CHUNK_M``, default 2048)."""
        try:
            return int(os.environ.get("SEAMLESS_TP_BS_CHUNK_M", "2048"))
        except ValueError:
            return 2048

    def _chunked_tp_linear_compute_cfg(self) -> ttnn.DeviceComputeKernelConfig:
        # Multiple LoFi block-sharded matmul chunks compound bf16 noise; HiFi2 matches TP=1 chunked path PCC.
        if self._chunked_tp_compute_cfg is None:
            self._chunked_tp_compute_cfg = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
        return self._chunked_tp_compute_cfg

    def _width_shard_mem_config(self, token_rows: int, channels: int, out_channels: int) -> ttnn.MemoryConfig:
        key = (token_rows, channels, out_channels)
        cached = self._width_shard_mem_cache.get(key)
        if cached is not None:
            return cached
        cached = dram_linear_input_mem_config(self.device, token_rows, channels, out_channels)
        self._width_shard_mem_cache[key] = cached
        return cached

    def _to_matmul_width_sharded(
        self, x: ttnn.Tensor, token_rows: int, channels: int, out_channels: int
    ) -> ttnn.Tensor:
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (token_rows, channels))
        return ensure_l1_width_sharded_activation(self.device, x, token_rows, channels, out_channels)

    @staticmethod
    def _width_sharded_to_3d(x: ttnn.Tensor, batch: int, seq: int, channels: int) -> ttnn.Tensor:
        return ensure_interleaved_bsh(x, batch=batch, seq=seq, channels=channels)

    def _sdpa_program_config(self, seq_q: int, seq_k: int) -> ttnn.SDPAProgramConfig:
        return sdpa_program_config(self.device, seq_q, seq_k, self._sdpa_pc_cache)

    @staticmethod
    def _linear_token_rows(x: ttnn.Tensor) -> int:
        if len(x.shape) == 3:
            return int(x.shape[0]) * int(x.shape[1])
        if len(x.shape) == 2:
            return int(x.shape[0])
        return int(x.shape[-2])

    def _dram_matmul_pc(
        self,
        m: int,
        k: int,
        n: int,
        *,
        fused_activation=None,
    ) -> ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig:
        key = (m, k, n, fused_activation)
        cached = self._dram_matmul_pc_cache.get(key)
        if cached is not None:
            return cached
        cached = dram_matmul_program_config(
            self.device,
            m,
            k,
            n,
            fused_activation=fused_activation,
        )
        self._dram_matmul_pc_cache[key] = cached
        return cached

    @staticmethod
    def _bias_token_rows(bias: ttnn.Tensor) -> int:
        if len(bias.shape) == 4:
            return int(bias.shape[2])
        return 32

    @staticmethod
    def _pad_token_rows(x: ttnn.Tensor, m_actual: int, m_padded: int) -> ttnn.Tensor:
        if m_actual >= m_padded:
            return x
        k = int(x.shape[-1])
        pad_rows = m_padded - m_actual
        pad = ttnn.full(
            [pad_rows, k],
            0.0,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=x.device(),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        padded = ttnn.concat([x, pad], dim=0, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pad)
        return padded

    def _finalize_dram_sharded_linear(
        self,
        x: ttnn.Tensor,
        *,
        batch: int,
        seq: int,
        m_actual: int,
        out_dim: int,
    ) -> ttnn.Tensor:
        x = width_sharded_to_l1_interleaved(x)
        if len(x.shape) == 4 and int(x.shape[1]) == 1:
            x = ttnn.reshape(x, (batch, seq, int(x.shape[-1])))
        if len(x.shape) == 2 and int(x.shape[0]) > m_actual:
            x = ttnn.slice(x, [0, 0], [m_actual, int(x.shape[-1])], [1, 1])
        padded_n = int(x.shape[-1])
        if padded_n > out_dim:
            if len(x.shape) == 2:
                x = ttnn.slice(x, [0, 0], [m_actual, out_dim], [1, 1])
            elif len(x.shape) == 3:
                x = ttnn.slice(x, [0, 0, 0], [batch, seq, out_dim], [1, 1, 1])
        if len(x.shape) == 2:
            return ttnn.reshape(x, (batch, seq, out_dim))
        if len(x.shape) == 3 and (int(x.shape[0]) != batch or int(x.shape[1]) != seq):
            x = ttnn.slice(x, [0, 0, 0], [batch, seq, out_dim], [1, 1, 1])
            return ttnn.reshape(x, (batch, seq, out_dim))
        return x

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: Optional[str] = None,
        logical_out_dim: Optional[int] = None,
        keep_sharded_output: bool = False,
        accept_sharded_input: bool = False,
        batch: Optional[int] = None,
        seq: Optional[int] = None,
    ) -> ttnn.Tensor:
        k = int(weight.shape[-2])
        n = int(weight.shape[-1])
        out_dim = logical_out_dim if logical_out_dim is not None else n

        if (accept_sharded_input or (len(x.shape) == 2 and ttnn.is_sharded(x))) and ttnn.is_sharded(x):
            if batch is None or seq is None:
                raise ValueError("batch and seq are required for sharded linear input")
            x_flat = x
            m_actual = batch * seq
            m = self._bias_token_rows(bias)
            x_sharded = ensure_l1_width_sharded_activation(self.device, x_flat, m, k, n) if m_actual <= TILE else None
        elif len(x.shape) == 3:
            batch = int(x.shape[0])
            seq = int(x.shape[1])
            x_flat = ttnn.reshape(x, (batch * seq, k))
            m_actual = batch * seq
            x_sharded = None
        elif len(x.shape) == 2:
            batch = batch if batch is not None else int(x.shape[0])
            seq = seq if seq is not None else 1
            x_flat = x
            m_actual = int(x.shape[0])
            x_sharded = None
        else:
            batch = batch if batch is not None else 1
            seq = seq if seq is not None else 1
            x_flat = x
            m_actual = self._linear_token_rows(x)
            x_sharded = None

        m = self._bias_token_rows(bias)
        if m_actual > m:
            # Long-seq prefill (m_actual > TILE): the DRAM-sharded matmul kernel is hard-coded to
            # M == TILE, so chunk the matmul into ``ceil(m_actual / TILE)`` calls of that kernel.
            # Each chunk runs the bit-identical fast-path matmul, so PCC is unchanged.
            return self._linear_chunked(
                x_flat if x_sharded is None else x,
                weight,
                bias,
                activation=activation,
                logical_out_dim=out_dim,
                batch=batch,
                seq=seq,
                m_actual=m_actual,
                m=m,
                k=k,
                n=n,
            )

        if x_sharded is None:
            x_flat = self._pad_token_rows(x_flat, m_actual, m)
            x_sharded = ensure_l1_width_sharded_activation(self.device, x_flat, m, k, n)
        fused_activation = ttnn.UnaryOpType.RELU if activation == "relu" else None
        out = ttnn.linear(
            x_sharded,
            weight,
            bias=bias,
            program_config=self._dram_matmul_pc(m, k, n, fused_activation=fused_activation),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )
        if x_sharded is not x_flat and x_sharded is not x:
            ttnn.deallocate(x_sharded)
        if keep_sharded_output:
            return out
        return self._finalize_dram_sharded_linear(
            out,
            batch=batch,
            seq=seq,
            m_actual=m_actual,
            out_dim=out_dim,
        )

    def _linear_chunked(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: Optional[str],
        logical_out_dim: int,
        batch: int,
        seq: int,
        m_actual: int,
        m: int,
        k: int,
        n: int,
    ) -> ttnn.Tensor:
        """Chunked long-seq variant of ``_linear``.

        Slices ``x`` into ``ceil(m_actual / TILE)`` chunks of ``m == TILE`` rows, runs the existing
        DRAM-sharded matmul kernel on each, and concatenates the per-chunk interleaved outputs.

        Each chunk goes through bit-identical math to the short-seq fast path, so PCC is
        preserved. Output is interleaved ``[batch, seq, logical_out_dim]``.
        """
        fused_activation = ttnn.UnaryOpType.RELU if activation == "relu" else None
        pc = self._dram_matmul_pc(m, k, n, fused_activation=fused_activation)
        # Per-chunk matmul accumulates bf16 partials. The short-seq fast path runs a single matmul
        # on TILE rows (per_core_M=1) at ``LoFi``; the chunked path issues ``ceil(m_actual/TILE)``
        # matmuls and that LoFi noise compounds. ``HiFi2`` keeps PCC at parity with the short-seq
        # path on Blackhole.
        if not hasattr(self, "_chunked_linear_compute_cfg"):
            self._chunked_linear_compute_cfg = ttnn.init_device_compute_kernel_config(
                self.device.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

        # Normalize the input to a 2-D ``[m_actual, k]`` interleaved tensor for slicing.
        # Keep x_inter in DRAM so that L1 interleaved chunks can be freed before the matmul
        # program is dispatched without triggering validate_circular_buffer_region.
        if ttnn.is_sharded(x):
            x_inter = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        else:
            x_inter = x
        if len(x_inter.shape) == 3:
            x_inter = ttnn.reshape(x_inter, (m_actual, k))
        elif len(x_inter.shape) != 2:
            x_inter = ttnn.reshape(x_inter, (m_actual, k))
        if x_inter.memory_config().buffer_type == ttnn.BufferType.L1:
            _x_inter_prev = x_inter
            x_inter = ttnn.to_memory_config(x_inter, ttnn.DRAM_MEMORY_CONFIG)
            if _x_inter_prev is not x:
                ttnn.deallocate(_x_inter_prev)

        chunks: list[ttnn.Tensor] = []
        num_chunks = (m_actual + m - 1) // m
        for i in range(num_chunks):
            start = i * m
            end = min(start + m, m_actual)
            chunk_rows = end - start

            chunk = ttnn.slice(x_inter, [start, 0], [end, k], [1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
            if chunk_rows < m:
                chunk = self._pad_token_rows(chunk, chunk_rows, m)
            chunk_sharded = ensure_l1_width_sharded_activation(self.device, chunk, m, k, n)
            if chunk_sharded is not chunk and chunk is not x_inter:
                ttnn.deallocate(chunk)
                chunk = None
            out_sharded = ttnn.linear(
                chunk_sharded,
                weight,
                bias=bias,
                program_config=pc,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                compute_kernel_config=self._chunked_linear_compute_cfg,
            )
            if chunk_sharded is not chunk:
                ttnn.deallocate(chunk_sharded)
            if chunk is not None and chunk is not x_inter:
                ttnn.deallocate(chunk)
            out_inter = width_sharded_to_l1_interleaved(out_sharded)
            if getattr(self, "_long_seq_mc", None) is ttnn.DRAM_MEMORY_CONFIG:
                out_dram = ttnn.to_memory_config(out_inter, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(out_inter)
                out_inter = out_dram
            ttnn.deallocate(out_sharded)
            # ``out_inter`` shape is 2-D ``[m, padded_n]`` or 4-D ``[1, 1, m, padded_n]`` —
            # normalize to 2-D for concat along dim 0. Trim the bottom pad rows on the last chunk.
            if len(out_inter.shape) == 4 and int(out_inter.shape[1]) == 1:
                out_inter = ttnn.reshape(out_inter, (int(out_inter.shape[2]), int(out_inter.shape[-1])))
            if chunk_rows < m:
                out_inter = ttnn.slice(
                    out_inter,
                    [0, 0],
                    [chunk_rows, int(out_inter.shape[-1])],
                    [1, 1],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            chunks.append(out_inter)

        if x_inter is not x:
            ttnn.deallocate(x_inter)

        # Chunk-level outputs land in L1 (small per chunk); the concatenated full activation may
        # not fit (``seq=4096, ffn_dim=8192`` → 64 MB). Route the concat output to DRAM when the
        # caller's long-seq policy says so.
        concat_mc = getattr(self, "_long_seq_mc", None) or ttnn.L1_MEMORY_CONFIG
        if len(chunks) == 1:
            out_concat = chunks[0]
            if concat_mc is ttnn.DRAM_MEMORY_CONFIG and out_concat.memory_config().buffer_type != ttnn.BufferType.DRAM:
                out_dram = ttnn.to_memory_config(out_concat, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(out_concat)
                out_concat = out_dram
        else:
            out_concat = ttnn.concat(chunks, dim=0, memory_config=concat_mc)
            for c in chunks:
                ttnn.deallocate(c)

        # Slice off the DRAM-pad columns (``padded_n - logical_out_dim``) and reshape to 3-D.
        padded_n = int(out_concat.shape[-1])
        if padded_n > logical_out_dim:
            out_concat = ttnn.slice(out_concat, [0, 0], [m_actual, logical_out_dim], [1, 1], memory_config=concat_mc)
        return ttnn.reshape(out_concat, (batch, seq, logical_out_dim))

    @staticmethod
    def _slice_tp_matmul_rows(x: ttnn.Tensor, start: int, end: int, k: int) -> ttnn.Tensor:
        """Slice token rows from a block-sharded or interleaved activation."""
        rank = len(x.shape)
        if rank == 4:
            return ttnn.slice(x, [0, 0, start, 0], [1, 1, end, k], [1, 1, 1, 1])
        if rank == 3:
            return ttnn.slice(x, [0, start, 0], [1, end, k], [1, 1, 1])
        if rank == 2:
            return ttnn.slice(x, [start, 0], [end, k], [1, 1])
        raise ValueError(f"unsupported activation rank for TP matmul slice: {rank}")

    def _build_ln_sharded_config(self, m_tiles: int, n_tiles: int):
        return build_ln_sharded_config(self.device, m_tiles, n_tiles, self._ln_sharded_cache)

    def _layer_norm_sharded(
        self,
        x: ttnn.Tensor,
        *,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        m_tiles: int,
        n_tiles: int,
        input_sharded: bool = False,
        output_sharded: bool = False,
        matmul_fusion_m: Optional[int] = None,
        matmul_fusion_k: Optional[int] = None,
    ) -> ttnn.Tensor:
        """Sharded multicore LN. Set ``output_sharded=True`` to feed a matmul without an S2I.

        Short-seq (``m_tiles == 1``) uses WIDTH_SHARDED LN; long-seq uses BLOCK_SHARDED LN when
        ``SEAMLESS_TP_BS_LN`` is on (default) and ``output_sharded`` feeds a block-sharded matmul.
        When ``matmul_fusion_m`` / ``matmul_fusion_k`` are set, LN uses the same block-sharded
        layout as ``encoder_tp_matmul_in0_memory_config`` so the downstream matmul skips S2I.
        Set ``SEAMLESS_TP_BS_LN=0`` to use interleaved ``ttnn.layer_norm`` instead.
        """
        if m_tiles > 1:
            # Block-sharded long-seq LN (see method docstring): keep the LN sharded so the downstream
            # block-sharded matmul skips its interleaved->block reshard. Otherwise fall through to the
            # unsharded path below.
            if self._tp_bs_ln and output_sharded:
                fused_ln = None
                if matmul_fusion_m is not None and matmul_fusion_k is not None:
                    fused_ln = encoder_tp_matmul_in0_ln_config(
                        self.device, matmul_fusion_m, matmul_fusion_k, self._tp_ln_fusion_cache
                    )
                if fused_ln is not None:
                    sharded_mem_config, sharded_pc = fused_ln
                else:
                    sharded_mem_config, base_pc = self._build_ln_sharded_config(m_tiles, n_tiles)
                    sharded_pc = ttnn.LayerNormShardedMultiCoreProgramConfig(
                        compute_with_storage_grid_size=base_pc.compute_with_storage_grid_size,
                        subblock_w=base_pc.subblock_w,
                        block_h=base_pc.block_h,
                        block_w=base_pc.block_w,
                        inplace=True,
                    )
                if input_sharded and ttnn.is_sharded(x) and x.memory_config() == sharded_mem_config:
                    x_sharded = x
                else:
                    x_sharded = ttnn.to_memory_config(x, sharded_mem_config)
                normed = ttnn.layer_norm(
                    x_sharded,
                    weight=weight,
                    bias=bias,
                    epsilon=self.layer_norm_eps,
                    memory_config=sharded_mem_config,
                    program_config=sharded_pc,
                    compute_kernel_config=self._layernorm_compute_cfg,
                )
                return normed

            x_inter = x
            if ttnn.is_sharded(x):
                x_inter = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
            normed = ttnn.layer_norm(
                x_inter,
                weight=weight,
                bias=bias,
                epsilon=self.layer_norm_eps,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                compute_kernel_config=self._layernorm_compute_cfg,
            )
            if x_inter is not x:
                ttnn.deallocate(x_inter)
            return normed

        sharded_mem_config, sharded_pc = self._build_ln_sharded_config(m_tiles, n_tiles)

        if input_sharded and ttnn.is_sharded(x):
            x_sharded = x
        else:
            x_sharded = ttnn.to_memory_config(x, sharded_mem_config)
        normed_sharded = ttnn.layer_norm(
            x_sharded,
            weight=weight,
            bias=bias,
            epsilon=self.layer_norm_eps,
            memory_config=sharded_mem_config,
            program_config=sharded_pc,
            compute_kernel_config=self._layernorm_compute_cfg,
        )
        if x_sharded is not x:
            ttnn.deallocate(x_sharded)
        if output_sharded:
            return normed_sharded
        normed = ttnn.sharded_to_interleaved(normed_sharded, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        ttnn.deallocate(normed_sharded)
        return normed

    def _linear_tp_block_sharded(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        m: int,
        k: int,
        n: int,
        fused_activation,
        program_config,
        in0_mem: ttnn.MemoryConfig,
        out_mem: ttnn.MemoryConfig,
        memory_config: ttnn.MemoryConfig,
        compute_kernel_config,
    ) -> ttnn.Tensor:
        if ttnn.is_sharded(x) and x.memory_config() == in0_mem:
            x_bs = x
            owns_x_bs = False
        elif ttnn.is_sharded(x):
            x_bs = ttnn.to_memory_config(x, in0_mem)
            owns_x_bs = True
        else:
            x2d = x if len(x.shape) == 2 else ttnn.reshape(x, (m, k))
            x_bs = ttnn.to_memory_config(x2d, in0_mem)
            if x2d is not x:
                ttnn.deallocate(x2d)
            owns_x_bs = True
        out_bs = ttnn.linear(
            x_bs,
            weight,
            bias=bias,
            program_config=program_config,
            memory_config=out_mem,
            compute_kernel_config=compute_kernel_config,
        )
        if owns_x_bs:
            ttnn.deallocate(x_bs)
        out = ttnn.sharded_to_interleaved(out_bs, memory_config, output_dtype=ttnn.bfloat16)
        ttnn.deallocate(out_bs)
        if len(x.shape) >= 3:
            out = ttnn.reshape(out, (int(x.shape[0]), int(x.shape[1]), n))
        return out

    def _linear_tp_chunked(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        m: int,
        k: int,
        n: int,
        chunk_m: int,
        batch: int,
        seq: int,
        memory_config: ttnn.MemoryConfig,
        fused_activation,
        program_config,
        in0_mem: ttnn.MemoryConfig,
        out_mem: ttnn.MemoryConfig,
    ) -> ttnn.Tensor:
        """Long-seq TP block-sharded matmul via ``ceil(m / chunk_m)`` tuned chunk kernels."""
        num_chunks = (m + chunk_m - 1) // chunk_m
        compute_cfg = self._chunked_tp_linear_compute_cfg() if num_chunks > 1 else self._linear_ln_compute_cfg

        use_sharded_input = ttnn.is_sharded(x)
        x_inter = x
        owns_x_inter = False
        if not use_sharded_input:
            if len(x_inter.shape) == 3:
                reshaped = ttnn.reshape(x_inter, (m, k))
                if reshaped is not x_inter:
                    x_inter = reshaped
                    owns_x_inter = True
            elif len(x_inter.shape) != 2:
                reshaped = ttnn.reshape(x_inter, (m, k))
                if reshaped is not x_inter:
                    x_inter = reshaped
                    owns_x_inter = True
            if x_inter.memory_config().buffer_type == ttnn.BufferType.L1:
                x_dram = ttnn.to_memory_config(x_inter, ttnn.DRAM_MEMORY_CONFIG)
                if owns_x_inter:
                    ttnn.deallocate(x_inter)
                elif x_dram is not x:
                    owns_x_inter = True
                x_inter = x_dram

        chunks: list[ttnn.Tensor] = []
        for i in range(num_chunks):
            start = i * chunk_m
            end = min(start + chunk_m, m)
            chunk_rows = end - start

            if use_sharded_input:
                row_slice = self._slice_tp_matmul_rows(x, start, end, k)
                owns_row_slice = row_slice is not x
                if chunk_rows < chunk_m:
                    row_inter = ttnn.sharded_to_interleaved(
                        row_slice, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16
                    )
                    if owns_row_slice:
                        ttnn.deallocate(row_slice)
                    row_inter = self._pad_token_rows(row_inter, chunk_rows, chunk_m)
                    x_bs = ttnn.to_memory_config(row_inter, in0_mem)
                    ttnn.deallocate(row_inter)
                elif row_slice.memory_config() == in0_mem:
                    x_bs = row_slice
                    owns_row_slice = False
                else:
                    x_bs = ttnn.to_memory_config(row_slice, in0_mem)
                    if owns_row_slice:
                        ttnn.deallocate(row_slice)
            else:
                chunk = ttnn.slice(x_inter, [start, 0], [end, k], [1, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
                if chunk_rows < chunk_m:
                    chunk = self._pad_token_rows(chunk, chunk_rows, chunk_m)
                x2d = chunk if len(chunk.shape) == 2 else ttnn.reshape(chunk, (chunk_m, k))
                x_bs = ttnn.to_memory_config(x2d, in0_mem)
                if x2d is not chunk:
                    ttnn.deallocate(x2d)
                if chunk is not x_inter:
                    ttnn.deallocate(chunk)

            out_bs = ttnn.linear(
                x_bs,
                weight,
                bias=bias,
                program_config=program_config,
                memory_config=out_mem,
                compute_kernel_config=compute_cfg,
            )
            if x_bs is not x:
                ttnn.deallocate(x_bs)
            out_inter = ttnn.sharded_to_interleaved(out_bs, memory_config, output_dtype=ttnn.bfloat16)
            ttnn.deallocate(out_bs)
            if chunk_rows < chunk_m:
                out_inter = ttnn.slice(
                    out_inter,
                    [0, 0],
                    [chunk_rows, int(out_inter.shape[-1])],
                    [1, 1],
                    memory_config=memory_config,
                )
            chunks.append(out_inter)

        if owns_x_inter:
            ttnn.deallocate(x_inter)

        if len(chunks) == 1:
            out_concat = chunks[0]
        else:
            out_concat = ttnn.concat(chunks, dim=0, memory_config=memory_config)
            for c in chunks:
                ttnn.deallocate(c)
        return ttnn.reshape(out_concat, (batch, seq, n))

    def _linear_tp(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: Optional[str] = None,
        memory_config: Optional[ttnn.MemoryConfig] = None,
    ) -> ttnn.Tensor:
        """Regular (non-DRAM-sharded) linear for TP>1 path.

        Used when weights are ``ShardTensorToMesh`` distributed across devices.
        Each device computes a local partial result; the caller applies
        ``encoder_all_reduce_sum_replicate`` after row-parallel layers.

        Uses ``encoder_tp_block_sharded_matmul`` when shapes are tuned; long-seq
        (``m > SEAMLESS_TP_BS_CHUNK_M``) runs the same kernel on row chunks. Otherwise
        falls back to tuned 1D interleaved ``ttnn.linear``.
        """
        if memory_config is None:
            memory_config = getattr(self, "_activation_mc", ttnn.L1_MEMORY_CONFIG)
        k = int(weight.shape[-2])
        n = int(weight.shape[-1])
        m = self._linear_token_rows(x)
        chunk_m = self._tp_bs_chunk_rows()
        fused_activation = ttnn.UnaryOpType.RELU if activation == "relu" else None

        tuned_chunk = encoder_tp_block_sharded_matmul(self.device, chunk_m, k, n, fused_activation=fused_activation)
        if m > chunk_m and tuned_chunk is not None:
            program_config, in0_mem, out_mem = tuned_chunk
            batch = int(x.shape[0]) if len(x.shape) >= 3 else 1
            seq = int(x.shape[1]) if len(x.shape) >= 3 else m
            return self._linear_tp_chunked(
                x,
                weight,
                bias,
                m=m,
                k=k,
                n=n,
                chunk_m=chunk_m,
                batch=batch,
                seq=seq,
                memory_config=memory_config,
                fused_activation=fused_activation,
                program_config=program_config,
                in0_mem=in0_mem,
                out_mem=out_mem,
            )

        tuned = encoder_tp_block_sharded_matmul(self.device, m, k, n, fused_activation=fused_activation)
        if tuned is not None:
            program_config, in0_mem, out_mem = tuned
            return self._linear_tp_block_sharded(
                x,
                weight,
                bias,
                m=m,
                k=k,
                n=n,
                fused_activation=fused_activation,
                program_config=program_config,
                in0_mem=in0_mem,
                out_mem=out_mem,
                memory_config=memory_config,
                compute_kernel_config=self._linear_ln_compute_cfg,
            )

        # Untuned-shape fallback: block-sharded LN input is not valid here — convert to interleaved.
        x_in = x
        owns_x_in = False
        if ttnn.is_sharded(x):
            x_in = ttnn.sharded_to_interleaved(x, memory_config, output_dtype=ttnn.bfloat16)
            owns_x_in = True
        if len(x_in.shape) == 3:
            x_in = ttnn.reshape(x_in, (m, k))
        kwargs = {}
        if activation == "relu":
            kwargs["activation"] = "relu"
        fallback_pc = encoder_tp_interleaved_matmul_program_config(self.device, m, k, n)
        if fallback_pc is not None:
            kwargs["program_config"] = fallback_pc
        out = ttnn.linear(
            x_in,
            weight,
            bias=bias,
            memory_config=memory_config,
            compute_kernel_config=self._linear_ln_compute_cfg,
            **kwargs,
        )
        if owns_x_in:
            ttnn.deallocate(x_in)
        if len(x.shape) >= 3:
            out = ttnn.reshape(out, (int(x.shape[0]), int(x.shape[1]), n))
        return out

    def _attention(
        self,
        hidden_states: ttnn.Tensor,
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
    ) -> ttnn.Tensor:
        tp = self._tp
        num_local_heads = num_heads // tp  # = num_heads when tp == 1
        local_hidden = hidden_size // tp  # per-device head output dim when tp > 1

        if tp > 1:
            # TP path: column-parallel QKV (output dim = 3 * hidden_size // tp per device).
            qkv_dim = 3 * local_hidden
            qkv = self._linear_tp(
                hidden_states,
                attn_module.qkv.weight,
                attn_module.qkv.bias,
            )
            qkv_4d = ttnn.reshape(qkv, (batch, 1, seq_q, qkv_dim))
        else:
            # Single-device DRAM-sharded path.
            qkv_dim = 3 * hidden_size
            qkv = self._linear(
                hidden_states,
                attn_module.qkv.weight,
                attn_module.qkv.bias,
                logical_out_dim=qkv_dim,
                accept_sharded_input=ttnn.is_sharded(hidden_states),
                batch=batch,
                seq=seq_q,
            )
            qkv_4d = ttnn.reshape(qkv, (batch, 1, seq_q, qkv_dim))

        # ``nlp_create_qkv_heads`` consumes a 4-D ``[B, 1, S, 3*local_H]`` input and
        # returns Q/K/V already shaped as ``[B, num_local_heads, S, head_dim]``.
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_4d,
            num_heads=num_local_heads,
            num_kv_heads=num_local_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # ``reshape`` can be a view; keep base ``qkv`` alive until heads are materialized.
        if tp > 1:
            ttnn.deallocate(qkv)
        ttnn.deallocate(qkv_4d)

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            scale=1.0 / math.sqrt(head_dim),
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)

        if tp > 1:
            # TP: merged is [B, 1, S, H//tp]; reshape and row-parallel out_proj.
            merged = ttnn.reshape(merged_4d, (batch, seq_q, local_hidden))
            proj = self._linear_tp(
                merged,
                attn_module.out_proj.weight,
                attn_module.out_proj.bias,
            )
            ttnn.deallocate(merged_4d)
            # all_reduce: sum partial [B, S, H] across TP devices → replicated [B, S, H].
            proj = encoder_all_reduce_sum_replicate(
                proj,
                self.device,
                cluster_axis=self._cluster_axis,
                memory_config=getattr(self, "_activation_mc", ttnn.L1_MEMORY_CONFIG),
            )
            return proj
        else:
            # ``ttnn.reshape`` returns a view onto ``merged_4d`` storage. For short-seq prefill the
            # next matmul (DRAM-sharded fast path) consumes ``merged`` synchronously, so deallocating
            # ``merged_4d`` early is safe. The long-seq chunked matmul makes multiple intermediate
            # allocations before consuming ``merged``, which can overwrite the freed view; keep
            # ``merged_4d`` alive until after the projection.
            merged = ttnn.reshape(merged_4d, (batch, seq_q, hidden_size))
            proj = self._linear(
                merged,
                attn_module.out_proj.weight,
                attn_module.out_proj.bias,
                logical_out_dim=hidden_size,
                keep_sharded_output=True,
                batch=batch,
                seq=seq_q,
            )
            ttnn.deallocate(merged_4d)
            return proj

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor],
        position_ids: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        *,
        inputs_embeds: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """
        Args:
            input_ids: ``uint32`` ``[batch, seq]`` on device (mutually exclusive with ``inputs_embeds``).
            position_ids: ``uint32`` ``[batch, seq]`` on device (sinusoidal table indices).
            attention_mask: optional additive mask ``[batch, 1, seq, seq]`` (bfloat16).
            inputs_embeds: optional ``bfloat16`` ``[batch, seq, hidden_size]`` on device; matches HF
                ``SeamlessM4Tv2Encoder`` when ``inputs_embeds`` is passed instead of ``input_ids``.

        Returns:
            Last hidden states ``bfloat16`` ``[batch, seq, hidden_size]`` on device.
        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify only one of input_ids or inputs_embeds.")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("One of input_ids or inputs_embeds is required.")

        parameters = self.parameters
        num_heads = self.num_attention_heads
        hidden_size = self.hidden_size
        head_dim = hidden_size // num_heads
        num_layers = self.num_hidden_layers
        tp = self._tp

        if inputs_embeds is not None:
            batch = int(inputs_embeds.shape[0])
            seq = int(inputs_embeds.shape[1])
            pos = ttnn.embedding(
                position_ids,
                weight=parameters.embed_positions.weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            hidden = ttnn.add(inputs_embeds, pos, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(pos)
        else:
            batch = int(input_ids.shape[0])  # type: ignore[union-attr]
            seq = int(input_ids.shape[1])

            tok = ttnn.embedding(
                input_ids,
                weight=parameters.embed_tokens.weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            pos = ttnn.embedding(
                position_ids,
                weight=parameters.embed_positions.weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

            hidden = ttnn.add(tok, pos, memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(tok)
            ttnn.deallocate(pos)

        sdpa_self = self._sdpa_program_config(seq, seq)

        m_tiles = (batch * seq + 31) // 32
        n_tiles = hidden_size // 32
        ffn_dim = 8 * hidden_size
        token_rows = batch * seq
        if tp > 1:
            self._activation_mc = encoder_tp_activation_memory_config(token_rows)

        if tp > 1:
            sharded_hidden_mem = self._activation_mc
            self._long_seq_mc = None
            long_seq = False
        else:
            qkv_n = int(parameters.layers[0].self_attn.qkv.weight.shape[-1])
            # The L1 WIDTH-sharded activation layout used by the DRAM-sharded matmul kernel only
            # supports ``m == TILE`` (one input tile row). For long-seq prefill (``token_rows > TILE``)
            # the chunked path in ``_linear`` runs the matmul kernel ``ceil(m_actual / TILE)`` times and
            # produces interleaved output, so keep the per-layer hidden state interleaved.
            long_seq = token_rows > TILE
            long_seq_use_dram = long_seq and token_rows >= 256
            if long_seq:
                sharded_hidden_mem = ttnn.DRAM_MEMORY_CONFIG if long_seq_use_dram else ttnn.L1_MEMORY_CONFIG
            else:
                hidden = self._to_matmul_width_sharded(hidden, token_rows, hidden_size, qkv_n)
                sharded_hidden_mem = ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG
            self._long_seq_mc = sharded_hidden_mem if long_seq else None

        tp_fused_ln = self._tp_bs_ln and tp > 1 and m_tiles > 1

        for i in range(num_layers):
            layer = parameters.layers[i]

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.self_attn_layer_norm.weight,
                bias=layer.self_attn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
                input_sharded=ttnn.is_sharded(hidden),
                output_sharded=((not long_seq) and (tp == 1)) or tp_fused_ln,
                matmul_fusion_m=token_rows if tp_fused_ln else None,
                matmul_fusion_k=hidden_size if tp_fused_ln else None,
            )
            attn_out = self._attention(
                normed,
                layer.self_attn,
                attention_mask,
                batch=batch,
                seq_q=seq,
                seq_k=seq,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_cfg=sdpa_self,
            )
            ttnn.deallocate(normed)
            hidden = ttnn.add(hidden, attn_out, memory_config=sharded_hidden_mem)
            ttnn.deallocate(attn_out)

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.ffn_layer_norm.weight,
                bias=layer.ffn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
                input_sharded=ttnn.is_sharded(hidden),
                output_sharded=((not long_seq) and (tp == 1)) or tp_fused_ln,
                matmul_fusion_m=token_rows if tp_fused_ln else None,
                matmul_fusion_k=hidden_size if tp_fused_ln else None,
            )
            if tp > 1:
                # TP FFN: column-parallel fc1 (output = ffn_dim//tp), row-parallel fc2 (output = H).
                ff = self._linear_tp(normed, layer.ffn.fc1.weight, layer.ffn.fc1.bias, activation="relu")
                ttnn.deallocate(normed)
                ff = self._linear_tp(ff, layer.ffn.fc2.weight, layer.ffn.fc2.bias)
                ff = encoder_all_reduce_sum_replicate(
                    ff,
                    self.device,
                    cluster_axis=self._cluster_axis,
                    memory_config=self._activation_mc,
                )
            else:
                ff = self._linear(
                    normed,
                    layer.ffn.fc1.weight,
                    layer.ffn.fc1.bias,
                    activation="relu",
                    logical_out_dim=ffn_dim,
                    keep_sharded_output=not long_seq,
                    accept_sharded_input=not long_seq,
                    batch=batch,
                    seq=seq,
                )
                ttnn.deallocate(normed)
                ff = self._linear(
                    ff,
                    layer.ffn.fc2.weight,
                    layer.ffn.fc2.bias,
                    logical_out_dim=hidden_size,
                    accept_sharded_input=not long_seq,
                    keep_sharded_output=not long_seq,
                    batch=batch,
                    seq=seq,
                )
            hidden = ttnn.add(hidden, ff, memory_config=sharded_hidden_mem)
            ttnn.deallocate(ff)

        hidden = self._layer_norm_sharded(
            hidden,
            weight=parameters.layer_norm.weight,
            bias=parameters.layer_norm.bias,
            m_tiles=m_tiles,
            n_tiles=n_tiles,
            input_sharded=ttnn.is_sharded(hidden),
            output_sharded=False,
        )
        if ttnn.is_sharded(hidden):
            return self._width_sharded_to_3d(hidden, batch, seq, hidden_size)
        return ttnn.reshape(hidden, (batch, seq, hidden_size)) if len(hidden.shape) != 3 else hidden
