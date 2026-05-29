# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Encoder`] (prefill / inference)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import ttnn

from models.experimental.seamless_m4t_v2_large.tt.common import (
    TILE,
    all_reduce_sum_replicate,
    build_ln_sharded_config,
    dram_linear_input_mem_config,
    dram_matmul_program_config,
    ensure_interleaved_bsh,
    ensure_l1_width_sharded_activation,
    find_grid_for_k_n,
    gather_in0_matmul_program_config,
    sdpa_program_config,
    width_sharded_l1_memcfg,
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
        use_gather_in0: bool = False,
    ):
        self.device = device
        self.parameters = parameters
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self._tp = get_tp(device)
        self._cluster_axis = mesh_cluster_axis(device)
        # ``gather_in0`` opt: width-sharded activations + DRAM-width-sharded weights end-to-end on
        # tp > 1.  Falls back to the original interleaved-L1 TP path on tp == 1 (the existing
        # single-device DRAM-sharded path already covers that case).
        # Default OFF: the gather_in0 program-config has ``per_core_M = m_seq / TILE``, so output
        # activation per core scales linearly with input sequence length. At the PCC test's
        # max seq (4096), per-core output is ~768 KB and overflows Blackhole L1 (1.5 MB/core)
        # after kernel CBs are subtracted. The pattern is well-suited to small-M cases (decode,
        # text_decoder per-step inference) but not to the text_encoder's wide-prefill use case.
        # Caller can opt in via ``use_gather_in0=True`` for short-seq workloads (eg ≤ 256 tokens),
        # but the matching weight prep must be set the same way in ``create_text_encoder_parameters``.
        self._use_gather_in0 = use_gather_in0 and self._tp > 1
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
        self._sdpa_pc_cache: dict = {}
        self._dram_matmul_pc_cache: dict = {}
        self._width_shard_mem_cache: dict = {}

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

        # Normalize the input to a 2-D ``[m_actual, k]`` interleaved L1 tensor for slicing.
        if ttnn.is_sharded(x):
            x_inter = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        else:
            x_inter = x
        if len(x_inter.shape) == 3:
            x_inter = ttnn.reshape(x_inter, (m_actual, k))
        elif len(x_inter.shape) != 2:
            x_inter = ttnn.reshape(x_inter, (m_actual, k))

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
            if chunk is not x_inter:
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
    ) -> ttnn.Tensor:
        """Width-sharded multicore LN. Set ``output_sharded=True`` to feed matmul without S2I.

        For long-seq prefill (``m_tiles > 1``) the BLOCK_SHARDED LN path is fragile when the
        upstream tensor is WIDTH_SHARDED or 3-D interleaved (shape mismatches and layout
        conversions inside ttnn cause silent numerical drift). Fall back to the plain unsharded
        ``ttnn.layer_norm`` for that case — slower per call, but identical math to HF.
        """
        if m_tiles > 1:
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

    def _linear_tp(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        activation: Optional[str] = None,
        memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """Regular (non-DRAM-sharded) linear for TP>1 path.

        Used when weights are ``ShardTensorToMesh`` distributed across devices.
        Each device computes a local partial result; the caller applies
        ``all_reduce_sum_replicate`` after row-parallel layers.
        """
        kwargs = {}
        if activation == "relu":
            kwargs["activation"] = "relu"
        return ttnn.linear(
            x,
            weight,
            bias=bias,
            memory_config=memory_config,
            compute_kernel_config=self._linear_ln_compute_cfg,
            **kwargs,
        )

    def _gather_in0_grid(self, k_tiles: int, n_tiles: int) -> Tuple[int, int]:
        """Pick the compute grid for ``gather_in0`` matmul, cached per (k_tiles, n_tiles)."""
        key = ("grid", k_tiles, n_tiles)
        cached = self._dram_matmul_pc_cache.get(key)
        if cached is not None:
            return cached
        grid = find_grid_for_k_n(k_tiles, n_tiles)
        self._dram_matmul_pc_cache[key] = grid
        return grid

    def _linear_gather_in0(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        m: int,
        k: int,
        n: int,
        activation: Optional[str] = None,
    ) -> ttnn.Tensor:
        """``gather_in0`` matmul: L1-WIDTH-sharded ``x`` × DRAM-WIDTH-sharded ``weight`` → L1-WIDTH-sharded output.

        Args:
            x: input activation. May be interleaved-L1 (will be converted) or already
               WIDTH-sharded across K. Shape ``[..., k]`` where the flat token count is ``m``.
            weight: DRAM-width-sharded weight, ``[k, n]`` per device.
            bias: TILE bias, broadcasted on M.
            m, k, n: shape (k, n are per-device dims).
            activation: optional ``"relu"`` to fuse into the matmul.

        Caller is responsible for choosing weight layout that satisfies ``k % (TILE*num_cores) == 0``
        and ``n % (TILE*num_cores) == 0`` for the grid this helper picks via ``find_grid_for_k_n``.
        """
        if m % TILE != 0:
            raise ValueError(f"_linear_gather_in0: m={m} must be a multiple of TILE={TILE}")
        if k % TILE != 0 or n % TILE != 0:
            raise ValueError(f"_linear_gather_in0: k={k}, n={n} must be multiples of TILE")
        k_tiles = k // TILE
        n_tiles = n // TILE
        m_tiles = m // TILE
        grid_x, grid_y = self._gather_in0_grid(k_tiles, n_tiles)
        num_cores = grid_x * grid_y

        # Activation: L1 WIDTH_SHARDED [m_tiles*TILE, k_tiles*TILE] over the grid.
        in_mem = width_sharded_l1_memcfg(m_tiles, k_tiles, grid_x, grid_y)
        if not ttnn.is_sharded(x):
            if len(x.shape) > 2:
                x = ttnn.reshape(x, (m, k))
            x_sharded = ttnn.interleaved_to_sharded(x, in_mem)
        else:
            x_sharded = x

        # Output: L1 WIDTH_SHARDED [m, padded_n] over the same grid.
        out_mem = width_sharded_l1_memcfg(m_tiles, n_tiles, grid_x, grid_y)

        fused_activation = None
        if activation == "relu":
            fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)

        prog_cfg = gather_in0_matmul_program_config(
            grid_x=grid_x,
            grid_y=grid_y,
            m_seq=m,
            k_dim=k,
            n_dim=n,
            fuse_batch=True,
            fused_activation=fused_activation,
            fp32_dest_acc_en=True,
        )

        out = ttnn.linear(
            x_sharded,
            weight,
            bias=bias,
            dtype=ttnn.bfloat16,
            memory_config=out_mem,
            program_config=prog_cfg,
            compute_kernel_config=self._linear_ln_compute_cfg,
        )
        if x_sharded is not x:
            ttnn.deallocate(x_sharded)
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

        if tp > 1 and self._use_gather_in0:
            # gather_in0 path: width-sharded activations + DRAM-width-sharded col-parallel QKV.
            qkv_dim = 3 * local_hidden
            qkv_sharded = self._linear_gather_in0(
                hidden_states,
                attn_module.qkv.weight,
                attn_module.qkv.bias,
                m=batch * seq_q,
                k=hidden_size,
                n=qkv_dim,
            )
            # ``nlp_create_qkv_heads`` requires interleaved-L1 input; bridge here.
            qkv = width_sharded_to_l1_interleaved(qkv_sharded)
            ttnn.deallocate(qkv_sharded)
            qkv_4d = ttnn.reshape(qkv, (batch, 1, seq_q, qkv_dim))
        elif tp > 1:
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

        if tp > 1 and self._use_gather_in0:
            # gather_in0 path: row-parallel out_proj. Input is interleaved L1 ``[B, S, local_H]``;
            # the wrapper converts to width-sharded internally.
            merged = ttnn.reshape(merged_4d, (batch, seq_q, local_hidden))
            proj_sharded = self._linear_gather_in0(
                merged,
                attn_module.out_proj.weight,
                attn_module.out_proj.bias,
                m=batch * seq_q,
                k=local_hidden,
                n=hidden_size,
            )
            ttnn.deallocate(merged_4d)
            # Convert sharded → interleaved before all_reduce (the existing reducer expects
            # interleaved inputs; a sharded-native variant would save this round trip).
            proj = width_sharded_to_l1_interleaved(proj_sharded)
            ttnn.deallocate(proj_sharded)
            proj = all_reduce_sum_replicate(
                proj,
                self.device,
                cluster_axis=self._cluster_axis,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            return proj
        elif tp > 1:
            # TP: merged is [B, 1, S, H//tp]; reshape and row-parallel out_proj.
            merged = ttnn.reshape(merged_4d, (batch, seq_q, local_hidden))
            proj = self._linear_tp(
                merged,
                attn_module.out_proj.weight,
                attn_module.out_proj.bias,
            )
            ttnn.deallocate(merged_4d)
            # all_reduce: sum partial [B, S, H] across TP devices → replicated [B, S, H].
            proj = all_reduce_sum_replicate(
                proj,
                self.device,
                cluster_axis=self._cluster_axis,
                memory_config=ttnn.L1_MEMORY_CONFIG,
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
            # TP path: standard interleaved activations; DRAM-sharded not used.
            sharded_hidden_mem = ttnn.L1_MEMORY_CONFIG
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

        for i in range(num_layers):
            layer = parameters.layers[i]

            normed = self._layer_norm_sharded(
                hidden,
                weight=layer.self_attn_layer_norm.weight,
                bias=layer.self_attn_layer_norm.bias,
                m_tiles=m_tiles,
                n_tiles=n_tiles,
                input_sharded=ttnn.is_sharded(hidden),
                output_sharded=(not long_seq) and (tp == 1),
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
                output_sharded=(not long_seq) and (tp == 1),
            )
            if tp > 1 and self._use_gather_in0:
                # gather_in0 FFN: col-parallel fc1 (output local_ffn_dim, width-sharded), then
                # row-parallel fc2 (output H, partial sum, width-sharded). Conversions only at
                # the SDPA-free LN-input/LN-output boundaries and around all_reduce.
                local_ffn_dim = ffn_dim // tp
                ff_sharded = self._linear_gather_in0(
                    normed,
                    layer.ffn.fc1.weight,
                    layer.ffn.fc1.bias,
                    m=batch * seq,
                    k=hidden_size,
                    n=local_ffn_dim,
                    activation="relu",
                )
                ttnn.deallocate(normed)
                # fc2 input is width-sharded over K=local_ffn_dim; the wrapper accepts sharded
                # input directly and produces width-sharded output.
                ff_sharded2 = self._linear_gather_in0(
                    ff_sharded,
                    layer.ffn.fc2.weight,
                    layer.ffn.fc2.bias,
                    m=batch * seq,
                    k=local_ffn_dim,
                    n=hidden_size,
                )
                ttnn.deallocate(ff_sharded)
                ff = width_sharded_to_l1_interleaved(ff_sharded2)
                ttnn.deallocate(ff_sharded2)
                ff = all_reduce_sum_replicate(
                    ff,
                    self.device,
                    cluster_axis=self._cluster_axis,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            elif tp > 1:
                # TP FFN: column-parallel fc1 (output = ffn_dim//tp), row-parallel fc2 (output = H).
                ff = self._linear_tp(normed, layer.ffn.fc1.weight, layer.ffn.fc1.bias, activation="relu")
                ttnn.deallocate(normed)
                ff = self._linear_tp(ff, layer.ffn.fc2.weight, layer.ffn.fc2.bias)
                ff = all_reduce_sum_replicate(
                    ff,
                    self.device,
                    cluster_axis=self._cluster_axis,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
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
