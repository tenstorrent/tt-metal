# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN [`SeamlessM4Tv2Decoder`] with prefill and KV-cache decode.

Prefill and decode linears use interleaved L1 weights with
``MatmulMultiCoreReuseMultiCast1DProgramConfig`` via ``common.matmul_program_config`` (text
encoder / speech encoder pattern). Decode self-attn QKV uses a separate ``qkv_decode`` weight
tensor for KV-cache PCC.

KV-cache: batched prefill uses ``slice_write`` (self; cross on bf16), decode cross reuses DRAM
cache with Q-only ``nlp_create_qkv_heads`` when ``cross_attn_cache_valid``.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import ttnn

from models.common.utility_functions import is_blackhole, nearest_32
from models.experimental.seamless_m4t_v2_large.tt.common import (
    all_reduce_sum_replicate,
    build_ln_sharded_config,
    matmul_multicast_1d_program_config,
    matmul_program_config,
    sdpa_program_config,
    TILE,
    to_torch_replicated_first_shard,
)
from models.experimental.seamless_m4t_v2_large.tt.mesh_helpers import mesh_cluster_axis, get_tp


def _num_to_corerange(batch: int) -> ttnn.CoreRange:
    """Single rectangular core range for height-sharded decode tensors (batch ≤ 8 or multiple of 8)."""
    num_x = min(batch, 8)
    num_y = max(1, (batch + num_x - 1) // num_x)
    return ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_x - 1, num_y - 1))


# Drain on-device profiler markers every N decoder layers when device profiling is on.
_PROFILER_LAYER_DRAIN_INTERVAL = 8


def _drain_device_profiler(device: ttnn.Device, *, trace_no_profiler: bool) -> None:
    """Flush on-device profiler markers when profiling is enabled."""
    if trace_no_profiler:
        return
    if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
        ttnn.ReadDeviceProfiler(device)


def init_text_decoder_kv_cache(
    device: ttnn.Device,
    *,
    num_hidden_layers: int,
    num_attention_heads: int,
    hidden_size: int,
    max_batch_size: int,
    max_seq_len: int,
    encoder_seq_len: int,
    cache_dtype: ttnn.DataType = ttnn.bfloat16,
    tp: Optional[int] = None,
) -> tuple[list[list[ttnn.Tensor]], list[list[ttnn.Tensor]]]:
    """
    Allocate per-layer self-attention and cross-attention KV caches.

    When ``tp > 1``, each device holds caches for ``num_attention_heads // tp`` local heads.
    Caches are replicated (zero-initialized) and diverge during decode as each device
    writes its local head K/V.

    Returns:
        ``(kv_cache, cross_attn_cache)`` where each is a list of length ``num_hidden_layers``
        containing ``[K, V]`` device tensors.
    """
    if tp is None:
        tp = get_tp(device)
    num_local_heads = num_attention_heads // tp
    head_dim = hidden_size // num_attention_heads
    chunk_size = 256
    padded_max_seq_len = ((max_seq_len + chunk_size - 1) // chunk_size) * chunk_size

    # For multi-device, replicate the zero tensors to all devices.
    mm: Optional[object] = None
    try:
        if hasattr(device, "get_num_devices") and int(device.get_num_devices()) > 1:
            mm = ttnn.ReplicateTensorToMesh(device)
    except Exception:
        pass

    kv_cache: list[list[ttnn.Tensor]] = []
    cross_attn_cache: list[list[ttnn.Tensor]] = []

    for _ in range(num_hidden_layers):
        k_cache = torch.zeros((max_batch_size, num_local_heads, padded_max_seq_len, head_dim))
        v_cache = torch.zeros((max_batch_size, num_local_heads, padded_max_seq_len, head_dim))
        kv_cache.append(
            [
                ttnn.from_torch(
                    k_cache,
                    dtype=cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mm,
                ),
                ttnn.from_torch(
                    v_cache,
                    dtype=cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mm,
                ),
            ]
        )

        cross_k = torch.zeros((max_batch_size, num_local_heads, encoder_seq_len, head_dim))
        cross_v = torch.zeros((max_batch_size, num_local_heads, encoder_seq_len, head_dim))
        cross_attn_cache.append(
            [
                ttnn.from_torch(
                    cross_k,
                    dtype=cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mm,
                ),
                ttnn.from_torch(
                    cross_v,
                    dtype=cache_dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=mm,
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


def write_self_kv_prefill_to_cache(
    key_states: ttnn.Tensor,
    value_states: ttnn.Tensor,
    kv_cache: list[ttnn.Tensor],
    *,
    seq_len: int,
) -> None:
    """Bulk-write prefilled self-attention K/V ``[B, H, L, D]`` into KV cache (Whisper ``slice_write`` path).

    Writes only ``[0:seq_len)`` along the sequence axis so tile-padded prefill forwards do not
    pollute the cache past the real token count.
    """
    k_cache, v_cache = kv_cache
    bsz = int(key_states.shape[0])
    nh = int(key_states.shape[1])
    head_dim = int(key_states.shape[3])
    padded_seq = int(key_states.shape[2])
    begins = [0, 0, 0, 0]
    ends = [bsz, nh, seq_len, head_dim]
    strides = [1, 1, 1, 1]
    if padded_seq != seq_len:
        k_src = ttnn.slice(
            key_states, [0, 0, 0, 0], [bsz, nh, seq_len, head_dim], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v_src = ttnn.slice(
            value_states,
            [0, 0, 0, 0],
            [bsz, nh, seq_len, head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        k_src = key_states
        v_src = value_states
    cache_dtype = k_cache.dtype
    if k_src.dtype != cache_dtype:
        k_typed = ttnn.typecast(k_src, dtype=cache_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_typed = ttnn.typecast(v_src, dtype=cache_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        k_typed = k_src
        v_typed = v_src
    k_dram = ttnn.to_memory_config(k_typed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_dram = ttnn.to_memory_config(v_typed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.experimental.slice_write(k_dram, k_cache, begins, ends, strides)
    ttnn.experimental.slice_write(v_dram, v_cache, begins, ends, strides)
    if k_dram is not k_typed:
        ttnn.deallocate(k_dram)
    if v_dram is not v_typed:
        ttnn.deallocate(v_dram)
    if k_typed is not k_src:
        ttnn.deallocate(k_typed)
        ttnn.deallocate(v_typed)
    if padded_seq != seq_len:
        ttnn.deallocate(k_src)
        ttnn.deallocate(v_src)


def write_cross_kv_prefill_to_cache(
    key_states: ttnn.Tensor,
    value_states: ttnn.Tensor,
    cross_cache: list[ttnn.Tensor],
    *,
    seq_len: Optional[int] = None,
) -> None:
    """Bulk-write cross K/V into cross cache via ``slice_write`` (bf16 warm path; bf8 keeps ``copy``)."""
    k_cache, v_cache = cross_cache
    bsz = int(key_states.shape[0])
    nh = int(key_states.shape[1])
    head_dim = int(key_states.shape[3])
    padded_seq = int(key_states.shape[2])
    fill_len = seq_len if seq_len is not None else padded_seq
    begins = [0, 0, 0, 0]
    ends = [bsz, nh, fill_len, head_dim]
    strides = [1, 1, 1, 1]
    if padded_seq != fill_len:
        k_src = ttnn.slice(
            key_states, [0, 0, 0, 0], [bsz, nh, fill_len, head_dim], [1, 1, 1, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        v_src = ttnn.slice(
            value_states,
            [0, 0, 0, 0],
            [bsz, nh, fill_len, head_dim],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    else:
        k_src = key_states
        v_src = value_states
    cache_dtype = k_cache.dtype
    if k_src.dtype != cache_dtype:
        k_typed = ttnn.typecast(k_src, dtype=cache_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_typed = ttnn.typecast(v_src, dtype=cache_dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        k_typed = k_src
        v_typed = v_src
    k_dram = ttnn.to_memory_config(k_typed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_dram = ttnn.to_memory_config(v_typed, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.experimental.slice_write(k_dram, k_cache, begins, ends, strides)
    ttnn.experimental.slice_write(v_dram, v_cache, begins, ends, strides)
    if k_dram is not k_typed:
        ttnn.deallocate(k_dram)
    if v_dram is not v_typed:
        ttnn.deallocate(v_dram)
    if k_typed is not k_src:
        ttnn.deallocate(k_typed)
        ttnn.deallocate(v_typed)
    if padded_seq != fill_len:
        ttnn.deallocate(k_src)
        ttnn.deallocate(v_src)


def warm_text_decoder_kv_cache_prefill(
    decoder: "TTSeamlessM4Tv2Decoder",
    input_ids_tt: ttnn.Tensor,
    position_ids_tt: ttnn.Tensor,
    encoder_tt: ttnn.Tensor,
    causal_4d: ttnn.Tensor,
    cross_4d: ttnn.Tensor,
    kv_cache: list[list[ttnn.Tensor]],
    cross_attn_cache: list[list[ttnn.Tensor]],
    *,
    kv_cache_fill_len: Optional[int] = None,
    trace_no_profiler: bool = False,
) -> ttnn.Tensor:
    """One batched prefill forward to fill self/cross KV caches (``prefill_kv_cache_fill=True``).

    Returns decoder hidden states ``[B, S, H]`` so the caller can run ``lm_head`` on the last seed
    position without re-feeding the final seed token through decode (which would overwrite cache).
    """
    return decoder.forward(
        input_ids_tt,
        position_ids_tt,
        encoder_tt,
        causal_4d,
        cross_4d,
        kv_cache=kv_cache,
        cross_attn_cache=cross_attn_cache,
        prefill_kv_cache_fill=True,
        kv_cache_fill_len=kv_cache_fill_len,
        trace_no_profiler=trace_no_profiler,
    )


def _next_power_of_2_cap256(n: int) -> int:
    """Smallest power of 2 >= ``n``, capped at 256 (SDPA decode chunk / bucket size)."""
    if n <= 0:
        return 1
    if n > 256:
        return 256
    power = 1
    while power < n:
        power *= 2
    return power


def _effective_decode_sdpa_seq_len(active_seq_len: int, padded_max_seq_len: int) -> int:
    """Cap SDPA decode chunking to the live cache length (fewer empty tiles early in decode)."""
    live = max(32, _next_power_of_2_cap256(active_seq_len))
    return min(padded_max_seq_len, live)


def _get_decode_sdpa_configs(
    device: ttnn.Device,
    *,
    num_local_heads: int,
    head_dim: int,
    max_batch_size: int,
    max_seq_len: int,
    active_seq_len: int,
) -> tuple[
    ttnn.MemoryConfig,
    ttnn.SDPAProgramConfig,
    ttnn.DeviceComputeKernelConfig,
    ttnn.MemoryConfig,
    ttnn.MemoryConfig,
]:
    """Decode SDPA + head-op memory configs (``nlp_create_qkv_heads_decode`` / ``sdpa_decode`` / ``concat_heads_decode``)."""
    padded_num_heads = nearest_32(num_local_heads)

    grid_size = device.compute_with_storage_grid_size()
    batch_grid = ttnn.num_cores_to_corerangeset(max_batch_size, grid_size, row_wise=True)
    sdpa_batch_sharded_memcfg = ttnn.create_sharded_memory_config(
        shape=(padded_num_heads, head_dim),
        core_grid=batch_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    if is_blackhole():
        create_heads_memcfg = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, head_dim),
            core_grid=ttnn.CoreGrid(y=4, x=8),
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
    else:
        create_heads_memcfg = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG

    sdpa_output_memcfg = ttnn.create_sharded_memory_config(
        shape=(padded_num_heads, head_dim),
        core_grid=ttnn.CoreRangeSet({_num_to_corerange(max_batch_size)}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    chunk_size = 256
    padded_max_seq_len = ((max_seq_len + chunk_size - 1) // chunk_size) * chunk_size
    effective_seq = _effective_decode_sdpa_seq_len(active_seq_len, padded_max_seq_len)

    k_chunk_size = _next_power_of_2_cap256(effective_seq)
    q_chunk_size = _next_power_of_2_cap256(effective_seq)
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
    return (
        sdpa_batch_sharded_memcfg,
        sdpa_decode_progcfg,
        sdpa_decode_compute_cfg,
        create_heads_memcfg,
        sdpa_output_memcfg,
    )


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
        self._tp = get_tp(device)
        self._cluster_axis = mesh_cluster_axis(device)
        self._num_local_heads = num_attention_heads // self._tp
        self._sdpa_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        # Decode self-attn reads the full cached sequence; HiFi4 matches prefill linears for PCC.
        self._sdpa_decode_slice_compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
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
        # Experimental: emit matmul outputs sharded (1D->WIDTH, 2D->BLOCK) per the perf-sweep
        # winners, then sharded_to_interleaved back so downstream is unchanged. Applies to ALL
        # matmuls through _linear (prefill + decode). The sharded matmul kernel time lands in the
        # ops-perf CSV for comparison vs interleaved. NB: the sweep tuned the decode shapes (M=32);
        # prefill shapes inherit the same 1D/2D rule but were not separately swept.
        # See tests/perf/test_text_decoder_matmul_perf_report_sweep.py.
        self._sharded_decode_out = os.environ.get("SEAMLESS_DECODE_SHARDED_OUT", "1") != "0"
        self._ln_sharded_cache: dict = {}
        self._matmul_pc_cache: dict = {}
        self._tile_padded_batch_rows = TILE * ((max_batch_size + TILE - 1) // TILE)
        # Same rationale for SDPA chunk schedules: ``forward`` / greedy decode revisit the same
        # ``(seq_q, seq_k, large_chunks)`` keys many times (24 layers × steps); reuse one object.
        self._sdpa_pc_cache: dict = {}
        self._decode_sdpa_cache: dict = {}
        self._decode_pos_cache: dict[tuple[int, int], ttnn.Tensor] = {}

    def decode_trace_cache_seq_len(self, active_seq_len: int) -> int:
        """Return decode SDPA cache bucket (32/64/128/256, capped by ``max_seq_len``)."""
        chunk_size = 256
        padded_max = ((self.max_seq_len + chunk_size - 1) // chunk_size) * chunk_size
        return _effective_decode_sdpa_seq_len(active_seq_len, padded_max)

    def _decode_sdpa_configs(
        self, active_seq_len: int
    ) -> tuple[ttnn.MemoryConfig, ttnn.SDPAProgramConfig, ttnn.DeviceComputeKernelConfig]:
        effective_seq = self.decode_trace_cache_seq_len(active_seq_len)
        key = (self.max_batch_size, self.max_seq_len, effective_seq)
        cached = self._decode_sdpa_cache.get(key)
        if cached is None:
            cached = _get_decode_sdpa_configs(
                self.device,
                num_local_heads=self._num_local_heads,
                head_dim=self.hidden_size // self.num_attention_heads,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len,
                active_seq_len=effective_seq,
            )
            self._decode_sdpa_cache[key] = cached
        return cached

    def borrow_current_decode_pos_tensor(self, position: int, batch_size: int = 1) -> ttnn.Tensor:
        """Reuse a device ``[batch]`` index tensor for a fixed decode position (avoids per-forward alloc)."""
        key = (batch_size, position)
        cached = self._decode_pos_cache.get(key)
        if cached is None:
            cached = make_current_decode_pos_tensor(self.device, position, batch_size=batch_size)
            self._decode_pos_cache[key] = cached
        return cached

    def _sdpa_program_config(self, seq_q: int, seq_k: int, *, large_chunks: bool = True) -> ttnn.SDPAProgramConfig:
        """See ``common.sdpa_program_config`` — ``large_chunks=False`` for short speech encoder keys."""
        return sdpa_program_config(self.device, seq_q, seq_k, self._sdpa_pc_cache, large_chunks=large_chunks)

    def _decode_matmul_pc(self, in_dim: int, out_dim: int) -> ttnn.ProgramConfig:
        """Decode matmul PC (effective ``M=32`` tile rows).

        Delegates to ``common.matmul_multicast_1d_program_config`` — the same 1D multicast builder
        prefill uses for M<=128. Measured (perf sweep 2026_06_02, Blackhole) 1.0-1.48x faster than
        the prior hand-rolled ``(cg.x, 1)`` grid across every decoder shape (QKV 8.0→6.1µs, out_proj
        3.6→2.5µs, FFN fc1 10.9→8.9µs, fc2 14.0→11.0µs): the old grid wasted cores because cg.x=11
        does not divide the N/K tile counts, and its 2048-N 2D branch collapsed back to 11x1.
        ``_pick_matmul_1d_grid`` instead picks a divisor grid (8x3/8x8) with per_core_N≈1. Matmul
        math is grid-invariant so cached K/V PCC is preserved; DRAM-width-sharding was measured
        slower here (8-bank Blackhole pins it to 8 cores). See tests/perf/test_decode_dram_sharded_sweep.py.
        """
        key = ("decode", in_dim, out_dim)
        cached = self._matmul_pc_cache.get(key)
        if cached is not None:
            return cached
        result = matmul_multicast_1d_program_config(self.device, m=self._tile_padded_batch_rows, k=in_dim, n=out_dim)
        self._matmul_pc_cache[key] = result
        return result

    @staticmethod
    def _linear_token_rows(x: ttnn.Tensor) -> int:
        if len(x.shape) == 3:
            return int(x.shape[0]) * int(x.shape[1])
        if len(x.shape) == 2:
            return int(x.shape[0])
        return int(x.shape[-2])

    def _matmul_token_rows(self, x: ttnn.Tensor, *, is_decode: bool) -> int:
        rows = self._linear_token_rows(x)
        if is_decode and rows <= self.max_batch_size:
            return self._tile_padded_batch_rows
        return rows

    def _matmul_pc(self, token_rows: int, in_dim: int, out_dim: int) -> ttnn.ProgramConfig:
        key = (token_rows, in_dim, out_dim)
        cached = self._matmul_pc_cache.get(key)
        if cached is not None:
            return cached
        cached = matmul_program_config(
            self.device,
            token_rows=token_rows,
            in_dim=in_dim,
            out_dim=out_dim,
        )
        self._matmul_pc_cache[key] = cached
        return cached

    def _sharded_decode_out_memcfg(self, program_config) -> Optional[ttnn.MemoryConfig]:
        """L1 sharded output memcfg matching ``program_config``'s grid/per-core tiling.

        1D multicast (projections) -> WIDTH_SHARDED; 2D multicast (KV-enc fill) -> BLOCK_SHARDED.
        Returns ``None`` for unsupported program-config types (caller keeps interleaved output).
        """
        if isinstance(program_config, ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig):
            layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        elif isinstance(program_config, ttnn.MatmulMultiCoreReuseMultiCastProgramConfig):
            layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            return None
        grid = program_config.compute_with_storage_grid_size
        gx, gy = int(grid.x), int(grid.y)
        shard = [int(program_config.per_core_M) * TILE, int(program_config.per_core_N) * TILE]
        core_range = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
        return ttnn.MemoryConfig(
            layout, ttnn.BufferType.L1, ttnn.ShardSpec(core_range, shard, ttnn.ShardOrientation.ROW_MAJOR)
        )

    def _linear(
        self,
        x: ttnn.Tensor,
        weight: ttnn.Tensor,
        bias: ttnn.Tensor,
        *,
        compute_cfg: Optional[ttnn.DeviceComputeKernelConfig] = None,
        memory_config: ttnn.MemoryConfig = ttnn.L1_MEMORY_CONFIG,
        program_config: Optional[ttnn.ProgramConfig] = None,
        activation: Optional[str] = None,
        is_decode: bool = False,
    ) -> ttnn.Tensor:
        ck = compute_cfg if compute_cfg is not None else self._linear_ln_compute_cfg
        if program_config is None:
            program_config = self._matmul_pc(
                self._matmul_token_rows(x, is_decode=is_decode),
                int(weight.shape[-2]),
                int(weight.shape[-1]),
            )
        sharded_out_mem = None
        if self._sharded_decode_out:
            sharded_out_mem = self._sharded_decode_out_memcfg(program_config)
        out = ttnn.linear(
            x,
            weight,
            bias=bias,
            program_config=program_config,
            memory_config=sharded_out_mem if sharded_out_mem is not None else memory_config,
            compute_kernel_config=ck,
            activation=activation,
        )
        if sharded_out_mem is not None:
            # Convert back so downstream ops (reshape / create_heads / residual add) see the
            # interleaved layout they expect; the sharded matmul itself is what the CSV captures.
            out = ttnn.sharded_to_interleaved(out, memory_config, output_dtype=ttnn.bfloat16)
        # TP 1D multicast matmul may return ``[B, 1, S, N]`` or ``[1, 1, M, N]``; downstream
        # attention slices expect ``[B, S, N]``.
        if len(x.shape) == 3:
            batch = int(x.shape[0])
            seq = int(x.shape[1])
            if len(out.shape) == 4 and int(out.shape[1]) == 1:
                out = ttnn.reshape(out, (batch, seq, int(out.shape[-1])))
            elif len(out.shape) == 2 and int(out.shape[0]) == batch * seq:
                out = ttnn.reshape(out, (batch, seq, int(out.shape[-1])))
        return out

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

    @staticmethod
    def _heads_fused(x: ttnn.Tensor, num_heads: int) -> ttnn.Tensor:
        """Heads-major ``[B, num_heads, S, head_dim]`` via fused ``nlp_create_qkv_heads``.

        Replaces ``_heads`` (``reshape`` + ``permute``) for the cross-attn K/V at ``enc_seq=512``,
        where the reshape materializes a tile-padded ``[B, S, num_heads, head_dim]`` (heads padded
        4->32) — ~27µs each. ``num_kv_heads=0`` runs the kernel on a single tensor and the Q-slot
        output is already non-transposed ``[B, num_heads, S, head_dim]`` (same idiom as the Whisper
        cross-attention KV path), so SDPA sees the identical layout with no separate permute.
        ``x``: ``[B, S, num_heads*head_dim]`` (3D) or ``[B, 1, S, num_heads*head_dim]`` (4D).
        """
        if len(x.shape) == 3:
            x = ttnn.unsqueeze(x, 1)
        return ttnn.experimental.nlp_create_qkv_heads(
            x, num_heads=num_heads, num_kv_heads=0, memory_config=ttnn.L1_MEMORY_CONFIG
        )[0]

    @staticmethod
    def _cross_q_heads_decode(q: ttnn.Tensor, *, num_heads: int) -> ttnn.Tensor:
        """Decode cross Q via ``nlp_create_qkv_heads`` (``num_kv_heads=0``)."""
        if len(q.shape) == 3:
            q = ttnn.unsqueeze(q, 1)
        qh, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q,
            num_heads=num_heads,
            num_kv_heads=0,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        return qh

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
        sdpa_decode_bundle: tuple[
            ttnn.MemoryConfig,
            ttnn.SDPAProgramConfig,
            ttnn.DeviceComputeKernelConfig,
            ttnn.MemoryConfig,
            ttnn.MemoryConfig,
        ],
        pc_qkv,
        pc_out,
        attn_scale: float,
    ) -> ttnn.Tensor:
        """Single-token self-attention with KV cache (decode head ops + ``sdpa_decode``)."""
        tp = self._tp
        num_local_heads = num_heads // tp
        local_hidden = hidden_size // tp  # head output dim per device when tp > 1
        seq_q = 1
        padded_batch = nearest_32(self.max_batch_size)
        qkv_local_dim = 3 * local_hidden  # 3*H for tp=1, 3*H//tp for tp>1

        qkv_w = attn_module.qkv_decode
        qkv = self._linear(
            hidden_states,
            qkv_w.weight,
            qkv_w.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=pc_qkv,
            is_decode=True,
        )
        qkv_4d = ttnn.reshape(
            qkv,
            (1, 1, batch, qkv_local_dim),
            (1, 1, padded_batch, qkv_local_dim),
        )

        _, sdpa_decode_progcfg, sdpa_decode_compute_cfg, create_heads_memcfg, sdpa_output_memcfg = sdpa_decode_bundle
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv_4d,
            num_heads=num_local_heads,
            num_kv_heads=num_local_heads,
            memory_config=create_heads_memcfg,
        )
        ttnn.deallocate(qkv)
        ttnn.deallocate(qkv_4d)

        k_cache, v_cache = kv_cache
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=current_decode_pos, page_table=None)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=current_decode_pos, page_table=None)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
            q,
            k_cache,
            v_cache,
            cur_pos_tensor=current_decode_pos,
            scale=attn_scale,
            program_config=sdpa_decode_progcfg,
            compute_kernel_config=sdpa_decode_compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)

        attn_out = ttnn.to_memory_config(attn_out, memory_config=sdpa_output_memcfg)
        merged_4d = ttnn.experimental.nlp_concat_heads_decode(attn_out, num_heads=num_local_heads)
        ttnn.deallocate(attn_out)
        if padded_batch != batch:
            merged_4d = ttnn.slice(merged_4d, [0, 0, 0, 0], [1, 1, batch, local_hidden], [1, 1, 1, 1])
        merged = ttnn.sharded_to_interleaved(merged_4d, ttnn.L1_MEMORY_CONFIG, output_dtype=ttnn.bfloat16)
        ttnn.deallocate(merged_4d)
        merged = ttnn.reshape(merged, (batch, seq_q, local_hidden))
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=pc_out,
            is_decode=True,
        )
        ttnn.deallocate(merged)
        if tp > 1:
            proj = all_reduce_sum_replicate(
                proj,
                self.device,
                cluster_axis=self._cluster_axis,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
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
        pc_q,
        pc_out,
        pc_kv_enc: Optional[object],
        attn_scale: float,
    ) -> ttnn.Tensor:
        tp = self._tp
        num_local_heads = num_heads // tp
        local_hidden = hidden_size // tp  # Q output dim per device when tp > 1
        seq_q = 1

        if cross_attn_cache is not None and cross_attn_cache_valid:
            q = self._linear(
                hidden_states,
                attn_module.q_proj.weight,
                attn_module.q_proj.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_q,
                is_decode=True,
            )
            qh = self._cross_q_heads_decode(q, num_heads=num_local_heads)
            kh, vh = cross_attn_cache[0], cross_attn_cache[1]
        else:
            q = self._linear(
                hidden_states,
                attn_module.q_proj.weight,
                attn_module.q_proj.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_q,
                is_decode=True,
            )
            assert pc_kv_enc is not None
            kv_packed = self._linear(
                encoder_hidden_states,
                attn_module.kv.weight,
                attn_module.kv.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_kv_enc,
                is_decode=True,
            )
            # TP: kv output dim is 2*local_hidden; split at local_hidden
            kv_half = int(kv_packed.shape[-1]) // 2
            k = ttnn.slice(kv_packed, [0, 0, 0], [batch, enc_seq, kv_half], [1, 1, 1])
            v = ttnn.slice(kv_packed, [0, 0, kv_half], [batch, enc_seq, 2 * kv_half], [1, 1, 1])
            ttnn.deallocate(kv_packed)
            qh = self._heads(q, batch, seq_q, num_local_heads, head_dim)
            kh = self._heads(k, batch, enc_seq, num_local_heads, head_dim)
            vh = self._heads(v, batch, enc_seq, num_local_heads, head_dim)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            if cross_attn_cache is not None:
                ttnn.copy(kh, cross_attn_cache[0])
                ttnn.copy(vh, cross_attn_cache[1])
                kh, vh = cross_attn_cache[0], cross_attn_cache[1]

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=cross_attention_mask,
            is_causal=False,
            scale=attn_scale,
            program_config=sdpa_cfg,
            compute_kernel_config=self._sdpa_decode_slice_compute_cfg,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qh)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_out)
        merged = ttnn.reshape(merged_4d, (batch, seq_q, local_hidden))
        ttnn.deallocate(merged_4d)
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=pc_out,
            is_decode=True,
        )
        ttnn.deallocate(merged)
        if tp > 1:
            proj = all_reduce_sum_replicate(
                proj,
                self.device,
                cluster_axis=self._cluster_axis,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
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
        cache_seq_len: Optional[int] = None,
        prefill_kv_cache_fill: bool = False,
        kv_cache_fill_len: Optional[int] = None,
        sdpa_decode_bundle: Optional[
            tuple[
                ttnn.MemoryConfig,
                ttnn.SDPAProgramConfig,
                ttnn.DeviceComputeKernelConfig,
                ttnn.MemoryConfig,
                ttnn.MemoryConfig,
            ]
        ] = None,
        decode_attn_pcs: Optional[dict] = None,
        attn_scale: Optional[float] = None,
    ) -> ttnn.Tensor:
        is_decode = kv_cache is not None and current_decode_pos is not None
        is_cross_attn = "kv" in attn_module and "qkv" not in attn_module
        if is_decode and encoder_hidden_states is None and not is_cross_attn:
            assert cache_seq_len is not None
            assert sdpa_decode_bundle is not None and decode_attn_pcs is not None and attn_scale is not None
            return self._self_attention_decode(
                hidden_states,
                attn_module,
                kv_cache,
                current_decode_pos,
                batch=batch,
                num_heads=num_heads,
                head_dim=head_dim,
                hidden_size=hidden_size,
                sdpa_decode_bundle=sdpa_decode_bundle,
                pc_qkv=decode_attn_pcs["qkv"],
                pc_out=decode_attn_pcs["out"],
                attn_scale=attn_scale,
            )
        if is_decode and (encoder_hidden_states is not None or (is_cross_attn and cross_attn_cache_valid)):
            assert decode_attn_pcs is not None and attn_scale is not None
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
                pc_q=decode_attn_pcs["q_cross"],
                pc_out=decode_attn_pcs["out_cross"],
                pc_kv_enc=decode_attn_pcs.get("kv_enc"),
                attn_scale=attn_scale,
            )

        tp = self._tp
        num_local_heads = num_heads // tp
        local_hidden = hidden_size // tp  # per-device output dim

        q_src = hidden_states
        kv_src = hidden_states if encoder_hidden_states is None else encoder_hidden_states

        if encoder_hidden_states is None and hasattr(attn_module, "qkv"):
            # Self-attention fused QKV. For TP: output is 3*local_hidden per device.
            qkv_out_dim = int(attn_module.qkv.weight.shape[-1])  # 3*H or 3*H//tp
            pc_qkv = self._matmul_pc(batch * seq_q, hidden_size, qkv_out_dim)
            qkv = self._linear(
                q_src,
                attn_module.qkv.weight,
                attn_module.qkv.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_qkv,
            )
            qkv_4d = ttnn.reshape(qkv, (batch, 1, seq_q, qkv_out_dim))
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                qkv_4d,
                num_heads=num_local_heads,
                num_kv_heads=num_local_heads,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv)
            ttnn.deallocate(qkv_4d)
            qh = ttnn.multiply(q, 1.0 / math.sqrt(head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(q)
            kh, vh = k, v
        else:
            # Cross-attention (separate Q and fused KV). For TP: Q output = local_hidden, KV = 2*local_hidden.
            q_out_dim = int(attn_module.q_proj.weight.shape[-1])  # H or H//tp
            pc_q = self._matmul_pc(batch * seq_q, hidden_size, q_out_dim)

            q = self._linear(
                q_src,
                attn_module.q_proj.weight,
                attn_module.q_proj.bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_q,
            )
            kv_packed = None
            if hasattr(attn_module, "kv"):
                kv_out_dim = int(attn_module.kv.weight.shape[-1])  # 2H or 2H//tp
                pc_kv2 = self._matmul_pc(batch * seq_k, hidden_size, kv_out_dim)
                kv_packed = self._linear(
                    kv_src,
                    attn_module.kv.weight,
                    attn_module.kv.bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    program_config=pc_kv2,
                )
                kv_half = int(kv_packed.shape[-1]) // 2
                k = ttnn.slice(kv_packed, [0, 0, 0], [batch, seq_k, kv_half], [1, 1, 1])
                v = ttnn.slice(kv_packed, [0, 0, kv_half], [batch, seq_k, 2 * kv_half], [1, 1, 1])
            else:
                k_out_dim = int(attn_module.k_proj.weight.shape[-1])
                pc_kv_single = self._matmul_pc(batch * seq_k, hidden_size, k_out_dim)
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

            # Fused head-split (avoids the ~27µs reshape+permute per K/V at enc_seq=512).
            qh = self._heads_fused(q, num_local_heads)
            kh = self._heads_fused(k, num_local_heads)
            vh = self._heads_fused(v, num_local_heads)

            ttnn.deallocate(q)
            if kv_packed is not None:
                ttnn.deallocate(kv_packed)
            else:
                ttnn.deallocate(k)
                ttnn.deallocate(v)

            qh = ttnn.multiply(qh, 1.0 / math.sqrt(head_dim), memory_config=ttnn.L1_MEMORY_CONFIG)

        is_causal = encoder_hidden_states is None and attn_mask is None

        if prefill_kv_cache_fill and kv_cache is not None and encoder_hidden_states is None and "qkv" in attn_module:
            fill_len = kv_cache_fill_len if kv_cache_fill_len is not None else seq_q
            write_self_kv_prefill_to_cache(kh, vh, kv_cache, seq_len=fill_len)

        cross_kv_in_cache = False
        if prefill_kv_cache_fill and cross_attn_cache is not None and encoder_hidden_states is not None:
            if cross_attn_cache[0].dtype == ttnn.bfloat16:
                write_cross_kv_prefill_to_cache(kh, vh, cross_attn_cache)
                kh, vh = cross_attn_cache[0], cross_attn_cache[1]
                cross_kv_in_cache = True
            else:
                ttnn.copy(kh, cross_attn_cache[0])
                ttnn.copy(vh, cross_attn_cache[1])

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
        if not cross_kv_in_cache:
            ttnn.deallocate(kh)
            ttnn.deallocate(vh)

        ls = attn_out.shape
        ps = attn_out.padded_shape
        if (
            len(ls) == 4
            and int(ls[0]) == batch
            and int(ls[1]) == num_local_heads
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
                [batch, num_local_heads, seq_q, head_dim],
                [1, 1, 1, 1],
            )
            ttnn.deallocate(attn_out)

        merged_4d = ttnn.experimental.nlp_concat_heads(attn_for_concat, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn_for_concat)
        merged = ttnn.reshape(merged_4d, (batch, seq_q, local_hidden))
        ttnn.deallocate(merged_4d)
        out_proj_in_dim = int(attn_module.out_proj.weight.shape[-2])  # H or H//tp
        proj = self._linear(
            merged,
            attn_module.out_proj.weight,
            attn_module.out_proj.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            program_config=self._matmul_pc(batch * seq_q, out_proj_in_dim, hidden_size),
        )
        ttnn.deallocate(merged)
        if tp > 1:
            proj = all_reduce_sum_replicate(
                proj,
                self.device,
                cluster_axis=self._cluster_axis,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
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
        cache_seq_len: Optional[int] = None,
        prefill_kv_cache_fill: bool = False,
        kv_cache_fill_len: Optional[int] = None,
        trace_no_profiler: bool = False,
    ) -> ttnn.Tensor:
        parameters = self.parameters
        num_heads = self.num_attention_heads
        hidden_size = self.hidden_size
        head_dim = hidden_size // num_heads
        num_layers = self.num_hidden_layers
        is_decode = kv_cache is not None and current_decode_pos is not None
        if is_decode:
            if cache_seq_len is None:
                cache_seq_len = int(to_torch_replicated_first_shard(current_decode_pos).reshape(-1)[0].item()) + 1
        else:
            cache_seq_len = None

        sdpa_self = self._sdpa_program_config(seq, seq, large_chunks=True)
        # Encoder keys are often 32-wide after subsampling; k_chunk=64 mis-schedules SDPA vs PyTorch.
        sdpa_cross = self._sdpa_program_config(seq, enc_seq, large_chunks=(enc_seq >= 64))

        # fc1.weight shape is [in, out//tp] for TP (column-parallel). shape[-1] = local ffn dim.
        ffn_intermediate = int(parameters.layers[0].ffn.fc1.weight.shape[-1])
        token_m = batch * seq
        m_tiles = (batch * seq + 31) // 32
        n_tiles = hidden_size // 32
        use_kv_path = kv_cache is not None
        ffn_fc2_cfg = self._linear_ln_compute_cfg if use_kv_path else self._ffn_fc2_compute_cfg

        if is_decode:
            pc_ffn_fc1 = self._decode_matmul_pc(hidden_size, ffn_intermediate)
            pc_ffn_fc2 = self._decode_matmul_pc(ffn_intermediate, hidden_size)
        else:
            pc_ffn_fc1 = self._matmul_pc(token_m, hidden_size, ffn_intermediate)
            pc_ffn_fc2 = self._matmul_pc(token_m, ffn_intermediate, hidden_size)

        sdpa_decode_bundle = None
        decode_attn_pcs = None
        attn_scale = None
        if is_decode:
            assert cache_seq_len is not None
            sdpa_decode_bundle = self._decode_sdpa_configs(cache_seq_len)
            attn_scale = 1.0 / math.sqrt(head_dim)
            # Read TP-aware dims from actual weight shapes (correct for both tp=1 and tp>1).
            qkv_decode_out = int(parameters.layers[0].self_attn.qkv_decode.weight.shape[-1])
            self_out_in = int(parameters.layers[0].self_attn.out_proj.weight.shape[-2])
            cross_q_out = int(parameters.layers[0].cross_attention.q_proj.weight.shape[-1])
            cross_out_in = int(parameters.layers[0].cross_attention.out_proj.weight.shape[-2])
            decode_attn_pcs = {
                "qkv": self._decode_matmul_pc(hidden_size, qkv_decode_out),
                "out": self._decode_matmul_pc(self_out_in, hidden_size),
                "q_cross": self._decode_matmul_pc(hidden_size, cross_q_out),
                "out_cross": self._decode_matmul_pc(cross_out_in, hidden_size),
            }
            if not cross_attn_cache_valid:
                cross_kv_out = int(parameters.layers[0].cross_attention.kv.weight.shape[-1])
                decode_attn_pcs["kv_enc"] = self._matmul_pc(batch * enc_seq, hidden_size, cross_kv_out)

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
                cache_seq_len=cache_seq_len,
                prefill_kv_cache_fill=prefill_kv_cache_fill,
                kv_cache_fill_len=kv_cache_fill_len,
                sdpa_decode_bundle=sdpa_decode_bundle,
                decode_attn_pcs=decode_attn_pcs,
                attn_scale=attn_scale,
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
            enc_states = None if is_decode and cross_attn_cache_valid else encoder_hidden_states
            attn_out = self._attention(
                normed,
                enc_states,
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
                cache_seq_len=cache_seq_len,
                prefill_kv_cache_fill=prefill_kv_cache_fill,
                kv_cache_fill_len=kv_cache_fill_len,
                sdpa_decode_bundle=sdpa_decode_bundle,
                decode_attn_pcs=decode_attn_pcs,
                attn_scale=attn_scale,
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
                program_config=pc_ffn_fc1,
                activation="relu",
                is_decode=is_decode,
            )
            ttnn.deallocate(normed)
            ff_in = ff
            ff = self._linear(
                ff_in,
                layer.ffn.fc2.weight,
                layer.ffn.fc2.bias,
                compute_cfg=ffn_fc2_cfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                program_config=pc_ffn_fc2,
                is_decode=is_decode,
            )
            ttnn.deallocate(ff_in)
            if self._tp > 1:
                ff = all_reduce_sum_replicate(
                    ff,
                    self.device,
                    cluster_axis=self._cluster_axis,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            residual = hidden
            hidden = ttnn.add(residual, ff, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(residual)
            ttnn.deallocate(ff)

            if (i + 1) % _PROFILER_LAYER_DRAIN_INTERVAL == 0:
                _drain_device_profiler(self.device, trace_no_profiler=trace_no_profiler)

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
        cache_seq_len: Optional[int] = None,
        prefill_kv_cache_fill: bool = False,
        kv_cache_fill_len: Optional[int] = None,
        trace_no_profiler: bool = False,
    ) -> ttnn.Tensor:
        """
        Prefill when ``kv_cache`` is ``None``.

        Decode when ``kv_cache`` and ``current_decode_pos`` are set (``seq_len=1``, no causal mask).
        Pass ``cache_seq_len=position + 1`` on decode to avoid a host read of ``current_decode_pos``.

        Set ``prefill_kv_cache_fill=True`` with ``kv_cache`` on a full-sequence prefill forward to
        populate caches from batch attention (faster and better PCC than token-by-token cache warm-up).
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
            cache_seq_len=cache_seq_len,
            prefill_kv_cache_fill=prefill_kv_cache_fill,
            kv_cache_fill_len=kv_cache_fill_len,
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

        _drain_device_profiler(self.device, trace_no_profiler=trace_no_profiler)

        return out
