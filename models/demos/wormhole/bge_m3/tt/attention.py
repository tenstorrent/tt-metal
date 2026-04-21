# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight
from models.demos.wormhole.bge_m3.tt.device_kernels import (
    bge_m3_linear_activation_memory_config,
    bge_m3_matmul_compute_kernel_config,
    bge_m3_matmul_core_grid,
    bge_m3_sdpa_compute_kernel_config,
    bge_m3_weight_dram_memory_config,
    max_qkv_mm_chunk_seq_len,
    max_wo_mm_chunk_seq_len,
)

# SDPA tiling (``main``-compatible path): fixed **Q chunk = 128** and largest **K** in (256, 128) that
# divides ``seq_len``; ``exp_approx_mode=True``. That pairing passes full-model PCC at S8192 on Wormhole.
# Picking **Q=256** + ``exp_approx_mode=False`` (an optimization on this branch) regresses S8192 PCC.
#
# For S32/S64 (not divisible by 128), use flexible Q/K from (256..32) so tiles divide the runtime length.
_SDPA_Q_CHUNK_MAIN = 128
_SDPA_K_CANDIDATES_MAIN = (256, 128)
_SDPA_Q_CHUNKS_FLEX = (256, 128, 64, 32)
_SDPA_K_CHUNKS_FLEX = (256, 128, 64, 32)


def _sdpa_chunks_for_seq_len(seq_len: int) -> tuple[int, int]:
    """Q/K chunk sizes for SDPA. ``main`` uses fixed Q=128 for all 128-token-aligned lengths."""
    if seq_len % 128 == 0:
        for k_chunk in _SDPA_K_CANDIDATES_MAIN:
            if k_chunk <= seq_len and seq_len % k_chunk == 0:
                return _SDPA_Q_CHUNK_MAIN, k_chunk
        raise ValueError(f"Unable to pick k_chunk_size for seq_len={seq_len} (expected a multiple of 128)")
    if seq_len % 32 != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by 32 (tile height)")
    if seq_len > 128 and seq_len % 128 != 0:
        raise ValueError(f"seq_len {seq_len} must be divisible by 128 when seq_len > 128")
    q_chunk = next(q for q in _SDPA_Q_CHUNKS_FLEX if q <= seq_len and seq_len % q == 0)
    k_chunk = next(k for k in _SDPA_K_CHUNKS_FLEX if k <= seq_len and seq_len % k == 0)
    return q_chunk, k_chunk


def _sdpa_exp_approx_for_seq_len(seq_len: int) -> bool:
    """``main`` sets ``exp_approx_mode=True`` for 128-aligned encoder runs (incl. S8192); short S32/S64 use False."""
    return seq_len % 128 == 0


def _sdpa_storage_grid(mesh_device: ttnn.MeshDevice | None):
    """Use the device's CoreCoord for SDPA (matches unit tests; avoids tuple/grid mismatches)."""
    if mesh_device is None:
        return (8, 8)
    try:
        return mesh_device.compute_with_storage_grid_size()
    except Exception:
        return (8, 8)


def _sdpa_compute_grid_for_seq_len(_seq_len: int, mesh_device: ttnn.MeshDevice | None):
    """``compute_with_storage_grid_size`` for ``SDPAProgramConfig``.

    Use the full device compute grid for every sequence length. An older **S512-only** cap
    (smaller worker grid) hurt 512-token throughput; chunk sizes and ``exp_approx_mode`` still
    follow ``_sdpa_chunks_for_seq_len`` / ``_sdpa_exp_approx_for_seq_len`` and are unchanged for S8192.
    """
    return _sdpa_storage_grid(mesh_device)


def _sdpa_program_config_for_seq_len(seq_len: int, mesh_device: ttnn.MeshDevice | None) -> ttnn.SDPAProgramConfig:
    q_chunk, k_chunk = _sdpa_chunks_for_seq_len(seq_len)
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=_sdpa_compute_grid_for_seq_len(seq_len, mesh_device),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=_sdpa_exp_approx_for_seq_len(seq_len),
    )


@dataclass
class BgeM3AttentionConfig:
    # Required weights
    wqkv: LazyWeight
    wo_weight: LazyWeight

    # Model dimensions
    hidden_size: int
    num_heads: int
    head_dim: int

    # Optional weights
    bqkv: LazyWeight | None = None
    wo_bias: LazyWeight | None = None
    mesh_device: ttnn.MeshDevice | None = None

    # Attention
    attention_scale: float | None = None

    # Runtime-resolved dtype and memory fields
    qkv_dtype: ttnn.DataType | None = None
    score_dtype: ttnn.DataType | None = None
    output_dtype: ttnn.DataType | None = None
    qkv_memcfg: ttnn.MemoryConfig | None = None
    score_memcfg: ttnn.MemoryConfig | None = None
    output_memcfg: ttnn.MemoryConfig | None = None

    # Optional runtime program and compute knobs
    qkv_prg_config: object | None = None
    score_prg_config: object | None = None
    output_prg_config: object | None = None
    qkv_compute_kernel_cfg: object | None = None
    score_compute_kernel_cfg: object | None = None
    output_compute_kernel_cfg: object | None = None
    # Compile-time max sequence length (selects Wormhole HiFi4 at S8192 for PCC).
    max_seq_len: int | None = None
    max_batch_size: int | None = None

    @property
    def qkv_out_dim(self) -> int:
        return 3 * self.hidden_size


class BgeM3Attention(LightweightModule):
    """
    BGE-M3 encoder self-attention module.

    Public API is intentionally minimal and encoder-only:
      - __init__
      - from_config
      - load_device_weights
      - forward

    Explicitly out of scope:
      - decode/prefill split
      - KV cache / paged attention
      - rotary embedding
      - collectives
    """

    def __init__(
        self,
        wqkv: LazyWeight,
        wo_weight: LazyWeight,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        bqkv: LazyWeight | None = None,
        wo_bias: LazyWeight | None = None,
        attention_scale: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.config = _resolve_attention_config(
            BgeM3AttentionConfig(
                wqkv=wqkv,
                wo_weight=wo_weight,
                hidden_size=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                bqkv=bqkv,
                wo_bias=wo_bias,
                attention_scale=attention_scale,
                max_seq_len=max_seq_len,
            )
        )
        self._device_weights_loaded = False

    @classmethod
    def from_config(cls, config: BgeM3AttentionConfig) -> "BgeM3Attention":
        instance = object.__new__(cls)
        super(BgeM3Attention, instance).__init__()
        instance.config = _resolve_attention_config(config)
        instance._device_weights_loaded = False
        return instance

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return

        self.wqkv = self.config.wqkv.get_device_weight()
        self.bqkv = self.config.bqkv.get_device_weight() if self.config.bqkv is not None else None
        self.wo_weight = self.config.wo_weight.get_device_weight()
        self.wo_bias = self.config.wo_bias.get_device_weight() if self.config.wo_bias is not None else None
        self._device_weights_loaded = True

    def _masked_fill_scores(
        self,
        attention_scores: ttnn.Tensor,
        pad_mask: ttnn.Tensor | None,
        masked_value: float = -1e9,
    ) -> ttnn.Tensor:
        """
        TT equivalent of:
            attention_scores.masked_fill_(pad_mask, masked_value)

        where:
          - attention_scores has shape [B, N, S_q, S_k]
          - pad_mask has shape [B, 1, 1, S_k]
        """
        if pad_mask is None:
            return attention_scores

        num_heads = attention_scores.shape[1]
        query_seq_len = attention_scores.shape[2]
        expanded_mask = ttnn.expand(pad_mask, [-1, num_heads, query_seq_len, -1])

        target_memcfg = attention_scores.memory_config()

        if expanded_mask.memory_config() != target_memcfg:
            expanded_mask = ttnn.to_memory_config(expanded_mask, target_memcfg)

        masked_positions = ttnn.gt(expanded_mask, 0)
        if masked_positions.memory_config() != target_memcfg:
            masked_positions = ttnn.to_memory_config(masked_positions, target_memcfg)

        return ttnn.where(masked_positions, masked_value, attention_scores)

    def forward(self, hidden_states: ttnn.Tensor, attention_mask: ttnn.Tensor | None = None) -> ttnn.Tensor:
        self.load_device_weights()

        batch_size, _, seq_len, _ = hidden_states.shape

        assert seq_len > 0, "seq_len must be positive"
        assert seq_len % 32 == 0, "seq_len must be divisible by 32 (tile height) for TILE_LAYOUT"
        if seq_len > 128:
            assert seq_len % 128 == 0, "seq_len must be divisible by 128 when seq_len > 128"

        cfg = self.config
        core_grid = bge_m3_matmul_core_grid(cfg.mesh_device, seq_len, batch_size)

        max_qkv = max_qkv_mm_chunk_seq_len(cfg.mesh_device)
        if seq_len > max_qkv:
            if seq_len % max_qkv != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {max_qkv}")
            hidden_states = ttnn.reshape(
                hidden_states,
                [batch_size, seq_len // max_qkv, max_qkv, -1],
            )

        # Stage 1: fused QKV projection.
        qkv_fused = ttnn.linear(
            hidden_states,
            self.wqkv,
            memory_config=cfg.qkv_memcfg,
            dtype=cfg.qkv_dtype,
            bias=self.bqkv,
            program_config=cfg.qkv_prg_config,
            compute_kernel_config=cfg.qkv_compute_kernel_cfg,
            core_grid=core_grid,
        )
        if seq_len > max_qkv:
            qkv_fused = ttnn.reshape(qkv_fused, [batch_size, 1, seq_len, -1])

        # Stage 2: split Q/K/V heads.
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv_fused,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_heads,
            transpose_k_heads=False,
            memory_config=cfg.score_memcfg,
        )
        ttnn.deallocate(qkv_fused)

        # Stage 3: optional deterministic cast to score dtype.
        if cfg.score_dtype is not None and q.dtype != cfg.score_dtype:
            q_cast = ttnn.typecast(q, dtype=cfg.score_dtype)
            ttnn.deallocate(q)
            q = q_cast
        if cfg.score_dtype is not None and k.dtype != cfg.score_dtype:
            k_cast = ttnn.typecast(k, dtype=cfg.score_dtype)
            ttnn.deallocate(k)
            k = k_cast
        if cfg.score_dtype is not None and v.dtype != cfg.score_dtype:
            v_cast = ttnn.typecast(v, dtype=cfg.score_dtype)
            ttnn.deallocate(v)
            v = v_cast

        sdpa_mask = attention_mask
        if sdpa_mask is not None:
            if len(sdpa_mask.shape) != 4:
                raise ValueError(f"attention_mask must have rank 4 [B, 1, 1, S], got shape={sdpa_mask.shape}")
            if (
                sdpa_mask.shape[0] != batch_size
                or sdpa_mask.shape[1] != 1
                or sdpa_mask.shape[2] != 1
                or sdpa_mask.shape[3] != seq_len
            ):
                raise ValueError(
                    f"attention_mask must have shape [B, 1, 1, S]=[{batch_size}, 1, 1, {seq_len}], "
                    f"got shape={sdpa_mask.shape}"
                )
            sdpa_mask = ttnn.expand(sdpa_mask, [-1, -1, seq_len, -1])

            if cfg.score_dtype is not None and sdpa_mask.dtype != cfg.score_dtype:
                sdpa_mask = ttnn.typecast(sdpa_mask, dtype=cfg.score_dtype)

            # SDPA requires attn_mask in DRAM (ttnn sdpa_device_operation); L1 score_memcfg must not apply here.
            if sdpa_mask.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                sdpa_mask = ttnn.to_memory_config(sdpa_mask, ttnn.DRAM_MEMORY_CONFIG)

        # Stage 4: encoder SDPA (chunk sizes must divide the actual sequence length, including S<128).
        sdpa_program_config = _sdpa_program_config_for_seq_len(seq_len, cfg.mesh_device)
        context = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=sdpa_mask,
            scale=cfg.attention_scale,
            program_config=sdpa_program_config,
            compute_kernel_config=cfg.score_compute_kernel_cfg,
            memory_config=cfg.score_memcfg,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Stage 5: concat heads back to [B, 1, S, D].
        context = ttnn.experimental.nlp_concat_heads(context, memory_config=cfg.output_memcfg)

        max_wo = max_wo_mm_chunk_seq_len(cfg.mesh_device)
        if seq_len > max_wo:
            if seq_len % max_wo != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {max_wo}")
            context = ttnn.reshape(context, [batch_size, seq_len // max_wo, max_wo, -1])

        # Stage 6: output projection.
        output = ttnn.linear(
            context,
            self.wo_weight,
            memory_config=cfg.output_memcfg,
            dtype=cfg.output_dtype,
            bias=self.wo_bias,
            program_config=cfg.output_prg_config,
            compute_kernel_config=cfg.output_compute_kernel_cfg,
            core_grid=core_grid,
        )
        ttnn.deallocate(context)

        if seq_len > max_wo:
            output = ttnn.reshape(output, [batch_size, 1, seq_len, -1])

        return output


def _resolve_attention_config(config: BgeM3AttentionConfig) -> BgeM3AttentionConfig:
    """
    Resolve attention defaults and materialize LazyWeight metadata.
    """

    # Phase A: fail-fast shape and required-weight validation.
    if config.hidden_size != config.num_heads * config.head_dim:
        raise ValueError(
            "Invalid head geometry: hidden_size must equal num_heads * head_dim "
            f"(got hidden_size={config.hidden_size}, num_heads={config.num_heads}, head_dim={config.head_dim})"
        )
    if config.wqkv is None or config.wo_weight is None:
        raise ValueError("Both wqkv and wo_weight must be provided for BgeM3Attention")

    to_set: dict[str, object] = {}

    # Phase B: numerics defaults.
    if config.attention_scale is None:
        to_set["attention_scale"] = config.head_dim**-0.5
    if config.qkv_dtype is None:
        to_set["qkv_dtype"] = ttnn.bfloat16
    if config.score_dtype is None:
        to_set["score_dtype"] = ttnn.bfloat16
    if config.output_dtype is None:
        to_set["output_dtype"] = ttnn.bfloat16

    max_seq = config.max_seq_len
    max_batch = config.max_batch_size if config.max_batch_size is not None else 1

    # Phase C: resolve single target device.
    param_devices = [
        param.device
        for param in (config.wqkv, config.bqkv, config.wo_weight, config.wo_bias)
        if param is not None and param.device is not None
    ]
    if param_devices and any(device != param_devices[0] for device in param_devices):
        raise ValueError("All attention parameters must target the same device")
    if config.mesh_device is not None and param_devices and param_devices[0] != config.mesh_device:
        raise ValueError("All attention parameters must target the configured mesh_device")

    mesh_device = (
        config.mesh_device
        if config.mesh_device is not None
        else (param_devices[0] if param_devices else ttnn.GetDefaultDevice())
    )
    if mesh_device is None:
        raise ValueError("Unable to resolve target device for BgeM3Attention")

    if config.mesh_device is None:
        to_set["mesh_device"] = mesh_device

    # Phase D: activation memory (single envelope: batch×seq + seq cap — device_kernels).
    act_mem = bge_m3_linear_activation_memory_config(max_seq, max_batch)
    if config.qkv_memcfg is None:
        to_set["qkv_memcfg"] = act_mem
    if config.score_memcfg is None:
        to_set["score_memcfg"] = act_mem
    if config.output_memcfg is None:
        to_set["output_memcfg"] = act_mem

    if config.score_prg_config is None:
        q0, k0 = _sdpa_chunks_for_seq_len(128)
        to_set["score_prg_config"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=_sdpa_compute_grid_for_seq_len(128, mesh_device),
            q_chunk_size=q0,
            k_chunk_size=k0,
            exp_approx_mode=_sdpa_exp_approx_for_seq_len(128),
        )

    if config.qkv_compute_kernel_cfg is None:
        to_set["qkv_compute_kernel_cfg"] = bge_m3_matmul_compute_kernel_config(
            mesh_device, max_seq_len=max_seq, max_batch_size=max_batch
        )
    if config.output_compute_kernel_cfg is None:
        to_set["output_compute_kernel_cfg"] = bge_m3_matmul_compute_kernel_config(
            mesh_device, max_seq_len=max_seq, max_batch_size=max_batch
        )
    if config.score_compute_kernel_cfg is None:
        to_set["score_compute_kernel_cfg"] = bge_m3_sdpa_compute_kernel_config(
            mesh_device, max_seq_len=max_seq, max_batch_size=max_batch
        )

    # Phase E: resolve LazyWeights with resolved dtype + memory config.
    qkv_dtype = to_set.get("qkv_dtype", config.qkv_dtype)
    output_dtype = to_set.get("output_dtype", config.output_dtype)
    weight_dram = bge_m3_weight_dram_memory_config()

    to_set["wqkv"] = resolve_lazy_weight(
        config.wqkv,
        device=mesh_device,
        dtype=qkv_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_dram,
        mesh_mapper_config=None,
    )
    to_set["wo_weight"] = resolve_lazy_weight(
        config.wo_weight,
        device=mesh_device,
        dtype=output_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=weight_dram,
        mesh_mapper_config=None,
    )
    if config.bqkv is not None:
        to_set["bqkv"] = resolve_lazy_weight(
            config.bqkv,
            device=mesh_device,
            dtype=qkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_dram,
            mesh_mapper_config=None,
        )
    if config.wo_bias is not None:
        to_set["wo_bias"] = resolve_lazy_weight(
            config.wo_bias,
            device=mesh_device,
            dtype=output_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=weight_dram,
            mesh_mapper_config=None,
        )

    return replace(config, **to_set)
