# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight

# Match Attention1D long-sequence QKV matmul guard.
MAX_QKV_MM_SEQ_LEN = 2048
# Match Attention1D long-sequence WO matmul guard.
MAX_MM_SEQ_LEN = 1024


def _hifi2_mm_kernel() -> ttnn.WormholeComputeKernelConfig:
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
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

        assert seq_len % 128 == 0 and seq_len > 0, "seq_len must be divisible by 128"

        cfg = self.config

        if seq_len > MAX_QKV_MM_SEQ_LEN:
            qkv_ck = _hifi2_mm_kernel()
            out_ck = _hifi2_mm_kernel()
            score_ck = _hifi2_mm_kernel()
        else:
            qkv_ck = cfg.qkv_compute_kernel_cfg
            out_ck = cfg.output_compute_kernel_cfg
            score_ck = cfg.score_compute_kernel_cfg

        if seq_len > MAX_QKV_MM_SEQ_LEN:
            if seq_len % MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {MAX_QKV_MM_SEQ_LEN}")
            hidden_states = ttnn.reshape(
                hidden_states,
                [batch_size, seq_len // MAX_QKV_MM_SEQ_LEN, MAX_QKV_MM_SEQ_LEN, -1],
            )

        # Stage 1: fused QKV projection.
        qkv_fused = ttnn.linear(
            hidden_states,
            self.wqkv,
            memory_config=cfg.qkv_memcfg,
            dtype=cfg.qkv_dtype,
            bias=self.bqkv,
            program_config=cfg.qkv_prg_config,
            compute_kernel_config=qkv_ck,
        )
        if seq_len > MAX_QKV_MM_SEQ_LEN:
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

        # sdpa_mask = attention_mask
        # if sdpa_mask is not None:
        #     if len(sdpa_mask.shape) != 4:
        #         raise ValueError(f"attention_mask must have rank 4 [B, 1, 1, S], got shape={sdpa_mask.shape}")
        #     if (
        #         sdpa_mask.shape[0] != batch_size
        #         or sdpa_mask.shape[1] != 1
        #         or sdpa_mask.shape[2] != 1
        #         or sdpa_mask.shape[3] != seq_len
        #     ):
        #         raise ValueError(
        #             f"attention_mask must have shape [B, 1, 1, S]=[{batch_size}, 1, 1, {seq_len}], "
        #             f"got shape={sdpa_mask.shape}"
        #         )
        #     sdpa_mask = ttnn.expand(sdpa_mask, [-1, -1, seq_len, -1])
        sdpa_mask = attention_mask
        if sdpa_mask is not None:
            if len(sdpa_mask.shape) != 4:
                raise ValueError(f"attention_mask must have rank 4, got shape={sdpa_mask.shape}")
            if sdpa_mask.shape[0] != batch_size or sdpa_mask.shape[1] != 1 or sdpa_mask.shape[3] != seq_len:
                raise ValueError(
                    f"attention_mask must be [B, 1, ?, S] with B={batch_size}, S={seq_len}, "
                    f"got shape={sdpa_mask.shape}"
                )
            if sdpa_mask.shape[2] == 1:
                sdpa_mask = ttnn.expand(sdpa_mask, [-1, -1, seq_len, -1])
            # elif sdpa_mask.shape[2] != seq_len:
            #     raise ValueError(
            #         f"attention_mask dim 2 must be 1 or {seq_len}, got shape={sdpa_mask.shape}"
            #     )

            if cfg.score_dtype is not None and sdpa_mask.dtype != cfg.score_dtype:
                sdpa_mask = ttnn.typecast(sdpa_mask, dtype=cfg.score_dtype)

            # score_memcfg = cfg.score_memcfg or ttnn.DRAM_MEMORY_CONFIG
            score_memcfg = ttnn.DRAM_MEMORY_CONFIG

            if sdpa_mask.memory_config() != score_memcfg:
                sdpa_mask = ttnn.to_memory_config(sdpa_mask, score_memcfg)

        # Stage 4: encoder SDPA.
        context = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            attn_mask=sdpa_mask,
            scale=cfg.attention_scale,
            program_config=cfg.score_prg_config,
            compute_kernel_config=score_ck,
            memory_config=cfg.score_memcfg,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Stage 5: concat heads back to [B, 1, S, D].
        context = ttnn.experimental.nlp_concat_heads(context, memory_config=cfg.output_memcfg)

        if seq_len > MAX_MM_SEQ_LEN:
            if seq_len % MAX_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {MAX_MM_SEQ_LEN}")
            context = ttnn.reshape(context, [batch_size, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        # Stage 6: output projection.
        output = ttnn.linear(
            context,
            self.wo_weight,
            memory_config=cfg.output_memcfg,
            dtype=cfg.output_dtype,
            bias=self.wo_bias,
            program_config=cfg.output_prg_config,
            compute_kernel_config=out_ck,
        )
        ttnn.deallocate(context)

        if seq_len > MAX_MM_SEQ_LEN:
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

    # Phase C: memory config defaults.
    if config.qkv_memcfg is None:
        to_set["qkv_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.score_memcfg is None:
        to_set["score_memcfg"] = ttnn.DRAM_MEMORY_CONFIG
    if config.output_memcfg is None:
        to_set["output_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    if config.qkv_compute_kernel_cfg is None:
        to_set["qkv_compute_kernel_cfg"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,  # HiFi4 to2
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
    if config.output_compute_kernel_cfg is None:
        to_set["output_compute_kernel_cfg"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,  # HiFi4 to2
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    if config.score_prg_config is None:
        to_set["score_prg_config"] = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=128,
            k_chunk_size=512,
            # exp_approx_mode=False,
            exp_approx_mode=True,
        )

    # Phase D: resolve single target device.
    param_devices = [
        param.device
        for param in (config.wqkv, config.bqkv, config.wo_weight, config.wo_bias)
        if param is not None and param.device is not None
    ]
    if param_devices and any(device != param_devices[0] for device in param_devices):
        raise ValueError("All attention parameters must target the same device")
    mesh_device = param_devices[0] if param_devices else ttnn.GetDefaultDevice()
    if mesh_device is None:
        raise ValueError("Unable to resolve target device for BgeM3Attention")

    # Phase E: resolve LazyWeights with resolved dtype + memory config.
    qkv_dtype = to_set.get("qkv_dtype", config.qkv_dtype)
    output_dtype = to_set.get("output_dtype", config.output_dtype)
    qkv_memcfg = to_set.get("qkv_memcfg", config.qkv_memcfg)
    output_memcfg = to_set.get("output_memcfg", config.output_memcfg)

    to_set["wqkv"] = resolve_lazy_weight(
        config.wqkv,
        device=mesh_device,
        dtype=qkv_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=qkv_memcfg,
        mesh_mapper_config=None,
    )
    to_set["wo_weight"] = resolve_lazy_weight(
        config.wo_weight,
        device=mesh_device,
        dtype=output_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=output_memcfg,
        mesh_mapper_config=None,
    )
    if config.bqkv is not None:
        to_set["bqkv"] = resolve_lazy_weight(
            config.bqkv,
            device=mesh_device,
            dtype=qkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=qkv_memcfg,
            mesh_mapper_config=None,
        )
    if config.wo_bias is not None:
        to_set["wo_bias"] = resolve_lazy_weight(
            config.wo_bias,
            device=mesh_device,
            dtype=output_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=output_memcfg,
            mesh_mapper_config=None,
        )

    return replace(config, **to_set)
