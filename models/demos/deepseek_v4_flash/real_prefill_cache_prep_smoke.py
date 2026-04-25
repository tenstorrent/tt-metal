# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import window_topk_indices
from models.demos.deepseek_v4_flash.real_attention_projection_smoke import (
    ATTENTION_FP8_BLOCK_SIZE,
    ATTENTION_TTNN_TILE_MULTIPLE,
    build_torch_attention_projection_reference,
    decode_real_attention_projection_weights,
    deterministic_attention_activation,
    layer_attention_projection_keys,
)
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex, TensorMetadata
from models.demos.deepseek_v4_flash.real_kv_projection_smoke import (
    KV_FP8_BLOCK_SIZE,
    KvProjectionWeights,
    build_torch_kv_projection_reference,
    decode_real_kv_projection_weights,
    layer_kv_projection_keys,
)
from models.demos.deepseek_v4_flash.ttnn_attention_projection import AttentionProjectionWeights, TtAttentionProjection

REAL_PREFILL_CACHE_PREP_SMOKE_SCHEMA_VERSION = 1
DEFAULT_LAYER_PREFILL_CACHE_PREP_LAYER = 3
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_PREFILL_CACHE_PREP_MAX_TENSORS = 9
DEFAULT_PREFILL_CACHE_PREP_MAX_BYTES = 64 * 1024 * 1024
PREFILL_CACHE_PREP_TTNN_TILE_MULTIPLE = ATTENTION_TTNN_TILE_MULTIPLE


def run_real_prefill_cache_prep_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_LAYER_PREFILL_CACHE_PREP_LAYER,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    start_pos: int = 0,
    max_tensors: int = DEFAULT_PREFILL_CACHE_PREP_MAX_TENSORS,
    max_bytes: int = DEFAULT_PREFILL_CACHE_PREP_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    attn_norm_pcc: float = 0.999,
    q_rank_pcc: float = 0.999,
    q_output_pcc: float = 0.99,
    kv_linear_pcc: float = 0.99,
    kv_output_pcc: float = 0.99,
    cache_prep_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
    cache_prep_atol: float = 2e-1,
) -> dict[str, Any]:
    """Run the first real DeepSeek V4 Flash prefill attention/cache-prep slice."""

    _validate_smoke_args(
        layer=layer,
        seq_len=seq_len,
        start_pos=start_pos,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
        pcc_thresholds={
            "attn_norm_pcc": attn_norm_pcc,
            "q_rank_pcc": q_rank_pcc,
            "q_output_pcc": q_output_pcc,
            "kv_linear_pcc": kv_linear_pcc,
            "kv_output_pcc": kv_output_pcc,
            "cache_prep_pcc": cache_prep_pcc,
        },
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    tensors, metadata = load_real_prefill_cache_prep_slice(
        snapshot_dir,
        layer=layer,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    q_weights = decode_real_attention_projection_weights(tensors, config=config, layer=layer)
    kv_weights = decode_real_kv_projection_weights(tensors, config=config, layer=layer)

    activation = deterministic_attention_activation(hidden_size=config.hidden_size, seq_len=seq_len)
    reference = build_torch_prefill_cache_prep_reference(
        tensors,
        q_weights,
        kv_weights,
        config=config,
        layer=layer,
        activation=activation,
        start_pos=start_pos,
    )
    result = _base_result(
        snapshot_dir=snapshot_dir,
        layer=layer,
        seq_len=seq_len,
        start_pos=start_pos,
        config=config,
        metadata=metadata,
        tensors=tensors,
        q_weights=q_weights,
        kv_weights=kv_weights,
        activation=activation,
        reference=reference,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )

    if cpu_only:
        result["mode"] = "cpu-reference"
        result["passed"] = True
        return result

    if seq_len % PREFILL_CACHE_PREP_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(
            f"TTNN smoke seq_len must be a multiple of {PREFILL_CACHE_PREP_TTNN_TILE_MULTIPLE}, got {seq_len}"
        )

    ttnn_outputs = _run_ttnn_prefill_cache_prep(
        tensors,
        q_weights,
        kv_weights,
        config=config,
        layer=layer,
        activation=activation,
        start_pos=start_pos,
        device_id=device_id,
    )
    result["mode"] = "ttnn"
    result["device_id"] = int(device_id)
    result["ttnn_ops"] = [
        "ttnn.rms_norm(attn_norm)",
        "TtAttentionProjection.project_q_rank",
        "ttnn.linear(wq_a)",
        "ttnn.rms_norm(q_norm)",
        "TtAttentionProjection.project_q_from_rank",
        "ttnn.linear(wq_b)",
        "ttnn.linear(wkv)",
        "ttnn.rms_norm(kv_norm)",
        "ttnn.reshape(q_output -> q_heads)",
        "ttnn.rms_norm(q_heads)",
        "ttnn.slice(q_nope/q_rope)",
        "ttnn.reshape(kv_output -> kv_heads)",
        "ttnn.slice(kv_nope/kv_rope)",
    ]
    result["ttnn"] = {
        name: _tensor_summary(tensor) for name, tensor in ttnn_outputs.items() if tensor.dtype.is_floating_point
    }
    result["host_cache_prep"] = {
        "window_topk_idxs": _int_tensor_summary(ttnn_outputs["window_topk_idxs"]),
    }
    accuracy = {
        "attn_norm_output": _accuracy_summary(
            reference["attn_norm_output"],
            ttnn_outputs["attn_norm_output"],
            pcc_threshold=attn_norm_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "q_rank_norm": _accuracy_summary(
            reference["q_rank_norm"],
            ttnn_outputs["q_rank_norm"],
            pcc_threshold=q_rank_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "q_output": _accuracy_summary(
            reference["q_output"],
            ttnn_outputs["q_output"],
            pcc_threshold=q_output_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "kv_linear": _accuracy_summary(
            reference["kv_linear"],
            ttnn_outputs["kv_linear"],
            pcc_threshold=kv_linear_pcc,
            rtol=rtol,
            atol=atol,
        ),
        "kv_output": _accuracy_summary(
            reference["kv_output"],
            ttnn_outputs["kv_output"],
            pcc_threshold=kv_output_pcc,
            rtol=rtol,
            atol=atol,
        ),
    }
    for name in (
        "q_heads_pre_norm",
        "q_heads",
        "q_nope",
        "q_rope",
        "q_rope_rotated",
        "q_prefill",
        "kv_heads",
        "kv_nope",
        "kv_rope",
        "kv_rope_rotated",
        "kv_cache_ready",
        "sliding_window_cache",
    ):
        accuracy[name] = _accuracy_summary(
            reference[name],
            ttnn_outputs[name],
            pcc_threshold=cache_prep_pcc,
            rtol=rtol,
            atol=cache_prep_atol,
        )
    accuracy["window_topk_idxs"] = _int_equality_summary(
        reference["window_topk_idxs"], ttnn_outputs["window_topk_idxs"]
    )
    result["accuracy"] = accuracy
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def layer_prefill_cache_prep_keys(index: RealCheckpointTensorIndex, *, layer: int) -> list[str]:
    """Return the strict real tensor set for Q plus K/V projection cache prep."""

    keys: list[str] = []
    for key in layer_attention_projection_keys(index, layer=layer) + layer_kv_projection_keys(index, layer=layer):
        if key not in keys:
            keys.append(key)
    return keys


def load_real_prefill_cache_prep_slice(
    snapshot_dir: str | Path,
    *,
    layer: int,
    max_tensors: int = DEFAULT_PREFILL_CACHE_PREP_MAX_TENSORS,
    max_bytes: int = DEFAULT_PREFILL_CACHE_PREP_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    keys = layer_prefill_cache_prep_keys(index, layer=layer)
    return index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)


def build_torch_prefill_cache_prep_reference(
    tensors: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    kv_weights: KvProjectionWeights,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    start_pos: int = 0,
) -> dict[str, torch.Tensor]:
    attention_reference = build_torch_attention_projection_reference(
        tensors,
        q_weights,
        config=config,
        layer=layer,
        activation=activation,
    )
    kv_reference = build_torch_kv_projection_reference(
        tensors,
        kv_weights,
        config=config,
        layer=layer,
        activation=activation,
    )
    cache_reference = build_prefill_cache_prep_from_projected(
        attention_reference["q_output"].to(torch.bfloat16),
        kv_reference["kv_output"].to(torch.bfloat16),
        config=config,
        layer=layer,
        start_pos=start_pos,
    )
    return {
        **attention_reference,
        **kv_reference,
        **cache_reference,
    }


def build_prefill_cache_prep_from_projected(
    q_output: torch.Tensor,
    kv_output: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    start_pos: int = 0,
) -> dict[str, torch.Tensor]:
    """Convert projected Q and K/V tensors to the local prefill/cache boundary."""

    _validate_projected_q_kv(q_output, kv_output, config=config)
    batch_size, _, seq_len, _ = q_output.shape
    head_dim = int(config.head_dim)
    rope_dim = int(config.qk_rope_head_dim)
    nope_dim = head_dim - rope_dim
    q_heads_pre_norm = q_output[:, 0].reshape(batch_size, seq_len, config.num_attention_heads, head_dim).contiguous()
    q_heads = _unit_rms_norm(q_heads_pre_norm, eps=config.rms_norm_eps)
    q_nope, q_rope = q_heads.split([nope_dim, rope_dim], dim=-1)

    kv_heads = kv_output[:, 0].reshape(batch_size, seq_len, config.num_key_value_heads, head_dim).contiguous()
    if config.num_key_value_heads != 1:
        raise ValueError(f"DeepSeek V4 Flash cache-prep smoke expects one K/V head, got {config.num_key_value_heads}")
    kv_cache_projection = kv_heads[:, :, 0].contiguous()
    kv_nope, kv_rope = kv_cache_projection.split([nope_dim, rope_dim], dim=-1)

    freqs_cis = precompute_deepseek_v4_rope_frequencies(config, layer=layer, seq_len=start_pos + seq_len)[
        start_pos : start_pos + seq_len
    ]
    q_rope_rotated = apply_deepseek_v4_rotary(q_rope.contiguous(), freqs_cis)
    kv_rope_rotated = apply_deepseek_v4_rotary(kv_rope.contiguous(), freqs_cis)
    q_prefill = torch.cat([q_nope, q_rope_rotated], dim=-1).contiguous()
    kv_cache_ready = torch.cat([kv_nope, kv_rope_rotated], dim=-1).contiguous()
    sliding_window_cache = _sliding_window_cache(
        kv_cache_ready, sliding_window=config.sliding_window, start_pos=start_pos
    )
    window_topk_idxs = window_topk_indices(config.sliding_window, batch_size, seq_len, start_pos).to(torch.int32)
    return {
        "q_heads_pre_norm": q_heads_pre_norm,
        "q_heads": q_heads,
        "q_nope": q_nope.contiguous(),
        "q_rope": q_rope.contiguous(),
        "q_rope_rotated": q_rope_rotated,
        "q_prefill": q_prefill,
        "kv_heads": kv_heads,
        "kv_nope": kv_nope.contiguous(),
        "kv_rope": kv_rope.contiguous(),
        "kv_rope_rotated": kv_rope_rotated,
        "kv_cache_ready": kv_cache_ready,
        "sliding_window_cache": sliding_window_cache,
        "window_topk_idxs": window_topk_idxs,
        "rope_freqs_cis": freqs_cis,
    }


def precompute_deepseek_v4_rope_frequencies(
    config: DeepSeekV4FlashConfig,
    *,
    layer: int,
    seq_len: int,
) -> torch.Tensor:
    """Match the local DeepSeek V4 Flash YaRN RoPE convention."""

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if layer < 0 or layer >= len(config.compress_ratios):
        raise ValueError(f"layer {layer} is outside compress_ratios length {len(config.compress_ratios)}")
    if config.compress_ratios[layer]:
        original_seq_len = int(config.rope_scaling["original_max_position_embeddings"])
        base = float(config.compress_rope_theta)
    else:
        original_seq_len = 0
        base = float(config.rope_theta)
    return _precompute_freqs_cis(
        dim=int(config.qk_rope_head_dim),
        seq_len=seq_len,
        original_seq_len=original_seq_len,
        base=base,
        factor=float(config.rope_scaling["factor"]),
        beta_fast=float(config.rope_scaling["beta_fast"]),
        beta_slow=float(config.rope_scaling["beta_slow"]),
    )


def apply_deepseek_v4_rotary(x: torch.Tensor, freqs_cis: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    if x.shape[-1] % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {x.shape[-1]}")
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    elif x_complex.ndim == 4:
        freqs = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    else:
        raise ValueError(f"Expected RoPE tensor rank 3 or 4 after complex view, got {x_complex.ndim}")
    rotated = torch.view_as_real(x_complex * freqs).flatten(-2)
    return rotated.to(x.dtype).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real DeepSeek V4 Flash prefill attention/cache-prep TTNN smoke path."
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER_PREFILL_CACHE_PREP_LAYER)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_PREFILL_CACHE_PREP_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_PREFILL_CACHE_PREP_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--attn-norm-pcc", type=float, default=0.999)
    parser.add_argument("--q-rank-pcc", type=float, default=0.999)
    parser.add_argument("--q-output-pcc", type=float, default=0.99)
    parser.add_argument("--kv-linear-pcc", type=float, default=0.99)
    parser.add_argument("--kv-output-pcc", type=float, default=0.99)
    parser.add_argument("--cache-prep-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--cache-prep-atol", type=float, default=2e-1)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_prefill_cache_prep_smoke(
        args.snapshot_dir,
        layer=args.layer,
        seq_len=args.seq_len,
        start_pos=args.start_pos,
        max_tensors=args.max_tensors,
        max_bytes=args.max_bytes,
        cpu_only=args.cpu_only,
        device_id=args.device_id,
        attn_norm_pcc=args.attn_norm_pcc,
        q_rank_pcc=args.q_rank_pcc,
        q_output_pcc=args.q_output_pcc,
        kv_linear_pcc=args.kv_linear_pcc,
        kv_output_pcc=args.kv_output_pcc,
        cache_prep_pcc=args.cache_prep_pcc,
        rtol=args.rtol,
        atol=args.atol,
        cache_prep_atol=args.cache_prep_atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _run_ttnn_prefill_cache_prep(
    tensors: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    kv_weights: KvProjectionWeights,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    start_pos: int,
    device_id: int,
) -> dict[str, torch.Tensor]:
    import ttnn
    from models.demos.deepseek_v4_flash.real_kv_projection_smoke import _to_tt_linear_weight

    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        tt_input = ttnn.from_torch(
            activation,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_attn_norm_weight = ttnn.from_torch(
            tensors[f"layers.{layer}.attn_norm.weight"].contiguous().to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_attn_norm_output = ttnn.rms_norm(
            tt_input,
            weight=tt_attn_norm_weight,
            epsilon=config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q_module = TtAttentionProjection(
            device=device,
            weights=q_weights,
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            norm_eps=config.rms_norm_eps,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_q_rank_norm = q_module.project_q_rank(tt_attn_norm_output)
        tt_q_output = q_module.project_q_from_rank(tt_q_rank_norm)

        tt_wkv = _to_tt_linear_weight(
            kv_weights.wkv,
            device=device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv_norm_weight = ttnn.from_torch(
            kv_weights.kv_norm.contiguous().to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv_linear = ttnn.linear(tt_attn_norm_output, tt_wkv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_kv_output = ttnn.rms_norm(
            tt_kv_linear,
            weight=tt_kv_norm_weight,
            epsilon=config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        seq_len = int(activation.shape[2])
        head_dim = int(config.head_dim)
        rope_dim = int(config.qk_rope_head_dim)
        nope_dim = head_dim - rope_dim
        compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        tt_q_heads_pre_norm = ttnn.reshape(
            tt_q_output,
            [1, seq_len, int(config.num_attention_heads), head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_q_head_norm_weight = ttnn.from_torch(
            torch.ones(head_dim, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_q_heads = ttnn.rms_norm(
            tt_q_heads_pre_norm,
            weight=tt_q_head_norm_weight,
            epsilon=config.rms_norm_eps,
            compute_kernel_config=compute_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_q_nope = ttnn.slice(
            tt_q_heads,
            (0, 0, 0, 0),
            (1, seq_len, int(config.num_attention_heads), nope_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_q_rope = ttnn.slice(
            tt_q_heads,
            (0, 0, 0, nope_dim),
            (1, seq_len, int(config.num_attention_heads), head_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv_heads = ttnn.reshape(
            tt_kv_output,
            [1, seq_len, int(config.num_key_value_heads), head_dim],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv_nope = ttnn.slice(
            tt_kv_heads,
            (0, 0, 0, 0),
            (1, seq_len, int(config.num_key_value_heads), nope_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv_rope = ttnn.slice(
            tt_kv_heads,
            (0, 0, 0, nope_dim),
            (1, seq_len, int(config.num_key_value_heads), head_dim),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        outputs = {
            "attn_norm_output": ttnn.to_torch(tt_attn_norm_output).contiguous(),
            "q_rank_norm": ttnn.to_torch(tt_q_rank_norm).contiguous(),
            "q_output": ttnn.to_torch(tt_q_output).contiguous(),
            "kv_linear": ttnn.to_torch(tt_kv_linear).contiguous(),
            "kv_output": ttnn.to_torch(tt_kv_output).contiguous(),
            "q_heads_pre_norm": ttnn.to_torch(tt_q_heads_pre_norm).contiguous(),
            "q_heads": ttnn.to_torch(tt_q_heads).contiguous(),
            "q_nope": ttnn.to_torch(tt_q_nope).contiguous(),
            "q_rope": ttnn.to_torch(tt_q_rope).contiguous(),
            "kv_heads": ttnn.to_torch(tt_kv_heads).contiguous(),
            "kv_nope": ttnn.to_torch(tt_kv_nope).contiguous()[:, :, 0],
            "kv_rope": ttnn.to_torch(tt_kv_rope).contiguous()[:, :, 0],
        }
        rope_host = _host_rope_and_cache_prep_from_ttnn_splits(
            outputs["q_nope"],
            outputs["q_rope"],
            outputs["kv_nope"],
            outputs["kv_rope"],
            config=config,
            layer=layer,
            start_pos=start_pos,
        )
        outputs.update(rope_host)
        return outputs
    finally:
        ttnn.close_device(device)


def _host_rope_and_cache_prep_from_ttnn_splits(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    start_pos: int,
) -> dict[str, torch.Tensor]:
    seq_len = int(q_nope.shape[1])
    freqs_cis = precompute_deepseek_v4_rope_frequencies(config, layer=layer, seq_len=start_pos + seq_len)[
        start_pos : start_pos + seq_len
    ]
    q_rope_rotated = apply_deepseek_v4_rotary(q_rope.contiguous(), freqs_cis)
    kv_rope_rotated = apply_deepseek_v4_rotary(kv_rope.contiguous(), freqs_cis)
    q_prefill = torch.cat([q_nope, q_rope_rotated], dim=-1).contiguous()
    kv_cache_ready = torch.cat([kv_nope, kv_rope_rotated], dim=-1).contiguous()
    return {
        "q_rope_rotated": q_rope_rotated,
        "q_prefill": q_prefill,
        "kv_rope_rotated": kv_rope_rotated,
        "kv_cache_ready": kv_cache_ready,
        "sliding_window_cache": _sliding_window_cache(
            kv_cache_ready,
            sliding_window=config.sliding_window,
            start_pos=start_pos,
        ),
        "window_topk_idxs": window_topk_indices(
            config.sliding_window,
            int(kv_cache_ready.shape[0]),
            int(kv_cache_ready.shape[1]),
            start_pos,
        ).to(torch.int32),
    }


def _base_result(
    *,
    snapshot_dir: Path,
    layer: int,
    seq_len: int,
    start_pos: int,
    config: DeepSeekV4FlashConfig,
    metadata: Sequence[TensorMetadata],
    tensors: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    kv_weights: KvProjectionWeights,
    activation: torch.Tensor,
    reference: dict[str, torch.Tensor],
    max_tensors: int,
    max_bytes: int,
) -> dict[str, Any]:
    payload_bytes = _payload_byte_split(metadata)
    compress_ratio = int(config.compress_ratios[layer])
    rope_base = config.compress_rope_theta if compress_ratio else config.rope_theta
    rope_original_seq_len = int(config.rope_scaling["original_max_position_embeddings"]) if compress_ratio else 0
    return {
        "schema_version": REAL_PREFILL_CACHE_PREP_SMOKE_SCHEMA_VERSION,
        "mode": "unexecuted",
        "snapshot_dir": str(snapshot_dir),
        "layer": int(layer),
        "sequence_length": int(seq_len),
        "start_pos": int(start_pos),
        "model": {
            "hidden_size": config.hidden_size,
            "q_lora_rank": config.q_lora_rank,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "q_output_dim": config.num_attention_heads * config.head_dim,
            "kv_output_dim": config.num_key_value_heads * config.head_dim,
            "kv_nope_head_dim": config.head_dim - config.qk_rope_head_dim,
            "qk_rope_head_dim": config.qk_rope_head_dim,
            "compress_ratio": compress_ratio,
            "sliding_window": config.sliding_window,
            "rms_norm_eps": config.rms_norm_eps,
            "torch_dtype": config.torch_dtype,
            "quantization_config": config.quantization_config,
        },
        "projection_scope": {
            "path": "attention_norm -> real Q projection + real K/V projection -> q/kv reshape -> split -> RoPE",
            "not_full_attention": True,
            "sparse_attention": "excluded",
            "output_projection": "excluded",
            "cache_update": "excluded; this smoke builds host-visible prefill/cache-ready tensors only",
            "ttnn_tensor_caches": "not used",
        },
        "rope": {
            "applied": True,
            "base": float(rope_base),
            "original_seq_len": int(rope_original_seq_len),
            "factor": float(config.rope_scaling["factor"]),
            "beta_fast": float(config.rope_scaling["beta_fast"]),
            "beta_slow": float(config.rope_scaling["beta_slow"]),
            "source": "DeepSeek V4 Flash inference/model.py precompute_freqs_cis/apply_rotary_emb convention",
        },
        "selected_source_keys": [item.source_key for item in metadata],
        "loaded_tensors": [_metadata_summary(item) for item in metadata],
        "payload_bytes": payload_bytes,
        "budget": {
            "max_tensors": int(max_tensors),
            "max_bytes": int(max_bytes),
            "selected_tensors": len(metadata),
            "selected_payload_bytes": payload_bytes["total"],
        },
        "source_formats": {
            "attention_weight_block_size": list(ATTENTION_FP8_BLOCK_SIZE),
            "kv_weight_block_size": list(KV_FP8_BLOCK_SIZE),
            "decoded_tensors": {
                "wq_a": _tensor_summary(q_weights.wq_a),
                "q_norm": _tensor_summary(q_weights.q_norm),
                "wq_b": _tensor_summary(q_weights.wq_b),
                "wkv": _tensor_summary(kv_weights.wkv),
                "kv_norm": _tensor_summary(kv_weights.kv_norm),
            },
            "normalization_tensors": {
                "attn_norm": _tensor_summary(tensors[f"layers.{layer}.attn_norm.weight"].to(torch.bfloat16)),
                "q_norm": _tensor_summary(q_weights.q_norm),
                "kv_norm": _tensor_summary(kv_weights.kv_norm),
            },
        },
        "output_shapes": {
            name: list(reference[name].shape)
            for name in (
                "attn_norm_output",
                "q_rank_norm",
                "q_output",
                "kv_linear",
                "kv_output",
                "q_heads_pre_norm",
                "q_heads",
                "q_nope",
                "q_rope",
                "q_rope_rotated",
                "q_prefill",
                "kv_heads",
                "kv_nope",
                "kv_rope",
                "kv_rope_rotated",
                "kv_cache_ready",
                "sliding_window_cache",
                "window_topk_idxs",
            )
        },
        "cache_prep": {
            "q_shape_contract": "[batch, seq_len, num_attention_heads, head_dim]",
            "kv_cache_ready_shape_contract": "[batch, cache_len, head_dim]",
            "split_boundary": {
                "kv_nope_head_dim": config.head_dim - config.qk_rope_head_dim,
                "qk_rope_head_dim": config.qk_rope_head_dim,
            },
            "window_topk_source": "cpu_reference.window_topk_indices",
            "compressed_cache": (
                "not materialized in this slice; seq_len is below layer compress_ratio"
                if compress_ratio and seq_len < compress_ratio
                else "excluded from this first Q/KV cache-prep slice"
            ),
        },
        "host_boundaries": [
            {
                "name": "projection_fp8_decode_to_bf16",
                "location": "before TTNN projection modules",
                "description": "selected real FP8 query and K/V weights are decoded on host to BF16",
            },
            {
                "name": "activation_host_to_device",
                "location": "smoke input",
                "description": "deterministic BF16 activation is generated on host and uploaded to TTNN",
            },
            {
                "name": "projection_and_split_readback",
                "location": "after TTNN projection/reshape/split",
                "description": "TTNN outputs are copied to host for numerical smoke accuracy checks",
            },
            {
                "name": "rope_cache_prep_host",
                "location": "after TTNN q/kv split",
                "description": "local DeepSeek V4 Flash complex RoPE and final cache-ready concat run on host",
            },
        ],
        "reference_ops": [
            "decode_attention_projection_weights",
            "decode_kv_projection_weight",
            "torch.rms_norm_reference(attn_norm)",
            "torch.linear(wq_a)",
            "torch.rms_norm_reference(q_norm)",
            "torch.linear(wq_b)",
            "torch.linear(wkv)",
            "torch.rms_norm_reference(kv_norm)",
            "torch.reshape/split(q/kv)",
            "torch.unit_rms_norm(q_heads)",
            "DeepSeek V4 Flash RoPE(q_rope, kv_rope)",
            "window_topk_indices",
        ],
        "ttnn_ops": [],
        "inputs": {"activation": _tensor_summary(activation)},
        "reference": {
            name: _tensor_summary(tensor) for name, tensor in reference.items() if tensor.dtype.is_floating_point
        },
        "reference_int": {"window_topk_idxs": _int_tensor_summary(reference["window_topk_idxs"])},
        "ttnn": {},
        "host_cache_prep": {},
        "accuracy": {},
        "passed": False,
    }


def _payload_byte_split(metadata: Sequence[TensorMetadata]) -> dict[str, int]:
    split = {
        "attn_norm": 0,
        "q_norm": 0,
        "kv_norm": 0,
        "wq_a_weight": 0,
        "wq_a_scale": 0,
        "wq_b_weight": 0,
        "wq_b_scale": 0,
        "wkv_weight": 0,
        "wkv_scale": 0,
    }
    for item in metadata:
        key = item.canonical_key
        if key.endswith(".attn_norm.weight"):
            split["attn_norm"] += item.nbytes
        elif key.endswith(".attn.q_norm.weight"):
            split["q_norm"] += item.nbytes
        elif key.endswith(".attn.kv_norm.weight"):
            split["kv_norm"] += item.nbytes
        elif key.endswith(".attn.wq_a.weight"):
            split["wq_a_weight"] += item.nbytes
        elif key.endswith(".attn.wq_a.scale"):
            split["wq_a_scale"] += item.nbytes
        elif key.endswith(".attn.wq_b.weight"):
            split["wq_b_weight"] += item.nbytes
        elif key.endswith(".attn.wq_b.scale"):
            split["wq_b_scale"] += item.nbytes
        elif key.endswith(".attn.wkv.weight"):
            split["wkv_weight"] += item.nbytes
        elif key.endswith(".attn.wkv.scale"):
            split["wkv_scale"] += item.nbytes
        else:
            raise ValueError(f"Unexpected tensor in prefill cache-prep slice: {key}")
    split["norms"] = split["attn_norm"] + split["q_norm"] + split["kv_norm"]
    split["q_low_rank"] = split["wq_a_weight"] + split["wq_a_scale"]
    split["q_output"] = split["wq_b_weight"] + split["wq_b_scale"]
    split["kv_projection"] = split["wkv_weight"] + split["wkv_scale"]
    split["weights"] = split["wq_a_weight"] + split["wq_b_weight"] + split["wkv_weight"]
    split["scales"] = split["wq_a_scale"] + split["wq_b_scale"] + split["wkv_scale"]
    split["total"] = split["norms"] + split["q_low_rank"] + split["q_output"] + split["kv_projection"]
    return split


def _precompute_freqs_cis(
    *,
    dim: int,
    seq_len: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    if dim <= 0 or dim % 2 != 0:
        raise ValueError(f"RoPE dim must be a positive even integer, got {dim}")

    def find_correction_dim(num_rotations: float) -> float:
        return dim * math.log(original_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range() -> tuple[int, int]:
        low = math.floor(find_correction_dim(beta_fast))
        high = math.ceil(find_correction_dim(beta_slow))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_value: int, max_value: int, ramp_dim: int) -> torch.Tensor:
        if min_value == max_value:
            max_value += 0.001
        ramp = (torch.arange(ramp_dim, dtype=torch.float32) - min_value) / (max_value - min_value)
        return torch.clamp(ramp, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range()
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _unit_rms_norm(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    return (x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)).to(x.dtype).contiguous()


def _sliding_window_cache(kv_cache_ready: torch.Tensor, *, sliding_window: int, start_pos: int) -> torch.Tensor:
    if start_pos != 0:
        raise ValueError("This prefill cache-prep smoke only supports start_pos=0")
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    if kv_cache_ready.shape[1] <= sliding_window:
        return kv_cache_ready.contiguous()
    cutoff = kv_cache_ready.shape[1] % sliding_window
    window = kv_cache_ready.new_empty(kv_cache_ready.shape[0], sliding_window, kv_cache_ready.shape[-1])
    left, right = kv_cache_ready[:, -sliding_window:].split([sliding_window - cutoff, cutoff], dim=1)
    window[:, cutoff:sliding_window] = left
    window[:, :cutoff] = right
    return window.contiguous()


def _validate_projected_q_kv(
    q_output: torch.Tensor,
    kv_output: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
) -> None:
    if q_output.ndim != 4 or q_output.shape[1] != 1:
        raise ValueError(f"Expected q_output shape [batch, 1, seq_len, heads * head_dim], got {tuple(q_output.shape)}")
    if kv_output.ndim != 4 or kv_output.shape[1] != 1:
        raise ValueError(f"Expected kv_output shape [batch, 1, seq_len, head_dim], got {tuple(kv_output.shape)}")
    expected_q_width = int(config.num_attention_heads) * int(config.head_dim)
    expected_kv_width = int(config.num_key_value_heads) * int(config.head_dim)
    if q_output.shape[-1] != expected_q_width:
        raise ValueError(f"Expected q_output width {expected_q_width}, got {q_output.shape[-1]}")
    if kv_output.shape[-1] != expected_kv_width:
        raise ValueError(f"Expected kv_output width {expected_kv_width}, got {kv_output.shape[-1]}")
    if q_output.shape[0] != kv_output.shape[0] or q_output.shape[2] != kv_output.shape[2]:
        raise ValueError(
            f"Q/KV batch and seq dims must match, got {tuple(q_output.shape)} and {tuple(kv_output.shape)}"
        )


def _metadata_summary(item: TensorMetadata) -> dict[str, Any]:
    return {
        "canonical_key": item.canonical_key,
        "source_key": item.source_key,
        "shard": item.shard_name,
        "dtype": item.dtype,
        "shape": list(item.shape),
        "nbytes": item.nbytes,
    }


def _tensor_summary(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": 0}
    tensor_float = tensor.float()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(tensor_float.min().item()),
        "max": float(tensor_float.max().item()),
        "mean": float(tensor_float.mean().item()),
    }


def _int_tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    if tensor.numel() == 0:
        return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "numel": 0}
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": int(tensor.min().item()),
        "max": int(tensor.max().item()),
    }


def _accuracy_summary(
    expected: torch.Tensor,
    actual: torch.Tensor,
    *,
    pcc_threshold: float,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if tuple(actual.shape) != tuple(expected.shape):
        return {
            "passed": False,
            "reason": f"shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}",
        }

    expected_float = expected.float()
    actual_float = actual.float()
    abs_diff = (actual_float - expected_float).abs()
    pcc = _pcc(expected_float, actual_float)
    allclose = bool(torch.allclose(actual_float, expected_float, rtol=rtol, atol=atol))
    return {
        "passed": bool(pcc >= pcc_threshold and allclose),
        "pcc": float(pcc),
        "pcc_threshold": float(pcc_threshold),
        "allclose": allclose,
        "rtol": float(rtol),
        "atol": float(atol),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
    }


def _int_equality_summary(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, Any]:
    if tuple(actual.shape) != tuple(expected.shape):
        return {
            "passed": False,
            "reason": f"shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}",
        }
    equal = bool(torch.equal(actual.cpu(), expected.cpu()))
    return {
        "passed": equal,
        "equal": equal,
        "mismatches": int((actual.cpu() != expected.cpu()).sum().item()),
    }


def _pcc(expected: torch.Tensor, actual: torch.Tensor) -> float:
    expected_flat = expected.reshape(-1).float()
    actual_flat = actual.reshape(-1).float()
    expected_centered = expected_flat - expected_flat.mean()
    actual_centered = actual_flat - actual_flat.mean()
    denominator = torch.linalg.vector_norm(expected_centered) * torch.linalg.vector_norm(actual_centered)
    if float(denominator.item()) == 0.0:
        return 1.0 if torch.allclose(expected_flat, actual_flat) else 0.0
    return float((expected_centered * actual_centered).sum().div(denominator).item())


def _validate_smoke_args(
    *,
    layer: int,
    seq_len: int,
    start_pos: int,
    max_tensors: int,
    max_bytes: int,
    pcc_thresholds: Mapping[str, float],
) -> None:
    if layer < 0:
        raise ValueError(f"layer must be non-negative, got {layer}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if start_pos != 0:
        raise ValueError("This prefill cache-prep smoke only supports start_pos=0")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    for name, value in pcc_thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


if __name__ == "__main__":
    main()
