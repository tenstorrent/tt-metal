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
import torch.nn.functional as F

from models.demos.deepseek_v4_flash.config import DeepSeekV4FlashConfig
from models.demos.deepseek_v4_flash.cpu_reference import compress_topk_indices, sparse_attention
from models.demos.deepseek_v4_flash.real_attention_projection_smoke import (
    ATTENTION_FP8_BLOCK_SIZE,
    ATTENTION_TTNN_TILE_MULTIPLE,
    decode_real_attention_projection_weights,
    deterministic_attention_activation,
    layer_attention_projection_keys,
)
from models.demos.deepseek_v4_flash.real_checkpoint_loader import RealCheckpointTensorIndex, TensorMetadata
from models.demos.deepseek_v4_flash.real_kv_projection_smoke import (
    KV_FP8_BLOCK_SIZE,
    KvProjectionWeights,
    decode_real_kv_projection_weights,
    layer_kv_projection_keys,
)
from models.demos.deepseek_v4_flash.real_prefill_cache_prep_smoke import (
    DEFAULT_LAYER_PREFILL_CACHE_PREP_LAYER,
    DEFAULT_SEQUENCE_LENGTH,
    PREFILL_CACHE_PREP_TTNN_TILE_MULTIPLE,
    _accuracy_summary,
    _int_equality_summary,
    _int_tensor_summary,
    _metadata_summary,
    _run_ttnn_prefill_cache_prep,
    _tensor_summary,
    apply_deepseek_v4_rotary,
    build_torch_prefill_cache_prep_reference,
    precompute_deepseek_v4_rope_frequencies,
)
from models.demos.deepseek_v4_flash.real_shared_expert_smoke import decode_fp8_block_scaled_weight
from models.demos.deepseek_v4_flash.ttnn_attention_projection import (
    AttentionProjectionWeights,
    TtAttentionProjection,
    grouped_output_projection_a,
)
from models.demos.deepseek_v4_flash.ttnn_sparse_attention import TtSparsePrefillAttention

REAL_PREFILL_ATTENTION_RUNTIME_SMOKE_SCHEMA_VERSION = 1
DEFAULT_PREFILL_ATTENTION_RUNTIME_LAYER = DEFAULT_LAYER_PREFILL_CACHE_PREP_LAYER
DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_TENSORS = 18
DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES = 128 * 1024 * 1024
PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE = ATTENTION_TTNN_TILE_MULTIPLE
ATTENTION_OUTPUT_PROJECTIONS = ("wo_a", "wo_b")
_DIRECT_WEIGHT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def run_real_prefill_attention_runtime_smoke(
    snapshot_dir: str | Path,
    *,
    layer: int = DEFAULT_PREFILL_ATTENTION_RUNTIME_LAYER,
    seq_len: int = DEFAULT_SEQUENCE_LENGTH,
    start_pos: int = 0,
    max_tensors: int = DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_TENSORS,
    max_bytes: int = DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES,
    cpu_only: bool = False,
    device_id: int = 0,
    attn_norm_pcc: float = 0.999,
    q_rank_pcc: float = 0.999,
    q_output_pcc: float = 0.99,
    kv_linear_pcc: float = 0.99,
    kv_output_pcc: float = 0.99,
    cache_prep_pcc: float = 0.99,
    attention_pcc: float = 0.99,
    output_pcc: float = 0.99,
    rtol: float = 8e-2,
    atol: float = 8e-2,
    cache_prep_atol: float = 2e-1,
    output_atol: float = 2e-1,
) -> dict[str, Any]:
    """Run the first real layer attention runtime slice after Q/KV cache prep."""

    _validate_runtime_smoke_args(
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
            "attention_pcc": attention_pcc,
            "output_pcc": output_pcc,
        },
    )
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    _validate_runtime_config(config, layer=layer, seq_len=seq_len, start_pos=start_pos)
    tensors, metadata = load_real_prefill_attention_runtime_slice(
        snapshot_dir,
        config=config,
        layer=layer,
        max_tensors=max_tensors,
        max_bytes=max_bytes,
    )
    q_weights = decode_real_prefill_attention_projection_weights(tensors, config=config, layer=layer)
    kv_weights = decode_real_kv_projection_weights(tensors, config=config, layer=layer)

    activation = deterministic_attention_activation(hidden_size=config.hidden_size, seq_len=seq_len)
    reference = build_torch_prefill_attention_runtime_reference(
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

    if seq_len % PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE != 0:
        raise ValueError(
            f"TTNN smoke seq_len must be a multiple of {PREFILL_ATTENTION_RUNTIME_TTNN_TILE_MULTIPLE}, "
            f"got {seq_len}"
        )

    cache_q_weights = _query_projection_weights_only(q_weights)
    cache_outputs = _run_ttnn_prefill_cache_prep(
        tensors,
        cache_q_weights,
        kv_weights,
        config=config,
        layer=layer,
        activation=activation,
        start_pos=start_pos,
        device_id=device_id,
    )
    ttnn_outputs = _run_ttnn_attention_runtime_from_cache_boundary(
        cache_outputs,
        q_weights,
        tensors[f"layers.{layer}.attn.attn_sink"],
        config=config,
        layer=layer,
        start_pos=start_pos,
        device_id=device_id,
    )
    ttnn_outputs.update(cache_outputs)
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
        "ttnn.reshape/slice(q and kv heads)",
        "TtSparsePrefillAttention",
        "ttnn.from_torch(attention_output_after_inverse_rope)",
        "ttnn.linear(wo_b)",
    ]
    result["ttnn"] = {
        name: _tensor_summary(tensor)
        for name, tensor in ttnn_outputs.items()
        if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point
    }
    result["ttnn_int"] = {
        "window_topk_idxs": _int_tensor_summary(ttnn_outputs["window_topk_idxs"]),
        "compress_topk_idxs": _int_tensor_summary(ttnn_outputs["compress_topk_idxs"]),
        "runtime_topk_idxs": _int_tensor_summary(ttnn_outputs["runtime_topk_idxs"]),
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
        "q_prefill",
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
    for name in ("attention_output_rotary", "attention_output", "attention_output_flat", "output_rank"):
        accuracy[name] = _accuracy_summary(
            reference[name],
            ttnn_outputs[name],
            pcc_threshold=attention_pcc,
            rtol=rtol,
            atol=output_atol,
        )
    accuracy["attention_output_projected"] = _accuracy_summary(
        reference["attention_output_projected"],
        ttnn_outputs["attention_output_projected"],
        pcc_threshold=output_pcc,
        rtol=rtol,
        atol=output_atol,
    )
    for name in ("window_topk_idxs", "compress_topk_idxs", "runtime_topk_idxs"):
        accuracy[name] = _int_equality_summary(reference[name], ttnn_outputs[name])
    result["accuracy"] = accuracy
    result["passed"] = all(item["passed"] for item in result["accuracy"].values())
    return result


def layer_prefill_attention_runtime_keys(
    index: RealCheckpointTensorIndex,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
) -> list[str]:
    """Return the strict tensor set for the first real layer prefill-attention runtime slice."""

    keys: list[str] = []
    for key in layer_attention_projection_keys(index, layer=layer) + layer_kv_projection_keys(index, layer=layer):
        if key not in keys:
            keys.append(key)

    prefix = f"layers.{layer}.attn"
    for key in (f"{prefix}.attn_sink",):
        index.location(key)
        keys.append(key)

    if config.compress_ratios[layer]:
        compressor_keys = [
            f"{prefix}.compressor.ape",
            f"{prefix}.compressor.wkv.weight",
            f"{prefix}.compressor.wgate.weight",
            f"{prefix}.compressor.norm.weight",
        ]
        for key in compressor_keys:
            index.location(key)
            keys.append(key)
        if config.compress_ratios[layer] == 4:
            keys.extend(_layer_prefill_indexer_keys(index, layer=layer))

    for key in _layer_attention_output_projection_keys(index, layer=layer):
        if key not in keys:
            keys.append(key)
    return keys


def load_real_prefill_attention_runtime_slice(
    snapshot_dir: str | Path,
    *,
    config: DeepSeekV4FlashConfig | None = None,
    layer: int,
    max_tensors: int = DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_TENSORS,
    max_bytes: int = DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES,
) -> tuple[dict[str, torch.Tensor], list[TensorMetadata]]:
    snapshot_dir = Path(snapshot_dir).expanduser().resolve()
    if config is None:
        config = DeepSeekV4FlashConfig.from_model_path(snapshot_dir)
    index = RealCheckpointTensorIndex.from_snapshot(snapshot_dir)
    keys = layer_prefill_attention_runtime_keys(index, config=config, layer=layer)
    return index.load_tensors(keys, max_tensors=max_tensors, max_bytes=max_bytes)


def decode_real_prefill_attention_projection_weights(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    dtype: torch.dtype = torch.bfloat16,
) -> AttentionProjectionWeights:
    query_weights = decode_real_attention_projection_weights(tensors, config=config, layer=layer, dtype=dtype)
    wo_a = _decode_attention_projection_weight(tensors, config=config, layer=layer, projection="wo_a", dtype=dtype)
    wo_b = _decode_attention_projection_weight(tensors, config=config, layer=layer, projection="wo_b", dtype=dtype)
    validate_real_attention_output_projection_slice(
        {"wo_a": wo_a, "wo_b": wo_b},
        config=config,
    )
    return AttentionProjectionWeights(
        wq_a=query_weights.wq_a,
        q_norm=query_weights.q_norm,
        wq_b=query_weights.wq_b,
        wo_a=wo_a,
        wo_b=wo_b,
    )


def validate_real_attention_output_projection_slice(
    decoded: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
) -> None:
    q_output_dim = int(config.num_attention_heads) * int(config.head_dim)
    if q_output_dim % int(config.o_groups) != 0:
        raise ValueError(f"q output dim {q_output_dim} must be divisible by o_groups {config.o_groups}")
    expected_shapes = {
        "wo_a": (int(config.o_groups) * int(config.o_lora_rank), q_output_dim // int(config.o_groups)),
        "wo_b": (int(config.hidden_size), int(config.o_groups) * int(config.o_lora_rank)),
    }
    for name, expected_shape in expected_shapes.items():
        if name not in decoded:
            raise KeyError(f"Missing decoded output projection tensor {name!r}")
        if tuple(decoded[name].shape) != expected_shape:
            raise ValueError(f"Expected decoded {name} shape {expected_shape}, got {tuple(decoded[name].shape)}")


def build_torch_prefill_attention_runtime_reference(
    tensors: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    kv_weights: KvProjectionWeights,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    activation: torch.Tensor,
    start_pos: int = 0,
) -> dict[str, torch.Tensor]:
    cache_reference = build_torch_prefill_cache_prep_reference(
        tensors,
        q_weights,
        kv_weights,
        config=config,
        layer=layer,
        activation=activation,
        start_pos=start_pos,
    )
    batch_size, seq_len, num_heads, head_dim = cache_reference["q_prefill"].shape
    attn_sink = tensors[f"layers.{layer}.attn.attn_sink"].float().contiguous()
    window_topk_idxs = cache_reference["window_topk_idxs"]
    compress_topk_idxs = _empty_compress_topk_indices(
        int(config.compress_ratios[layer]),
        batch_size=batch_size,
        seq_len=seq_len,
        start_pos=start_pos,
        offset=seq_len,
    )
    runtime_topk_idxs = torch.cat([window_topk_idxs, compress_topk_idxs], dim=-1).to(torch.int32)
    attention_output_rotary = sparse_attention(
        cache_reference["q_prefill"],
        cache_reference["kv_cache_ready"],
        attn_sink,
        runtime_topk_idxs,
        head_dim**-0.5,
    ).contiguous()
    attention_output = _inverse_attention_rope(
        attention_output_rotary,
        config=config,
        layer=layer,
        start_pos=start_pos,
    )
    attention_output_flat = attention_output.reshape(batch_size, seq_len, num_heads * head_dim).to(torch.bfloat16)
    if q_weights.wo_a is None or q_weights.wo_b is None:
        raise ValueError("Output projection weights are required for prefill attention runtime reference")
    output_rank = grouped_output_projection_a(attention_output_flat, q_weights.wo_a, o_groups=int(config.o_groups))
    attention_output_projected = F.linear(output_rank.float(), q_weights.wo_b.float()).unsqueeze(1)
    return {
        **cache_reference,
        "compress_topk_idxs": compress_topk_idxs,
        "runtime_topk_idxs": runtime_topk_idxs,
        "attention_output_rotary": attention_output_rotary,
        "attention_output": attention_output,
        "attention_output_flat": attention_output_flat,
        "output_rank": output_rank,
        "attention_output_projected": attention_output_projected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real DeepSeek V4 Flash prefill attention runtime TTNN smoke path."
    )
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=DEFAULT_PREFILL_ATTENTION_RUNTIME_LAYER)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--start-pos", type=int, default=0)
    parser.add_argument("--max-tensors", type=int, default=DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_TENSORS)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_PREFILL_ATTENTION_RUNTIME_MAX_BYTES)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--attn-norm-pcc", type=float, default=0.999)
    parser.add_argument("--q-rank-pcc", type=float, default=0.999)
    parser.add_argument("--q-output-pcc", type=float, default=0.99)
    parser.add_argument("--kv-linear-pcc", type=float, default=0.99)
    parser.add_argument("--kv-output-pcc", type=float, default=0.99)
    parser.add_argument("--cache-prep-pcc", type=float, default=0.99)
    parser.add_argument("--attention-pcc", type=float, default=0.99)
    parser.add_argument("--output-pcc", type=float, default=0.99)
    parser.add_argument("--rtol", type=float, default=8e-2)
    parser.add_argument("--atol", type=float, default=8e-2)
    parser.add_argument("--cache-prep-atol", type=float, default=2e-1)
    parser.add_argument("--output-atol", type=float, default=2e-1)
    parser.add_argument("--verbose-logs", action="store_true")
    args = parser.parse_args()

    if not args.verbose_logs:
        os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")
        os.environ.setdefault("LOGGER_LEVEL", "Warning")

    result = run_real_prefill_attention_runtime_smoke(
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
        attention_pcc=args.attention_pcc,
        output_pcc=args.output_pcc,
        rtol=args.rtol,
        atol=args.atol,
        cache_prep_atol=args.cache_prep_atol,
        output_atol=args.output_atol,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


def _run_ttnn_attention_runtime_from_cache_boundary(
    cache_outputs: Mapping[str, torch.Tensor],
    q_weights: AttentionProjectionWeights,
    attn_sink: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    start_pos: int,
    device_id: int,
) -> dict[str, torch.Tensor]:
    import ttnn

    q_prefill = cache_outputs["q_prefill"].contiguous()
    kv_cache_ready = cache_outputs["kv_cache_ready"].contiguous()
    batch_size, seq_len, num_heads, head_dim = q_prefill.shape
    compress_topk_idxs = _empty_compress_topk_indices(
        int(config.compress_ratios[layer]),
        batch_size=batch_size,
        seq_len=seq_len,
        start_pos=start_pos,
        offset=seq_len,
    )
    runtime_topk_idxs = torch.cat([cache_outputs["window_topk_idxs"], compress_topk_idxs], dim=-1).to(torch.int32)
    device = ttnn.open_device(device_id=device_id, num_command_queues=1)
    try:
        tt_q = ttnn.from_torch(
            q_prefill.reshape(batch_size, seq_len, num_heads * head_dim).unsqueeze(1).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_kv = ttnn.from_torch(
            kv_cache_ready.unsqueeze(1).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        sparse_module = TtSparsePrefillAttention(
            device=device,
            num_heads=int(config.num_attention_heads),
            head_dim=int(config.head_dim),
            softmax_scale=int(config.head_dim) ** -0.5,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_attention_output_rotary = sparse_module(
            tt_q,
            tt_kv,
            attn_sink=attn_sink,
            topk_idxs=runtime_topk_idxs,
        )
        attention_output_rotary_flat = ttnn.to_torch(tt_attention_output_rotary).contiguous()[:, 0]
        attention_output_rotary = attention_output_rotary_flat.reshape(batch_size, seq_len, num_heads, head_dim)
        attention_output = _inverse_attention_rope(
            attention_output_rotary,
            config=config,
            layer=layer,
            start_pos=start_pos,
        )
        attention_output_flat = attention_output.reshape(batch_size, seq_len, num_heads * head_dim).to(torch.bfloat16)
        output_rank = grouped_output_projection_a(attention_output_flat, q_weights.wo_a, o_groups=int(config.o_groups))
        tt_attention_output = ttnn.from_torch(
            attention_output_flat.unsqueeze(1).to(torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        projection_module = TtAttentionProjection(
            device=device,
            weights=q_weights,
            hidden_size=int(config.hidden_size),
            q_lora_rank=int(config.q_lora_rank),
            num_heads=int(config.num_attention_heads),
            head_dim=int(config.head_dim),
            norm_eps=float(config.rms_norm_eps),
            o_groups=int(config.o_groups),
            o_lora_rank=int(config.o_lora_rank),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_projected = projection_module.project_output(tt_attention_output)
        return {
            "compress_topk_idxs": compress_topk_idxs,
            "runtime_topk_idxs": runtime_topk_idxs,
            "attention_output_rotary": attention_output_rotary.contiguous(),
            "attention_output": attention_output.contiguous(),
            "attention_output_flat": attention_output_flat.contiguous(),
            "output_rank": output_rank.contiguous(),
            "attention_output_projected": ttnn.to_torch(tt_projected).contiguous(),
        }
    finally:
        ttnn.close_device(device)


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
    indexer_expected = compress_ratio == 4
    indexer_selected = any(".attn.indexer." in item.canonical_key for item in metadata)
    compressor_selected = any(".attn.compressor." in item.canonical_key for item in metadata)
    compressed_tokens = int(reference["compress_topk_idxs"].shape[-1])
    return {
        "schema_version": REAL_PREFILL_ATTENTION_RUNTIME_SMOKE_SCHEMA_VERSION,
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
            "o_groups": config.o_groups,
            "o_lora_rank": config.o_lora_rank,
            "output_rank_dim": config.o_groups * config.o_lora_rank,
            "compress_ratio": compress_ratio,
            "sliding_window": config.sliding_window,
            "index_n_heads": config.index_n_heads,
            "index_head_dim": config.index_head_dim,
            "index_topk": config.index_topk,
            "rms_norm_eps": config.rms_norm_eps,
            "torch_dtype": config.torch_dtype,
            "quantization_config": config.quantization_config,
        },
        "runtime_scope": {
            "path": (
                "attn_norm -> real Q/KV cache prep -> sliding-window sparse attention -> "
                "inverse RoPE -> grouped wo_a -> wo_b"
            ),
            "not_full_decoder_layer": True,
            "residual": "excluded",
            "ffn": "excluded",
            "cache_update": "excluded; seq_len=32 path uses host-visible first-prefill tensors only",
            "ttnn_tensor_caches": "not used",
        },
        "sparse_attention_inputs": {
            "window_topk_source": "cpu_reference.window_topk_indices",
            "compress_topk_source": (
                "none for this slice because seq_len < compress_ratio"
                if compressed_tokens == 0
                else "cpu_reference.compress_topk_indices"
            ),
            "runtime_topk_concat": True,
            "compressor_tensors_loaded": compressor_selected,
            "compressor_executed": False,
            "compressor_skip_reason": (
                f"seq_len {seq_len} is below layer compress_ratio {compress_ratio}"
                if compress_ratio and seq_len < compress_ratio
                else "compressed prefill contribution is outside this focused layer-3 smoke"
            ),
            "indexer_expected_for_layer": indexer_expected,
            "indexer_tensors_loaded": indexer_selected,
            "indexer_skip_reason": (
                "layer compress_ratio is 128; HF only has learned indexer tensors for compress_ratio=4 layers"
                if not indexer_expected
                else "indexer is not needed while seq_len is below compress_ratio"
            ),
            "attention_cache_source": "kv_cache_ready from the existing real prefill/cache-prep path",
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
                "wo_a": _tensor_summary(q_weights.wo_a),
                "wo_b": _tensor_summary(q_weights.wo_b),
            },
            "normalization_tensors": {
                "attn_norm": _tensor_summary(tensors[f"layers.{layer}.attn_norm.weight"].to(torch.bfloat16)),
                "q_norm": _tensor_summary(q_weights.q_norm),
                "kv_norm": _tensor_summary(kv_weights.kv_norm),
            },
            "attention_sink": _tensor_summary(tensors[f"layers.{layer}.attn.attn_sink"]),
        },
        "output_shapes": {
            name: list(reference[name].shape)
            for name in (
                "q_prefill",
                "kv_cache_ready",
                "window_topk_idxs",
                "compress_topk_idxs",
                "runtime_topk_idxs",
                "attention_output_rotary",
                "attention_output",
                "attention_output_flat",
                "output_rank",
                "attention_output_projected",
            )
        },
        "host_boundaries": [
            {
                "name": "projection_fp8_decode_to_bf16",
                "location": "before TTNN projection modules",
                "description": "selected real FP8 Q, K/V, wo_a, and wo_b weights are decoded on host to BF16",
            },
            {
                "name": "activation_host_to_device",
                "location": "smoke input",
                "description": "deterministic BF16 activation is generated on host and uploaded to TTNN",
            },
            {
                "name": "cache_prep_readback",
                "location": "after real Q/KV prefill cache prep",
                "description": "q_prefill and kv_cache_ready are copied to host at the existing cache-prep boundary",
            },
            {
                "name": "topk_host",
                "location": "before sparse attention",
                "description": "sliding-window top-k indices are generated on host",
            },
            {
                "name": "sparse_attention_host_fallback",
                "location": "inside TtSparsePrefillAttention",
                "description": "indexed gather, attention-sink softmax, and weighted reduction run on host",
            },
            {
                "name": "inverse_rope_host",
                "location": "after sparse attention",
                "description": "attention output RoPE dimensions are inverse-rotated on host before output projection",
            },
            {
                "name": "grouped_wo_a_host",
                "location": "inside output projection",
                "description": "grouped wo_a projection runs on host before TTNN wo_b",
            },
            {
                "name": "output_readback",
                "location": "after wo_b",
                "description": "final attention block output is copied to host for smoke accuracy",
            },
        ],
        "reference_ops": [
            "decode_attention_projection_weights",
            "decode_kv_projection_weight",
            "decode_attention_output_projection_weights",
            "torch.rms_norm_reference(attn_norm)",
            "torch.linear(wq_a)",
            "torch.rms_norm_reference(q_norm)",
            "torch.linear(wq_b)",
            "torch.linear(wkv)",
            "torch.rms_norm_reference(kv_norm)",
            "torch.reshape/split(q/kv)",
            "DeepSeek V4 Flash RoPE(q_rope, kv_rope)",
            "window_topk_indices",
            "sparse_attention",
            "DeepSeek V4 Flash inverse RoPE(attention_output)",
            "grouped_output_projection_a",
            "torch.linear(wo_b)",
        ],
        "ttnn_ops": [],
        "inputs": {"activation": _tensor_summary(activation)},
        "reference": {
            name: _tensor_summary(tensor) for name, tensor in reference.items() if tensor.dtype.is_floating_point
        },
        "reference_int": {
            "window_topk_idxs": _int_tensor_summary(reference["window_topk_idxs"]),
            "compress_topk_idxs": _int_tensor_summary(reference["compress_topk_idxs"]),
            "runtime_topk_idxs": _int_tensor_summary(reference["runtime_topk_idxs"]),
        },
        "ttnn": {},
        "ttnn_int": {},
        "accuracy": {},
        "passed": False,
    }


def _layer_attention_output_projection_keys(index: RealCheckpointTensorIndex, *, layer: int) -> list[str]:
    prefix = f"layers.{layer}.attn"
    projection_weight_keys = [f"{prefix}.{projection}.weight" for projection in ATTENTION_OUTPUT_PROJECTIONS]
    projection_metadata = {item.canonical_key: item for item in index.metadata_for_keys(projection_weight_keys)}
    keys: list[str] = []
    for projection, weight_key in zip(ATTENTION_OUTPUT_PROJECTIONS, projection_weight_keys):
        keys.append(weight_key)
        if _metadata_dtype_is_fp8(projection_metadata[weight_key].dtype):
            scale_key = f"{prefix}.{projection}.scale"
            index.location(scale_key)
            keys.append(scale_key)
    return keys


def _layer_prefill_indexer_keys(index: RealCheckpointTensorIndex, *, layer: int) -> list[str]:
    prefix = f"layers.{layer}.attn.indexer"
    keys = [
        f"{prefix}.wq_b.weight",
        f"{prefix}.weights_proj.weight",
        f"{prefix}.compressor.wkv.weight",
        f"{prefix}.compressor.wgate.weight",
        f"{prefix}.compressor.ape",
        f"{prefix}.compressor.norm.weight",
    ]
    for key in keys:
        index.location(key)
    metadata = {item.canonical_key: item for item in index.metadata_for_keys([f"{prefix}.wq_b.weight"])}
    if _metadata_dtype_is_fp8(metadata[f"{prefix}.wq_b.weight"].dtype):
        scale_key = f"{prefix}.wq_b.scale"
        index.location(scale_key)
        keys.insert(1, scale_key)
    return keys


def _decode_attention_projection_weight(
    tensors: Mapping[str, torch.Tensor],
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    projection: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if projection not in ATTENTION_OUTPUT_PROJECTIONS:
        raise ValueError(f"Unsupported output projection {projection!r}")
    prefix = f"layers.{layer}.attn.{projection}"
    weight = tensors[f"{prefix}.weight"]
    scale = tensors.get(f"{prefix}.scale")
    expected_shape = _output_projection_shape(config, projection=projection)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(f"Expected {prefix}.weight shape {expected_shape}, got {tuple(weight.shape)}")
    if _is_fp8_tensor(weight, scale):
        expected_scale_shape = (
            math.ceil(weight.shape[0] / ATTENTION_FP8_BLOCK_SIZE[0]),
            math.ceil(weight.shape[1] / ATTENTION_FP8_BLOCK_SIZE[1]),
        )
        if tuple(scale.shape) != expected_scale_shape:
            raise ValueError(f"Expected {prefix}.scale shape {expected_scale_shape}, got {tuple(scale.shape)}")
        return decode_fp8_block_scaled_weight(
            weight,
            scale,
            block_size=ATTENTION_FP8_BLOCK_SIZE,
            dtype=dtype,
        )
    if weight.dtype in _DIRECT_WEIGHT_DTYPES:
        if scale is not None and scale.ndim != 2:
            raise ValueError(f"Expected {prefix}.scale to be rank 2, got {tuple(scale.shape)}")
        return weight.contiguous().to(dtype)
    scale_dtype = None if scale is None else scale.dtype
    raise TypeError(
        f"Unsupported attention output projection format for {prefix}: "
        f"weight dtype {weight.dtype}, scale dtype {scale_dtype}"
    )


def _output_projection_shape(config: DeepSeekV4FlashConfig, *, projection: str) -> tuple[int, int]:
    q_output_dim = int(config.num_attention_heads) * int(config.head_dim)
    output_rank_dim = int(config.o_groups) * int(config.o_lora_rank)
    if projection == "wo_a":
        return output_rank_dim, q_output_dim // int(config.o_groups)
    if projection == "wo_b":
        return int(config.hidden_size), output_rank_dim
    raise ValueError(f"Unsupported output projection {projection!r}")


def _inverse_attention_rope(
    attention_output: torch.Tensor,
    *,
    config: DeepSeekV4FlashConfig,
    layer: int,
    start_pos: int,
) -> torch.Tensor:
    head_dim = int(config.head_dim)
    rope_dim = int(config.qk_rope_head_dim)
    nope_dim = head_dim - rope_dim
    attention_nope, attention_rope = attention_output.split([nope_dim, rope_dim], dim=-1)
    seq_len = int(attention_output.shape[1])
    freqs_cis = precompute_deepseek_v4_rope_frequencies(config, layer=layer, seq_len=start_pos + seq_len)[
        start_pos : start_pos + seq_len
    ]
    attention_rope_unrotated = apply_deepseek_v4_rotary(attention_rope.contiguous(), freqs_cis, inverse=True)
    return torch.cat([attention_nope, attention_rope_unrotated], dim=-1).contiguous()


def _query_projection_weights_only(weights: AttentionProjectionWeights) -> AttentionProjectionWeights:
    return AttentionProjectionWeights(
        wq_a=weights.wq_a,
        q_norm=weights.q_norm,
        wq_b=weights.wq_b,
    )


def _empty_compress_topk_indices(
    ratio: int,
    *,
    batch_size: int,
    seq_len: int,
    start_pos: int,
    offset: int,
) -> torch.Tensor:
    if ratio <= 0 or seq_len < ratio:
        return torch.empty(batch_size, seq_len, 0, dtype=torch.int32)
    return compress_topk_indices(ratio, batch_size, seq_len, start_pos, offset).to(torch.int32)


def _is_fp8_tensor(weight: torch.Tensor, scale: torch.Tensor | None) -> bool:
    return weight.dtype == torch.float8_e4m3fn and scale is not None and scale.dtype == torch.float8_e8m0fnu


def _metadata_dtype_is_fp8(dtype: str) -> bool:
    return dtype.startswith("F8_")


def _payload_byte_split(metadata: Sequence[TensorMetadata]) -> dict[str, int]:
    split = {
        "attn_norm": 0,
        "attn_sink": 0,
        "q_norm": 0,
        "kv_norm": 0,
        "wq_a_weight": 0,
        "wq_a_scale": 0,
        "wq_b_weight": 0,
        "wq_b_scale": 0,
        "wkv_weight": 0,
        "wkv_scale": 0,
        "compressor_ape": 0,
        "compressor_wkv_weight": 0,
        "compressor_wgate_weight": 0,
        "compressor_norm": 0,
        "indexer": 0,
        "wo_a_weight": 0,
        "wo_a_scale": 0,
        "wo_b_weight": 0,
        "wo_b_scale": 0,
    }
    for item in metadata:
        key = item.canonical_key
        if key.endswith(".attn_norm.weight"):
            split["attn_norm"] += item.nbytes
        elif key.endswith(".attn.attn_sink"):
            split["attn_sink"] += item.nbytes
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
        elif key.endswith(".attn.compressor.ape"):
            split["compressor_ape"] += item.nbytes
        elif key.endswith(".attn.compressor.wkv.weight"):
            split["compressor_wkv_weight"] += item.nbytes
        elif key.endswith(".attn.compressor.wgate.weight"):
            split["compressor_wgate_weight"] += item.nbytes
        elif key.endswith(".attn.compressor.norm.weight"):
            split["compressor_norm"] += item.nbytes
        elif ".attn.indexer." in key:
            split["indexer"] += item.nbytes
        elif key.endswith(".attn.wo_a.weight"):
            split["wo_a_weight"] += item.nbytes
        elif key.endswith(".attn.wo_a.scale"):
            split["wo_a_scale"] += item.nbytes
        elif key.endswith(".attn.wo_b.weight"):
            split["wo_b_weight"] += item.nbytes
        elif key.endswith(".attn.wo_b.scale"):
            split["wo_b_scale"] += item.nbytes
        else:
            raise ValueError(f"Unexpected tensor in prefill attention runtime slice: {key}")
    split["norms"] = split["attn_norm"] + split["q_norm"] + split["kv_norm"]
    split["q_low_rank"] = split["wq_a_weight"] + split["wq_a_scale"]
    split["q_output"] = split["wq_b_weight"] + split["wq_b_scale"]
    split["kv_projection"] = split["wkv_weight"] + split["wkv_scale"]
    split["compressor"] = (
        split["compressor_ape"]
        + split["compressor_wkv_weight"]
        + split["compressor_wgate_weight"]
        + split["compressor_norm"]
    )
    split["output_projection"] = split["wo_a_weight"] + split["wo_a_scale"] + split["wo_b_weight"] + split["wo_b_scale"]
    split["weights"] = (
        split["wq_a_weight"]
        + split["wq_b_weight"]
        + split["wkv_weight"]
        + split["compressor_wkv_weight"]
        + split["compressor_wgate_weight"]
        + split["wo_a_weight"]
        + split["wo_b_weight"]
    )
    split["scales"] = (
        split["wq_a_scale"] + split["wq_b_scale"] + split["wkv_scale"] + split["wo_a_scale"] + split["wo_b_scale"]
    )
    split["total"] = (
        split["norms"]
        + split["attn_sink"]
        + split["q_low_rank"]
        + split["q_output"]
        + split["kv_projection"]
        + split["compressor"]
        + split["indexer"]
        + split["output_projection"]
    )
    return split


def _validate_runtime_config(
    config: DeepSeekV4FlashConfig,
    *,
    layer: int,
    seq_len: int,
    start_pos: int,
) -> None:
    if layer >= len(config.compress_ratios):
        raise ValueError(f"layer {layer} is outside compress_ratios length {len(config.compress_ratios)}")
    if start_pos != 0:
        raise ValueError("This prefill attention runtime smoke only supports start_pos=0")
    if seq_len > int(config.sliding_window):
        raise ValueError(
            f"This first runtime smoke supports seq_len <= sliding_window {config.sliding_window}, got {seq_len}"
        )
    compress_ratio = int(config.compress_ratios[layer])
    if compress_ratio and seq_len >= compress_ratio:
        raise ValueError(
            "This first runtime smoke is scoped to the no-compressed-cache prefill segment; "
            f"got seq_len {seq_len} and compress_ratio {compress_ratio}"
        )
    if int(config.num_key_value_heads) != 1:
        raise ValueError(f"Expected one K/V head, got {config.num_key_value_heads}")


def _validate_runtime_smoke_args(
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
        raise ValueError("This prefill attention runtime smoke only supports start_pos=0")
    if max_tensors <= 0:
        raise ValueError(f"max_tensors must be positive, got {max_tensors}")
    if max_bytes <= 0:
        raise ValueError(f"max_bytes must be positive, got {max_bytes}")
    if PREFILL_CACHE_PREP_TTNN_TILE_MULTIPLE != ATTENTION_TTNN_TILE_MULTIPLE:
        raise ValueError("Prefill cache-prep and attention tile multiples disagree")
    for name, value in pcc_thresholds.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")


if __name__ == "__main__":
    main()
