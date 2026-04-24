# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.cpu_reference import indexer_topk
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.ttnn_attention_projection import TtAttentionProjection
from models.demos.deepseek_v4_flash.ttnn_prefill_compressor import (
    PrefillCompressorWeights,
    _compress_projected_prefill,
    _to_tt_linear_weight,
    _ttnn_prefill_to_torch_3d,
    _validate_compressor_weights,
)


@dataclass(frozen=True)
class PrefillIndexerWeights:
    wq_b: torch.Tensor
    weights_proj: torch.Tensor
    compressor: PrefillCompressorWeights


class TtPrefillIndexer(LightweightModule):
    """Single-device DeepSeek V4 Flash prefill indexer stepping stone.

    Data contract:
    - hidden_states: TTNN tensor shaped [batch, 1, seq_len, hidden]
    - output: host torch int tensor shaped [batch, seq_len, topk]

    The q-rank path is reused from ``TtAttentionProjection``. The indexer q,
    per-head weights, and compressed KV projections run on TTNN; the ratio
    pooling and final top-k selection intentionally remain host-side through
    ``cpu_reference.indexer_topk``. This preserves the sparse attention ABI while
    the full device-side index selection path is still being brought up.
    """

    def __init__(
        self,
        *,
        attention_projection: TtAttentionProjection,
        weights: PrefillIndexerWeights,
        index_n_heads: int,
        index_head_dim: int,
        index_topk: int,
        compress_ratio: int,
        norm_eps: float = 1e-6,
        overlap: bool | None = None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ):
        if overlap is None:
            overlap = compress_ratio == 4
        validate_prefill_indexer_config(
            attention_projection=attention_projection,
            weights=weights,
            index_n_heads=index_n_heads,
            index_head_dim=index_head_dim,
            index_topk=index_topk,
            compress_ratio=compress_ratio,
            overlap=overlap,
            dtype=dtype,
            memory_config=memory_config,
        )

        self.attention_projection = attention_projection
        self.device = attention_projection.device
        self.dtype = dtype
        self.memory_config = memory_config
        self.hidden_size = attention_projection.hidden_size
        self.q_lora_rank = attention_projection.q_lora_rank
        self.index_n_heads = int(index_n_heads)
        self.index_head_dim = int(index_head_dim)
        self.index_topk = int(index_topk)
        self.compress_ratio = int(compress_ratio)
        self.norm_eps = float(norm_eps)
        self.overlap = bool(overlap)
        self.index_weight_scale = self.index_head_dim**-0.5 * self.index_n_heads**-0.5
        self.compressor_ape = weights.compressor.ape.float().contiguous()
        self.compressor_norm_weight = weights.compressor.norm_weight.float().contiguous()

        self.wq_b = _to_tt_linear_weight(
            weights.wq_b,
            device=self.device,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.weights_proj = _to_tt_linear_weight(
            weights.weights_proj,
            device=self.device,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.compressor_wkv = _to_tt_linear_weight(
            weights.compressor.wkv,
            device=self.device,
            dtype=dtype,
            memory_config=memory_config,
        )
        self.compressor_wgate = _to_tt_linear_weight(
            weights.compressor.wgate,
            device=self.device,
            dtype=dtype,
            memory_config=memory_config,
        )

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        *,
        device,
        layer: int,
        attention_projection: TtAttentionProjection | None = None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        overlap: bool | None = None,
    ) -> "TtPrefillIndexer":
        preprocessed_path = Path(preprocessed_path)
        manifest = load_tt_manifest(preprocessed_path)
        config = manifest["config"]
        compress_ratio = int(config["compress_ratios"][layer])
        if compress_ratio == 0:
            raise ValueError(f"Layer {layer} has compress_ratio=0 and no prefill indexer")
        if attention_projection is None:
            attention_projection = TtAttentionProjection.from_preprocessed(
                preprocessed_path,
                device=device,
                layer=layer,
                include_output_projection=False,
                dtype=dtype,
                memory_config=memory_config,
            )
        return cls(
            attention_projection=attention_projection,
            weights=load_prefill_indexer_weights(preprocessed_path, manifest=manifest, layer=layer),
            index_n_heads=int(config["index_n_heads"]),
            index_head_dim=int(config["index_head_dim"]),
            index_topk=int(config["index_topk"]),
            compress_ratio=compress_ratio,
            norm_eps=float(config["rms_norm_eps"]),
            overlap=overlap,
            dtype=dtype,
            memory_config=memory_config,
        )

    def forward(self, hidden_states, *, start_pos: int = 0, offset: int = 0) -> torch.Tensor:
        q_rank = self.attention_projection.project_q_rank(hidden_states)
        return self.topk_from_q_rank(hidden_states, q_rank=q_rank, start_pos=start_pos, offset=offset)

    def topk_from_q_rank(self, hidden_states, *, q_rank, start_pos: int = 0, offset: int = 0) -> torch.Tensor:
        validate_prefill_indexer_input(
            hidden_states,
            q_rank=q_rank,
            hidden_size=self.hidden_size,
            q_lora_rank=self.q_lora_rank,
            compress_ratio=self.compress_ratio,
            start_pos=start_pos,
            offset=offset,
        )
        index_q = ttnn.linear(q_rank, self.wq_b, memory_config=self.memory_config)
        weights = ttnn.linear(hidden_states, self.weights_proj, memory_config=self.memory_config)
        kv = ttnn.linear(hidden_states, self.compressor_wkv, memory_config=self.memory_config)
        score = ttnn.linear(hidden_states, self.compressor_wgate, memory_config=self.memory_config)

        q_host = _ttnn_prefill_to_torch_3d(index_q).reshape(
            -1,
            int(hidden_states.shape[2]),
            self.index_n_heads,
            self.index_head_dim,
        )
        weights_host = _ttnn_prefill_to_torch_3d(weights).float() * self.index_weight_scale
        compressed_kv = _compress_projected_prefill(
            _ttnn_prefill_to_torch_3d(kv),
            _ttnn_prefill_to_torch_3d(score),
            ape=self.compressor_ape,
            norm_weight=self.compressor_norm_weight,
            compress_ratio=self.compress_ratio,
            head_dim=self.index_head_dim,
            norm_eps=self.norm_eps,
            overlap=self.overlap,
        )
        return indexer_topk(
            q_host,
            compressed_kv,
            weights_host,
            index_topk=self.index_topk,
            compress_ratio=self.compress_ratio,
            start_pos=start_pos,
            offset=offset,
        )


def load_prefill_indexer_weights(
    preprocessed_path: str | Path,
    *,
    manifest: dict | None = None,
    layer: int,
) -> PrefillIndexerWeights:
    preprocessed_path = Path(preprocessed_path)
    if manifest is None:
        manifest = load_tt_manifest(preprocessed_path)
    prefix = f"layers.{layer}.attn.indexer"
    keys = {
        "wq_b": f"{prefix}.wq_b.weight",
        "weights_proj": f"{prefix}.weights_proj.weight",
        "compressor_wkv": f"{prefix}.compressor.wkv.weight",
        "compressor_wgate": f"{prefix}.compressor.wgate.weight",
        "compressor_ape": f"{prefix}.compressor.ape",
        "compressor_norm_weight": f"{prefix}.compressor.norm.weight",
    }
    loaded: dict[str, torch.Tensor] = {}
    for artifact in manifest["artifacts"]["non_expert_safetensors"]:
        with safe_open(preprocessed_path / artifact, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for name, key in keys.items():
                if name not in loaded and key in available:
                    loaded[name] = handle.get_tensor(key).contiguous()
        if len(loaded) == len(keys):
            break

    missing = sorted(key for name, key in keys.items() if name not in loaded)
    if missing:
        raise KeyError(f"Missing prefill indexer weights for layer {layer}: {missing}")

    return PrefillIndexerWeights(
        wq_b=loaded["wq_b"],
        weights_proj=loaded["weights_proj"],
        compressor=PrefillCompressorWeights(
            wkv=loaded["compressor_wkv"],
            wgate=loaded["compressor_wgate"],
            ape=loaded["compressor_ape"],
            norm_weight=loaded["compressor_norm_weight"],
        ),
    )


def validate_prefill_indexer_config(
    *,
    attention_projection: TtAttentionProjection,
    weights: PrefillIndexerWeights,
    index_n_heads: int,
    index_head_dim: int,
    index_topk: int,
    compress_ratio: int,
    overlap: bool,
    dtype,
    memory_config,
) -> None:
    if attention_projection.dtype != dtype:
        raise ValueError("attention_projection and prefill indexer must use the same TTNN dtype")
    if attention_projection.memory_config != memory_config:
        raise ValueError("attention_projection and prefill indexer must use the same TTNN memory config")
    if index_n_heads <= 0:
        raise ValueError(f"index_n_heads must be positive, got {index_n_heads}")
    if index_head_dim <= 0:
        raise ValueError(f"index_head_dim must be positive, got {index_head_dim}")
    if index_topk <= 0:
        raise ValueError(f"index_topk must be positive, got {index_topk}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")

    expected_q_dim = index_n_heads * index_head_dim
    if tuple(weights.wq_b.shape) != (expected_q_dim, attention_projection.q_lora_rank):
        raise ValueError(
            f"Expected indexer wq_b shape {(expected_q_dim, attention_projection.q_lora_rank)}, "
            f"got {tuple(weights.wq_b.shape)}"
        )
    if tuple(weights.weights_proj.shape) != (index_n_heads, attention_projection.hidden_size):
        raise ValueError(
            f"Expected indexer weights_proj shape {(index_n_heads, attention_projection.hidden_size)}, "
            f"got {tuple(weights.weights_proj.shape)}"
        )
    _validate_compressor_weights(
        weights.compressor,
        compress_ratio=compress_ratio,
        head_dim=index_head_dim,
        overlap=overlap,
    )


def validate_prefill_indexer_input(
    hidden_states,
    *,
    q_rank=None,
    hidden_size: int,
    q_lora_rank: int,
    compress_ratio: int,
    start_pos: int = 0,
    offset: int = 0,
) -> None:
    shape = tuple(hidden_states.shape)
    if len(shape) != 4 or shape[1] != 1:
        raise ValueError(f"Expected hidden_states shape [batch, 1, seq_len, hidden], got {shape}")
    if shape[-1] != hidden_size:
        raise ValueError(f"Expected hidden_states hidden size {hidden_size}, got {shape[-1]}")
    if shape[-2] < compress_ratio:
        raise ValueError(f"Expected seq_len >= compress_ratio {compress_ratio}, got {shape[-2]}")
    if start_pos != 0:
        raise ValueError("Prefill indexer stepping stone only supports start_pos=0 without tensor caches")
    if offset != 0:
        raise ValueError("Prefill indexer output indexes its own compressed KV, so offset must be 0")
    if q_rank is None:
        return
    q_rank_shape = tuple(q_rank.shape)
    expected_shape = (shape[0], 1, shape[2], q_lora_rank)
    if q_rank_shape != expected_shape:
        raise ValueError(f"Expected q_rank shape {expected_shape}, got {q_rank_shape}")
