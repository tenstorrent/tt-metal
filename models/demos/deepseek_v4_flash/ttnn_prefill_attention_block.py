# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v4_flash.manifest import load_tt_manifest
from models.demos.deepseek_v4_flash.ttnn_attention_projection import TtAttentionProjection
from models.demos.deepseek_v4_flash.ttnn_prefill_compressor import TtPrefillCompressor
from models.demos.deepseek_v4_flash.ttnn_prefill_indexer import TtPrefillIndexer
from models.demos.deepseek_v4_flash.ttnn_sparse_attention import TtSparsePrefillAttention


class TtPrefillAttentionBlock(LightweightModule):
    """Single-device DeepSeek V4 Flash prefill attention block stepping stone.

    Callable path:
    ``hidden_states -> q projection -> compressor + indexer -> sparse attention -> output projection``.

    ``hidden_states`` must be a TTNN tensor shaped ``[batch, 1, seq_len, hidden]``.
    ``topk_idxs`` may still be provided explicitly as a host torch tensor shaped
    ``[batch, seq_len, topk]``; when omitted, the block computes compressed-KV
    sparse indices through ``TtPrefillIndexer``. Compressor overlap pooling,
    indexer top-k, sparse gather plus sink-softmax reduction, and grouped
    ``wo_a`` remain host-side fallbacks owned by the underlying stepping-stone
    modules.
    """

    def __init__(
        self,
        *,
        attention_projection: TtAttentionProjection,
        compressor: TtPrefillCompressor,
        indexer: TtPrefillIndexer,
        sparse_attention: TtSparsePrefillAttention,
        attn_sink: torch.Tensor,
    ):
        validate_prefill_attention_block_config(
            attention_projection=attention_projection,
            compressor=compressor,
            indexer=indexer,
            sparse_attention=sparse_attention,
            attn_sink=attn_sink,
        )
        self.attention_projection = attention_projection
        self.compressor = compressor
        self.indexer = indexer
        self.sparse_attention = sparse_attention
        self.attn_sink = attn_sink.float().contiguous()

    @classmethod
    def from_preprocessed(
        cls,
        preprocessed_path: str | Path,
        *,
        device,
        layer: int,
        softmax_scale: float | None = None,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> "TtPrefillAttentionBlock":
        preprocessed_path = Path(preprocessed_path)
        manifest = load_tt_manifest(preprocessed_path)
        config = manifest["config"]
        num_heads = int(config["num_attention_heads"])
        head_dim = int(config["head_dim"])
        if softmax_scale is None:
            softmax_scale = head_dim**-0.5
        attention_projection = TtAttentionProjection.from_preprocessed(
            preprocessed_path,
            device=device,
            layer=layer,
            include_output_projection=True,
            dtype=dtype,
            memory_config=memory_config,
        )
        return cls(
            attention_projection=attention_projection,
            compressor=TtPrefillCompressor.from_preprocessed(
                preprocessed_path,
                device=device,
                layer=layer,
                dtype=dtype,
                memory_config=memory_config,
            ),
            indexer=TtPrefillIndexer.from_preprocessed(
                preprocessed_path,
                device=device,
                layer=layer,
                attention_projection=attention_projection,
                dtype=dtype,
                memory_config=memory_config,
            ),
            sparse_attention=TtSparsePrefillAttention(
                device=device,
                num_heads=num_heads,
                head_dim=head_dim,
                softmax_scale=softmax_scale,
                dtype=dtype,
                memory_config=memory_config,
            ),
            attn_sink=load_attention_sink(preprocessed_path, manifest=manifest, layer=layer),
        )

    def forward(self, hidden_states, *, topk_idxs: torch.Tensor | None = None):
        validate_prefill_attention_block_input(
            hidden_states,
            topk_idxs,
            hidden_size=self.attention_projection.hidden_size,
            compress_ratio=self.compressor.compress_ratio,
        )
        q_rank = self.attention_projection.project_q_rank(hidden_states)
        q = self.attention_projection.project_q_from_rank(q_rank)
        compressed_kv = self.compressor(hidden_states)
        if topk_idxs is None:
            topk_idxs = self.indexer.topk_from_q_rank(hidden_states, q_rank=q_rank)
        attention_output = self.sparse_attention(
            q,
            compressed_kv,
            attn_sink=self.attn_sink,
            topk_idxs=topk_idxs,
        )
        return self.attention_projection.project_output(attention_output)


def load_attention_sink(
    preprocessed_path: str | Path,
    *,
    manifest: dict | None = None,
    layer: int,
) -> torch.Tensor:
    preprocessed_path = Path(preprocessed_path)
    if manifest is None:
        manifest = load_tt_manifest(preprocessed_path)
    key = f"layers.{layer}.attn.attn_sink"
    for artifact in manifest["artifacts"]["non_expert_safetensors"]:
        with safe_open(preprocessed_path / artifact, framework="pt", device="cpu") as handle:
            if key in set(handle.keys()):
                return handle.get_tensor(key).contiguous()
    raise KeyError(f"Missing attention sink for layer {layer}: {key}")


def validate_prefill_attention_block_config(
    *,
    attention_projection: TtAttentionProjection,
    compressor: TtPrefillCompressor,
    indexer: TtPrefillIndexer,
    sparse_attention: TtSparsePrefillAttention,
    attn_sink: torch.Tensor,
) -> None:
    if (
        attention_projection.device != compressor.device
        or attention_projection.device != indexer.device
        or attention_projection.device != sparse_attention.device
    ):
        raise ValueError(
            "attention_projection, compressor, indexer, and sparse_attention must use the same TTNN device"
        )
    if (
        attention_projection.dtype != compressor.dtype
        or attention_projection.dtype != indexer.dtype
        or attention_projection.dtype != sparse_attention.dtype
    ):
        raise ValueError("attention_projection, compressor, indexer, and sparse_attention must use the same TTNN dtype")
    if attention_projection.memory_config != compressor.memory_config:
        raise ValueError("attention_projection and compressor must use the same TTNN memory config")
    if attention_projection.memory_config != indexer.memory_config:
        raise ValueError("attention_projection and indexer must use the same TTNN memory config")
    if attention_projection.memory_config != sparse_attention.memory_config:
        raise ValueError("attention_projection and sparse_attention must use the same TTNN memory config")
    if attention_projection.num_heads != sparse_attention.num_heads:
        raise ValueError(
            f"projection num_heads {attention_projection.num_heads} must match sparse attention "
            f"num_heads {sparse_attention.num_heads}"
        )
    if attention_projection.head_dim != sparse_attention.head_dim:
        raise ValueError(
            f"projection head_dim {attention_projection.head_dim} must match sparse attention "
            f"head_dim {sparse_attention.head_dim}"
        )
    if compressor.head_dim != sparse_attention.head_dim:
        raise ValueError(
            f"compressor head_dim {compressor.head_dim} must match sparse attention head_dim "
            f"{sparse_attention.head_dim}"
        )
    if indexer.hidden_size != attention_projection.hidden_size:
        raise ValueError(
            f"indexer hidden_size {indexer.hidden_size} must match projection hidden_size "
            f"{attention_projection.hidden_size}"
        )
    if indexer.q_lora_rank != attention_projection.q_lora_rank:
        raise ValueError(
            f"indexer q_lora_rank {indexer.q_lora_rank} must match projection q_lora_rank "
            f"{attention_projection.q_lora_rank}"
        )
    if indexer.compress_ratio != compressor.compress_ratio:
        raise ValueError(
            f"indexer compress_ratio {indexer.compress_ratio} must match compressor compress_ratio "
            f"{compressor.compress_ratio}"
        )
    if attn_sink.ndim != 1 or tuple(attn_sink.shape) != (sparse_attention.num_heads,):
        raise ValueError(f"Expected attn_sink shape {(sparse_attention.num_heads,)}, got {tuple(attn_sink.shape)}")


def validate_prefill_attention_block_input(
    hidden_states,
    topk_idxs: torch.Tensor | None,
    *,
    hidden_size: int,
    compress_ratio: int,
) -> None:
    shape = tuple(hidden_states.shape)
    if len(shape) != 4 or shape[1] != 1:
        raise ValueError(f"Expected hidden_states shape [batch, 1, seq_len, hidden], got {shape}")
    batch_size, _, seq_len, width = shape
    if width != hidden_size:
        raise ValueError(f"Expected hidden_states hidden size {hidden_size}, got {width}")
    if seq_len < compress_ratio:
        raise ValueError(f"Expected seq_len >= compress_ratio {compress_ratio}, got {seq_len}")
    if topk_idxs is None:
        return
    if topk_idxs.ndim != 3:
        raise ValueError(f"Expected topk_idxs shape [batch, seq_len, topk], got {tuple(topk_idxs.shape)}")
    if topk_idxs.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"Expected topk_idxs dtype int32 or int64, got {topk_idxs.dtype}")
    if tuple(topk_idxs.shape[:2]) != (batch_size, seq_len):
        raise ValueError(f"Expected topk_idxs batch/seq {(batch_size, seq_len)}, got {tuple(topk_idxs.shape[:2])}")
