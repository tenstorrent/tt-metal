# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""TextModel (HF `EmbeddingGemma2TextModel`) and the device-side text embedding forward.

    e    = embed_tokens(ids) * sqrt(512)
    ple  = PLE projection(e)                                  [B,1,S,24*512]
    h    = e
    for i in 0..23:  h = layer_i(h, RMSNorm(ple[..., i]), mask[kind_i])
    last = norm(h) @ W_emb_proj                                [B,1,S,768]
    emb  = L2normalize(masked_mean(last))                      [B,1,1,768]   (Pooler)

Contract (forward)
    input_ids [B, S] uint32 ROW_MAJOR device; keep [B, 1, 1, S] bf16 TILE device (1 valid / 0 pad).
    S must be a prepared bucket (`prepare(S)` at setup). Positions are arange(S) for every row.
    Returns (last_hidden [B,1,S,768] bf16, embedding [B,1,1,768] fp32). Rows of `last_hidden` at
    padded positions are not meaningful. forward() issues device ops only (no host round trip),
    so it can be captured in a trace.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.embedding_gemma2.tt.config import SLIDING, RMSNormConfig, TextModelConfig, compute_kernel_config_hifi4
from models.demos.embedding_gemma2.tt.embedding import TextEmbedding
from models.demos.embedding_gemma2.tt.encoder_layer import EncoderLayer
from models.demos.embedding_gemma2.tt.linear import linear
from models.demos.embedding_gemma2.tt.masks import AttentionMasks
from models.demos.embedding_gemma2.tt.per_layer_input import PerLayerInput
from models.demos.embedding_gemma2.tt.pooler import Pooler
from models.demos.embedding_gemma2.tt.precision import PrecisionConfig, resolve_precision
from models.demos.embedding_gemma2.tt.rms_norm import RMSNorm
from models.demos.embedding_gemma2.tt.rope import RotaryTables
from models.demos.embedding_gemma2.tt.weights import as_weight, linear_weight, substate


class TextModel(LightweightModule):
    def __init__(
        self,
        config: TextModelConfig,
        device,
        embedding: TextEmbedding,
        per_layer_input: PerLayerInput,
        layers: list[EncoderLayer],
        norm: RMSNorm,
        w_out: ttnn.Tensor,
        rope: RotaryTables,
        masks: AttentionMasks,
        precision: PrecisionConfig | None = None,
    ):
        super().__init__()
        self.precision = precision or resolve_precision(None)
        self.out_policy = self.precision.text.matmul("final_proj")
        self.config = config
        self.device = device
        self.embedding = embedding
        self.per_layer_input = per_layer_input
        self.layers = layers
        self.norm = norm
        self.w_out = w_out
        self.rope = rope
        self.masks = masks
        self.pooler = Pooler(config.embedding_dim)
        self.compute_kernel_config = compute_kernel_config_hifi4()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor],
        config: TextModelConfig,
        device,
        *,
        cache_dir: Path | None = None,
        num_layers: int | None = None,
        precision=None,
    ) -> "TextModel":
        """`state_dict`: HF text-tower keys without the `language_model.` prefix.

        num_layers < config.num_layers builds a truncated stack (debug / per-layer tests only).
        precision: None (default) loads `doc/datatype_sweep/selected_precision_config.json` (or the
        file named by EMBEDDING_GEMMA2_PRECISION); also a path, a dict or a PrecisionConfig.
        """
        precision = resolve_precision(precision)
        tp = precision.text
        n = config.num_layers if num_layers is None else num_layers
        rope = RotaryTables.build(config, device)
        masks = AttentionMasks(
            device,
            config.sliding_window,
            sliding_head_dim=config.head_dim_for(SLIDING),
            num_heads=config.layers[0].attention.num_heads,
            dtype=tp.mask_ttnn_dtype,
        )
        embedding = TextEmbedding.from_state_dict(state_dict, config, device, cache_dir=cache_dir)
        ple = PerLayerInput.from_state_dict(
            substate(state_dict, "ple"), config.per_layer_input, device, cache_dir=cache_dir, precision=tp
        )
        layers = [
            EncoderLayer.from_state_dict(
                substate(state_dict, f"layers.{i}"),
                config.layers[i],
                device,
                rope,
                cache_dir=cache_dir,
                masks=masks,
                precision=tp,
            )
            for i in range(n)
        ]
        norm = RMSNorm.from_state_dict(
            substate(state_dict, "norm"),
            RMSNormConfig(dim=config.hidden_size, eps=config.rms_norm_eps),
            device,
            cache_dir=cache_dir,
            name="norm",
            compute_kernel_config=tp.norms.compute_kernel_config,
        )
        w = state_dict["embedding_projection.weight"]
        if tuple(w.shape) != (config.embedding_dim, config.hidden_size):
            raise ValueError(f"embedding_projection: unexpected shape {tuple(w.shape)}")
        w_out = as_weight(
            linear_weight(w),
            device,
            cache_dir=cache_dir,
            name="embedding_projection.weight",
            dtype=tp.matmul("final_proj").ttnn_dtype,
        )
        return cls(config, device, embedding, ple, layers, norm, w_out, rope, masks, precision)

    def prepare(self, seq_len: int) -> None:
        """Setup-time per-bucket constants (band mask). Must run before forward / trace capture."""
        if seq_len % 32 != 0 or seq_len > self.config.max_seq_len:
            raise ValueError(f"physical seq_len {seq_len} must be a multiple of 32 and <= {self.config.max_seq_len}")
        self.masks.prepare(seq_len)
        self.rope.prepare(seq_len)

    def forward_hidden(self, input_ids: ttnn.Tensor, keep: ttnn.Tensor) -> ttnn.Tensor:
        """Embedding + PLE + encoder stack, before the final norm. [B,1,S,hidden]."""
        masks = self.masks(keep)
        h = self.embedding(input_ids)
        return self.encode_embeds(h, masks)

    def encode_embeds(self, embeds: ttnn.Tensor, masks: dict[str, ttnn.Tensor]) -> ttnn.Tensor:
        """PLE + encoder stack from input embeddings (consumed) and `self.masks(keep)` (consumed).

        Every intermediate is freed at its last use: the PLE source (the wide projection, or the
        embeddings in the narrowed per-layer mode) after the last layer input is built, each mask
        after the last layer of its kind, each layer's input after that layer."""
        n = len(self.layers)
        kinds = [self.config.layer_types[i] for i in range(n)]
        last_use = {kind: max(i for i in range(n) if kinds[i] == kind) for kind in set(kinds)}
        ple = self.per_layer_input(embeds)  # `embeds` itself in the per-layer mode
        h = embeds
        for i, layer in enumerate(self.layers):
            pli = self.per_layer_input.layer_input(ple, i)
            if i == n - 1 and ple is not h:
                ttnn.deallocate(ple)
            h_next = layer(h, pli, masks[kinds[i]])
            ttnn.deallocate(pli)
            if h is not ple or i == n - 1:
                ttnn.deallocate(h)
            h = h_next
            for kind, mask in list(masks.items()):
                if last_use.get(kind, -1) == i and all(m is not mask for k, m in masks.items() if k != kind):
                    ttnn.deallocate(mask)
                    del masks[kind]
        for m in {id(v): v for v in masks.values()}.values():
            ttnn.deallocate(m)
        return h

    def head(self, h: ttnn.Tensor) -> ttnn.Tensor:
        """Final norm + per-token 512->768 projection. [B,1,S,768]."""
        n = self.norm(h)
        out = linear(
            n,
            self.w_out,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            policy=self.out_policy,
            dtype=self.out_policy.out_ttnn_dtype,
        )
        ttnn.deallocate(n)
        return out

    def forward(self, input_ids: ttnn.Tensor, keep: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        h = self.forward_hidden(input_ids, keep)
        last = self.head(h)
        ttnn.deallocate(h)
        emb = self.pooler(last, keep)
        return last, emb
