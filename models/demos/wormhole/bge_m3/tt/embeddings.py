# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.modules.lazy_weight import LazyWeight, resolve_lazy_weight


@dataclass
class BgeM3EmbeddingsConfig:
    word_embeddings_weight: LazyWeight
    position_embeddings_weight: LazyWeight
    token_type_embeddings_weight: LazyWeight
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    pad_token_id: int
    mesh_device: ttnn.MeshDevice | None = None
    embedding_dtype: ttnn.DataType | None = None
    embedding_memcfg: ttnn.MemoryConfig | None = None


class BgeM3Embedding(LightweightModule):
    """
    BGE-M3 embeddings public API contract.

    This module will produce a sum of word + position + token-type embeddings
    and return shape [B, 1, S, D].
    """

    def __init__(
        self,
        word_embeddings_weight: LazyWeight,
        position_embeddings_weight: LazyWeight,
        token_type_embeddings_weight: LazyWeight,
        vocab_size: int,
        max_position_embeddings: int,
        hidden_size: int,
        pad_token_id: int,
    ):
        super().__init__()

        self.config = _resolve_embeddings_config(
            BgeM3EmbeddingsConfig(
                word_embeddings_weight=word_embeddings_weight,
                position_embeddings_weight=position_embeddings_weight,
                token_type_embeddings_weight=token_type_embeddings_weight,
                vocab_size=vocab_size,
                max_position_embeddings=max_position_embeddings,
                hidden_size=hidden_size,
                pad_token_id=pad_token_id,
            )
        )
        self._device_weights_loaded = False
        self._fold_token_type = False
        self.pos: ttnn.Tensor | None = None

    @classmethod
    def from_config(cls, config: BgeM3EmbeddingsConfig) -> "BgeM3Embedding":
        instance = object.__new__(cls)
        super(BgeM3Embedding, instance).__init__()
        instance.config = _resolve_embeddings_config(config)
        instance._device_weights_loaded = False
        instance._fold_token_type = False
        instance.pos = None
        return instance

    def load_device_weights(self) -> None:
        if self._device_weights_loaded:
            return

        # When type_vocab_size == 1 (BGE-M3), fold token_type_emb[0] into
        # word_embeddings at load time. Saves ~35 µs/forward (embed + add).
        tt_src = self.config.token_type_embeddings_weight.source
        self._fold_token_type = tt_src is not None and tt_src.shape[0] == 1
        if self._fold_token_type:
            word_lw = self.config.word_embeddings_weight
            word_src = word_lw.source
            tt_row = tt_src.to(dtype=word_src.dtype).reshape(-1)
            folded = word_src.clone()
            folded.add_(tt_row.to(folded.device))
            word_lw.source = folded
            if word_lw.cache_dir_weight_name is not None:
                cdir, wname = word_lw.cache_dir_weight_name
                word_lw.cache_dir_weight_name = (cdir, wname + "_tt_folded")

        self.word_embeddings_weight = self.config.word_embeddings_weight.get_device_weight()
        self.position_embeddings_weight = self.config.position_embeddings_weight.get_device_weight()
        if not self._fold_token_type:
            self.token_type_embeddings_weight = self.config.token_type_embeddings_weight.get_device_weight()
        else:
            self.token_type_embeddings_weight = None
        self._device_weights_loaded = True

    def forward(
        self,
        input_ids: ttnn.Tensor,
        token_type_ids: ttnn.Tensor | None = None,
        position_ids: ttnn.Tensor | None = None,
        defer_position_add: bool = False,
    ) -> ttnn.Tensor | tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Public forward API.

        Args:
            input_ids: rank-2 IDs tensor with shape [B, S].
            token_type_ids: optional rank-2 IDs tensor with shape [B, S].
            position_ids: optional rank-2 IDs tensor with shape [B, S].
            defer_position_add: when True, return (word+token_type, position) separately
                so the caller can fold the add into the next LayerNorm as residual.

        Returns:
            Embedding activations with shape [B, 1, S, D] (or tuple if defer_position_add).
        """
        if token_type_ids is None:
            token_type_ids = ttnn.subtract(input_ids, input_ids)

        self.load_device_weights()

        if position_ids is None:
            batch_size, seq_len = input_ids.shape
            position_ids = self._build_position_ids(batch_size=batch_size, seq_len=seq_len)

        word_embeddings = ttnn.embedding(
            input_ids,
            weight=self.word_embeddings_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.config.embedding_memcfg,
            padding_idx=self.config.pad_token_id,
        )
        position_embeddings = ttnn.embedding(
            position_ids,
            weight=self.position_embeddings_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=self.config.embedding_memcfg,
        )

        if self._fold_token_type:
            # token_type[0] already baked into word_embeddings_weight
            main = word_embeddings
        else:
            token_type_embeddings = ttnn.embedding(
                token_type_ids,
                weight=self.token_type_embeddings_weight,
                layout=ttnn.TILE_LAYOUT,
                memory_config=self.config.embedding_memcfg,
            )
            main = ttnn.add(word_embeddings, token_type_embeddings)
            ttnn.deallocate(word_embeddings)
            ttnn.deallocate(token_type_embeddings)

        if defer_position_add:
            main = ttnn.unsqueeze(main, dim=1)
            position_embeddings = ttnn.unsqueeze(position_embeddings, dim=1)
            return main, position_embeddings

        embeddings = ttnn.add(main, position_embeddings)
        ttnn.deallocate(main)
        ttnn.deallocate(position_embeddings)
        embeddings = ttnn.unsqueeze(embeddings, dim=1)
        return embeddings

    def _build_position_ids(self, batch_size: int, seq_len: int) -> ttnn.Tensor:
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(f"seq_len={seq_len} exceeds max_position_embeddings={self.config.max_position_embeddings}")

        if self.pos is None:
            self.pos = ttnn.arange(
                0,
                self.config.max_position_embeddings,
                1,
                dtype=ttnn.uint32,
                device=self.config.word_embeddings_weight.device,
                memory_config=self.config.embedding_memcfg,
            )
            self.pos = ttnn.reshape(self.pos, (1, -1))

        # Mirror reference.py: arange(max_pos) -> slice to seq_len -> repeat by batch.
        pos_prefix = ttnn.slice(
            self.pos,
            (0, 0),
            (1, seq_len),
            memory_config=self.config.embedding_memcfg,
        )
        if batch_size == 1:
            # Keep the slice alive for embedding; repeat(1, 1) may alias input.
            return pos_prefix

        position_ids = ttnn.repeat(
            pos_prefix,
            (batch_size, 1),
            memory_config=self.config.embedding_memcfg,
        )
        ttnn.deallocate(pos_prefix)
        return position_ids


def _resolve_embeddings_config(config: BgeM3EmbeddingsConfig) -> BgeM3EmbeddingsConfig:
    to_set: dict[str, object] = {}
    if config.embedding_dtype is None:
        to_set["embedding_dtype"] = ttnn.bfloat16
    if config.embedding_memcfg is None:
        to_set["embedding_memcfg"] = ttnn.DRAM_MEMORY_CONFIG

    weight_devices = [
        device
        for device in (
            config.word_embeddings_weight.device,
            config.position_embeddings_weight.device,
            config.token_type_embeddings_weight.device,
        )
        if device is not None
    ]
    if weight_devices and any(device != weight_devices[0] for device in weight_devices):
        raise ValueError("All embedding weights must target the same device")
    if config.mesh_device is not None and weight_devices and weight_devices[0] != config.mesh_device:
        raise ValueError("All embedding weights must target the configured mesh_device")

    mesh_device = (
        config.mesh_device
        if config.mesh_device is not None
        else (weight_devices[0] if weight_devices else ttnn.GetDefaultDevice())
    )

    embedding_dtype = to_set.get("embedding_dtype", config.embedding_dtype)
    embedding_memcfg = to_set.get("embedding_memcfg", config.embedding_memcfg)

    to_set["word_embeddings_weight"] = resolve_lazy_weight(
        config.word_embeddings_weight,
        device=mesh_device,
        dtype=embedding_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=embedding_memcfg,
        mesh_mapper_config=None,
    )
    to_set["position_embeddings_weight"] = resolve_lazy_weight(
        config.position_embeddings_weight,
        device=mesh_device,
        dtype=embedding_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=embedding_memcfg,
        mesh_mapper_config=None,
    )
    to_set["token_type_embeddings_weight"] = resolve_lazy_weight(
        config.token_type_embeddings_weight,
        device=mesh_device,
        dtype=embedding_dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=embedding_memcfg,
        mesh_mapper_config=None,
    )
    return replace(config, **to_set)
