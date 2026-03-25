# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.wormhole.bge_m3.tt.embeddings import BgeM3Embedding, BgeM3EmbeddingsConfig
from models.demos.wormhole.bge_m3.tt.encoder import BgeM3TransformerBlock
from models.demos.wormhole.bge_m3.tt.norm import LayerNorm1D, LayerNorm1DConfig
from models.demos.wormhole.bge_m3.tt.weight_adapter import LayerNormWeights, build_embedding_weights


class BgeM3Model(LightweightModule):
    """
    End-to-end BGE-M3 encoder returning last hidden state [B, 1, S, D].
    """

    _ADDITIVE_MASKED_VALUE = -100000.0
    _ADDITIVE_UNMASKED_VALUE = 0.0
    _MASK_DTYPE = ttnn.bfloat16

    def __init__(self, args, mesh_device, dtype, state_dict):
        super().__init__()
        self.pad_token_id = int(args.pad_token_id)

        embedding_weights = build_embedding_weights(state_dict, ttnn.bfloat16)
        self.embeddings = BgeM3Embedding.from_config(
            BgeM3EmbeddingsConfig(
                word_embeddings_weight=embedding_weights.word_embeddings_weight,
                position_embeddings_weight=embedding_weights.position_embeddings_weight,
                token_type_embeddings_weight=embedding_weights.token_type_embeddings_weight,
                vocab_size=args.vocab_size,
                max_position_embeddings=args.max_context_len,
                hidden_size=args.dim,
                pad_token_id=args.pad_token_id,
                mesh_device=mesh_device,
                embedding_dtype=ttnn.bfloat16,
            )
        )
        self.embedding_norm = _build_optional_layer_norm(
            embedding_weights.layer_norm,
            eps=args.norm_eps,
            mesh_device=mesh_device,
        )
        self.layers = [
            BgeM3TransformerBlock(
                args=args,
                mesh_device=mesh_device,
                dtype=dtype,
                state_dict=state_dict,
                layer_num=layer_num,
            )
            for layer_num in range(args.n_layers)
        ]

    def create_position_ids_from_input_ids(
        self,
        input_ids: ttnn.Tensor,
        padding_idx: int,
        past_key_values_length: int = 0,
    ) -> ttnn.Tensor:
        """
        HuggingFace RoBERTa-compatible, padding-aware position ID derivation.
        """
        self._require_rank2(input_ids, "input_ids")

        mask = ttnn.ne(input_ids, padding_idx)
        if mask.layout != ttnn.TILE_LAYOUT:
            mask = ttnn.to_layout(mask, ttnn.TILE_LAYOUT)
        mask = ttnn.typecast(mask, dtype=ttnn.int32)

        incremental_indices = ttnn.cumsum(mask, dim=1, dtype=ttnn.int32)
        if past_key_values_length:
            incremental_indices = incremental_indices + int(past_key_values_length)
        incremental_indices = incremental_indices * mask

        position_ids = incremental_indices + int(padding_idx)
        if position_ids.dtype != ttnn.uint32:
            position_ids = ttnn.typecast(position_ids, dtype=ttnn.uint32)
        if position_ids.layout != input_ids.layout:
            position_ids = ttnn.to_layout(position_ids, input_ids.layout)
        return position_ids

    def _prepare_attention_mask(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None,
    ) -> ttnn.Tensor | None:
        """
        Normalize mask to additive [B, 1, 1, S] with {0.0, -100000.0}.
        Return None when there are no masked positions.
        """
        self._require_rank2(input_ids, "input_ids")
        seq_len = input_ids.shape[1]

        if attention_mask is None:
            pad_mask = ttnn.eq(input_ids, self.pad_token_id)
            if not self._has_any_masked_positions(pad_mask):
                return None
            additive_mask = self._build_additive_attention_mask(pad_mask)
        else:
            rank = len(attention_mask.shape)
            if rank == 2:
                # HF convention: 1=keep, 0=pad.
                pad_mask = ttnn.eq(attention_mask, 0)
                additive_mask = self._build_additive_attention_mask(pad_mask)
            elif rank == 4:
                if attention_mask.shape[1] != 1 or attention_mask.shape[2] != 1 or attention_mask.shape[3] != seq_len:
                    raise ValueError(
                        f"attention_mask rank-4 shape must be [B, 1, 1, S] with S={seq_len}, got {attention_mask.shape}"
                    )
                additive_mask = attention_mask
            else:
                raise ValueError(f"attention_mask rank must be 2 or 4, got shape={attention_mask.shape}")

        if getattr(additive_mask, "layout", None) != ttnn.TILE_LAYOUT:
            additive_mask = ttnn.to_layout(additive_mask, ttnn.TILE_LAYOUT)
        if additive_mask.dtype != self._MASK_DTYPE:
            additive_mask = ttnn.typecast(additive_mask, self._MASK_DTYPE)

        memory_config_fn = getattr(additive_mask, "memory_config", None)
        if callable(memory_config_fn) and memory_config_fn() != ttnn.DRAM_MEMORY_CONFIG:
            additive_mask = ttnn.to_memory_config(additive_mask, ttnn.DRAM_MEMORY_CONFIG)

        return additive_mask

    def _build_additive_attention_mask(self, pad_mask: ttnn.Tensor) -> ttnn.Tensor:
        while len(pad_mask.shape) < 4:
            pad_mask = ttnn.unsqueeze(pad_mask, dim=1)
        return ttnn.where(
            pad_mask,
            self._ADDITIVE_MASKED_VALUE,
            self._ADDITIVE_UNMASKED_VALUE,
        )

    @staticmethod
    def _has_any_masked_positions(mask: ttnn.Tensor) -> bool:
        mask_int = ttnn.typecast(mask, dtype=ttnn.uint32)
        return int(ttnn.sum(mask_int).item()) > 0

    def forward(
        self,
        input_ids: ttnn.Tensor,
        attention_mask: ttnn.Tensor | None = None,
        token_type_ids: ttnn.Tensor | None = None,
        position_ids: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        self._require_rank2(input_ids, "input_ids")

        if token_type_ids is None:
            token_type_ids = ttnn.subtract(input_ids, input_ids)
        else:
            self._require_rank2(token_type_ids, "token_type_ids")

        if position_ids is None:
            position_ids = self.create_position_ids_from_input_ids(
                input_ids=input_ids,
                padding_idx=self.pad_token_id,
                past_key_values_length=0,
            )

        prepared_attention_mask = self._prepare_attention_mask(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        if self.embedding_norm is not None:
            hidden_states = self.embedding_norm(hidden_states)

        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, attention_mask=prepared_attention_mask)

        return hidden_states

    @staticmethod
    def _require_rank2(tensor: ttnn.Tensor, name: str) -> None:
        if len(tensor.shape) != 2:
            raise ValueError(f"{name} must have rank 2 [B, S], got shape={tensor.shape}")


def _build_optional_layer_norm(
    layer_norm_weights: LayerNormWeights | None,
    eps: float,
    mesh_device,
) -> LayerNorm1D | None:
    if layer_norm_weights is None:
        return None

    return LayerNorm1D.from_config(
        LayerNorm1DConfig(
            weight=layer_norm_weights.weight,
            bias=layer_norm_weights.bias,
            eps=eps,
            mesh_device=mesh_device,
        )
    )


BGEModel = BgeM3Model
