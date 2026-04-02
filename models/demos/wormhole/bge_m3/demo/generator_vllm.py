# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch
import transformers

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.demos.wormhole.bge_m3.demo.m3_scores import _get_special_token_ids, _sparse_embedding_scatter_ttnn
from models.demos.wormhole.bge_m3.tt.common import create_tt_model
from models.demos.wormhole.bge_m3.tt.model_config import get_padded_sequence_length


class BgeM3ForEmbedding:
    """
    vLLM-facing embedding wrapper for direct BGE-M3 encoder execution.

    This implementation intentionally keeps a single-device execution path.
    """

    def __init__(
        self,
        device: ttnn.Device = None,
        max_batch_size: int = 8,
        max_seq_len: int = 8192,
        dtype=ttnn.bfloat16,
        model_name: str = "BAAI/bge-m3",
        vllm_config=None,
        prefix: str = "",
        tt_data_parallel: int = 1,
        **kwargs,
    ):
        self.sentence_pooling_method = kwargs.pop("sentence_pooling_method", "mean")
        self.normalize_embeddings = kwargs.pop("normalize_embeddings", False)
        self.return_dense = kwargs.pop("return_dense", True)
        self.return_sparse = kwargs.pop("return_sparse", False)
        self.return_colbert = kwargs.pop("return_colbert", False)

        del prefix, kwargs

        if vllm_config is not None and device is None:
            device = vllm_config.device_config.device

        if device is None:
            raise ValueError("Either 'device' or 'vllm_config' must be provided")

        self.device = device
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        # Accepted for API compatibility; execution stays single-device.
        self.tt_data_parallel = tt_data_parallel
        self.dtype = dtype
        self.model_name = model_name

        if vllm_config is not None:
            self.vllm_config = vllm_config

        self.config = transformers.AutoConfig.from_pretrained(model_name)
        self.pooler = None
        self._is_initialized = False

        self.model_args = None
        self.model = None
        # Compatibility placeholders for callers that probe the newer API.
        self.model_args_list = None
        self.models = None
        self.data_parallel = None
        self.submeshes = None
        self.state_dict = None
        self.tokenizer = None

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config: transformers.PretrainedConfig,
        mesh_device: ttnn.Device,
        max_batch_size: int,
        max_seq_len: Optional[int] = 8192,
        model_location_generator=None,
        tt_data_parallel=1,
        optimizations: Optional[str] = None,
        vllm_config=None,
        dtype=ttnn.bfloat16,
        **kwargs,
    ) -> "BgeM3ForEmbedding":
        if optimizations is not None:
            raise ValueError("Optimizations are not supported for BGE-M3")

        if vllm_config is not None:
            if (
                not hasattr(vllm_config.model_config, "override_tt_config")
                or vllm_config.model_config.override_tt_config is None
            ):
                vllm_config.model_config.override_tt_config = {}
            vllm_config.model_config.override_tt_config["is_embedding_model"] = True

            return cls(
                device=mesh_device,
                model_location_generator=model_location_generator,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                vllm_config=vllm_config,
                tt_data_parallel=tt_data_parallel,
                dtype=dtype,
            )

        return cls(
            device=mesh_device,
            model_location_generator=model_location_generator,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            tt_data_parallel=tt_data_parallel,
            dtype=dtype,
        )

    def _initialize_model(self) -> None:
        if self._is_initialized and self.model is not None:
            return

        self.model_args, self.model, self.state_dict = create_tt_model(
            mesh_device=self.device,
            max_batch_size=self.max_batch_size,
            max_seq_len=self.max_seq_len,
            dtype=self.dtype,
            state_dict=self.state_dict,
            hf_model_name=self.model_name,
        )
        self.tokenizer = self.model_args.tokenizer
        self._is_initialized = True

    @staticmethod
    def _get_padded_seq_len(seq_len: int) -> int:
        return get_padded_sequence_length(seq_len)

    def _validate_request(self, batch_size: int, padded_seq_len: int) -> None:
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds max_batch_size {self.max_batch_size}")
        if padded_seq_len > self.max_seq_len:
            raise ValueError(f"Padded sequence length {padded_seq_len} exceeds max_seq_len {self.max_seq_len}")

    @staticmethod
    def _pad_tensor(tensor: torch.Tensor, padded_seq_len: int, pad_value: int = 0) -> torch.Tensor:
        batch_size, seq_len = tensor.shape
        if seq_len == padded_seq_len:
            return tensor

        padded = torch.full(
            (batch_size, padded_seq_len),
            fill_value=pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:, :seq_len] = tensor
        return padded

    @staticmethod
    def _pad_batch_tensor(tensor: torch.Tensor, padded_batch_size: int, pad_value: int = 0) -> torch.Tensor:
        batch_size = tensor.shape[0]
        if batch_size == padded_batch_size:
            return tensor

        padded = torch.full(
            (padded_batch_size, *tensor.shape[1:]),
            fill_value=pad_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        padded[:batch_size] = tensor
        return padded

    def _pad_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        padded_seq_len = self._get_padded_seq_len(input_ids.shape[1])

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        padded_inputs = {
            "input_ids": self._pad_batch_tensor(
                self._pad_tensor(input_ids, padded_seq_len, pad_value=self.tokenizer.pad_token_id),
                self.max_batch_size,
                pad_value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": self._pad_batch_tensor(
                self._pad_tensor(attention_mask, padded_seq_len, pad_value=0),
                self.max_batch_size,
                pad_value=0,
            ),
            "token_type_ids": self._pad_batch_tensor(
                self._pad_tensor(token_type_ids, padded_seq_len, pad_value=0),
                self.max_batch_size,
                pad_value=0,
            )
            if token_type_ids is not None
            else None,
            "position_ids": self._pad_batch_tensor(
                self._pad_tensor(position_ids, padded_seq_len, pad_value=self.tokenizer.pad_token_id),
                self.max_batch_size,
                pad_value=self.tokenizer.pad_token_id,
            )
            if position_ids is not None
            else None,
        }

        return padded_inputs

    def _to_ttnn_ids(self, ids: torch.Tensor, *, device: Optional[ttnn.Device] = None) -> ttnn.Tensor:
        target_device = self.device if device is None else device
        return ttnn.from_torch(
            ids.to(torch.int32),
            device=target_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )

    @staticmethod
    def _crop_hidden_state_ttnn(hidden_state: ttnn.Tensor, batch_size: int, seq_len: int) -> ttnn.Tensor:
        shape = hidden_state.shape
        if len(shape) == 4:
            return ttnn.slice(
                hidden_state,
                [0, 0, 0, 0],
                [batch_size, shape[1], seq_len, shape[3]],
            )
        if len(shape) == 3:
            return ttnn.slice(
                hidden_state,
                [0, 0, 0],
                [batch_size, seq_len, shape[2]],
            )
        raise ValueError(f"Unsupported hidden_state rank: shape={shape}")

    @staticmethod
    def _flatten_sparse_token_weights_ttnn(token_weights: ttnn.Tensor) -> ttnn.Tensor:
        token_weights = ttnn.to_memory_config(token_weights, ttnn.DRAM_MEMORY_CONFIG)
        if token_weights.layout != ttnn.TILE_LAYOUT:
            token_weights = ttnn.to_layout(token_weights, ttnn.TILE_LAYOUT)
        shape = token_weights.shape
        if len(shape) == 4:
            batch_size, heads, seq_len, width = map(int, shape)
            if heads != 1 or width != 1:
                raise ValueError(f"Unsupported sparse token weight shape: {shape}")
            return ttnn.reshape(token_weights, [batch_size, seq_len])
        if len(shape) == 3:
            batch_size, dim1, dim2 = map(int, shape)
            if dim1 == 1:
                return ttnn.reshape(token_weights, [batch_size, dim2])
            if dim2 == 1:
                return ttnn.reshape(token_weights, [batch_size, dim1])
            raise ValueError(f"Unsupported sparse token weight shape: {shape}")
        if len(shape) == 2:
            return token_weights
        raise ValueError(f"Unsupported sparse token weight rank: shape={shape}")

    def _dense_embedding(self, last_hidden_state: ttnn.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        tt_hidden = self._crop_hidden_state_ttnn(last_hidden_state, batch_size, seq_len)
        if len(tt_hidden.shape) == 3:
            tt_hidden = ttnn.unsqueeze(tt_hidden, dim=1)

        tt_hidden = ttnn.to_memory_config(tt_hidden, ttnn.DRAM_MEMORY_CONFIG)
        B, _, S, D = tt_hidden.shape

        if self.sentence_pooling_method == "cls":
            pooled_tt = ttnn.slice(tt_hidden, [0, 0, 0, 0], [B, 1, 1, D])
            pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
            pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
        elif self.sentence_pooling_method == "mean":
            mask_torch = attention_mask[:, :S].unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)
            mask_tt = ttnn.from_torch(
                mask_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
            summed_tt = ttnn.sum(ttnn.multiply(tt_hidden, mask_tt), dim=2)
            counts_torch = attention_mask[:, :S].sum(dim=1, keepdim=True).clamp(min=1).to(torch.bfloat16).unsqueeze(1)
            counts_tt = ttnn.from_torch(
                counts_torch,
                device=self.device,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
            )
            pooled_tt = ttnn.divide(summed_tt, counts_tt)
            pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
        elif self.sentence_pooling_method == "last_token":
            left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
            if left_padding:
                pooled_tt = ttnn.slice(tt_hidden, [0, 0, S - 1, 0], [B, 1, S, D])
                pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
                pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
            else:
                selector = torch.zeros((batch_size, seq_len), dtype=torch.bfloat16)
                selector[torch.arange(batch_size), attention_mask.sum(dim=1) - 1] = 1
                selector_tt = ttnn.from_torch(
                    selector.unsqueeze(1).unsqueeze(-1),
                    device=self.device,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                )
                pooled_tt = ttnn.sum(ttnn.multiply(tt_hidden, selector_tt), dim=2)
                pooled_tt = ttnn.squeeze(pooled_tt, dim=1)
        else:
            raise NotImplementedError(f"pooling method {self.sentence_pooling_method} not implemented")

        pooled = to_torch_auto_compose(pooled_tt, device=self.device)
        if pooled.dim() == 3 and pooled.shape[1] == 1:
            pooled = pooled.squeeze(1)
        return pooled.to(torch.float32)

    def _sparse_embedding(
        self,
        hidden_state: ttnn.Tensor,
        input_ids: torch.Tensor,
        return_embedding: bool = True,
    ) -> torch.Tensor:
        self._initialize_model()
        if self.model is None or self.model.sparse_linear is None:
            raise ValueError("Sparse linear head is not initialized")

        batch_size, seq_len = input_ids.shape
        token_weights_tt = self.model.sparse_linear(hidden_state)
        token_weights_tt = self._crop_hidden_state_ttnn(token_weights_tt, batch_size, seq_len)

        if not return_embedding:
            token_weights = to_torch_auto_compose(token_weights_tt, device=self.device)
            if token_weights.dim() == 4 and token_weights.shape[1] == 1:
                token_weights = token_weights.squeeze(1)
            return token_weights.to(torch.float32)

        token_weights_tt = self._flatten_sparse_token_weights_ttnn(token_weights_tt)
        input_ids_tt = ttnn.from_torch(
            input_ids.long().to(torch.int32),
            device=self.device,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        unused_tokens = _get_special_token_ids(self.tokenizer, self.config.vocab_size)
        sparse_embedding_tt = _sparse_embedding_scatter_ttnn(
            self.device,
            token_weights_tt,
            input_ids_tt,
            self.config.vocab_size,
            unused_tokens,
        )
        sparse_embedding = to_torch_auto_compose(sparse_embedding_tt, device=self.device)
        return sparse_embedding.to(torch.float32)

    def _colbert_embedding(self, hidden_state: ttnn.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        self._initialize_model()
        if self.model is None or self.model.colbert_linear is None:
            raise ValueError("ColBERT linear head is not initialized")

        batch_size, seq_len = attention_mask.shape
        tt_hidden = self._crop_hidden_state_ttnn(hidden_state, batch_size, seq_len)
        colbert_tt = self.model.colbert_linear(tt_hidden)
        colbert_tt = ttnn.to_memory_config(colbert_tt, ttnn.DRAM_MEMORY_CONFIG)

        if len(colbert_tt.shape) == 4:
            B, one, S, D = colbert_tt.shape
            colbert_tt = ttnn.slice(colbert_tt, [0, 0, 1, 0], [B, one, S, D])
            mask_torch = attention_mask[:, 1:S].unsqueeze(1).unsqueeze(-1).to(torch.bfloat16)
        else:
            B, S, D = colbert_tt.shape
            colbert_tt = ttnn.slice(colbert_tt, [0, 1, 0], [B, S, D])
            mask_torch = attention_mask[:, 1:S].unsqueeze(-1).to(torch.bfloat16)

        mask_tt = ttnn.from_torch(
            mask_torch,
            device=self.device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        colbert_tt = ttnn.multiply(colbert_tt, mask_tt)

        colbert_vecs = to_torch_auto_compose(colbert_tt, device=self.device)
        if colbert_vecs.dim() == 4 and colbert_vecs.shape[1] == 1:
            colbert_vecs = colbert_vecs.squeeze(1)
        return colbert_vecs.to(torch.float32)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        padded_seq_len = self._get_padded_seq_len(seq_len)

        self._validate_request(batch_size, padded_seq_len)
        self._initialize_model()

        padded_inputs = self._pad_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        output = self.model(
            input_ids=self._to_ttnn_ids(padded_inputs["input_ids"]),
            attention_mask=self._to_ttnn_ids(padded_inputs["attention_mask"]),
            token_type_ids=(
                self._to_ttnn_ids(padded_inputs["token_type_ids"])
                if padded_inputs["token_type_ids"] is not None
                else None
            ),
            position_ids=(
                self._to_ttnn_ids(padded_inputs["position_ids"]) if padded_inputs["position_ids"] is not None else None
            ),
        )

        if output.layout != ttnn.TILE_LAYOUT:
            output = ttnn.to_layout(output, ttnn.TILE_LAYOUT)

        return_dict = {}
        if self.return_dense:
            return_dict["dense_vecs"] = self._dense_embedding(output, padded_inputs["attention_mask"])
        if self.return_sparse:
            return_dict["sparse_vecs"] = self._sparse_embedding(output, padded_inputs["input_ids"])
        if self.return_colbert:
            return_dict["colbert_vecs"] = self._colbert_embedding(output, padded_inputs["attention_mask"])

        return return_dict

    def get_embedding_dim(self) -> int:
        return self.config.hidden_size

    def get_max_seq_len(self) -> int:
        return self.max_seq_len

    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    def _init_pooler(self, vllm_config, prefix: str = "") -> None:
        del vllm_config, prefix
        self.pooler = None


def register_model() -> None:
    try:
        from vllm.model_executor.model_loader import ModelRegistry

        ModelRegistry.register_model(
            "BAAI/bge-m3",
            BgeM3ForEmbedding,
            architecture="RobertaModel",
        )
    except ImportError:
        return
