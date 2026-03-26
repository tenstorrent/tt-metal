# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import torch
import transformers

import ttnn
from models.common.auto_compose import to_torch_auto_compose
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
            "input_ids": self._pad_tensor(input_ids, padded_seq_len, pad_value=self.tokenizer.pad_token_id),
            "attention_mask": self._pad_tensor(attention_mask, padded_seq_len, pad_value=0),
            "token_type_ids": self._pad_tensor(token_type_ids, padded_seq_len, pad_value=0)
            if token_type_ids is not None
            else None,
            "position_ids": self._pad_tensor(position_ids, padded_seq_len, pad_value=self.tokenizer.pad_token_id)
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
    def _pool_embeddings(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

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

        last_hidden_state = to_torch_auto_compose(output, device=self.device)
        if last_hidden_state.dim() == 4 and last_hidden_state.shape[1] == 1:
            last_hidden_state = last_hidden_state.squeeze(1)

        return last_hidden_state.to(torch.float32)

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
