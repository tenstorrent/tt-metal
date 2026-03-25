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
from models.tt_transformers.tt.generator import create_submeshes


class BgeM3ForEmbedding:
    """
    vLLM-facing embedding wrapper for direct BGE-M3 encoder execution.

    `tt_data_parallel=1` keeps inputs replicated across the mesh.
    `tt_data_parallel=num_devices` enables simple full-mesh data parallelism by
    padding the batch, sharding along batch dim 0, and composing hidden states
    back on host.
    """

    def __init__(
        self,
        device: ttnn.Device = None,
        model_location_generator=None,
        max_batch_size: int = 8,
        max_seq_len: int = 8192,
        tt_data_parallel: int = 1,
        dtype=ttnn.bfloat16,
        model_name: str = "BAAI/bge-m3",
        vllm_config=None,
        prefix: str = "",
        **kwargs,
    ):
        del prefix, kwargs

        if vllm_config is not None and device is None:
            device = vllm_config.device_config.device

        if device is None:
            raise ValueError("Either 'device' or 'vllm_config' must be provided")

        self.device = device
        self.model_location_generator = model_location_generator
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
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
        self.model_args_list = None
        self.models = None
        self.submeshes = None
        self.state_dict = None
        self.tokenizer = None

    # Reference: models/tt_transformers/tt/generator.py::create_submeshes
    def _get_num_devices(self) -> int:
        get_num_devices = getattr(self.device, "get_num_devices", None)
        if callable(get_num_devices):
            return int(get_num_devices())
        return 1

    # Reference: models/tt_transformers/tt/generator.py::create_submeshes
    def _get_effective_tt_data_parallel(self) -> int:
        if self.tt_data_parallel < 1:
            raise ValueError(f"tt_data_parallel must be >= 1, got {self.tt_data_parallel}")

        num_devices = self._get_num_devices()
        if self.tt_data_parallel == 1:
            return 1
        if self.tt_data_parallel == num_devices:
            return num_devices

        raise ValueError(
            f"BGE-M3 currently supports tt_data_parallel=1 or tt_data_parallel=num_devices ({num_devices}), "
            f"got {self.tt_data_parallel}"
        )

    def _get_pad_token_id(self) -> int:
        tokenizer_pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if tokenizer_pad_token_id is not None:
            return int(tokenizer_pad_token_id)

        config_pad_token_id = getattr(self.config, "pad_token_id", 0)
        return 0 if config_pad_token_id is None else int(config_pad_token_id)

    # Reference: models/tt_transformers/tt/generator_vllm.py::initialize_vllm_text_transformer
    def _get_padded_batch_size(self, batch_size: int) -> int:
        tt_data_parallel = self._get_effective_tt_data_parallel()
        if tt_data_parallel <= 1:
            return batch_size
        return ((batch_size + tt_data_parallel - 1) // tt_data_parallel) * tt_data_parallel

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
        **kwargs,
    ) -> "BgeM3ForEmbedding":
        # del hf_config, tt_data_parallel, optimizations, kwargs

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
                tt_data_parallel=tt_data_parallel,
                vllm_config=vllm_config,
            )

        return cls(
            device=mesh_device,
            model_location_generator=model_location_generator,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
            tt_data_parallel=tt_data_parallel,
        )

    # Reference: models/tt_transformers/tt/generator_vllm.py::initialize_vllm_text_transformer
    def _initialize_model(self) -> None:
        if self._is_initialized and self.models is not None and self.model_args_list is not None:
            return

        tt_data_parallel = self._get_effective_tt_data_parallel()
        submeshes = create_submeshes(self.device, tt_data_parallel)
        model_max_batch_size = self._get_padded_batch_size(self.max_batch_size)
        per_submesh_max_batch_size = model_max_batch_size // tt_data_parallel

        model_args_list = []
        models = []
        state_dict = self.state_dict
        for submesh in submeshes:
            model_args_i, model_i, state_dict = create_tt_model(
                mesh_device=submesh,
                max_batch_size=per_submesh_max_batch_size,
                max_seq_len=self.max_seq_len,
                dtype=self.dtype,
                state_dict=state_dict,
                hf_model_name=self.model_name,
            )
            model_args_list.append(model_args_i)
            models.append(model_i)

        self.submeshes = submeshes
        self.model_args_list = model_args_list
        self.models = models
        self.tokenizer = self.model_args_list[0].tokenizer

        # Preserve current external callers until forward/demo paths switch to list-backed state.
        self.model_args = self.model_args_list[0]
        self.model = self.models[0]
        self.state_dict = state_dict
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

    @staticmethod
    def _normalize_position_ids(position_ids: Optional[torch.Tensor], batch_size: int) -> Optional[torch.Tensor]:
        if position_ids is None:
            return None
        if position_ids.shape[0] == batch_size:
            return position_ids
        if position_ids.shape[0] == 1 and batch_size > 1:
            return position_ids.expand(batch_size, -1).clone()
        raise ValueError(
            f"position_ids batch dimension must be 1 or match input batch size {batch_size}, "
            f"got shape={tuple(position_ids.shape)}"
        )

    def _pad_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> dict[str, Optional[torch.Tensor]]:
        batch_size = input_ids.shape[0]
        padded_seq_len = self._get_padded_seq_len(input_ids.shape[1])
        padded_batch_size = self._get_padded_batch_size(batch_size)
        pad_token_id = self._get_pad_token_id()

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = self._normalize_position_ids(position_ids, batch_size)

        padded_inputs = {
            "input_ids": self._pad_batch_tensor(
                self._pad_tensor(input_ids, padded_seq_len, pad_value=pad_token_id),
                padded_batch_size,
                pad_value=pad_token_id,
            ),
            "attention_mask": self._pad_batch_tensor(
                self._pad_tensor(attention_mask, padded_seq_len, pad_value=0),
                padded_batch_size,
                pad_value=0,
            ),
            "token_type_ids": self._pad_batch_tensor(
                self._pad_tensor(token_type_ids, padded_seq_len, pad_value=0), padded_batch_size, pad_value=0
            )
            if token_type_ids is not None
            else None,
            "position_ids": self._pad_batch_tensor(
                self._pad_tensor(position_ids, padded_seq_len, pad_value=0), padded_batch_size, pad_value=0
            )
            if position_ids is not None
            else None,
        }

        return padded_inputs

    # Reference: models/tt_transformers/tt/generator.py batch chunking via data_parallel
    def _to_ttnn_ids(
        self, ids: torch.Tensor, *, device: Optional[ttnn.Device] = None, shard_batch: bool = False
    ) -> ttnn.Tensor:
        target_device = self.device if device is None else device
        from_torch_kwargs = {
            "device": target_device,
            "dtype": ttnn.uint32,
            "layout": ttnn.ROW_MAJOR_LAYOUT,
        }
        if shard_batch and device is None and self._get_effective_tt_data_parallel() > 1:
            from_torch_kwargs["mesh_mapper"] = ttnn.ShardTensorToMesh(self.device, dim=0)

        return ttnn.from_torch(ids.to(torch.int32), **from_torch_kwargs)

    @staticmethod
    def _pool_embeddings(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)
        return summed / counts

    # Reference: models/tt_transformers/tt/generator.py::torch.chunk(..., self.data_parallel, 0)
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

        models = self.models if self.models is not None else [self.model]
        submeshes = self.submeshes if self.submeshes is not None else [self.device]
        input_ids_chunks = torch.chunk(padded_inputs["input_ids"], len(models), dim=0)
        attention_mask_chunks = torch.chunk(padded_inputs["attention_mask"], len(models), dim=0)
        token_type_id_chunks = (
            torch.chunk(padded_inputs["token_type_ids"], len(models), dim=0)
            if padded_inputs["token_type_ids"] is not None
            else [None] * len(models)
        )
        position_id_chunks = (
            torch.chunk(padded_inputs["position_ids"], len(models), dim=0)
            if padded_inputs["position_ids"] is not None
            else [None] * len(models)
        )

        output_chunks = []
        for model, submesh, input_ids_chunk, attention_mask_chunk, token_type_ids_chunk, position_ids_chunk in zip(
            models,
            submeshes,
            input_ids_chunks,
            attention_mask_chunks,
            token_type_id_chunks,
            position_id_chunks,
        ):
            output = model(
                input_ids=self._to_ttnn_ids(input_ids_chunk, device=submesh),
                attention_mask=self._to_ttnn_ids(attention_mask_chunk, device=submesh),
                token_type_ids=(
                    self._to_ttnn_ids(token_type_ids_chunk, device=submesh)
                    if token_type_ids_chunk is not None
                    else None
                ),
                position_ids=(
                    self._to_ttnn_ids(position_ids_chunk, device=submesh) if position_ids_chunk is not None else None
                ),
            )

            last_hidden_state = to_torch_auto_compose(output, device=submesh)
            if last_hidden_state.dim() == 4 and last_hidden_state.shape[1] == 1:
                last_hidden_state = last_hidden_state.squeeze(1)
            output_chunks.append(last_hidden_state)

        last_hidden_state = torch.cat(output_chunks, dim=0) if len(output_chunks) > 1 else output_chunks[0]

        if last_hidden_state.shape[0] > batch_size:
            last_hidden_state = last_hidden_state[:batch_size]

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
