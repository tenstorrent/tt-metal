# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Full TP4 autoregressive Mistral-Small-24B-Instruct-2501 model.

The terminal stack deliberately wraps the accepted ``MultichipDecoder``
without changing its tensor-parallel dense kernels, activation/KV dtypes,
collectives, or stacked residual layouts.  Embedding and the untied LM head
are tensor-parallel, paged caches are caller-owned, and decode exposes the
device-resident state needed for split model/sampler traces.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch

import ttnn
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.multichip_decoder import TP_DEGREE, MultichipDecoder
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.optimized_decoder import (
    _dram_matmul_program_config,
    _dram_sharded_weight_memory_config,
    _l1_width_sharded_memory_config,
)
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.precision_policy import dtype_name, fidelity_name
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.modules.tt_ccl import TT_CCL, get_tt_ccl

HF_CONTEXT_LENGTH = 32768
HF_VOCAB_SIZE = 131072
PAGED_BLOCK_SIZE = 32
DEFAULT_PREFILL_CHUNK_SIZE = 576
LM_HEAD_COLUMNS_PER_DEVICE = 8192
LM_HEAD_INPUT_CORES = 10
LM_HEAD_OUTPUT_CORES = 64


def _config_value(config, name: str, default=None):
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _terminal_state_tensor(state_dict: Mapping[str, torch.Tensor], suffix: str) -> torch.Tensor:
    candidates = (suffix, f"model.{suffix}", f"module.{suffix}")
    for name in candidates:
        try:
            return state_dict[name]
        except (KeyError, TypeError):
            continue
    matches = [name for name in state_dict if name.endswith(suffix)]
    if len(matches) == 1:
        return state_dict[matches[0]]
    raise KeyError(f"unable to resolve terminal weight {suffix!r}; matches={matches[:8]}")


def _mesh_tensor(
    tensor: torch.Tensor,
    mesh_device,
    *,
    mesh_mapper,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    return ttnn.from_torch(
        tensor.contiguous(),
        device=mesh_device,
        mesh_mapper=mesh_mapper,
        layout=layout,
        dtype=dtype,
        memory_config=memory_config,
    )


@dataclass(frozen=True)
class FullModelConfig:
    """Accepted full-stack policy derived from the optimized TP4 decoder."""

    max_batch_size: int = 1
    max_context_len: int = HF_CONTEXT_LENGTH
    num_blocks: int = HF_CONTEXT_LENGTH // PAGED_BLOCK_SIZE
    prefill_chunk_size: int = DEFAULT_PREFILL_CHUNK_SIZE
    precision_config_id: str = "inherited_optimized_default"
    precision_config_path: str | None = None
    embedding_weight_dtype: object = ttnn.bfloat16
    norm_weight_dtype: object = ttnn.bfloat16
    attention_weight_dtype: object = ttnn.bfloat4_b
    mlp_gate_up_weight_dtype: object = ttnn.bfloat4_b
    mlp_down_weight_dtype: object = ttnn.bfloat4_b
    attention_math_fidelity: object = ttnn.MathFidelity.LoFi
    mlp_math_fidelity: object = ttnn.MathFidelity.LoFi
    mlp_down_math_fidelity: object = ttnn.MathFidelity.LoFi
    sdpa_math_fidelity: object = ttnn.MathFidelity.HiFi4
    attention_activation_dtype: object = ttnn.bfloat16
    mlp_activation_dtype: object = ttnn.bfloat16
    residual_dtype: object = ttnn.bfloat16
    decode_collective_dtype: object = ttnn.bfloat8_b
    prefill_collective_dtype: object = ttnn.bfloat16
    collective_workspace_dtype: object = ttnn.bfloat16
    attention_geometry: tuple[int, int, int, int, int, int] = (10, 12, 16, 8, 10, 4)
    mlp_geometry: tuple[int, int, int, int, int] = (10, 32, 40, 16, 16)
    kv_cache_dtype: object = ttnn.bfloat8_b
    lm_head_weight_dtype: object = ttnn.bfloat16
    lm_head_math_fidelity: object = ttnn.MathFidelity.HiFi2
    lm_head_columns_per_device: int = LM_HEAD_COLUMNS_PER_DEVICE
    lm_head_input_cores: int = LM_HEAD_INPUT_CORES
    lm_head_output_cores: int = LM_HEAD_OUTPUT_CORES
    lm_head_max_in0_block_w: int = 4
    logits_dtype: object = ttnn.bfloat16
    sampling_logits_dtype: object = ttnn.bfloat16
    sampling_params_dtype: object = ttnn.bfloat16
    sampling_index_dtype: object = ttnn.uint32
    layer_exceptions: Mapping[int, Mapping[str, object]] = field(default_factory=dict)
    override_num_layers: int | None = None

    def validate(self, hf_config) -> None:
        hf_context = int(_config_value(hf_config, "max_position_embeddings", HF_CONTEXT_LENGTH))
        if self.max_batch_size < 1 or self.max_batch_size > 32:
            raise ValueError("max_batch_size must be in [1, 32]")
        if self.max_context_len < 1 or self.max_context_len > hf_context:
            raise ValueError(f"max_context_len must be in [1, {hf_context}]")
        if self.prefill_chunk_size < 32 or self.prefill_chunk_size % 32:
            raise ValueError("prefill_chunk_size must be a positive multiple of 32")
        required_blocks = self.max_batch_size * math.ceil(self.max_context_len / PAGED_BLOCK_SIZE)
        if self.num_blocks < required_blocks:
            raise ValueError(
                f"num_blocks={self.num_blocks} cannot cover batch={self.max_batch_size}, "
                f"context={self.max_context_len}; need at least {required_blocks}"
            )
        if self.kv_cache_dtype not in (ttnn.bfloat8_b, ttnn.bfloat16):
            raise ValueError("the paged-cache split permits only BFP8 or BF16")
        if self.lm_head_weight_dtype not in (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b):
            raise ValueError("LM-head weights must be BF16, BFP8, or BFP4")
        weight_dtypes = (ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b)
        for name in ("attention_weight_dtype", "mlp_gate_up_weight_dtype", "mlp_down_weight_dtype"):
            if getattr(self, name) not in weight_dtypes:
                raise ValueError(f"{name} must be BF16, BFP8, or BFP4")
        if self.embedding_weight_dtype != ttnn.bfloat16 or self.norm_weight_dtype != ttnn.bfloat16:
            raise ValueError("the accepted embedding and norm kernels require BF16 weights")
        if self.residual_dtype != ttnn.bfloat16:
            raise ValueError("the stack-preserved residual contract currently requires BF16")
        if self.collective_workspace_dtype != ttnn.bfloat16:
            raise ValueError("persistent collective workspaces currently require BF16")
        if self.logits_dtype != self.sampling_logits_dtype:
            raise ValueError("Sampling1D input dtype must match the LM-head logits dtype")
        if self.logits_dtype != ttnn.bfloat16 or self.sampling_params_dtype != ttnn.bfloat16:
            raise ValueError("the accepted Sampling1D path requires BF16 logits and parameters")
        if self.sampling_index_dtype != ttnn.uint32:
            raise ValueError("the accepted Sampling1D token/index contract requires UINT32")
        if self.mlp_down_math_fidelity != self.mlp_math_fidelity:
            raise ValueError("the current fused MLP compute config requires gate/up/down to share math fidelity")
        for layer_idx in self.layer_exceptions:
            if int(layer_idx) < 0 or int(layer_idx) >= int(_config_value(hf_config, "num_hidden_layers")):
                raise ValueError(f"layer exception index {layer_idx} is out of range")
        local_vocab_size = int(_config_value(hf_config, "vocab_size")) // TP_DEGREE
        if self.lm_head_columns_per_device < ttnn.TILE_SIZE or (local_vocab_size % self.lm_head_columns_per_device):
            raise ValueError("lm_head_columns_per_device must divide the rank-local vocabulary")
        if self.lm_head_max_in0_block_w < 1:
            raise ValueError("lm_head_max_in0_block_w must be positive")

    def decoder_kwargs(self, layer_idx: int) -> dict[str, object]:
        kwargs = {
            "attention_weight_dtype": self.attention_weight_dtype,
            "mlp_weight_dtype": self.mlp_gate_up_weight_dtype,
            "down_weight_dtype": self.mlp_down_weight_dtype,
            "attention_math_fidelity": self.attention_math_fidelity,
            "mlp_math_fidelity": self.mlp_math_fidelity,
            "sdpa_math_fidelity": self.sdpa_math_fidelity,
            "attention_activation_dtype": dtype_name(self.attention_activation_dtype).lower().replace("_b", ""),
            "mlp_activation_dtype": dtype_name(self.mlp_activation_dtype).lower().replace("_b", ""),
            "residual_dtype": self.residual_dtype,
            "collective_dtype": dtype_name(self.decode_collective_dtype).lower().replace("_b", ""),
            "prefill_collective_dtype": dtype_name(self.prefill_collective_dtype).lower().replace("_b", ""),
            "collective_workspace_dtype": self.collective_workspace_dtype,
            "norm_weight_dtype": self.norm_weight_dtype,
            "attention_geometry": self.attention_geometry,
            "mlp_geometry": self.mlp_geometry,
        }
        kwargs.update(self.layer_exceptions.get(layer_idx, {}))
        return kwargs


class MistralSmall24BFullModel:
    """Embedding, optimized 40-layer TP4 decoder, final norm, LM head, sampler."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config,
        config: FullModelConfig,
        embedding_weight,
        final_norm_weight,
        lm_head_weights,
        layers,
        tt_ccl: TT_CCL,
        sampler: Sampling1D,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.config = config
        self.embedding_weight = embedding_weight
        self.final_norm_weight = final_norm_weight
        self.lm_head_weights = lm_head_weights
        self.layers = layers
        self.tt_ccl = tt_ccl
        self.sampler = sampler

        self.batch = config.max_batch_size
        self.hidden_size = int(_config_value(hf_config, "hidden_size"))
        self.num_heads = int(_config_value(hf_config, "num_attention_heads"))
        self.num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        self.head_dim = int(_config_value(hf_config, "head_dim", self.hidden_size // self.num_heads))
        self.vocab_size = int(_config_value(hf_config, "vocab_size"))
        self.local_vocab_size = self.vocab_size // TP_DEGREE
        self.num_layers = len(layers)
        self.lm_head_columns_per_device = config.lm_head_columns_per_device

        self.lm_head_input_mem_config = _l1_width_sharded_memory_config(
            mesh_device,
            ttnn.TILE_SIZE,
            self.hidden_size,
            config.lm_head_input_cores,
        )
        self.lm_head_program_configs = [
            _dram_matmul_program_config(
                ttnn.TILE_SIZE,
                self.hidden_size,
                config.lm_head_columns_per_device,
                config.lm_head_input_cores,
                config.lm_head_output_cores,
                # Eight K tiles grows terminal circular buffers to 1,782,528
                # bytes on Blackhole. Four stays below the 1,572,864-byte L1
                # limit while retaining the DRAM-sharded LM-head strategy.
                max_in0_block_w=config.lm_head_max_in0_block_w,
            )
            for _ in self.lm_head_weights
        ]
        self.lm_head_compute_kernel = ttnn.WormholeComputeKernelConfig(
            math_fidelity=config.lm_head_math_fidelity,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        hf_config,
        mesh_device,
        full_model_config: FullModelConfig | None = None,
    ) -> "MistralSmall24BFullModel":
        config = full_model_config or FullModelConfig()
        config.validate(hf_config)
        if mesh_device.get_num_devices() != TP_DEGREE or tuple(mesh_device.shape) != (1, TP_DEGREE):
            raise ValueError(f"full model requires a (1, {TP_DEGREE}) mesh")

        hidden_size = int(_config_value(hf_config, "hidden_size"))
        num_heads = int(_config_value(hf_config, "num_attention_heads"))
        num_kv_heads = int(_config_value(hf_config, "num_key_value_heads"))
        vocab_size = int(_config_value(hf_config, "vocab_size"))
        num_layers_total = int(_config_value(hf_config, "num_hidden_layers"))
        if (hidden_size, num_heads, num_kv_heads, vocab_size, num_layers_total) != (
            5120,
            32,
            8,
            HF_VOCAB_SIZE,
            40,
        ):
            raise ValueError("this policy is specific to Mistral-Small-24B-Instruct-2501")
        if vocab_size % TP_DEGREE:
            raise ValueError("vocabulary must divide evenly across TP ranks")

        replicate = ttnn.ReplicateTensorToMesh(mesh_device)
        shard_hidden = ttnn.ShardTensorToMesh(mesh_device, dim=1)
        tt_ccl = get_tt_ccl(mesh_device)

        embedding = _terminal_state_tensor(state_dict, "embed_tokens.weight").to(torch.bfloat16)
        if tuple(embedding.shape) != (vocab_size, hidden_size):
            raise ValueError(f"unexpected embedding shape {tuple(embedding.shape)}")
        embedding_weight = _mesh_tensor(
            embedding,
            mesh_device,
            mesh_mapper=shard_hidden,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=config.embedding_weight_dtype,
        )
        del embedding

        final_norm = _terminal_state_tensor(state_dict, "norm.weight").to(torch.bfloat16)
        final_norm_weight = _mesh_tensor(final_norm, mesh_device, mesh_mapper=replicate, dtype=config.norm_weight_dtype)
        del final_norm

        lm_head = _terminal_state_tensor(state_dict, "lm_head.weight").to(torch.bfloat16).transpose(0, 1)
        if tuple(lm_head.shape) != (hidden_size, vocab_size):
            raise ValueError(f"unexpected LM-head shape {tuple(lm_head.shape)}")
        local_vocab = vocab_size // TP_DEGREE
        lm_head_weights = []
        for split_start in range(0, local_vocab, config.lm_head_columns_per_device):
            rank_splits = [
                lm_head[
                    :,
                    rank * local_vocab
                    + split_start : rank * local_vocab
                    + split_start
                    + config.lm_head_columns_per_device,
                ]
                for rank in range(TP_DEGREE)
            ]
            combined = torch.cat(rank_splits, dim=1)
            weight_memory_config = _dram_sharded_weight_memory_config(
                mesh_device,
                hidden_size,
                config.lm_head_columns_per_device,
            )
            if config.lm_head_columns_per_device > LM_HEAD_COLUMNS_PER_DEVICE:
                # Direct host tilize into a 16K-column DRAM shard requires a
                # 2,208,512-byte static CB on one worker. Stage through tiled
                # interleaved DRAM so this load-time limitation does not reject
                # an otherwise legal runtime matmul candidate.
                weight = _mesh_tensor(
                    combined,
                    mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
                    dtype=config.lm_head_weight_dtype,
                )
                weight = ttnn.to_memory_config(weight, weight_memory_config)
            else:
                weight = _mesh_tensor(
                    combined,
                    mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
                    dtype=config.lm_head_weight_dtype,
                    memory_config=weight_memory_config,
                )
            lm_head_weights.append(weight)
        del lm_head

        num_layers = config.override_num_layers if config.override_num_layers is not None else num_layers_total
        if num_layers < 1 or num_layers > num_layers_total:
            raise ValueError(f"override_num_layers must be in [1, {num_layers_total}]")
        layers = []
        shared_rope = None
        shared_collective = None
        shared_prefill_collective = None
        for layer_idx in range(num_layers):
            layer = MultichipDecoder.from_state_dict(
                state_dict,
                hf_config=hf_config,
                layer_idx=layer_idx,
                mesh_device=mesh_device,
                batch=config.max_batch_size,
                max_cache_len=config.max_context_len,
                shared_rope=shared_rope,
                shared_collective=shared_collective,
                shared_prefill_collective=shared_prefill_collective,
                **config.decoder_kwargs(layer_idx),
            )
            shared_rope = layer.shared_rope
            shared_collective = layer.shared_collective
            shared_prefill_collective = layer.shared_prefill_collective
            layers.append(layer)

        sampler = Sampling1D.from_config(
            Sampling1DConfig(
                vocab_size=vocab_size,
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                max_batch_size=32,
                max_top_k=32,
                num_gather_links=2,
                sampling_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                allow_force_argmax=True,
                num_argmax_gather_links=2,
                ag_topology=ttnn.Topology.Linear,
                pad_to_power_of_2=True,
            )
        )
        return cls(
            mesh_device=mesh_device,
            hf_config=hf_config,
            config=config,
            embedding_weight=embedding_weight,
            final_norm_weight=final_norm_weight,
            lm_head_weights=lm_head_weights,
            layers=layers,
            tt_ccl=tt_ccl,
            sampler=sampler,
        )

    def allocate_kv_cache(self, *, num_blocks: int | None = None, dtype=None):
        """Allocate local KV-head shards directly on every mesh rank."""

        num_blocks = self.config.num_blocks if num_blocks is None else int(num_blocks)
        dtype = self.config.kv_cache_dtype if dtype is None else dtype
        if num_blocks < 1 or num_blocks > self.config.num_blocks:
            raise ValueError(f"num_blocks must be in [1, {self.config.num_blocks}]")
        if dtype not in (ttnn.bfloat8_b, ttnn.bfloat16):
            raise ValueError("KV cache dtype must be BFP8 or BF16")
        local_shape = ttnn.Shape([num_blocks, self.num_kv_heads // TP_DEGREE, PAGED_BLOCK_SIZE, self.head_dim])
        return [
            [
                ttnn.allocate_tensor_on_device(
                    local_shape,
                    dtype,
                    ttnn.TILE_LAYOUT,
                    self.mesh_device,
                    ttnn.DRAM_MEMORY_CONFIG,
                )
                for _ in range(2)
            ]
            for _ in range(self.num_layers)
        ]

    def precision_policy_summary(self) -> dict[str, object]:
        first = self.layers[0]
        last = self.layers[-1]

        def layer_summary(layer):
            return {
                "layer_idx": layer.layer_idx,
                "attention_weight_dtype": dtype_name(layer.qkv_weight.dtype),
                "mlp_gate_up_weight_dtype": dtype_name(layer.gate_weight.dtype),
                "mlp_down_weight_dtype": dtype_name(layer.down_weight.dtype),
                "attention_math_fidelity": fidelity_name(layer.attention_compute_kernel_config.math_fidelity),
                "mlp_math_fidelity": fidelity_name(layer.mlp_compute_kernel_config.math_fidelity),
                "sdpa_math_fidelity": fidelity_name(layer.sdpa_compute_kernel_config.math_fidelity),
                "attention_activation_dtype": dtype_name(layer.attention_activation_dtype),
                "mlp_activation_dtype": dtype_name(layer.mlp_activation_dtype),
                "residual_dtype": dtype_name(layer.residual_dtype),
                "decode_collective_dtype": dtype_name(layer.collective_dtype),
                "prefill_collective_dtype": dtype_name(layer.prefill_collective_dtype),
                "collective_workspace_dtype": dtype_name(layer.collective_workspace.dtype),
                "attention_geometry": list(layer.attention_geometry),
                "mlp_geometry": list(layer.mlp_geometry),
            }

        return {
            "config_id": self.config.precision_config_id,
            "config_path": self.config.precision_config_path,
            "embedding_weight_dtype": dtype_name(self.embedding_weight.dtype),
            "norm_weight_dtype": dtype_name(self.final_norm_weight.dtype),
            "kv_cache_dtype": dtype_name(self.config.kv_cache_dtype),
            "lm_head_weight_dtype": dtype_name(self.lm_head_weights[0].dtype),
            "lm_head_math_fidelity": fidelity_name(self.lm_head_compute_kernel.math_fidelity),
            "logits_dtype": dtype_name(self.config.logits_dtype),
            "sampling_logits_dtype": dtype_name(self.config.sampling_logits_dtype),
            "sampling_params_dtype": dtype_name(self.config.sampling_params_dtype),
            "sampling_index_dtype": dtype_name(self.config.sampling_index_dtype),
            "first_layer": layer_summary(first),
            "last_layer": layer_summary(last),
            "layer_exceptions": sorted(int(index) for index in self.config.layer_exceptions),
        }

    @staticmethod
    def reset_kv_cache(kv_cache) -> None:
        """Zero caller-owned cache buffers without changing trace-bound addresses."""

        for layer_cache in kv_cache:
            for cache in layer_cache:
                ttnn.fill(cache, 0.0, output_tensor=cache)

    def release_prefill_weights(self) -> None:
        """Release only the decoder's large-M duplicate weights for long decode."""

        for layer in self.layers:
            layer.release_prefill_weights()

    @property
    def prefill_weights_released(self) -> bool:
        return all(layer.prefill_weights_released for layer in self.layers)

    def _all_gather_hidden(self, local_hidden):
        return ttnn.all_gather(
            local_hidden,
            dim=3,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )

    def embed_prefill(self, tokens):
        local = ttnn.embedding(
            tokens,
            self.embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return self._all_gather_hidden(ttnn.unsqueeze_to_4D(local))

    def embed_decode(self, tokens):
        local = ttnn.embedding(
            tokens,
            self.embedding_weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden = self._all_gather_hidden(ttnn.unsqueeze_to_4D(local))
        hidden = ttnn.slice(
            hidden,
            [0, 0, 0, 0],
            [1, 1, self.batch, self.hidden_size],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        return ttnn.reshape(hidden, [1, self.batch, 1, self.hidden_size])

    def _final_norm_prefill(self, hidden):
        return ttnn.rms_norm(
            hidden,
            epsilon=float(_config_value(self.hf_config, "rms_norm_eps")),
            weight=self.final_norm_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _final_norm_decode(self, hidden):
        last = self.layers[-1]
        return ttnn.rms_norm(
            hidden,
            epsilon=float(_config_value(self.hf_config, "rms_norm_eps")),
            weight=self.final_norm_weight,
            program_config=last.decode_norm_program_config,
            memory_config=last.decode_norm_mem_config,
        )

    def _lm_head_tile_rows(self, hidden):
        """Run the DRAM-sharded LM head over at most one physical tile row."""

        rows = int(hidden.shape[-2])
        if rows < 1 or rows > ttnn.TILE_SIZE:
            raise ValueError(f"LM-head tile helper expects 1..{ttnn.TILE_SIZE} rows, got {rows}")
        hidden = ttnn.to_memory_config(hidden, self.lm_head_input_mem_config)
        outputs = []
        for weight, program_config in zip(self.lm_head_weights, self.lm_head_program_configs):
            split = ttnn.linear(
                hidden,
                weight,
                dtype=self.config.logits_dtype,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                program_config=program_config,
                compute_kernel_config=self.lm_head_compute_kernel,
            )
            outputs.append(ttnn.sharded_to_interleaved(split, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        return ttnn.concat(outputs, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _lm_head_many_rows(self, hidden):
        rows = int(hidden.shape[-2])
        outputs = []
        for start in range(0, rows, ttnn.TILE_SIZE):
            end = min(start + ttnn.TILE_SIZE, rows)
            tile = ttnn.slice(
                hidden,
                [0, 0, start, 0],
                [1, 1, end, self.hidden_size],
                [1, 1, 1, 1],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            outputs.append(self._lm_head_tile_rows(tile))
        return outputs[0] if len(outputs) == 1 else ttnn.concat(outputs, dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def select_prefill_token_hidden(self, hidden, user_idx: int, token_idx: int):
        if user_idx < 0 or user_idx >= int(hidden.shape[1]):
            raise ValueError("prefill user index is out of range")
        if token_idx < 0 or token_idx >= int(hidden.shape[2]):
            raise ValueError("prefill token index is out of range")
        return ttnn.slice(
            hidden,
            [0, user_idx, token_idx, 0],
            [1, user_idx + 1, token_idx + 1, self.hidden_size],
            [1, 1, 1, 1],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prefill_selected_hidden_logits(self, selected_rows, *, fixed_sampling_rows: bool = False):
        if not selected_rows or len(selected_rows) > self.batch:
            raise ValueError(f"expected between 1 and {self.batch} selected rows")
        selected = selected_rows[0] if len(selected_rows) == 1 else ttnn.concat(selected_rows, dim=2)
        if fixed_sampling_rows and int(selected.shape[-2]) < ttnn.TILE_SIZE:
            selected = ttnn.pad(
                selected,
                [(0, 0), (0, 0), (0, ttnn.TILE_SIZE - int(selected.shape[-2])), (0, 0)],
                value=0.0,
            )
        return self._lm_head_tile_rows(self._final_norm_prefill(selected))

    def prefill_hidden_logits(self, hidden):
        batch = int(hidden.shape[1])
        seq_len = int(hidden.shape[2])
        normed = self._final_norm_prefill(hidden)
        flat = ttnn.reshape(normed, [1, 1, batch * seq_len, self.hidden_size])
        logits = self._lm_head_many_rows(flat)
        return ttnn.reshape(logits, [1, batch, seq_len, self.local_vocab_size])

    def prefill_forward(self, tokens, *, page_table, kv_cache, return_hidden: bool = False):
        batch, seq_len = int(tokens.shape[-2]), int(tokens.shape[-1])
        if batch != self.batch:
            raise ValueError(f"tokens batch={batch} does not match configured batch={self.batch}")
        hidden = self.embed_prefill(tokens)
        hidden, logical_seq_len = self.layers[0].prepare_prefill_residual(hidden)
        for layer, (key_cache, value_cache) in zip(self.layers, kv_cache):
            hidden = layer.prefill_forward_stacked(
                hidden,
                key_cache,
                value_cache,
                logical_seq_len=logical_seq_len,
                page_table=page_table,
            )
        hidden = self.layers[-1].finish_prefill_residual(hidden, logical_seq_len=logical_seq_len)
        if return_hidden:
            return hidden
        return self.prefill_hidden_logits(hidden)

    def decode_forward(
        self,
        tokens,
        current_pos,
        rotary_position,
        *,
        page_table,
        kv_cache,
        advance_positions: bool = True,
    ):
        hidden = self.embed_decode(tokens)
        hidden = self.layers[0].prepare_decode_residual(hidden)
        for layer, (key_cache, value_cache) in zip(self.layers, kv_cache):
            hidden = layer.decode_forward_stacked(
                hidden,
                key_cache,
                value_cache,
                current_pos_tensor=current_pos,
                rotary_pos_tensor=rotary_position,
                page_table=page_table,
            )
        hidden = self._final_norm_decode(hidden)
        # Sampling1D owns fixed 32-row index/seed buffers. Pad only at the
        # terminal boundary so batch-1 remains batch-1 through all 40 decoder
        # layers and the LM-head/sampler trace has a stable physical contract.
        if self.batch < ttnn.TILE_SIZE:
            hidden = ttnn.pad(
                hidden,
                [(0, 0), (0, 0), (0, ttnn.TILE_SIZE - self.batch), (0, 0)],
                value=0.0,
            )
        logits = self._lm_head_tile_rows(hidden)
        if advance_positions:
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(rotary_position)
        return logits

    def sample_split(self, logits, *, k, p, temp, tt_out_tok=None):
        """Canonical local-top-k split sampler; k=1,p=0,temp=1 is greedy."""

        return self.sampler.decode_forward(
            logits,
            k=k,
            p=p,
            temp=temp,
            tt_out_tok=tt_out_tok,
            enable_log_probs=False,
        )[0]


__all__ = [
    "DEFAULT_PREFILL_CHUNK_SIZE",
    "FullModelConfig",
    "HF_CONTEXT_LENGTH",
    "HF_VOCAB_SIZE",
    "LM_HEAD_COLUMNS_PER_DEVICE",
    "MistralSmall24BFullModel",
    "PAGED_BLOCK_SIZE",
]
