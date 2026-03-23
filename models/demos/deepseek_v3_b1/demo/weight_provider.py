# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight providers for the DeepSeek V3 B1 demo pipeline.
CacheWeightProvider loads from disk; SyntheticWeightProvider builds deterministic synthetic weights;
StateDictWeightProvider loads HuggingFace safetensors and runs the same prepare_* path as synthetic;
CachingStateDictWeightProvider wraps StateDictWeightProvider to persist prepared weights to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

import ttnn
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3_b1.prepare_weights import (
    NUM_ROUTED_EXPERTS,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    load_dense_decoder_layer,
    load_embedding_weights,
    load_lm_head_weights,
    load_moe_decoder_layer,
    load_moe_routed_experts,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    save_decoder_layer,
    save_embedding_weights,
    save_lm_head_weights,
)


class WeightProvider(Protocol):
    """Provides embedding and LM head weights on demand; each host loads only what its stage needs."""

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        ...

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        ...

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        ...

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        ...


class LogicalModelDimensions:
    """HF / logical tensor dimensions for DeepSeek V3 B1. Must match prepare_weights and stage expectations."""

    HIDDEN_SIZE = 7168
    VOCAB_SIZE = 129280
    Q_A_DIM = 1536
    Q_B_OUT = 24576
    KV_A_DIM = 576
    KV_B_LORA_RANK = 512
    KV_B_PROJ_OUT = 32768
    O_PROJ_OUT = 16384
    MOE_INTERMEDIATE_SIZE = 2048
    INTERMEDIATE_SIZE = 18432
    GATE_NUM_INDICES = 256


def _layer_key(layer_id: int, suffix: str) -> str:
    """State dict key under model.layers.{layer_id}."""
    return f"model.layers.{layer_id}.{suffix}"


def _build_synthetic_moe_state_dict(
    layer_id: int,
    *,
    num_routed_experts: int = NUM_ROUTED_EXPERTS,
) -> dict[str, torch.Tensor]:
    """Build a synthetic MoE layer state dict with HF tensor shapes (randn for weights, ones for norms)."""
    state_dict: dict[str, torch.Tensor] = {}
    dtype = torch.bfloat16

    # Attention weights (HF shapes)
    state_dict[_layer_key(layer_id, "self_attn.q_a_proj.weight")] = torch.randn(
        LogicalModelDimensions.Q_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.q_b_proj.weight")] = torch.randn(
        LogicalModelDimensions.Q_B_OUT, LogicalModelDimensions.Q_A_DIM, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_a_proj_with_mqa.weight")] = torch.randn(
        LogicalModelDimensions.KV_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_b_proj.weight")] = torch.randn(
        LogicalModelDimensions.KV_B_PROJ_OUT, LogicalModelDimensions.KV_B_LORA_RANK, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.o_proj.weight")] = torch.randn(
        LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.O_PROJ_OUT, dtype=dtype
    )

    # Norms (ones per plan)
    state_dict[_layer_key(layer_id, "input_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.q_a_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.Q_A_DIM, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_a_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.KV_B_LORA_RANK, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "post_attention_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )

    # MoE gate
    state_dict[_layer_key(layer_id, "mlp.gate.weight")] = torch.randn(
        LogicalModelDimensions.GATE_NUM_INDICES, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.gate.e_score_correction_bias")] = torch.randn(
        LogicalModelDimensions.GATE_NUM_INDICES, dtype=dtype
    )

    # Shared experts
    state_dict[_layer_key(layer_id, "mlp.shared_experts.gate_proj.weight")] = torch.randn(
        LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.shared_experts.up_proj.weight")] = torch.randn(
        LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.shared_experts.down_proj.weight")] = torch.randn(
        LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, dtype=dtype
    )

    # Routed experts
    for e in range(num_routed_experts):
        state_dict[_layer_key(layer_id, f"mlp.experts.{e}.gate_proj.weight")] = torch.randn(
            LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
        )
        state_dict[_layer_key(layer_id, f"mlp.experts.{e}.up_proj.weight")] = torch.randn(
            LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
        )
        state_dict[_layer_key(layer_id, f"mlp.experts.{e}.down_proj.weight")] = torch.randn(
            LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.MOE_INTERMEDIATE_SIZE, dtype=dtype
        )

    return state_dict


def _build_synthetic_dense_state_dict(layer_id: int) -> dict[str, torch.Tensor]:
    """Build a synthetic dense layer state dict with HF tensor shapes (randn for weights, ones for norms)."""
    state_dict: dict[str, torch.Tensor] = {}
    dtype = torch.bfloat16

    # Attention weights (HF shapes)
    state_dict[_layer_key(layer_id, "self_attn.q_a_proj.weight")] = torch.randn(
        LogicalModelDimensions.Q_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.q_b_proj.weight")] = torch.randn(
        LogicalModelDimensions.Q_B_OUT, LogicalModelDimensions.Q_A_DIM, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_a_proj_with_mqa.weight")] = torch.randn(
        LogicalModelDimensions.KV_A_DIM, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_b_proj.weight")] = torch.randn(
        LogicalModelDimensions.KV_B_PROJ_OUT, LogicalModelDimensions.KV_B_LORA_RANK, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.o_proj.weight")] = torch.randn(
        LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.O_PROJ_OUT, dtype=dtype
    )

    # Norms (ones per plan)
    state_dict[_layer_key(layer_id, "input_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.q_a_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.Q_A_DIM, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "self_attn.kv_a_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.KV_B_LORA_RANK, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "post_attention_layernorm.weight")] = torch.ones(
        LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )

    # Single MLP (used for both shared and routed in dense)
    state_dict[_layer_key(layer_id, "mlp.gate_proj.weight")] = torch.randn(
        LogicalModelDimensions.INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.up_proj.weight")] = torch.randn(
        LogicalModelDimensions.INTERMEDIATE_SIZE, LogicalModelDimensions.HIDDEN_SIZE, dtype=dtype
    )
    state_dict[_layer_key(layer_id, "mlp.down_proj.weight")] = torch.randn(
        LogicalModelDimensions.HIDDEN_SIZE, LogicalModelDimensions.INTERMEDIATE_SIZE, dtype=dtype
    )

    return state_dict


class CacheWeightProvider:
    """Load embedding and LM head weights from cache; each host loads only what its stage needs."""

    def __init__(self, cache_path: Path) -> None:
        assert cache_path.exists(), f"Cache path does not exist: {cache_path}"
        assert cache_path.is_dir(), f"Cache path is not a directory: {cache_path}"
        self._path = cache_path

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        return load_embedding_weights(self._path, device)

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        return load_lm_head_weights(self._path, device)

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        with ttnn.device.setup_fast_dispatch(device):
            preloaded_experts = load_moe_routed_experts(self._path, device, layer_id)
        ttnn.enable_asynchronous_slow_dispatch(device)
        return load_moe_decoder_layer(self._path, device, layer_id, preloaded_routed_experts=preloaded_experts)

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        return load_dense_decoder_layer(self._path, device, layer_id)


class SyntheticWeightProvider:
    """Create deterministic synthetic embedding and LM head weights in place (no cache)."""

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        emb_w = torch.zeros(
            (LogicalModelDimensions.VOCAB_SIZE, LogicalModelDimensions.HIDDEN_SIZE), dtype=torch.bfloat16
        )
        emb_w[
            torch.arange(LogicalModelDimensions.VOCAB_SIZE),
            torch.arange(LogicalModelDimensions.VOCAB_SIZE, dtype=torch.int64) % LogicalModelDimensions.HIDDEN_SIZE,
        ] = 1
        return prepare_embedding_weights({"model.embed_tokens.weight": emb_w}, device, move_to_device=True)

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        # Stride for synthetic one-hot pattern: 101 matmul cores × 160 per core (matches LM head sampling op layout).
        _lm_head_n_synthetic = 101 * 160
        lm_w = torch.full(
            (LogicalModelDimensions.VOCAB_SIZE, LogicalModelDimensions.HIDDEN_SIZE), -1.0, dtype=torch.bfloat16
        )
        lm_w[
            torch.arange(LogicalModelDimensions.HIDDEN_SIZE, dtype=torch.int64) % _lm_head_n_synthetic,
            torch.arange(LogicalModelDimensions.HIDDEN_SIZE),
        ] = 1
        return prepare_lm_head_weights(
            {
                "lm_head.weight": lm_w,
                "model.norm.weight": torch.ones(LogicalModelDimensions.HIDDEN_SIZE, dtype=torch.bfloat16),
            },
            device,
            move_to_device=True,
        )

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

        sd = _build_synthetic_moe_state_dict(layer_id, num_routed_experts=NUM_ROUTED_EXPERTS)
        bdw = BlitzDecodeWeights(device)
        return prepare_moe_layer_weights(bdw, sd, layer_id, num_routed_experts=NUM_ROUTED_EXPERTS, move_to_device=True)

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

        sd = _build_synthetic_dense_state_dict(layer_id)
        bdw = BlitzDecodeWeights(device)
        return prepare_dense_layer_weights(bdw, sd, layer_id, move_to_device=True)


class StateDictWeightProvider:
    """Load real HF safetensors via LazyStateDict and prepare weights at runtime (no tensorbin cache)."""

    def __init__(self, model_path: Path) -> None:
        model_path = Path(model_path)
        assert model_path.exists(), f"Model path does not exist: {model_path}"
        assert model_path.is_dir(), f"Model path is not a directory: {model_path}"
        self._state_dict = LazyStateDict(model_path)

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        return prepare_embedding_weights(self._state_dict, device, move_to_device=True)

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        return prepare_lm_head_weights(self._state_dict, device, move_to_device=True)

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

        bdw = BlitzDecodeWeights(device)
        return prepare_moe_layer_weights(
            bdw,
            self._state_dict,
            layer_id,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            move_to_device=True,
        )

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        from models.demos.deepseek_v3_b1.blitz_decode_weights import BlitzDecodeWeights

        bdw = BlitzDecodeWeights(device)
        return prepare_dense_layer_weights(bdw, self._state_dict, layer_id, move_to_device=True)


class CachingStateDictWeightProvider:
    """Wraps StateDictWeightProvider: loads via prepare path, then saves to cache."""

    def __init__(self, inner: StateDictWeightProvider, cache_path: Path) -> None:
        self._inner = inner
        self._cache_path = Path(cache_path)
        _exists_err = FileExistsError(f"Cache path already exists (refusing to overwrite): {self._cache_path}.")
        if ttnn.distributed_context_is_initialized():
            rank = int(ttnn.distributed_context_get_rank())
            if rank == 0:
                if self._cache_path.exists():
                    raise _exists_err
                self._cache_path.mkdir(parents=True, exist_ok=False)
            ttnn.distributed_context_barrier()
            if not self._cache_path.is_dir():
                raise RuntimeError(f"Cache directory missing after distributed setup: {self._cache_path}")
        else:
            if self._cache_path.exists():
                raise _exists_err
            self._cache_path.mkdir(parents=True, exist_ok=False)

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        weights = self._inner.load_embedding(device)
        save_embedding_weights(weights, self._cache_path)
        return weights

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        weights = self._inner.load_lm_head(device)
        save_lm_head_weights(weights, self._cache_path)
        return weights

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        weights = self._inner.load_dense_layer(layer_id, device)
        save_decoder_layer(
            weights,
            self._cache_path,
            layer_id,
            hf_model_name="",
            hf_state_dict_name="",
        )
        return weights

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        weights = self._inner.load_moe_layer(layer_id, device)
        save_decoder_layer(
            weights,
            self._cache_path,
            layer_id,
            hf_model_name="",
            hf_state_dict_name="",
        )
        return weights
