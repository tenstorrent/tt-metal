# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight providers for the DeepSeek V3 B1 demo pipeline.
CacheWeightProvider loads from disk; SyntheticWeightProvider builds deterministic synthetic weights;
StateDictWeightProvider loads HuggingFace safetensors and runs the same prepare_* path as synthetic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

import ttnn
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3_b1.prepare_weights import (
    CACHE_TYPE_OVERLAPPED,
    CACHE_TYPE_TENSOR,
    CACHE_TYPE_TENSOR_LIST,
    NUM_ROUTED_EXPERTS,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    embedding_fingerprint,
    layer_fingerprints,
    lm_head_fingerprints,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
)
from models.demos.deepseek_v3_b1.tensor_cache import CacheConfig, TensorCache


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
    """Load weights from a TensorCache (content-addressed) on disk."""

    HF_MODEL_ID = "deepseek-ai/DeepSeek-V3"
    HF_REVISION = "main"

    def __init__(self, cache_path: Path) -> None:
        cache_path = Path(cache_path)
        assert cache_path.exists(), f"Cache path does not exist: {cache_path}"
        assert cache_path.is_dir(), f"Cache path is not a directory: {cache_path}"
        self._cache = TensorCache(cache_path)
        self._cc = CacheConfig(cache=self._cache, hf_model_id=self.HF_MODEL_ID, hf_revision=self.HF_REVISION)

    def _mesh_shape(self, device: ttnn.MeshDevice) -> tuple[int, int]:
        return (device.shape[0], device.shape[1])

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        fp = embedding_fingerprint(self._cc, self._mesh_shape(device))
        return DeepSeekV3EmbeddingLayerWeights(embedding=self._cache.load_tensor(fp, device=device))

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        fps = lm_head_fingerprints(self._cc, self._mesh_shape(device))
        lm_fp, _ = fps["lm_head"]
        norm_fp, _ = fps["final_norm"]
        return DeepSeekV3LMHeadWeights(
            lm_head=self._cache.load_tensor(lm_fp, device=device),
            final_norm=self._cache.load_tensor(norm_fp, device=device),
        )

    def _load_layer_groups(
        self, layer_id: int, device: ttnn.MeshDevice, is_moe: bool
    ) -> dict[str, dict | ttnn.Tensor | list[ttnn.Tensor]]:
        fps = layer_fingerprints(self._cc, self._mesh_shape(device), layer_id, is_moe=is_moe)
        result: dict = {}
        for name, (fp, ctype) in fps.items():
            if ctype == CACHE_TYPE_OVERLAPPED:
                result[name] = self._cache.load_overlapped(fp, device=device)
            elif ctype == CACHE_TYPE_TENSOR:
                result[name] = self._cache.load_tensor(fp, device=device)
            elif ctype == CACHE_TYPE_TENSOR_LIST:
                result[name] = self._cache.load_tensor_list(fp, device=device)
        return result

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        g = self._load_layer_groups(layer_id, device, is_moe=True)
        return DeepSeekV3MoELayerWeights(
            q_a_proj=g["q_ab_kv_a"]["q_a_proj"],
            q_b_proj=g["q_ab_kv_a"]["q_b_proj"],
            kv_a_proj=g["q_ab_kv_a"]["kv_a_proj"],
            o_proj=g["o_proj_gate_mm_norms"]["o_proj"],
            gate_mm=g["o_proj_gate_mm_norms"]["gate_mm"],
            attn_norm=g["o_proj_gate_mm_norms"]["attn_norm"],
            q_norm=g["o_proj_gate_mm_norms"]["q_norm"],
            kv_norm=g["o_proj_gate_mm_norms"]["kv_norm"],
            ffn_norm=g["o_proj_gate_mm_norms"]["ffn_norm"],
            gate_bias=g["gate_bias"],
            kv_b1_proj=g["kv_b12"]["kv_b1_proj"],
            kv_b2_proj=g["kv_b12"]["kv_b2_proj"],
            shared_gate_proj=g["gate_up"]["gate_proj"],
            shared_up_proj=g["gate_up"]["up_proj"],
            shared_down_proj=g["shared_down_proj"],
            routed_gate_proj=g["routed_gate_proj"],
            routed_up_proj=g["routed_up_proj"],
            routed_down_proj=g["routed_down_proj"],
        )

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        g = self._load_layer_groups(layer_id, device, is_moe=False)
        return DeepSeekV3DenseLayerWeights(
            q_a_proj=g["q_ab_kv_a"]["q_a_proj"],
            q_b_proj=g["q_ab_kv_a"]["q_b_proj"],
            kv_a_proj=g["q_ab_kv_a"]["kv_a_proj"],
            o_proj=g["o_proj_gate_mm_norms"]["o_proj"],
            attn_norm=g["o_proj_gate_mm_norms"]["attn_norm"],
            q_norm=g["o_proj_gate_mm_norms"]["q_norm"],
            kv_norm=g["o_proj_gate_mm_norms"]["kv_norm"],
            ffn_norm=g["o_proj_gate_mm_norms"]["ffn_norm"],
            kv_b1_proj=g["kv_b12"]["kv_b1_proj"],
            kv_b2_proj=g["kv_b12"]["kv_b2_proj"],
            shared_gate_proj=g["gate_up"]["gate_proj"],
            shared_up_proj=g["gate_up"]["up_proj"],
            shared_down_proj=g["shared_down_proj"],
            routed_gate_proj=g["routed_gate_proj"],
            routed_up_proj=g["routed_up_proj"],
            routed_down_proj=g["routed_down_proj"],
        )


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
        sd = _build_synthetic_moe_state_dict(layer_id, num_routed_experts=NUM_ROUTED_EXPERTS)
        return prepare_moe_layer_weights(device, sd, layer_id, num_routed_experts=NUM_ROUTED_EXPERTS)

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        sd = _build_synthetic_dense_state_dict(layer_id)
        return prepare_dense_layer_weights(device, sd, layer_id, move_to_device=True)


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
        return prepare_moe_layer_weights(
            device,
            self._state_dict,
            layer_id,
            num_routed_experts=NUM_ROUTED_EXPERTS,
            move_to_device=True,
        )

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        return prepare_dense_layer_weights(device, self._state_dict, layer_id, move_to_device=True)
