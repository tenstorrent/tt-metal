# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Weight providers for the DeepSeek V3 B1 demo pipeline.
CacheWeightProvider loads from disk; SyntheticWeightProvider builds deterministic synthetic weights;
StateDictWeightProvider loads HuggingFace safetensors and runs the same prepare_* path as synthetic.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.utils.lazy_state_dict import LazyStateDict
from models.demos.deepseek_v3_b1.model_dimensions import LogicalModelDimensions
from models.demos.deepseek_v3_b1.weights.cache import CacheConfig, CacheContext, TensorCache
from models.demos.deepseek_v3_b1.weights.prepare import (
    _MTP_LAYER_IDX,
    NUM_ROUTED_EXPERTS,
    DeepSeekV3DenseLayerWeights,
    DeepSeekV3EmbeddingLayerWeights,
    DeepSeekV3LMHeadWeights,
    DeepSeekV3MoELayerWeights,
    DeepSeekV3MTPWeights,
    DenseRoutedExpertWeights,
    MoERoutedExpertWeights,
    OverlappedTensor,
    prepare_attention_weights,
    prepare_dense_layer_weights,
    prepare_embedding_weights,
    prepare_lm_head_weights,
    prepare_moe_layer_weights,
    prepare_mtp_weights,
    prepare_q_ab_kv_a_weights,
    prepare_routed_expert_weights,
    prepare_shared_expert_weights,
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

    def load_mtp(self, device: ttnn.MeshDevice) -> DeepSeekV3MTPWeights:
        ...


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


def _build_synthetic_mtp_state_dict(mtp_layer_idx: int = _MTP_LAYER_IDX) -> dict[str, torch.Tensor]:
    """Build a synthetic MTP state dict with only the lightweight MTP projection/norm tensors."""
    dtype = torch.bfloat16
    H = LogicalModelDimensions.HIDDEN_SIZE

    return {
        _layer_key(mtp_layer_idx, "hnorm.weight"): torch.ones(H, dtype=dtype),
        _layer_key(mtp_layer_idx, "enorm.weight"): torch.ones(H, dtype=dtype),
        _layer_key(mtp_layer_idx, "eh_proj.weight"): torch.randn(H, 2 * H, dtype=dtype),
    }


class CacheWeightProvider:
    """Load weights through TensorCache-backed ``prepare_*`` calls with LazyStateDict miss source.

    The cache directory is created on first use if it does not already exist.
    """

    def __init__(
        self,
        cache_path: Path,
        model_path: Path,
        *,
        hf_model_id: str | None = None,
        hf_revision: str = "local",
        schema_version: int = 1,
    ) -> None:
        cache_path = Path(cache_path)
        model_path = Path(model_path)
        assert model_path.exists(), f"Model path does not exist: {model_path}"
        assert model_path.is_dir(), f"Model path is not a directory: {model_path}"
        self._cache = TensorCache(cache_path)
        self._state_dict = LazyStateDict(model_path)
        self._schema_version = schema_version
        self._hf_model_id = hf_model_id or model_path.name
        self._hf_revision = hf_revision

    def _cache_config(self, device: ttnn.MeshDevice) -> CacheConfig:
        context = CacheContext(
            schema_version=self._schema_version,
            hf_model_id=self._hf_model_id,
            hf_revision=self._hf_revision,
            mesh_shape=(device.shape[0], device.shape[1]),
        )
        return CacheConfig(cache=self._cache, context=context)

    def load_embedding(self, device: ttnn.MeshDevice) -> DeepSeekV3EmbeddingLayerWeights:
        return prepare_embedding_weights(self._state_dict, device, cache_config=self._cache_config(device))

    def load_lm_head(self, device: ttnn.MeshDevice) -> DeepSeekV3LMHeadWeights:
        return prepare_lm_head_weights(self._state_dict, device, cache_config=self._cache_config(device))

    def load_moe_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3MoELayerWeights:
        """Load MoE layer from tensor cache; FD-safe weights use fast dispatch, rest uses slow dispatch."""
        t_load = time.perf_counter()
        cache_config = self._cache_config(device)
        setup_s = time.perf_counter() - t_load

        t_before_with = time.perf_counter()
        with ttnn.device.setup_fast_dispatch(device):
            t_after_fd_init = time.perf_counter()
            fd_init_s = t_after_fd_init - t_before_with
            t0 = time.perf_counter()
            routed = prepare_routed_expert_weights(
                device,
                self._state_dict,
                layer_id,
                is_moe=True,
                num_routed_experts=NUM_ROUTED_EXPERTS,
                move_to_device=True,
                cache_config=cache_config,
            )
            routed_prepare_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            q_ab = prepare_q_ab_kv_a_weights(
                device,
                self._state_dict,
                layer_id,
                move_to_device=True,
                cache_config=cache_config,
            )
            q_ab_prepare_s = time.perf_counter() - t0
            t_before_teardown = time.perf_counter()
        t_after_with = time.perf_counter()
        fd_teardown_s = t_after_with - t_before_teardown

        ttnn.enable_asynchronous_slow_dispatch(device)

        logger.info(f"CacheWeightProvider MoE layer {layer_id}: setup (cache_config) {setup_s:.3f}s")
        logger.info(f"CacheWeightProvider MoE layer {layer_id}: fast_dispatch initialize {fd_init_s:.3f}s")
        logger.info(f"CacheWeightProvider MoE layer {layer_id}: prepare_routed_expert_weights {routed_prepare_s:.3f}s")
        logger.info(f"CacheWeightProvider MoE layer {layer_id}: prepare_q_ab_kv_a_weights (FD) {q_ab_prepare_s:.3f}s")
        logger.info(f"CacheWeightProvider MoE layer {layer_id}: fast_dispatch terminate {fd_teardown_s:.3f}s")

        t0 = time.perf_counter()
        attn = prepare_attention_weights(
            device,
            self._state_dict,
            layer_id,
            is_moe=True,
            move_to_device=True,
            cache_config=cache_config,
            preloaded_q_ab=q_ab,
        )
        attn_s = time.perf_counter() - t0
        logger.info(f"CacheWeightProvider MoE layer {layer_id}: prepare_attention_weights (SD) {attn_s:.3f}s")
        t0 = time.perf_counter()
        shared = prepare_shared_expert_weights(
            device,
            self._state_dict,
            layer_id,
            is_moe=True,
            move_to_device=True,
            cache_config=cache_config,
        )
        shared_s = time.perf_counter() - t0
        logger.info(f"CacheWeightProvider MoE layer {layer_id}: prepare_shared_expert_weights {shared_s:.3f}s")

        total_s = time.perf_counter() - t_load
        sum_parts = setup_s + fd_init_s + routed_prepare_s + q_ab_prepare_s + fd_teardown_s + attn_s + shared_s
        overhead_s = total_s - sum_parts
        logger.info(
            f"CacheWeightProvider MoE layer {layer_id}: load_moe_layer total {total_s:.3f}s "
            f"(sum of parts {sum_parts:.3f}s; unaccounted {overhead_s:+.3f}s — logging / small gaps)"
        )
        assert isinstance(attn.gate_mm, OverlappedTensor)
        assert attn.gate_bias is not None
        assert isinstance(routed, MoERoutedExpertWeights)
        return DeepSeekV3MoELayerWeights(
            q_a_proj=attn.q_a_proj,
            q_b_proj=attn.q_b_proj,
            kv_a_proj=attn.kv_a_proj,
            o_proj=attn.o_proj,
            gate_mm=attn.gate_mm,
            attn_norm=attn.attn_norm,
            q_norm=attn.q_norm,
            kv_norm=attn.kv_norm,
            ffn_norm=attn.ffn_norm,
            gate_bias=attn.gate_bias,
            kv_b1_proj=attn.kv_b1_proj,
            kv_b2_proj=attn.kv_b2_proj,
            shared_gate_proj=shared.shared_gate_proj,
            shared_up_proj=shared.shared_up_proj,
            shared_down_proj=shared.shared_down_proj,
            routed_gate_proj=routed.routed_gate_proj,
            routed_up_proj=routed.routed_up_proj,
            routed_down_proj=routed.routed_down_proj,
        )

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        """Load dense layer; FD-safe weights (routed experts, q_ab_kv_a) use fast dispatch."""
        t_load = time.perf_counter()
        cache_config = self._cache_config(device)

        with ttnn.device.setup_fast_dispatch(device):
            t0 = time.perf_counter()
            routed = prepare_routed_expert_weights(
                device,
                self._state_dict,
                layer_id,
                is_moe=False,
                move_to_device=True,
                cache_config=cache_config,
            )
            routed_s = time.perf_counter() - t0
            t0 = time.perf_counter()
            q_ab = prepare_q_ab_kv_a_weights(
                device,
                self._state_dict,
                layer_id,
                move_to_device=True,
                cache_config=cache_config,
            )
            q_ab_s = time.perf_counter() - t0

        ttnn.enable_asynchronous_slow_dispatch(device)

        t0 = time.perf_counter()
        attn = prepare_attention_weights(
            device,
            self._state_dict,
            layer_id,
            is_moe=False,
            move_to_device=True,
            cache_config=cache_config,
            preloaded_q_ab=q_ab,
        )
        attn_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        shared = prepare_shared_expert_weights(
            device,
            self._state_dict,
            layer_id,
            is_moe=False,
            move_to_device=True,
            cache_config=cache_config,
        )
        shared_s = time.perf_counter() - t0

        assert isinstance(routed, DenseRoutedExpertWeights)
        total_s = time.perf_counter() - t_load
        logger.info(
            f"CacheWeightProvider dense layer {layer_id}: total {total_s:.3f}s "
            f"(routed FD {routed_s:.3f}s, q_ab FD {q_ab_s:.3f}s, attn SD {attn_s:.3f}s, shared SD {shared_s:.3f}s)"
        )
        return DeepSeekV3DenseLayerWeights(
            q_a_proj=attn.q_a_proj,
            q_b_proj=attn.q_b_proj,
            kv_a_proj=attn.kv_a_proj,
            o_proj=attn.o_proj,
            attn_norm=attn.attn_norm,
            q_norm=attn.q_norm,
            kv_norm=attn.kv_norm,
            ffn_norm=attn.ffn_norm,
            kv_b1_proj=attn.kv_b1_proj,
            kv_b2_proj=attn.kv_b2_proj,
            shared_gate_proj=shared.shared_gate_proj,
            shared_up_proj=shared.shared_up_proj,
            shared_down_proj=shared.shared_down_proj,
            routed_gate_proj=routed.routed_gate_proj,
            routed_up_proj=routed.routed_up_proj,
            routed_down_proj=routed.routed_down_proj,
        )

    def load_mtp(self, device: ttnn.MeshDevice) -> DeepSeekV3MTPWeights:
        return prepare_mtp_weights(self._state_dict, device, cache_config=self._cache_config(device))


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
        return prepare_moe_layer_weights(
            device, sd, layer_id, num_routed_experts=NUM_ROUTED_EXPERTS, move_to_device=True
        )

    def load_dense_layer(self, layer_id: int, device: ttnn.MeshDevice) -> DeepSeekV3DenseLayerWeights:
        sd = _build_synthetic_dense_state_dict(layer_id)
        return prepare_dense_layer_weights(device, sd, layer_id, move_to_device=True)

    def load_mtp(self, device: ttnn.MeshDevice) -> DeepSeekV3MTPWeights:
        sd = _build_synthetic_mtp_state_dict()
        return prepare_mtp_weights(sd, device, move_to_device=True)


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

    def load_mtp(self, device: ttnn.MeshDevice) -> DeepSeekV3MTPWeights:
        return prepare_mtp_weights(self._state_dict, device, move_to_device=True)
