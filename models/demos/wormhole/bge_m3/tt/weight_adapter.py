from __future__ import annotations

from dataclasses import dataclass

import torch

from models.common.modules.lazy_weight import LazyWeight


def substate(state: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor]:
    """
    Extract a sub-dictionary from a state dict based on a dotted prefix.

    Example:
      substate({"a.b.c": t}, "a.b") -> {"c": t}
    """
    prefix = f"{key}."
    prefix_len = len(prefix)
    return {k[prefix_len:]: v for k, v in state.items() if k.startswith(prefix)}


@dataclass(frozen=True)
class LayerNormWeights:
    weight: LazyWeight
    bias: LazyWeight


@dataclass(frozen=True)
class EmbeddingWeights:
    word_embeddings_weight: LazyWeight
    position_embeddings_weight: LazyWeight
    token_type_embeddings_weight: LazyWeight
    layer_norm: LayerNormWeights | None


@dataclass(frozen=True)
class AttentionWeights:
    wqkv: LazyWeight
    wo_weight: LazyWeight
    bqkv: LazyWeight | None
    wo_bias: LazyWeight | None
    layer_norm: LayerNormWeights | None


@dataclass(frozen=True)
class MLPWeights:
    wi_weight: LazyWeight
    wo_weight: LazyWeight
    wi_bias: LazyWeight | None
    wo_bias: LazyWeight | None
    layer_norm: LayerNormWeights | None


def _to_lazy_weight(tensor: torch.Tensor, dtype: object) -> LazyWeight:
    return LazyWeight(source=tensor, dtype=dtype)


def _preprocess_linear_weight(weight: torch.Tensor) -> torch.Tensor:
    return weight.transpose(-1, -2).contiguous()


def _preprocess_linear_bias(bias: torch.Tensor) -> torch.Tensor:
    return bias.reshape(1, -1).contiguous()


def _require_tensor(state: dict[str, torch.Tensor], key: str, *, scope: str) -> torch.Tensor:
    if key not in state:
        raise KeyError(f"Missing required tensor '{scope}.{key}' in checkpoint state_dict")
    return state[key]


def _maybe_layer_norm_weights(state_dict_slice: dict[str, torch.Tensor], dtype: object) -> LayerNormWeights | None:
    has_weight = "weight" in state_dict_slice
    has_bias = "bias" in state_dict_slice

    if has_weight and has_bias:
        return LayerNormWeights(
            weight=_to_lazy_weight(state_dict_slice["weight"], dtype),
            bias=_to_lazy_weight(state_dict_slice["bias"], dtype),
        )
    if not has_weight and not has_bias:
        return None

    missing_key = "weight" if not has_weight else "bias"
    raise KeyError(
        "Broken LayerNorm checkpoint subtree: expected both 'weight' and 'bias', " f"but '{missing_key}' is missing"
    )


def build_embedding_weights(state_dict: dict[str, torch.Tensor], dtype: object) -> EmbeddingWeights:
    embeddings_state = substate(state_dict, "roberta.embeddings")
    layer_norm_state = substate(embeddings_state, "LayerNorm")

    return EmbeddingWeights(
        word_embeddings_weight=_to_lazy_weight(
            _require_tensor(embeddings_state, "word_embeddings.weight", scope="roberta.embeddings"),
            dtype,
        ),
        position_embeddings_weight=_to_lazy_weight(
            _require_tensor(embeddings_state, "position_embeddings.weight", scope="roberta.embeddings"),
            dtype,
        ),
        token_type_embeddings_weight=_to_lazy_weight(
            _require_tensor(embeddings_state, "token_type_embeddings.weight", scope="roberta.embeddings"),
            dtype,
        ),
        layer_norm=_maybe_layer_norm_weights(layer_norm_state, dtype),
    )


def build_attention_weights(state_dict: dict[str, torch.Tensor], layer_num: int, dtype: object) -> AttentionWeights:
    attention_scope = f"roberta.encoder.layer.{layer_num}.attention"
    attention_state = substate(state_dict, attention_scope)
    self_state = substate(attention_state, "self")
    output_state = substate(attention_state, "output")
    output_layer_norm_state = substate(output_state, "LayerNorm")

    q_weight = _preprocess_linear_weight(_require_tensor(self_state, "query.weight", scope=f"{attention_scope}.self"))
    k_weight = _preprocess_linear_weight(_require_tensor(self_state, "key.weight", scope=f"{attention_scope}.self"))
    v_weight = _preprocess_linear_weight(_require_tensor(self_state, "value.weight", scope=f"{attention_scope}.self"))
    wqkv = torch.cat((q_weight, k_weight, v_weight), dim=-1).contiguous()

    query_bias = self_state.get("query.bias")
    key_bias = self_state.get("key.bias")
    value_bias = self_state.get("value.bias")
    all_qkv_biases_present = all(bias is not None for bias in (query_bias, key_bias, value_bias))
    any_qkv_bias_present = any(bias is not None for bias in (query_bias, key_bias, value_bias))
    if all_qkv_biases_present:
        bqkv = torch.cat(
            (
                _preprocess_linear_bias(query_bias),
                _preprocess_linear_bias(key_bias),
                _preprocess_linear_bias(value_bias),
            ),
            dim=-1,
        ).contiguous()
    elif any_qkv_bias_present:
        raise KeyError("Broken attention checkpoint subtree: query/key/value bias must be all present or all absent")
    else:
        bqkv = None

    wo_weight = _preprocess_linear_weight(
        _require_tensor(output_state, "dense.weight", scope=f"{attention_scope}.output")
    )
    wo_bias = output_state.get("dense.bias")

    return AttentionWeights(
        wqkv=_to_lazy_weight(wqkv, dtype),
        wo_weight=_to_lazy_weight(wo_weight, dtype),
        bqkv=_to_lazy_weight(bqkv, dtype) if bqkv is not None else None,
        wo_bias=_to_lazy_weight(_preprocess_linear_bias(wo_bias), dtype) if wo_bias is not None else None,
        layer_norm=_maybe_layer_norm_weights(output_layer_norm_state, dtype),
    )


def build_mlp_weights(state_dict: dict[str, torch.Tensor], layer_num: int, dtype: object) -> MLPWeights:
    layer_scope = f"roberta.encoder.layer.{layer_num}"
    layer_state = substate(state_dict, layer_scope)
    intermediate_state = substate(layer_state, "intermediate")
    output_state = substate(layer_state, "output")
    intermediate_dense_state = substate(intermediate_state, "dense")
    output_dense_state = substate(output_state, "dense")
    output_layer_norm_state = substate(output_state, "LayerNorm")

    wi_weight = _preprocess_linear_weight(
        _require_tensor(intermediate_dense_state, "weight", scope=f"{layer_scope}.intermediate.dense")
    )
    wo_weight = _preprocess_linear_weight(
        _require_tensor(output_dense_state, "weight", scope=f"{layer_scope}.output.dense")
    )
    wi_bias = intermediate_dense_state.get("bias")
    wo_bias = output_dense_state.get("bias")

    return MLPWeights(
        wi_weight=_to_lazy_weight(wi_weight, dtype),
        wo_weight=_to_lazy_weight(wo_weight, dtype),
        wi_bias=_to_lazy_weight(_preprocess_linear_bias(wi_bias), dtype) if wi_bias is not None else None,
        wo_bias=_to_lazy_weight(_preprocess_linear_bias(wo_bias), dtype) if wo_bias is not None else None,
        layer_norm=_maybe_layer_norm_weights(output_layer_norm_state, dtype),
    )
