# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

import ttnn

ATTN_PARAM_NAMES = ("q_weight", "q_bias", "k_weight", "k_bias", "v_weight", "v_bias", "o_weight", "o_bias")
FFN_PARAM_NAMES = ("w1", "b1", "w2", "b2")


def to_float_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float()


class _ProjectionStateAdapter:
    """Adapter so model output projection participates in module-owned state loading."""

    def __init__(self, model):
        self.model = model

    def load_ttnn_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
        mapping = (
            ("weight", "proj_w", "proj_w_torch"),
            ("bias", "proj_b", "proj_b_torch"),
        )
        used: set[str] = set()
        missing: list[str] = []
        for key, tensor_attr, torch_attr in mapping:
            tensor = state.get(key)
            if tensor is None:
                missing.append(key)
                continue
            used.add(key)
            value = to_float_tensor(tensor)
            setattr(self.model, torch_attr, value)
            ref = getattr(self.model, tensor_attr)
            setattr(
                self.model,
                tensor_attr,
                ttnn.from_torch(value, device=self.model.device, dtype=ref.dtype, layout=ttnn.TILE_LAYOUT),
            )
        unexpected = sorted(k for k in state if k not in used)
        if strict and missing:
            raise ValueError(f"Missing projection weights: {missing}")
        return {"missing_keys": missing, "unexpected_keys": unexpected}


def export_ttnn_model_state(model) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    emb = model.embedding
    state["embedding.value.weight"] = emb.value_embedding.weight_torch.clone()
    state["embedding.value.bias"] = emb.value_embedding.bias_torch.clone()
    state["embedding.temporal.weight"] = emb.temporal_embedding.weight_torch.clone()
    state["embedding.temporal.bias"] = emb.temporal_embedding.bias_torch.clone()

    for i, layer in enumerate(model.encoder.layers):
        prefix = f"encoder.layers.{i}"
        attn = layer.attn
        for name in ATTN_PARAM_NAMES:
            state[f"{prefix}.attn.{name}"] = getattr(attn, f"{name}_torch").clone()

        for name in FFN_PARAM_NAMES:
            state[f"{prefix}.ffn.{name}"] = getattr(layer.ffn, f"{name}_torch").clone()

        for norm_name in ("norm1", "norm2"):
            norm = getattr(layer, norm_name)
            state[f"{prefix}.{norm_name}.weight"] = norm.weight_torch.clone()
            state[f"{prefix}.{norm_name}.bias"] = norm.bias_torch.clone()

    if model.encoder.distil_norm is not None:
        state["encoder.distil_norm.weight"] = model.encoder.distil_norm.weight_torch.clone()
        state["encoder.distil_norm.bias"] = model.encoder.distil_norm.bias_torch.clone()

    for i, layer in enumerate(model.decoder.layers):
        prefix = f"decoder.layers.{i}"
        for attn_name in ("self_attn", "cross_attn"):
            attn = getattr(layer, attn_name)
            for name in ATTN_PARAM_NAMES:
                state[f"{prefix}.{attn_name}.{name}"] = getattr(attn, f"{name}_torch").clone()

        for name in FFN_PARAM_NAMES:
            state[f"{prefix}.ffn.{name}"] = getattr(layer.ffn, f"{name}_torch").clone()

        for norm_name in ("norm1", "norm2", "norm3"):
            norm = getattr(layer, norm_name)
            state[f"{prefix}.{norm_name}.weight"] = norm.weight_torch.clone()
            state[f"{prefix}.{norm_name}.bias"] = norm.bias_torch.clone()

    state["projection.weight"] = model.proj_w_torch.clone()
    state["projection.bias"] = model.proj_b_torch.clone()
    return state


def load_ttnn_model_state(model, state: dict[str, torch.Tensor], *, strict: bool = True) -> dict[str, list[str]]:
    from .hf_runtime import PrefixedModuleLoadSpec, apply_prefixed_module_loads

    specs: list[PrefixedModuleLoadSpec] = [
        PrefixedModuleLoadSpec(
            module=model.embedding.value_embedding,
            prefix="embedding.value",
            load_method="load_ttnn_state_dict",
        ),
        PrefixedModuleLoadSpec(
            module=model.embedding.temporal_embedding,
            prefix="embedding.temporal",
            load_method="load_ttnn_state_dict",
        ),
    ]

    for index, layer in enumerate(model.encoder.layers):
        prefix = f"encoder.layers.{index}"
        specs.extend(
            [
                PrefixedModuleLoadSpec(module=layer.attn, prefix=f"{prefix}.attn", load_method="load_ttnn_state_dict"),
                PrefixedModuleLoadSpec(module=layer.ffn, prefix=f"{prefix}.ffn", load_method="load_ttnn_state_dict"),
                PrefixedModuleLoadSpec(
                    module=layer.norm1, prefix=f"{prefix}.norm1", load_method="load_ttnn_state_dict"
                ),
                PrefixedModuleLoadSpec(
                    module=layer.norm2, prefix=f"{prefix}.norm2", load_method="load_ttnn_state_dict"
                ),
            ]
        )

    if model.encoder.distil_norm is not None:
        specs.append(
            PrefixedModuleLoadSpec(
                module=model.encoder.distil_norm,
                prefix="encoder.distil_norm",
                load_method="load_ttnn_state_dict",
            )
        )

    for index, layer in enumerate(model.decoder.layers):
        prefix = f"decoder.layers.{index}"
        specs.extend(
            [
                PrefixedModuleLoadSpec(
                    module=layer.self_attn,
                    prefix=f"{prefix}.self_attn",
                    load_method="load_ttnn_state_dict",
                ),
                PrefixedModuleLoadSpec(
                    module=layer.cross_attn,
                    prefix=f"{prefix}.cross_attn",
                    load_method="load_ttnn_state_dict",
                ),
                PrefixedModuleLoadSpec(module=layer.ffn, prefix=f"{prefix}.ffn", load_method="load_ttnn_state_dict"),
                PrefixedModuleLoadSpec(
                    module=layer.norm1, prefix=f"{prefix}.norm1", load_method="load_ttnn_state_dict"
                ),
                PrefixedModuleLoadSpec(
                    module=layer.norm2, prefix=f"{prefix}.norm2", load_method="load_ttnn_state_dict"
                ),
                PrefixedModuleLoadSpec(
                    module=layer.norm3, prefix=f"{prefix}.norm3", load_method="load_ttnn_state_dict"
                ),
            ]
        )

    specs.append(
        PrefixedModuleLoadSpec(
            module=_ProjectionStateAdapter(model),
            prefix="projection",
            load_method="load_ttnn_state_dict",
        )
    )
    return apply_prefixed_module_loads(state, specs, strict=strict)


def export_torch_reference_state(torch_model) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    state["embedding.value.weight"] = torch_model.embedding.value_embedding.weight.detach().cpu()
    state["embedding.value.bias"] = torch_model.embedding.value_embedding.bias.detach().cpu()
    state["embedding.temporal.weight"] = torch_model.embedding.temporal_embedding.weight.detach().cpu()
    state["embedding.temporal.bias"] = torch_model.embedding.temporal_embedding.bias.detach().cpu()

    for i, layer in enumerate(torch_model.encoder.layers):
        prefix = f"encoder.layers.{i}"
        attn = layer.attn
        state[f"{prefix}.attn.q_weight"] = attn.q_proj.weight.detach().cpu()
        state[f"{prefix}.attn.q_bias"] = attn.q_proj.bias.detach().cpu()
        state[f"{prefix}.attn.k_weight"] = attn.k_proj.weight.detach().cpu()
        state[f"{prefix}.attn.k_bias"] = attn.k_proj.bias.detach().cpu()
        state[f"{prefix}.attn.v_weight"] = attn.v_proj.weight.detach().cpu()
        state[f"{prefix}.attn.v_bias"] = attn.v_proj.bias.detach().cpu()
        state[f"{prefix}.attn.o_weight"] = attn.o_proj.weight.detach().cpu()
        state[f"{prefix}.attn.o_bias"] = attn.o_proj.bias.detach().cpu()

        state[f"{prefix}.ffn.w1"] = layer.ffn.fc1.weight.detach().cpu()
        state[f"{prefix}.ffn.b1"] = layer.ffn.fc1.bias.detach().cpu()
        state[f"{prefix}.ffn.w2"] = layer.ffn.fc2.weight.detach().cpu()
        state[f"{prefix}.ffn.b2"] = layer.ffn.fc2.bias.detach().cpu()

        state[f"{prefix}.norm1.weight"] = layer.norm1.weight.detach().cpu()
        state[f"{prefix}.norm1.bias"] = layer.norm1.bias.detach().cpu()
        state[f"{prefix}.norm2.weight"] = layer.norm2.weight.detach().cpu()
        state[f"{prefix}.norm2.bias"] = layer.norm2.bias.detach().cpu()

    if torch_model.encoder.distil_norm is not None:
        state["encoder.distil_norm.weight"] = torch_model.encoder.distil_norm.weight.detach().cpu()
        state["encoder.distil_norm.bias"] = torch_model.encoder.distil_norm.bias.detach().cpu()

    for i, layer in enumerate(torch_model.decoder.layers):
        prefix = f"decoder.layers.{i}"
        attn = layer.self_attn
        state[f"{prefix}.self_attn.q_weight"] = attn.q_proj.weight.detach().cpu()
        state[f"{prefix}.self_attn.q_bias"] = attn.q_proj.bias.detach().cpu()
        state[f"{prefix}.self_attn.k_weight"] = attn.k_proj.weight.detach().cpu()
        state[f"{prefix}.self_attn.k_bias"] = attn.k_proj.bias.detach().cpu()
        state[f"{prefix}.self_attn.v_weight"] = attn.v_proj.weight.detach().cpu()
        state[f"{prefix}.self_attn.v_bias"] = attn.v_proj.bias.detach().cpu()
        state[f"{prefix}.self_attn.o_weight"] = attn.o_proj.weight.detach().cpu()
        state[f"{prefix}.self_attn.o_bias"] = attn.o_proj.bias.detach().cpu()

        attn = layer.cross_attn
        state[f"{prefix}.cross_attn.q_weight"] = attn.q_proj.weight.detach().cpu()
        state[f"{prefix}.cross_attn.q_bias"] = attn.q_proj.bias.detach().cpu()
        state[f"{prefix}.cross_attn.k_weight"] = attn.k_proj.weight.detach().cpu()
        state[f"{prefix}.cross_attn.k_bias"] = attn.k_proj.bias.detach().cpu()
        state[f"{prefix}.cross_attn.v_weight"] = attn.v_proj.weight.detach().cpu()
        state[f"{prefix}.cross_attn.v_bias"] = attn.v_proj.bias.detach().cpu()
        state[f"{prefix}.cross_attn.o_weight"] = attn.o_proj.weight.detach().cpu()
        state[f"{prefix}.cross_attn.o_bias"] = attn.o_proj.bias.detach().cpu()

        state[f"{prefix}.ffn.w1"] = layer.ffn.fc1.weight.detach().cpu()
        state[f"{prefix}.ffn.b1"] = layer.ffn.fc1.bias.detach().cpu()
        state[f"{prefix}.ffn.w2"] = layer.ffn.fc2.weight.detach().cpu()
        state[f"{prefix}.ffn.b2"] = layer.ffn.fc2.bias.detach().cpu()

        state[f"{prefix}.norm1.weight"] = layer.norm1.weight.detach().cpu()
        state[f"{prefix}.norm1.bias"] = layer.norm1.bias.detach().cpu()
        state[f"{prefix}.norm2.weight"] = layer.norm2.weight.detach().cpu()
        state[f"{prefix}.norm2.bias"] = layer.norm2.bias.detach().cpu()
        state[f"{prefix}.norm3.weight"] = layer.norm3.weight.detach().cpu()
        state[f"{prefix}.norm3.bias"] = layer.norm3.bias.detach().cpu()

    state["projection.weight"] = torch_model.proj.weight.detach().cpu()
    state["projection.bias"] = torch_model.proj.bias.detach().cpu()
    return state
