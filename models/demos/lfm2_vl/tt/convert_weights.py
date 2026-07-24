# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Load LiquidAI/LFM2.5-VL-1.6B safetensors into nested parameter objects for ttnn.

Checkpoint key layout (public HF release)::

    model.vision_tower.vision_model.*
    model.multi_modal_projector.linear_{1,2}.*
    model.language_model.embed_tokens / embedding_norm / layers.*
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import torch

import ttnn

from models.demos.lfm2_vl.tt.model_config import create_model_config


def _make_obj(**kwargs) -> Any:
    return type("Params", (), kwargs)()


def _tensor_to_ttnn(
    tensor: torch.Tensor,
    device,
    *,
    transpose: bool = False,
    dtype=ttnn.bfloat16,
) -> ttnn.Tensor:
    if transpose and tensor.ndim >= 2:
        tensor = tensor.transpose(-2, -1).contiguous()
    # Prefer TILE when last two dims are tile-aligned; otherwise ROW_MAJOR.
    layout = ttnn.TILE_LAYOUT
    if tensor.ndim >= 2 and (tensor.shape[-1] % 32 != 0 or tensor.shape[-2] % 32 != 0):
        layout = ttnn.ROW_MAJOR_LAYOUT
    if tensor.ndim == 1:
        layout = ttnn.ROW_MAJOR_LAYOUT
    return ttnn.from_torch(tensor, device=device, dtype=dtype, layout=layout)


def _open_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    """Load a single .safetensors file or a directory containing one."""
    path = model_path
    if os.path.isdir(path):
        candidates = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if name.endswith(".safetensors")
        ]
        if not candidates:
            raise FileNotFoundError(f"No .safetensors files under {path}")
        # Prefer a single non-index file
        non_index = [c for c in candidates if not c.endswith(".index.json")]
        path = sorted(non_index)[0] if non_index else candidates[0]

    if path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(path)

    if path.endswith((".bin", ".pt", ".pth")):
        return torch.load(path, map_location="cpu", weights_only=True)

    raise ValueError(f"Unsupported weight path: {model_path}")


def _require(state: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
    if key not in state:
        raise KeyError(
            f"Missing checkpoint key '{key}'. "
            "Expected LiquidAI/LFM2.5-VL-1.6B layout "
            "(model.vision_tower.*, model.multi_modal_projector.*, model.language_model.*)."
        )
    return state[key]


def _optional(state: Dict[str, torch.Tensor], key: str) -> Optional[torch.Tensor]:
    return state.get(key)


def _linear_params(state, weight_key: str, device, *, bias_key: Optional[str] = None, transpose: bool = True):
    weight = _tensor_to_ttnn(_require(state, weight_key), device, transpose=transpose)
    kwargs = {"weight": weight}
    if bias_key is not None:
        bias = _optional(state, bias_key)
        if bias is not None:
            kwargs["bias"] = _tensor_to_ttnn(bias, device, transpose=False)
    return _make_obj(**kwargs)


def _norm_params(state, weight_key: str, device, *, bias_key: Optional[str] = None):
    kwargs = {"weight": _tensor_to_ttnn(_require(state, weight_key), device, transpose=False)}
    if bias_key is not None and bias_key in state:
        kwargs["bias"] = _tensor_to_ttnn(state[bias_key], device, transpose=False)
    return _make_obj(**kwargs)


def convert_lfm2_weights(
    model_path: str,
    device,
    model_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Convert HF LFM2.5-VL weights into attribute-accessible parameter trees."""
    if model_config is None:
        model_config = create_model_config()

    state = _open_state_dict(model_path)
    # Normalize accidental top-level prefixes (some exports omit ``model.``)
    if not any(k.startswith("model.") for k in state):
        state = {f"model.{k}": v for k, v in state.items()}

    vision_cfg = model_config["vision_config"]
    num_layers = model_config["num_hidden_layers"]
    layer_types = model_config["layer_types"]

    # --- Language embeddings ---
    embed_tokens = _linear_params(
        state,
        "model.language_model.embed_tokens.weight",
        device,
        transpose=False,
    )
    embedding_norm = _norm_params(state, "model.language_model.embedding_norm.weight", device)

    # --- Vision tower ---
    patch_embedding = _linear_params(
        state,
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        device,
        bias_key="model.vision_tower.vision_model.embeddings.patch_embedding.bias",
        transpose=True,
    )
    position_embedding = _make_obj(
        weight=_tensor_to_ttnn(
            _require(state, "model.vision_tower.vision_model.embeddings.position_embedding.weight"),
            device,
            transpose=False,
        )
    )

    vision_layers = []
    for i in range(vision_cfg["num_hidden_layers"]):
        prefix = f"model.vision_tower.vision_model.encoder.layers.{i}"
        self_attn = _make_obj(
            q_proj=_linear_params(
                state,
                f"{prefix}.self_attn.q_proj.weight",
                device,
                bias_key=f"{prefix}.self_attn.q_proj.bias",
            ),
            k_proj=_linear_params(
                state,
                f"{prefix}.self_attn.k_proj.weight",
                device,
                bias_key=f"{prefix}.self_attn.k_proj.bias",
            ),
            v_proj=_linear_params(
                state,
                f"{prefix}.self_attn.v_proj.weight",
                device,
                bias_key=f"{prefix}.self_attn.v_proj.bias",
            ),
            out_proj=_linear_params(
                state,
                f"{prefix}.self_attn.out_proj.weight",
                device,
                bias_key=f"{prefix}.self_attn.out_proj.bias",
            ),
        )
        mlp = _make_obj(
            fc1=_linear_params(
                state,
                f"{prefix}.mlp.fc1.weight",
                device,
                bias_key=f"{prefix}.mlp.fc1.bias",
            ),
            fc2=_linear_params(
                state,
                f"{prefix}.mlp.fc2.weight",
                device,
                bias_key=f"{prefix}.mlp.fc2.bias",
            ),
        )
        vision_layers.append(
            _make_obj(
                layer_norm1=_norm_params(
                    state,
                    f"{prefix}.layer_norm1.weight",
                    device,
                    bias_key=f"{prefix}.layer_norm1.bias",
                ),
                layer_norm2=_norm_params(
                    state,
                    f"{prefix}.layer_norm2.weight",
                    device,
                    bias_key=f"{prefix}.layer_norm2.bias",
                ),
                self_attn=self_attn,
                mlp=mlp,
            )
        )

    post_layernorm = _norm_params(
        state,
        "model.vision_tower.vision_model.post_layernorm.weight",
        device,
        bias_key="model.vision_tower.vision_model.post_layernorm.bias",
    )
    vision_tower = _make_obj(
        patch_embedding=patch_embedding,
        position_embedding=position_embedding,
        layers=vision_layers,
        post_layernorm=post_layernorm,
    )

    # --- Multimodal projector ---
    multi_modal_projector = _make_obj(
        linear_1=_linear_params(
            state,
            "model.multi_modal_projector.linear_1.weight",
            device,
            bias_key="model.multi_modal_projector.linear_1.bias",
        ),
        linear_2=_linear_params(
            state,
            "model.multi_modal_projector.linear_2.weight",
            device,
            bias_key="model.multi_modal_projector.linear_2.bias",
        ),
    )

    # --- Language layers ---
    layers = []
    for i in range(num_layers):
        prefix = f"model.language_model.layers.{i}"
        layer_type = layer_types[i]
        operator_norm = _norm_params(state, f"{prefix}.operator_norm.weight", device)
        ffn_norm = _norm_params(state, f"{prefix}.ffn_norm.weight", device)
        feed_forward = _make_obj(
            w1=_linear_params(state, f"{prefix}.feed_forward.w1.weight", device),
            w2=_linear_params(state, f"{prefix}.feed_forward.w2.weight", device),
            w3=_linear_params(state, f"{prefix}.feed_forward.w3.weight", device),
        )

        if layer_type == "full_attention":
            self_attn = _make_obj(
                q_proj=_linear_params(state, f"{prefix}.self_attn.q_proj.weight", device),
                k_proj=_linear_params(state, f"{prefix}.self_attn.k_proj.weight", device),
                v_proj=_linear_params(state, f"{prefix}.self_attn.v_proj.weight", device),
                out_proj=_linear_params(state, f"{prefix}.self_attn.out_proj.weight", device),
                q_layernorm=_norm_params(state, f"{prefix}.self_attn.q_layernorm.weight", device),
                k_layernorm=_norm_params(state, f"{prefix}.self_attn.k_layernorm.weight", device),
            )
            layers.append(
                _make_obj(
                    operator_norm=operator_norm,
                    ffn_norm=ffn_norm,
                    feed_forward=feed_forward,
                    self_attn=self_attn,
                )
            )
        else:
            conv = _make_obj(
                in_proj=_linear_params(state, f"{prefix}.conv.in_proj.weight", device),
                # conv weight [C, 1, K] stays as-is for ttnn.conv1d
                conv=_make_obj(
                    weight=_tensor_to_ttnn(
                        _require(state, f"{prefix}.conv.conv.weight"),
                        device,
                        transpose=False,
                    )
                ),
                out_proj=_linear_params(state, f"{prefix}.conv.out_proj.weight", device),
            )
            layers.append(
                _make_obj(
                    operator_norm=operator_norm,
                    ffn_norm=ffn_norm,
                    feed_forward=feed_forward,
                    conv=conv,
                )
            )

    # Tied lm_head: reuse embed_tokens when lm_head is absent
    if "lm_head.weight" in state:
        lm_head = _linear_params(state, "lm_head.weight", device, transpose=True)
    else:
        lm_head = embed_tokens

    return _make_obj(
        embed_tokens=embed_tokens,
        embedding_norm=embedding_norm,
        vision_tower=vision_tower,
        multi_modal_projector=multi_modal_projector,
        layers=layers,
        lm_head=lm_head,
    )


def list_expected_keys(model_config: Optional[Dict[str, Any]] = None) -> Iterable[str]:
    """Return the HF keys required for a full conversion (for validation/tests)."""
    if model_config is None:
        model_config = create_model_config()
    keys = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.embedding_norm.weight",
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        "model.vision_tower.vision_model.embeddings.patch_embedding.bias",
        "model.vision_tower.vision_model.embeddings.position_embedding.weight",
        "model.vision_tower.vision_model.post_layernorm.weight",
        "model.vision_tower.vision_model.post_layernorm.bias",
        "model.multi_modal_projector.linear_1.weight",
        "model.multi_modal_projector.linear_1.bias",
        "model.multi_modal_projector.linear_2.weight",
        "model.multi_modal_projector.linear_2.bias",
    ]
    for i in range(model_config["vision_config"]["num_hidden_layers"]):
        p = f"model.vision_tower.vision_model.encoder.layers.{i}"
        keys.extend(
            [
                f"{p}.layer_norm1.weight",
                f"{p}.layer_norm1.bias",
                f"{p}.layer_norm2.weight",
                f"{p}.layer_norm2.bias",
                f"{p}.self_attn.q_proj.weight",
                f"{p}.self_attn.k_proj.weight",
                f"{p}.self_attn.v_proj.weight",
                f"{p}.self_attn.out_proj.weight",
                f"{p}.mlp.fc1.weight",
                f"{p}.mlp.fc2.weight",
            ]
        )
    for i, layer_type in enumerate(model_config["layer_types"]):
        p = f"model.language_model.layers.{i}"
        keys.extend(
            [
                f"{p}.operator_norm.weight",
                f"{p}.ffn_norm.weight",
                f"{p}.feed_forward.w1.weight",
                f"{p}.feed_forward.w2.weight",
                f"{p}.feed_forward.w3.weight",
            ]
        )
        if layer_type == "full_attention":
            keys.extend(
                [
                    f"{p}.self_attn.q_proj.weight",
                    f"{p}.self_attn.k_proj.weight",
                    f"{p}.self_attn.v_proj.weight",
                    f"{p}.self_attn.out_proj.weight",
                    f"{p}.self_attn.q_layernorm.weight",
                    f"{p}.self_attn.k_layernorm.weight",
                ]
            )
        else:
            keys.extend(
                [
                    f"{p}.conv.in_proj.weight",
                    f"{p}.conv.conv.weight",
                    f"{p}.conv.out_proj.weight",
                ]
            )
    return keys
