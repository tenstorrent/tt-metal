# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint save/load utilities: safetensors for model weights and optimizer state.

Model weights are saved in HuggingFace-compatible format by applying the inverse
permutation transforms (repermute_proj_rows, repermute_norm_weights) from
param_utils.py, so the checkpoint can be loaded directly by AutoModelForCausalLM.

Optimizer first/second moments are saved in the same HF-compatible format so
they survive topology changes between save and resume.

LoRA adapters (when active) are saved separately in raw TTML format for fast
resume; an optional merge step fuses them into the base weights for HF export.
"""

import json
import os
from typing import Dict, Optional

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

import ttml
import ttnn
from utils.param_utils import (
    build_weight_mapping_distributed,
    build_weight_mapping_single,
    repermute_norm_weights,
    repermute_proj_rows,
    unpermute_norm_weights,
    unpermute_proj_rows,
)


# =====================================================================
# HF tensor shape registry
# =====================================================================


def build_hf_shapes(config, tie_word_embeddings: bool) -> Dict[str, tuple]:
    """Return the expected HF (unpadded) shape for every parameter name."""
    shapes: Dict[str, tuple] = {}
    shapes["model.embed_tokens.weight"] = (config.vocab_size, config.hidden_size)
    if not tie_word_embeddings:
        shapes["lm_head.weight"] = (config.vocab_size, config.hidden_size)

    q_dim = config.num_attention_heads * config.head_dim
    k_dim = config.num_key_value_heads * config.head_dim
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}"
        shapes[f"{p}.self_attn.q_proj.weight"] = (q_dim, config.hidden_size)
        shapes[f"{p}.self_attn.k_proj.weight"] = (k_dim, config.hidden_size)
        shapes[f"{p}.self_attn.v_proj.weight"] = (k_dim, config.hidden_size)
        shapes[f"{p}.self_attn.o_proj.weight"] = (config.hidden_size, q_dim)

        if config.attention_bias:
            shapes[f"{p}.self_attn.q_proj.bias"] = (q_dim,)
            shapes[f"{p}.self_attn.k_proj.bias"] = (k_dim,)
            shapes[f"{p}.self_attn.v_proj.bias"] = (k_dim,)
            shapes[f"{p}.self_attn.o_proj.bias"] = (config.hidden_size,)

        shapes[f"{p}.self_attn.q_norm.weight"] = (config.head_dim,)
        shapes[f"{p}.self_attn.k_norm.weight"] = (config.head_dim,)
        shapes[f"{p}.input_layernorm.weight"] = (config.hidden_size,)
        shapes[f"{p}.post_attention_layernorm.weight"] = (config.hidden_size,)
        shapes[f"{p}.mlp.gate_proj.weight"] = (
            config.intermediate_size,
            config.hidden_size,
        )
        shapes[f"{p}.mlp.up_proj.weight"] = (
            config.intermediate_size,
            config.hidden_size,
        )
        shapes[f"{p}.mlp.down_proj.weight"] = (
            config.hidden_size,
            config.intermediate_size,
        )

    shapes["model.norm.weight"] = (config.hidden_size,)
    return shapes


# =====================================================================
# Low-level tensor extraction helpers
# =====================================================================


def _build_inv_transforms(forward_transforms: dict) -> dict:
    """Convert forward (HF→TTML) transforms to inverse (TTML→HF) transforms."""
    inv = {}
    for hf_name, tr in forward_transforms.items():
        if tr[0] == "unpermute_proj":
            inv[hf_name] = ("repermute_proj", tr[1])
        elif tr[0] == "unpermute_norm":
            inv[hf_name] = ("repermute_norm",)
    return inv


def _apply_inv_transform(
    weight: torch.Tensor, hf_name: str, inv_transforms: dict
) -> torch.Tensor:
    """Apply inverse permutation to convert TTML internal layout → HF layout."""
    if hf_name not in inv_transforms:
        return weight
    tr = inv_transforms[hf_name]
    if tr[0] == "repermute_proj":
        return repermute_proj_rows(weight, num_heads=tr[1])
    elif tr[0] == "repermute_norm":
        return repermute_norm_weights(weight)
    return weight


def _crop(weight: torch.Tensor, hf_shape: tuple) -> torch.Tensor:
    """Crop tile-padded ttml tensor to the original HF shape."""
    w = weight.squeeze()
    if w.dim() == 2:
        return w[: hf_shape[0], : hf_shape[1]]
    elif w.dim() == 1:
        return w[: hf_shape[0]]
    return w


def _gather_single(param) -> torch.Tensor:
    """Extract a parameter value from a single-device model. Returns float32."""
    return ttnn.to_torch(param.get_value()).to(torch.float32)


def _gather_distributed(
    param, shard_type: Optional[str], device, dp_size: int = 1
) -> torch.Tensor:
    """Extract a distributed parameter value, gathering TP shards. Returns float32."""
    val_tt = param.get_value()
    if shard_type == "col_w":
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 2)
        val = ttnn.to_torch(val_tt, mesh_composer=composer).to(torch.float32)
        if dp_size > 1:
            val = val[:, :, : val.shape[2] // dp_size, :]
    elif shard_type in ("row_w", "col_b"):
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 3)
        val = ttnn.to_torch(val_tt, mesh_composer=composer).to(torch.float32)
        if dp_size > 1:
            val = val[:, :, :, : val.shape[3] // dp_size]
    else:
        # None → replicated; take one copy from the first device
        composer = ttml.core.distributed.concat_mesh_to_tensor_composer(device, 0)
        val = ttnn.to_torch(val_tt, mesh_composer=composer).to(torch.float32)
        val = val[:1]
    return val


def _resolve_ttml_name(ttml_name: str, ttml_params: dict) -> Optional[str]:
    """Resolve the actual ttml param name, accounting for LoRA's /base_layer/ insertion.

    After inject_adapter_in_model the base weight path gains a /base_layer/ segment:
      .../q_proj/weight  →  .../q_proj/base_layer/weight
    """
    if ttml_name in ttml_params:
        return ttml_name
    parts = ttml_name.rsplit("/", 1)
    if len(parts) == 2:
        candidate = f"{parts[0]}/base_layer/{parts[1]}"
        if candidate in ttml_params:
            return candidate
    return None


def _is_col_parallel_lora_B(ttml_name: str) -> bool:
    """True when this lora_B belongs to a column-parallel projection (TP mode)."""
    for proj in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
        if f"/{proj}/lora_B" in ttml_name:
            return True
    return False


def _is_row_parallel_lora_A(ttml_name: str) -> bool:
    """True when this lora_A belongs to a row-parallel projection (TP mode)."""
    for proj in ("o_proj", "down_proj"):
        if f"/{proj}/lora_A" in ttml_name:
            return True
    return False


# =====================================================================
# Extract full model weights → HF-compatible state dict
# =====================================================================


def extract_hf_state_dict(
    ttml_model,
    config,
    tie_word_embeddings: bool,
    distributed: bool = False,
    device=None,
    shard_dim: Optional[int] = None,
    dp_size: int = 1,
    lora_config: Optional[dict] = None,
    merge_lora: bool = True,
) -> Dict[str, torch.Tensor]:
    """Extract model weights from a ttml model and convert to HF-compatible format.

    When LoRA adapters are active and merge_lora=True, the LoRA delta
    (lora_B @ lora_A * scaling) is added to the base weights before export so
    the result can be loaded directly as an HF Qwen3 model.

    When merge_lora=False, only the base weights are returned (useful for saving
    a training checkpoint that will later have LoRA re-injected on resume).
    """
    ttml_params = ttml_model.parameters()
    root_prefix = next(iter(ttml_params)).split("/")[0]
    hf_shapes = build_hf_shapes(config, tie_word_embeddings)

    if distributed:
        mapping, shard_types, fwd_transforms = build_weight_mapping_distributed(
            config, root_prefix, tie_word_embeddings
        )
    else:
        mapping, fwd_transforms = build_weight_mapping_single(
            config, root_prefix, tie_word_embeddings
        )

    inv_transforms = _build_inv_transforms(fwd_transforms)
    hf_state_dict: Dict[str, torch.Tensor] = {}

    for hf_name, ttml_name in tqdm(
        mapping.items(), desc="  Extracting weights", unit="w"
    ):
        actual_name = _resolve_ttml_name(ttml_name, ttml_params)
        if actual_name is None:
            continue
        hf_shape = hf_shapes.get(hf_name)
        if hf_shape is None:
            continue

        param = ttml_params[actual_name]
        if distributed:
            raw = _gather_distributed(param, shard_types.get(hf_name), device, dp_size)
        else:
            raw = _gather_single(param)

        weight = _apply_inv_transform(raw.squeeze(), hf_name, inv_transforms)
        hf_state_dict[hf_name] = _crop(weight, hf_shape).to(torch.bfloat16)

    if merge_lora and lora_config is not None:
        _merge_lora_inplace(
            hf_state_dict,
            ttml_params,
            config,
            lora_config,
            distributed,
            device,
            dp_size,
            inv_transforms,
            hf_shapes,
            root_prefix,
        )

    return hf_state_dict


def _merge_lora_inplace(
    hf_state_dict: Dict[str, torch.Tensor],
    ttml_params: dict,
    config,
    lora_config: dict,
    distributed: bool,
    device,
    dp_size: int,
    inv_transforms: dict,
    hf_shapes: dict,
    root_prefix: str,
) -> None:
    """Merge LoRA adapter weights into the base weights in hf_state_dict (in-place).

    The lora_B and lora_A tensors live in TTML layout.  We compute the delta
    (lora_B @ lora_A * scaling) in TTML layout, apply the same inverse
    permutation as the base weight, crop, and add it to hf_state_dict.
    """
    rank = lora_config["rank"]
    alpha = lora_config.get("alpha", rank)
    scaling = alpha / rank
    targets = set(lora_config.get("targets", []))

    col_parallel = {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}

    proj_locations = {
        "q_proj": "self_attn",
        "k_proj": "self_attn",
        "v_proj": "self_attn",
        "o_proj": "self_attn",
        "gate_proj": "mlp",
        "up_proj": "mlp",
        "down_proj": "mlp",
    }

    for i in range(config.num_hidden_layers):
        for proj_name in targets:
            sub = proj_locations.get(proj_name)
            if sub is None:
                continue
            hf_wname = f"model.layers.{i}.{sub}.{proj_name}.weight"
            ttml_base = f"{root_prefix}/model/layers/{i}/{sub}/{proj_name}"
            lora_A_name = f"{ttml_base}/lora_A"
            lora_B_name = f"{ttml_base}/lora_B"

            if lora_A_name not in ttml_params or lora_B_name not in ttml_params:
                continue
            if hf_wname not in hf_state_dict:
                continue

            if distributed and proj_name not in col_parallel:
                # Row-parallel: lora_A row-sharded (dim 3), lora_B replicated
                lora_A = _gather_distributed(
                    ttml_params[lora_A_name], "row_w", device, dp_size
                ).squeeze()
                lora_B = _gather_distributed(
                    ttml_params[lora_B_name], None, device, dp_size
                ).squeeze()
            elif distributed:
                # Column-parallel: lora_B col-sharded (dim 2), lora_A replicated
                lora_A = _gather_distributed(
                    ttml_params[lora_A_name], None, device, dp_size
                ).squeeze()
                lora_B = _gather_distributed(
                    ttml_params[lora_B_name], "col_w", device, dp_size
                ).squeeze()
            else:
                lora_A = _gather_single(ttml_params[lora_A_name]).squeeze()
                lora_B = _gather_single(ttml_params[lora_B_name]).squeeze()

            if lora_A.dim() != 2 or lora_B.dim() != 2:
                continue

            # delta is in TTML layout (same as the base weight before inv_transform)
            delta = (lora_B @ lora_A) * scaling
            delta = _apply_inv_transform(delta, hf_wname, inv_transforms)
            hf_shape = hf_shapes[hf_wname]
            delta = delta[: hf_shape[0], : hf_shape[1]]
            hf_state_dict[hf_wname] = (
                hf_state_dict[hf_wname].float() + delta.float()
            ).to(torch.bfloat16)


# =====================================================================
# Model save / load
# =====================================================================


def save_model_to_safetensors(
    ttml_model,
    config,
    save_dir: str,
    tie_word_embeddings: bool,
    distributed: bool = False,
    device=None,
    shard_dim: Optional[int] = None,
    dp_size: int = 1,
    lora_config: Optional[dict] = None,
    merge_lora: bool = True,
    filename: str = "model.safetensors",
) -> str:
    """Save ttml model weights as HF-compatible safetensors.

    When merge_lora=True and LoRA adapters are present, the adapters are merged
    into the base weights so the file can be loaded directly as an HF model.
    When merge_lora=False (training resume checkpoint), only the base weights are
    saved; LoRA adapters are saved separately by save_lora_adapters().

    Returns the path to the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    print(f"\nExtracting model weights → HF format...")
    hf_state_dict = extract_hf_state_dict(
        ttml_model,
        config,
        tie_word_embeddings,
        distributed=distributed,
        device=device,
        shard_dim=shard_dim,
        dp_size=dp_size,
        lora_config=lora_config,
        merge_lora=merge_lora,
    )
    print(f"  Saving {len(hf_state_dict)} tensors → {path}")
    save_file(hf_state_dict, path)
    print(f"  Done.")
    return path


def load_model_from_safetensors(
    load_dir: str, filename: str = "model.safetensors"
) -> dict:
    """Load model weights from safetensors → HF-format state dict (float32).

    The returned dict can be passed directly to load_weights_from_hf() or
    load_weights_from_hf_distributed() to populate a ttml model.
    """
    path = os.path.join(load_dir, filename)
    return {k: v.to(torch.float32) for k, v in load_file(path).items()}


def _load_hf_dict_into_ttml(
    ttml_model,
    hf_state_dict: dict,
    config,
    tie_word_embeddings: bool,
    distributed: bool = False,
    shard_dim: Optional[int] = None,
) -> None:
    """Load an HF-format state dict into a ttml model.

    Compared to the standard load_weights_from_hf[_distributed], this version
    uses _resolve_ttml_name to handle models where LoRA adapters have already
    been injected (so base weight paths include '/base_layer/').
    """
    if distributed:
        from model_qwen3_distributed import load_weights_from_hf_distributed

        load_weights_from_hf_distributed(
            ttml_model,
            hf_state_dict,
            config,
            tie_word_embeddings=tie_word_embeddings,
            shard_dim=shard_dim,
        )
    else:
        from model_qwen3 import load_weights_from_hf

        load_weights_from_hf(
            ttml_model, hf_state_dict, config, tie_word_embeddings=tie_word_embeddings
        )

    # Second pass: restore any base weights that were renamed by LoRA injection
    # (e.g., ".../q_proj/weight" → ".../q_proj/base_layer/weight")
    ttml_params = ttml_model.parameters()
    root_prefix = next(iter(ttml_params)).split("/")[0]

    if distributed:
        mapping, shard_types, fwd_transforms = build_weight_mapping_distributed(
            config, root_prefix, tie_word_embeddings
        )
    else:
        mapping, fwd_transforms = build_weight_mapping_single(
            config, root_prefix, tie_word_embeddings
        )
        shard_types = {}

    device = None
    if distributed:
        try:
            import ttml as _ttml

            device = _ttml.autograd.AutoContext.get_instance().get_device()
        except Exception:
            pass

    for hf_name, ttml_name in mapping.items():
        if ttml_name in ttml_params:
            continue  # already loaded in the first pass
        actual_name = _resolve_ttml_name(ttml_name, ttml_params)
        if actual_name is None or hf_name not in hf_state_dict:
            continue

        weight = hf_state_dict[hf_name].float()
        if hf_name in fwd_transforms:
            tr = fwd_transforms[hf_name]
            if tr[0] == "unpermute_proj":
                weight = unpermute_proj_rows(weight, num_heads=tr[1])
            elif tr[0] == "unpermute_norm":
                weight = unpermute_norm_weights(weight)

        ttml_shape = list(ttml_params[actual_name].shape())
        shard_type = shard_types.get(hf_name)

        if weight.dim() == 2:
            rows, cols = weight.shape
            tp_size = 1
            if distributed and device is not None:
                if shard_type == "col_w" and len(ttml_shape) >= 3:
                    tp_size = max(1, rows // ttml_shape[2]) if ttml_shape[2] else 1
                elif shard_type in ("row_w", "col_b") and len(ttml_shape) >= 4:
                    tp_size = max(1, cols // ttml_shape[3]) if ttml_shape[3] else 1
            tgt_rows = ttml_shape[2] * (tp_size if shard_type == "col_w" else 1)
            tgt_cols = ttml_shape[3] * (
                tp_size if shard_type in ("row_w", "col_b") else 1
            )
            if rows != tgt_rows or cols != tgt_cols:
                padded = torch.zeros(tgt_rows, tgt_cols, dtype=weight.dtype)
                padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = weight[
                    : min(rows, tgt_rows), : min(cols, tgt_cols)
                ]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0)
        elif weight.dim() == 1:
            dim = weight.shape[0]
            tgt_dim = ttml_shape[-1]
            if dim != tgt_dim:
                padded = torch.zeros(tgt_dim, dtype=weight.dtype)
                padded[: min(dim, tgt_dim)] = weight[: min(dim, tgt_dim)]
                weight = padded
            weight = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        w_np = weight.contiguous().numpy()
        if distributed and shard_type is not None and device is not None:
            dim = 2 if shard_type == "col_w" else 3
            mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
                device, dim, shard_dim
            )
            new_t = ttml.autograd.Tensor.from_numpy(
                w_np, ttnn.Layout.TILE, ttnn.bfloat16, mapper
            )
        else:
            new_t = ttml.autograd.Tensor.from_numpy(
                w_np, ttnn.Layout.TILE, ttnn.bfloat16
            )
        ttml_params[actual_name].assign(new_t)


# =====================================================================
# LoRA adapter save / load (for training resume without re-initialization)
# =====================================================================


def save_lora_adapters(
    ttml_model,
    save_dir: str,
    distributed: bool = False,
    device=None,
    dp_size: int = 1,
    filename: str = "lora_adapters.safetensors",
) -> str:
    """Save LoRA adapter weights for training resume.

    Weights are saved in squeezed float32 format using their ttml param path as
    the safetensors key.  The inverse permutation is NOT applied here because
    the adapters are in the correct TTML layout for re-injection on resume.

    Returns the path to the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    ttml_params = ttml_model.parameters()
    tensors: Dict[str, torch.Tensor] = {}

    for name, param in ttml_params.items():
        if "lora_A" not in name and "lora_B" not in name:
            continue
        if distributed:
            if "lora_B" in name and _is_col_parallel_lora_B(name):
                raw = _gather_distributed(param, "col_w", device, dp_size)
            elif "lora_A" in name and _is_row_parallel_lora_A(name):
                raw = _gather_distributed(param, "row_w", device, dp_size)
            else:
                raw = _gather_distributed(param, None, device, dp_size)
        else:
            raw = _gather_single(param)
        tensors[name] = raw.squeeze().to(torch.bfloat16)

    print(f"  Saving {len(tensors)} LoRA adapter tensors → {path}")
    save_file(tensors, path)
    return path


def load_lora_adapters(
    ttml_model,
    load_dir: str,
    distributed: bool = False,
    device=None,
    shard_dim: Optional[int] = None,
    dp_size: int = 1,
    filename: str = "lora_adapters.safetensors",
) -> None:
    """Load LoRA adapter weights into a ttml model that already has adapters injected.

    The model must have LoRA adapters already injected (via inject_adapter_in_model)
    before calling this function.
    """
    path = os.path.join(load_dir, filename)
    if not os.path.exists(path):
        print(f"  No LoRA adapters found at {path}, skipping.")
        return

    saved = load_file(path)
    ttml_params = ttml_model.parameters()
    loaded = 0

    for ttml_name, weight_t in saved.items():
        if ttml_name not in ttml_params:
            continue
        weight = weight_t.float()
        # Ensure 4D shape for tilization
        if weight.dim() == 2:
            w_np = weight.unsqueeze(0).unsqueeze(0).contiguous().numpy()
        elif weight.dim() == 1:
            w_np = weight.unsqueeze(0).unsqueeze(0).unsqueeze(0).contiguous().numpy()
        else:
            w_np = weight.contiguous().numpy()

        if distributed:
            if "lora_B" in ttml_name and _is_col_parallel_lora_B(ttml_name):
                mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
                    device, 2, shard_dim
                )
            elif "lora_A" in ttml_name and _is_row_parallel_lora_A(ttml_name):
                mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
                    device, 3, shard_dim
                )
            else:
                mapper = None

            new_t = ttml.autograd.Tensor.from_numpy(
                w_np,
                ttnn.Layout.TILE,
                ttnn.bfloat16,
                mapper if mapper is not None else None,
            )
        else:
            new_t = ttml.autograd.Tensor.from_numpy(
                w_np, ttnn.Layout.TILE, ttnn.bfloat16
            )

        ttml_params[ttml_name].assign(new_t)
        loaded += 1

    print(f"  LoRA adapters loaded: {loaded}/{len(saved)}")


# =====================================================================
# Helper: convert an HF-format tensor back to TTML numpy array
# =====================================================================


def _hf_to_ttml_np(
    weight: torch.Tensor,
    hf_name: str,
    fwd_transforms: dict,
    ttml_shape: list,
    shard_type: Optional[str] = None,
    tp_size: int = 1,
):
    """Apply forward transforms, padding, and return a numpy array ready for device upload.

    For distributed checkpoints, ttml_shape is the *per-device* shape; we use
    shard_type + tp_size to determine the required full-weight shape.
    """
    w = weight.float()
    if hf_name in fwd_transforms:
        tr = fwd_transforms[hf_name]
        if tr[0] == "unpermute_proj":
            w = unpermute_proj_rows(w, num_heads=tr[1])
        elif tr[0] == "unpermute_norm":
            w = unpermute_norm_weights(w)

    if w.dim() == 2:
        rows, cols = w.shape
        tgt_rows, tgt_cols = ttml_shape[2], ttml_shape[3]
        if shard_type == "col_w":
            tgt_rows *= tp_size
        elif shard_type in ("row_w", "col_b"):
            tgt_cols *= tp_size
        if rows != tgt_rows or cols != tgt_cols:
            padded = torch.zeros(tgt_rows, tgt_cols, dtype=w.dtype)
            padded[: min(rows, tgt_rows), : min(cols, tgt_cols)] = w[
                : min(rows, tgt_rows), : min(cols, tgt_cols)
            ]
            w = padded
        w = w.unsqueeze(0).unsqueeze(0)
    elif w.dim() == 1:
        dim = w.shape[0]
        tgt_dim = ttml_shape[-1]
        if shard_type == "col_b":
            tgt_dim *= tp_size
        if dim != tgt_dim:
            padded = torch.zeros(tgt_dim, dtype=w.dtype)
            padded[: min(dim, tgt_dim)] = w[: min(dim, tgt_dim)]
            w = padded
        w = w.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    return w.contiguous().numpy()


# =====================================================================
# Optimizer state save / load
# =====================================================================


def save_optimizer_state(
    optimizer,
    step: int,
    ttml_model,
    config,
    tie_word_embeddings: bool,
    save_dir: str,
    distributed: bool = False,
    device=None,
    shard_dim: Optional[int] = None,
    dp_size: int = 1,
    filename: str = "optimizer.safetensors",
) -> str:
    """Save optimizer first/second moments and step count to safetensors.

    Base-model moments are saved in HF-compatible format (same inverse
    permutation as model weights) so they can be restored on any TP topology.
    LoRA adapter moments are saved in raw TTML format under the
    "lora_first_moment/" and "lora_second_moment/" key prefixes.

    Returns the path to the saved file.
    """
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    ttml_params = ttml_model.parameters()
    root_prefix = next(iter(ttml_params)).split("/")[0]

    if distributed:
        mapping, shard_types, fwd_transforms = build_weight_mapping_distributed(
            config, root_prefix, tie_word_embeddings
        )
    else:
        mapping, fwd_transforms = build_weight_mapping_single(
            config, root_prefix, tie_word_embeddings
        )
        shard_types = {}

    inv_transforms = _build_inv_transforms(fwd_transforms)
    hf_shapes = build_hf_shapes(config, tie_word_embeddings)
    # Reverse lookup: ttml_name → hf_name (accounting for /base_layer/ in LoRA)
    ttml_to_hf: Dict[str, str] = {}
    for hf_name, ttml_name in mapping.items():
        ttml_to_hf[ttml_name] = hf_name
        # Also register the LoRA base_layer variant
        parts = ttml_name.rsplit("/", 1)
        if len(parts) == 2:
            ttml_to_hf[f"{parts[0]}/base_layer/{parts[1]}"] = hf_name

    opt_state = optimizer.get_state_dict()
    first_moment_map = opt_state["first_moment"]
    second_moment_map = opt_state["second_moment"]

    tensors: Dict[str, torch.Tensor] = {}
    tensors["_steps"] = torch.tensor([step], dtype=torch.int64)

    def _save_base_moments(moment_map, prefix):
        for param_name, moment_ptr in tqdm(
            list(moment_map.items()), desc=f"  {prefix}", unit="w"
        ):
            hf_name = ttml_to_hf.get(param_name)
            if hf_name is None:
                continue  # LoRA param or not in base mapping
            hf_shape = hf_shapes.get(hf_name)
            if hf_shape is None:
                continue
            if distributed:
                raw = _gather_distributed(
                    moment_ptr, shard_types.get(hf_name), device, dp_size
                )
            else:
                raw = _gather_single(moment_ptr)
            weight = _apply_inv_transform(raw.squeeze(), hf_name, inv_transforms)
            tensors[f"{prefix}/{hf_name}"] = _crop(weight, hf_shape).to(torch.bfloat16)

    def _save_lora_moments(moment_map, prefix):
        for param_name, moment_ptr in moment_map.items():
            if "lora_A" not in param_name and "lora_B" not in param_name:
                continue
            if distributed:
                if "lora_B" in param_name and _is_col_parallel_lora_B(param_name):
                    raw = _gather_distributed(moment_ptr, "col_w", device, dp_size)
                elif "lora_A" in param_name and _is_row_parallel_lora_A(param_name):
                    raw = _gather_distributed(moment_ptr, "row_w", device, dp_size)
                else:
                    raw = _gather_distributed(moment_ptr, None, device, dp_size)
            else:
                raw = _gather_single(moment_ptr)
            tensors[f"{prefix}/{param_name}"] = raw.squeeze().to(torch.bfloat16)

    print(f"\nExtracting optimizer state...")
    _save_base_moments(first_moment_map, "first_moment")
    _save_base_moments(second_moment_map, "second_moment")
    _save_lora_moments(first_moment_map, "lora_first_moment")
    _save_lora_moments(second_moment_map, "lora_second_moment")

    print(f"  Saving {len(tensors)} tensors → {path}")
    save_file(tensors, path)
    print(f"  Done.")
    return path


def load_optimizer_state(
    optimizer,
    load_dir: str,
    ttml_model,
    config,
    tie_word_embeddings: bool,
    distributed: bool = False,
    device=None,
    shard_dim: Optional[int] = None,
    dp_size: int = 1,
    filename: str = "optimizer.safetensors",
) -> int:
    """Restore optimizer first/second moments from a safetensors checkpoint.

    Moments are loaded into the optimizer's existing moment tensors in-place
    (exploiting shared_ptr semantics) so they are visible to the C++ optimizer
    without constructing new NamedParameters from Python.

    Finally, set_state_dict() is called with the updated moment maps and the
    saved step count to restore the AdamW bias-correction counter (m_steps).

    Returns the saved step count.
    """
    path = os.path.join(load_dir, filename)
    if not os.path.exists(path):
        print(f"  No optimizer state found at {path}, starting fresh.")
        return 0

    saved = {k: v for k, v in load_file(path).items()}
    step = int(saved.pop("_steps", torch.tensor([0]))[0])

    ttml_params = ttml_model.parameters()
    root_prefix = next(iter(ttml_params)).split("/")[0]

    if distributed:
        mapping, shard_types, fwd_transforms = build_weight_mapping_distributed(
            config, root_prefix, tie_word_embeddings
        )
        tp_size = 1
        # Determine tp_size from existing ttml param shapes vs hf shapes
        hf_shapes_ref = build_hf_shapes(config, tie_word_embeddings)
        for hf_name, ttml_name in mapping.items():
            actual = _resolve_ttml_name(ttml_name, ttml_params)
            if actual and hf_name in hf_shapes_ref:
                hf_shape = hf_shapes_ref[hf_name]
                ttml_shape = list(ttml_params[actual].shape())
                if shard_types.get(hf_name) == "col_w" and len(ttml_shape) >= 3:
                    tp_size = (
                        max(tp_size, hf_shape[0] // ttml_shape[2])
                        if ttml_shape[2]
                        else 1
                    )
                    break
    else:
        mapping, fwd_transforms = build_weight_mapping_single(
            config, root_prefix, tie_word_embeddings
        )
        shard_types = {}
        tp_size = 1

    # Retrieve the optimizer's moment NamedParameters (shared pointers)
    opt_state = optimizer.get_state_dict()
    first_moment_map = opt_state["first_moment"]
    second_moment_map = opt_state["second_moment"]

    # Build ttml_name → moment_ptr lookups for fast access
    sm_lookup = {name: ptr for name, ptr in second_moment_map.items()}

    def _restore_lora_moments(moment_lookup, prefix):
        loaded = 0
        for param_name in list(moment_lookup.keys()):
            if "lora_A" not in param_name and "lora_B" not in param_name:
                continue
            saved_key = f"{prefix}/{param_name}"
            if saved_key not in saved:
                continue
            weight = saved[saved_key].float()
            w_np = weight.unsqueeze(0).unsqueeze(0).contiguous().numpy()
            if distributed:
                if "lora_B" in param_name and _is_col_parallel_lora_B(param_name):
                    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
                        device, 2, shard_dim
                    )
                elif "lora_A" in param_name and _is_row_parallel_lora_A(param_name):
                    mapper = ttml.core.distributed.shard_tensor_to_mesh_mapper(
                        device, 3, shard_dim
                    )
                else:
                    mapper = None
                new_t = ttml.autograd.Tensor.from_numpy(
                    w_np,
                    ttnn.Layout.TILE,
                    ttnn.bfloat16,
                    mapper if mapper is not None else None,
                )
            else:
                new_t = ttml.autograd.Tensor.from_numpy(
                    w_np, ttnn.Layout.TILE, ttnn.bfloat16
                )
            moment_lookup[param_name].set_value(new_t.get_value())
            loaded += 1
        if loaded:
            print(f"  {prefix} (LoRA): {loaded} moments restored")

    print(f"\nRestoring optimizer state from {path}...")

    # NOTE: Moment restore via set_value is currently broken — the tensors
    # created by from_numpy have a different memory config than the original
    # zeros_like tensors, causing moreh_adamw to read garbage.  Skip moment
    # restore for now; the optimizer re-accumulates fresh moments in a few
    # steps with only a minor warmup penalty.
    # TODO: fix by copying data into existing moment buffers (preserving
    #       memory config) instead of replacing the tensor via set_value.
    print("  Skipping moment restore (fresh moments, step count only).")

    # Restore the step counter directly (avoids set_state_dict variant issues).
    optimizer.set_steps(step)
    print(f"  Step count restored: {step}")

    return step


# =====================================================================
# Convenience wrappers: full checkpoint save / load
# =====================================================================


def save_checkpoint(
    step: int,
    ttml_model,
    optimizer,
    config,
    save_dir: str,
    tie_word_embeddings: bool,
    distributed: bool = False,
    device=None,
    shard_dim: Optional[int] = None,
    dp_size: int = 1,
    lora_config: Optional[dict] = None,
    args_dict: Optional[dict] = None,
) -> None:
    """Save a complete training checkpoint.

    Creates the following files inside save_dir:
      model.safetensors          – HF-compatible base model weights (only when
                                   LoRA is NOT active; base weights are frozen
                                   during LoRA training so they don't need to
                                   be checkpointed — resume reloads from HF)
      lora_adapters.safetensors  – LoRA adapter weights (when LoRA is active)
      optimizer.safetensors      – AdamW moments + step count
      training_state.json        – step number and optional hyperparameter args
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Saving checkpoint at step {step} → {save_dir}")
    print(f"{'=' * 60}")

    if lora_config is None:
        # Full fine-tune: base weights have changed, save them.
        save_model_to_safetensors(
            ttml_model,
            config,
            save_dir,
            tie_word_embeddings,
            distributed=distributed,
            device=device,
            shard_dim=shard_dim,
            dp_size=dp_size,
            lora_config=None,
            merge_lora=False,
        )

    if lora_config is not None:
        save_lora_adapters(
            ttml_model,
            save_dir,
            distributed=distributed,
            device=device,
            dp_size=dp_size,
        )

    save_optimizer_state(
        optimizer,
        step,
        ttml_model,
        config,
        tie_word_embeddings,
        save_dir,
        distributed=distributed,
        device=device,
        shard_dim=shard_dim,
        dp_size=dp_size,
    )

    state = {"step": step}
    if args_dict:
        state["args"] = args_dict
    with open(os.path.join(save_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    print(f"Checkpoint saved: step={step}")


def load_checkpoint(
    ttml_model,
    optimizer,
    load_dir: str,
    config,
    tie_word_embeddings: bool,
    distributed: bool = False,
    device=None,
    shard_dim: Optional[int] = None,
    dp_size: int = 1,
    lora_config: Optional[dict] = None,
    skip_model_weights: bool = False,
) -> int:
    """Load a complete training checkpoint.

    Loads model weights (base + optional LoRA adapters) and optimizer state.
    Returns the saved step count so training can resume from the right step.

    When LoRA is active the base weights are frozen and are NOT saved in the
    checkpoint (see save_checkpoint).  The caller is responsible for loading
    the original HF weights into ttml_model *before* calling this function,
    and for injecting LoRA adapters via inject_adapter_in_model() so that the
    adapter parameter paths exist for the saved adapter weights to be assigned.

    When LoRA is NOT active, model.safetensors must be present in load_dir and
    is loaded here (unless skip_model_weights=True, meaning the caller already
    loaded them — e.g. to ensure the optimizer sees initialized tensors).
    """
    print(f"\n{'=' * 60}")
    print(f"Loading checkpoint from {load_dir}")
    print(f"{'=' * 60}")

    if lora_config is None and not skip_model_weights:
        model_path = os.path.join(load_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No model checkpoint found at {model_path}")

        hf_state_dict = load_model_from_safetensors(load_dir)
        print(f"\nLoading model weights from checkpoint...")
        _load_hf_dict_into_ttml(
            ttml_model,
            hf_state_dict,
            config,
            tie_word_embeddings=tie_word_embeddings,
            distributed=distributed,
            shard_dim=shard_dim,
        )
        print("  Model weights loaded.")
    elif lora_config is None and skip_model_weights:
        print(
            "\nModel weights already loaded from checkpoint (before optimizer creation)."
        )
    else:
        print(
            "\nLoRA checkpoint: base weights already loaded from HF model, skipping model.safetensors."
        )

    if lora_config is not None:
        print("\nLoading LoRA adapters from checkpoint...")
        load_lora_adapters(
            ttml_model,
            load_dir,
            distributed=distributed,
            device=device,
            shard_dim=shard_dim,
            dp_size=dp_size,
        )

    step = load_optimizer_state(
        optimizer,
        load_dir,
        ttml_model,
        config,
        tie_word_embeddings,
        distributed=distributed,
        device=device,
        shard_dim=shard_dim,
        dp_size=dp_size,
    )

    # Cross-check with training_state.json
    state_path = os.path.join(load_dir, "training_state.json")
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        json_step = state.get("step", step)
        if json_step != step:
            print(
                f"  Note: training_state.json says step={json_step} but "
                f"optimizer says step={step}. Using training_state.json."
            )
            step = json_step

    print(f"\nCheckpoint loaded: resuming from step {step + 1}")
    return step


def export_hf_model(
    ttml_model,
    config,
    save_dir: str,
    tie_word_embeddings: bool,
    original_model_path: str,
    distributed: bool = False,
    device=None,
    shard_dim: Optional[int] = None,
    dp_size: int = 1,
    lora_config: Optional[dict] = None,
) -> str:
    """Export the fine-tuned model in HF-compatible format.

    Saves merged model weights (LoRA merged if applicable) plus copies
    the tokenizer and model config from the original HF model path so
    the output directory can be loaded directly with AutoModelForCausalLM.

    Returns the export directory path.
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'=' * 60}")
    print(f"Exporting HF model → {save_dir}")
    print(f"{'=' * 60}")

    save_model_to_safetensors(
        ttml_model,
        config,
        save_dir,
        tie_word_embeddings,
        distributed=distributed,
        device=device,
        shard_dim=shard_dim,
        dp_size=dp_size,
        lora_config=lora_config,
        merge_lora=True,
        filename="model.safetensors",
    )

    # Copy tokenizer and config files from the original model
    import shutil

    config_files = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "generation_config.json",
        "vocab.json",
        "merges.txt",
    ]
    copied = []
    if os.path.isdir(original_model_path):
        for fname in config_files:
            src = os.path.join(original_model_path, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(save_dir, fname))
                copied.append(fname)
    if copied:
        print(f"  Copied from {original_model_path}: {copied}")
    else:
        print(
            f"  Note: could not copy tokenizer/config from {original_model_path}.\n"
            f"  Copy them manually or use the original model's tokenizer."
        )

    # Write a minimal model.safetensors.index.json for multi-shard compatibility
    # (single-shard model: all weights in model.safetensors)
    index = {
        "metadata": {"total_size": 0},
        "weight_map": {},
    }
    index_path = os.path.join(save_dir, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nHF export complete. Load with:")
    print(f'  AutoModelForCausalLM.from_pretrained("{save_dir}")')
    return save_dir
