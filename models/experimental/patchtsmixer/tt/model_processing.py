import torch

import ttnn


def preprocess_gated_attention(state_dict, path: str, device=None):
    """
    Preprocess weights for a PatchTSMixerGatedAttention module.
    Expects state_dict keys of the form
        f"attn_layer.weight" #[d_model, d_model]
        f"attn_layer.bias"   #[d_model]
    and returns TTNN tensors ready for use.
    """
    # 2D linear weight [out_features, in_features] from PyTorch
    weight = state_dict[f"attn_layer.weight"]  # (D, D)
    bias = state_dict["attn_layer.bias"]  # (D, )

    # ttnn.linear expects weight as [in_features, out_features], so transpose
    weight = weight.T  # (D, D) transposed

    # For TTNN, convert to tiled 4D: [1, 1, D, D]
    tt_weight = ttnn.from_torch(
        weight.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device  # (1, 1, D, D)
    )

    # Bias as [1, 1, 1, D] so it can broadcast along all non-features dims
    tt_bias = ttnn.from_torch(
        bias.view(1, 1, 1, -1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return {
        f"{path}.attn_layer.weight": tt_weight,
        f"{path}.attn_layer.bias": tt_bias,
    }


def preprocess_positional_encoding(state_dict, path: str, device=None):
    """
    Preprocess PatchTSMixerPositionalEncoding's `pe`.

    Expects:
        state_dict[f"pe"] has shape [N_p, D]

    Produces:
        TTNN tensor of shape [1, 1, N_p, D] for broadcast with (B, C, N_p, D)
    """
    pe = state_dict[f"pe"]  # (N_p, D)
    pe = pe.view(1, 1, pe.shape[0], pe.shape[1])  # (1, 1, N_p, D)

    tt_pe = ttnn.from_torch(
        pe,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return {f"{path}.pe": tt_pe}


def preprocess_norm_layer_batchnorm(state_dict, path: str, device=None):
    # Pytorch BN1d params are [D] reshape to [1, D, 1, 1]

    w = state_dict[f"norm.weight"].view(1, -1, 1, 1)
    b = state_dict[f"norm.bias"].view(1, -1, 1, 1)
    m = state_dict[f"norm.running_mean"].view(1, -1, 1, 1)
    v = state_dict[f"norm.running_var"].view(1, -1, 1, 1)

    tt_w = ttnn.from_torch(w, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_b = ttnn.from_torch(b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_m = ttnn.from_torch(m, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v = ttnn.from_torch(v, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    return {
        f"{path}.norm.weight": tt_w,
        f"{path}.norm.bias": tt_b,
        f"{path}.norm.running_mean": tt_m,
        f"{path}.norm.running_var": tt_v,
    }


def preprocess_layernorm(state_dict, path: str, device=None):
    # Pytorch ln has norm.weight [D] and norm.bias [D],
    # we will reshape them to [1,1,1,D] so that they broadcast to (B, C, N_p, D)
    gamma = state_dict[f"norm.weight"].view(1, 1, 1, -1)  # (1, 1, 1, D)
    beta = state_dict[f"norm.bias"].view(1, 1, 1, -1)  # (1, 1, 1, D)

    tt_gamma = ttnn.from_torch(gamma, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_beta = ttnn.from_torch(beta, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    return {
        f"{path}.norm.weight": tt_gamma,
        f"{path}.norm.bias": tt_beta,
    }


def preprocess_linear(state_dict, path: str, device=None, dtype=ttnn.bfloat16):
    """
    Converts a torch nn.Linear at `path` into TTNN
    Expects:
        state_dict[f"weight] shape (out, in)
        state_dict[f"bias] shape (out, )
    Produces:
        weight: (1, 1, in, out) TILE
        bias: (1, 1, 1, out) TILE
    """
    w = state_dict[f"weight"]  # (out, in)
    b = state_dict[f"bias"]  # (out, )

    w = torch.transpose(w, 0, 1)  # (in out)
    w = w.view(1, 1, w.shape[0], w.shape[1])  # (1, 1, in, out)
    b = b.view(1, 1, 1, b.shape[0])  # (1, 1, 1, out)

    tt_w = ttnn.from_torch(w, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tt_b = ttnn.from_torch(b, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return {f"{path}.weight": tt_w, f"{path}.bias": tt_b}


def preprocess_mlp(state_dict, path: str, device=None, dtype=ttnn.bfloat16):
    """
    process 2 layers mlp.
    """
    parameters = {}

    state_dict1 = {
        "weight": state_dict["fc1.weight"],
        "bias": state_dict["fc1.bias"],
    }

    state_dict2 = {
        "weight": state_dict["fc2.weight"],
        "bias": state_dict["fc2.bias"],
    }

    parameters.update(preprocess_linear(state_dict1, f"{path}.fc1", device=device, dtype=dtype))
    parameters.update(preprocess_linear(state_dict2, f"{path}.fc2", device=device, dtype=dtype))
    return parameters


def preprocess_feature_mixer_block(state_dict: dict, path: str, device, *, norm_type="LayerNorm", use_gated_attn=False):
    """
    Preprocess a feature/patch mixer block from PyTorch to TTNN.

    """
    parameters = {}

    # --- norm ---
    # Extract the norm.norm.* keys from PyTorch state_dict
    norm_state_dict = {
        "norm.weight": state_dict["norm.norm.weight"],
        "norm.bias": state_dict["norm.norm.bias"],
    }

    if "batch" in norm_type.lower():
        norm_state_dict["norm.running_mean"] = state_dict["norm.norm.running_mean"]
        norm_state_dict["norm.running_var"] = state_dict["norm.norm.running_var"]
        norm_params = preprocess_norm_layer_batchnorm(norm_state_dict, f"{path}.norm", device=device)
    else:
        norm_params = preprocess_layernorm(norm_state_dict, f"{path}.norm", device=device)

    parameters.update(norm_params)

    # --- mlp ---
    mlp_state_dict = {
        "fc1.weight": state_dict["mlp.fc1.weight"],
        "fc1.bias": state_dict["mlp.fc1.bias"],
        "fc2.weight": state_dict["mlp.fc2.weight"],
        "fc2.bias": state_dict["mlp.fc2.bias"],
    }
    mlp_params = preprocess_mlp(mlp_state_dict, f"{path}.mlp", device=device)
    parameters.update(mlp_params)

    # --- gate (optional) ---
    if use_gated_attn:
        gate_state_dict = {
            "attn_layer.weight": state_dict["gate.attn_layer.weight"],
            "attn_layer.bias": state_dict["gate.attn_layer.bias"],
        }
        gate_params = preprocess_gated_attention(gate_state_dict, f"{path}.gate", device=device)
        parameters.update(gate_params)

    return parameters


def preprocess_layer(
    state_dict: dict, path: str, device, *, mode="common_channel", norm_type="LayerNorm", use_gated_attn=False
):
    """
    Preprocess a PatchTSMixerLayer from PyTorch to TTNN.

    Args:
        state_dict: pyTorch state_dict with keys like "patch_mixer.norm.norm.weight", etc.
        path: Base path for TTNN parameters (e.g., "mixer_block.layers.0")
        device: TTNN device
        mode: "common_channel" or "mix_channel"
        norm_type: "LayerNorm" or "BatchNorm"
        use_gated_attn: Whether the layer uses gated attention

    Returns:
        Dict of TTNN parameters with prefixed keys
    """
    parameters = {}

    # Helper to extract sub-state_dict for a mixer block
    def extract_mixer_state_dict(mixer_name):
        prefix = f"{mixer_name}."
        return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}

    # Process patch_mixer
    patch_mixer_sd = extract_mixer_state_dict("patch_mixer")
    patch_mixer_params = preprocess_feature_mixer_block(
        patch_mixer_sd, f"{path}.patch_mixer", device, norm_type=norm_type, use_gated_attn=use_gated_attn
    )
    parameters.update(patch_mixer_params)

    # Process feature_mixer
    feature_mixer_sd = extract_mixer_state_dict("feature_mixer")
    feature_mixer_params = preprocess_feature_mixer_block(
        feature_mixer_sd, f"{path}.feature_mixer", device, norm_type=norm_type, use_gated_attn=use_gated_attn
    )
    parameters.update(feature_mixer_params)

    # Process channel_mixer if mode == "mix_channel"
    if mode == "mix_channel":
        channel_mixer_sd = extract_mixer_state_dict("channel_mixer")
        channel_mixer_params = preprocess_feature_mixer_block(
            channel_mixer_sd, f"{path}.channel_mixer", device, norm_type=norm_type, use_gated_attn=use_gated_attn
        )
        parameters.update(channel_mixer_params)

    return parameters


def preprocess_block(
    state_dict: dict,
    path: str,
    device,
    *,
    num_layers=1,
    mode="common_channel",
    norm_type="LayerNorm",
    use_gated_attn=False,
):
    """
    Preprocess a PatchTSMixerBlock from PyTorch to TTNN.

    Args:
        state_dict: Raw PyTorch state_dict with keys like "layers.0.patch_mixer.norm.norm.weight", etc.
        path: Base path for TTNN parameters (e.g., "mixer_block")
        device: TTNN device
        num_layers: Number of layers in the block
        mode: "common_channel" or "mix_channel"
        norm_type: "LayerNorm" or "BatchNorm"
        use_gated_attn: Whether the block uses gated attention

    """
    parameters = {}

    for i in range(num_layers):
        # Extract state_dict for this layer
        layer_prefix = f"layers.{i}."
        layer_state_dict = {k[len(layer_prefix) :]: v for k, v in state_dict.items() if k.startswith(layer_prefix)}

        # Preprocess the layer
        layer_params = preprocess_layer(
            layer_state_dict,
            f"{path}.layers.{i}",
            device,
            mode=mode,
            norm_type=norm_type,
            use_gated_attn=use_gated_attn,
        )
        parameters.update(layer_params)

    return parameters


def preprocess_forecast_head(state_dict, path: str, device):
    """
    Preprocess PatchTSMixerForecastHead from PyTorch to TTNN.

    """
    w = state_dict["proj.weight"]  # (H, in)
    b = state_dict["proj.bias"]  # (H,)

    # transpose to (in, H)
    w = w.transpose(0, 1).contiguous()

    w_tt = ttnn.from_torch(w.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b.view(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return {
        f"{path}.proj.weight": w_tt,
        f"{path}.proj.bias": b_tt,
    }


def preprocess_embedding_proj(state_dict, path: str, device):
    """
    Preprocess PatchTSMixerEmbedding projection from PyTorch to TTNN.

    """
    w = state_dict["proj.weight"]  # (d_model, patch_len)
    b = state_dict["proj.bias"]  # (d_model,)

    # transpose to (patch_len, d_model)
    w = w.transpose(0, 1).contiguous()

    w_tt = ttnn.from_torch(w.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b.view(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return {
        f"{path}.proj.weight": w_tt,
        f"{path}.proj.bias": b_tt,
    }


def preprocess_linear_head(state_dict, path: str, device):
    """
    Preprocess PatchTSMixerLinearHead for classification/regression.

    Expects:
        state_dict["projection.weight"]: (num_targets, in_features)
        state_dict["projection.bias"]: (num_targets,)
    """
    w = state_dict["projection.weight"]  # (num_targets, in_features)
    b = state_dict["projection.bias"]  # (num_targets,)

    # Transpose to (in_features, num_targets) for ttnn.linear
    w = w.transpose(0, 1).contiguous()

    # Convert to TTNN: (1, 1, in_features, num_targets)
    w_tt = ttnn.from_torch(w.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Bias: (1, 1, 1, num_targets)
    b_tt = ttnn.from_torch(b.view(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return {
        f"{path}.projection.weight": w_tt,
        f"{path}.projection.bias": b_tt,
    }


def preprocess_pretrain_head(state_dict, path: str, device):
    """
    Preprocess PatchTSMixerPretrainHead for pre-training.

    Expects:
        state_dict["projection.weight"]: (patch_length, d_model)
        state_dict["projection.bias"]: (patch_length,)
    """
    w = state_dict["projection.weight"]  # (patch_length, d_model)
    b = state_dict["projection.bias"]  # (patch_length,)

    # Transpose to (d_model, patch_length) for ttnn.linear
    w = w.transpose(0, 1).contiguous()

    # Convert to TTNN: (1, 1, d_model, patch_length)
    w_tt = ttnn.from_torch(w.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Bias: (1, 1, 1, patch_length)
    b_tt = ttnn.from_torch(b.view(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    return {
        f"{path}.projection.weight": w_tt,
        f"{path}.projection.bias": b_tt,
    }
