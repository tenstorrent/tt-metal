import ttnn


def preprocess_gated_attention(state_dict, path: str, device=None):
    """
    Preprocess weights for a PatchTSMixerGatedAttention module.
    Expects state_dict keys of the form
        f"{path}.attn_layer.weight" #[d_model, d_model]
        f"{path}.attn_layer.bias"   #[d_model]
    and returns TTNN tensors ready for use.
    """
    # 2D linear weight [out_features, in_features]
    weight = state_dict[f"{path}.attn_layer.weight"]  # (D, D)
    bias = state_dict[f"{path}.attn_layer.bias"]  # (D, )

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

    return tt_weight, tt_bias


def patchtsmixer_preprocessor(device, state_dict):
    parameters = {}

    gated_paths = [
        # Add more as we implement modules
        "mixer_block.layers.0.feature_mixer.gate",
        "mixer_bloc.layers.0.patch_mixer.gate",
    ]

    for path in gated_paths:
        w, b = preprocess_gated_attention(state_dict, path, device=device)
        parameters[f"{path}.attn_layer.weight"] = w
        parameters[f"{path}.attn_layer.bias"] = b

    return parameters
