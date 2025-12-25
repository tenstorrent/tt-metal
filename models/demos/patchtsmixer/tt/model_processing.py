import ttnn


def preprocess_gated_attention(state_dict, path: str, device=None):
    """
    Preprocess weights for a PatchTSMixerGatedAttention module.
    Expects state_dict keys of the form
        f"{path}.attn_layer.weight" #[d_model, d_model]
        f"{path}.attn_layer.bias"   #[d_model]
    and returns TTNN tensors ready for use.
    """
    # 2D linear weight [out_features, in_features] from PyTorch
    weight = state_dict[f"{path}.attn_layer.weight"]  # (D, D)
    bias = state_dict[f"{path}.attn_layer.bias"]  # (D, )

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

    return tt_weight, tt_bias


def preprocess_positional_encoding(state_dict, path: str, device=None):
    """
    Preprocess PatchTSMixerPositionalEncoding's `pe`.

    Expects:
        state_dict[f"{path}.pe"] has shape [N_p, D]

    Produces:
        TTNN tensor of shape [1, 1, N_p, D] for broadcast with (B, C, N_p, D)
    """
    pe = state_dict[f"{path}.pe"]  # (N_p, D)
    pe = pe.view(1, 1, pe.shape[0], pe.shape[1])  # (1, 1, N_p, D)

    tt_pe = ttnn.from_torch(
        pe,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    return tt_pe


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

    # --- positional encoding ---
    pos_enc_path = "pos_enc"
    parameters[f"{pos_enc_path}.pe"] = preprocess_positional_encoding(state_dict, pos_enc_path, device=device)

    return parameters
