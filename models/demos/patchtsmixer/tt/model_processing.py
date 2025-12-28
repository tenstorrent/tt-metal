import torch

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


def preprocess_norm_layer_batchnorm(state_dict, path: str, device=None):
    # Pytorch BN1d params are [D] reshape to [1, D, 1, 1]
    w = state_dict[f"{path}.norm.weight"].view(1, -1, 1, 1)
    b = state_dict[f"{path}.norm.bias"].view(1, -1, 1, 1)
    m = state_dict[f"{path}.norm.running_mean"].view(1, -1, 1, 1)
    v = state_dict[f"{path}.norm.running_var"].view(1, -1, 1, 1)

    tt_w = ttnn.from_torch(w, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_b = ttnn.from_torch(b, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_m = ttnn.from_torch(m, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_v = ttnn.from_torch(v, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return tt_w, tt_b, tt_m, tt_v


def preprocess_layernorm(state_dict, path: str, device=None):
    # Pytorch ln has norm.weight [D] and norm.bias [D],
    # we will reshape them to [1,1,1,D] so that they broadcast to (B, C, N_p, D)
    gamma = state_dict[f"{path}.norm.weight"].view(1, 1, 1, -1)  # (1, 1, 1, D)
    beta = state_dict[f"{path}.norm.bias"].view(1, 1, 1, -1)  # (1, 1, 1, D)

    tt_gamma = ttnn.from_torch(gamma, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_beta = ttnn.from_torch(beta, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return tt_gamma, tt_beta


def preprocess_linear(state_dict, path: str, device=None, dtype=ttnn.bfloat16):
    """
    Converts a torch nn.Linear at `path` into TTNN
    Expects:
        state_dict[f"{path}.weight] shape (out, in)
        state_dict[f"{path}.bias] shape (out, in)
    Produces:
        weight: (1, 1, in, out) TILE
        bias: (1, 1, 1, out) TILE
    """
    w = state_dict[f"{path}.weight"]  # (out, in)
    b = state_dict[f"{path}.bias"]  # (out, )

    w = torch.transpose(w, 0, 1)  # (in out)
    w = w.view(1, 1, w.shape[0], w.shape[1])  # (1, 1, in, out)
    b = b.view(1, 1, 1, b.shape[0])  # (1, 1, 1, out)

    tt_w = ttnn.from_torch(w, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    tt_b = ttnn.from_torch(b, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return tt_w, tt_b


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

    # Batch normalization
    batchnorm_path = "bn"
    w, b, m, v = preprocess_norm_layer_batchnorm(state_dict, batchnorm_path, device=device)
    parameters[f"{batchnorm_path}.norm.weight"] = w
    parameters[f"{batchnorm_path}.norm.bias"] = b
    parameters[f"{batchnorm_path}.norm.running_mean"] = m
    parameters[f"{batchnorm_path}.norm.running_var"] = v

    # Layer Normalization
    layernorm_path = "ln"
    g, b = preprocess_layernorm(state_dict, layernorm_path, device=device)
    parameters[f"{batchnorm_path}.norm.weight"] = g
    parameters[f"{batchnorm_path}.norm.bias"] = b

    return parameters
