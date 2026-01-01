import pytest
import torch

import ttnn
from models.demos.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerModelForForecasting
from models.demos.patchtsmixer.tt.model_processing import (
    preprocess_embedding_proj,
    preprocess_forecast_head,
    preprocess_gated_attention,
    preprocess_layernorm,
    preprocess_linear,
    preprocess_positional_encoding,
)
from models.demos.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerModelForForecasting
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("use_gated_attn", [False, True])
def test_patchtsmixer_model_for_forecasting(device, reset_seeds, use_gated_attn):
    torch.manual_seed(42)

    # ---- config ----
    B = 2
    L = 96
    C = 7
    patch_len = 16
    stride = 8
    D = 32
    H = 16
    num_layers = 2
    mode = "common_channel"  # Later add mix_channel.
    expansion = 2

    # ---- torch model ----
    torch_model = PatchTSMixerModelForForecasting(
        context_length=L,
        prediction_length=H,
        patch_length=patch_len,
        patch_stride=stride,
        num_channels=C,
        d_model=D,
        num_layers=num_layers,
        mode=mode,
        expansion=expansion,
        dropout=0.0,
        use_gated_attn=use_gated_attn,
        head_dropout=0.0,
        eps=1e-5,
    ).eval()

    base = "model"
    sd = torch_model.state_dict()

    # ---- fake nested torch-style state_dict (manual) ----
    state_dict = {}

    # (A) embedding proj
    state_dict[f"{base}.patch_embed.proj.weight"] = sd["patch_embed.proj.weight"]
    state_dict[f"{base}.patch_embed.proj.bias"] = sd["patch_embed.proj.bias"]

    # (B) positional encoding (align with your design)
    # If you store pe as buffer or parameter, add it here.
    # Commonly in your port you’ll preprocess it from torch model object,
    # but since you said "already processed", we just place it in state_dict.
    state_dict[f"{base}.pos_enc.pe"] = sd["pos_enc.pe"]

    # (C) mixer_block layers
    for i in range(num_layers):
        prefix = f"{base}.mixer_block.layers.{i}"

        # patch_mixer LN + MLP
        state_dict[f"{prefix}.patch_mixer.norm.norm.weight"] = sd[
            f"mixer_block.layers.{i}.patch_mixer.norm.norm.weight"
        ]
        state_dict[f"{prefix}.patch_mixer.norm.norm.bias"] = sd[f"mixer_block.layers.{i}.patch_mixer.norm.norm.bias"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc1.weight"] = sd[f"mixer_block.layers.{i}.patch_mixer.mlp.fc1.weight"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc1.bias"] = sd[f"mixer_block.layers.{i}.patch_mixer.mlp.fc1.bias"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc2.weight"] = sd[f"mixer_block.layers.{i}.patch_mixer.mlp.fc2.weight"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc2.bias"] = sd[f"mixer_block.layers.{i}.patch_mixer.mlp.fc2.bias"]

        # feature_mixer LN + MLP
        state_dict[f"{prefix}.feature_mixer.norm.norm.weight"] = sd[
            f"mixer_block.layers.{i}.feature_mixer.norm.norm.weight"
        ]
        state_dict[f"{prefix}.feature_mixer.norm.norm.bias"] = sd[
            f"mixer_block.layers.{i}.feature_mixer.norm.norm.bias"
        ]
        state_dict[f"{prefix}.feature_mixer.mlp.fc1.weight"] = sd[
            f"mixer_block.layers.{i}.feature_mixer.mlp.fc1.weight"
        ]
        state_dict[f"{prefix}.feature_mixer.mlp.fc1.bias"] = sd[f"mixer_block.layers.{i}.feature_mixer.mlp.fc1.bias"]
        state_dict[f"{prefix}.feature_mixer.mlp.fc2.weight"] = sd[
            f"mixer_block.layers.{i}.feature_mixer.mlp.fc2.weight"
        ]
        state_dict[f"{prefix}.feature_mixer.mlp.fc2.bias"] = sd[f"mixer_block.layers.{i}.feature_mixer.mlp.fc2.bias"]

        if use_gated_attn:
            state_dict[f"{prefix}.patch_mixer.gate.attn_layer.weight"] = sd[
                f"mixer_block.layers.{i}.patch_mixer.gate.attn_layer.weight"
            ]
            state_dict[f"{prefix}.patch_mixer.gate.attn_layer.bias"] = sd[
                f"mixer_block.layers.{i}.patch_mixer.gate.attn_layer.bias"
            ]
            state_dict[f"{prefix}.feature_mixer.gate.attn_layer.weight"] = sd[
                f"mixer_block.layers.{i}.feature_mixer.gate.attn_layer.weight"
            ]
            state_dict[f"{prefix}.feature_mixer.gate.attn_layer.bias"] = sd[
                f"mixer_block.layers.{i}.feature_mixer.gate.attn_layer.bias"
            ]

    # (D) head proj
    state_dict[f"{base}.head.proj.weight"] = sd["head.proj.weight"]
    state_dict[f"{base}.head.proj.bias"] = sd["head.proj.bias"]

    # ---- preprocess into TT parameters dict ----
    parameters = {}

    # embedding
    w_tt, b_tt = preprocess_embedding_proj(state_dict, f"{base}.patch_embed", device=device)
    parameters[f"{base}.patch_embed.proj.weight"] = w_tt
    parameters[f"{base}.patch_embed.proj.bias"] = b_tt

    tt_pe = preprocess_positional_encoding(state_dict, f"{base}.pos_enc", device=device)
    parameters[f"{base}.pos_enc.pe"] = tt_pe

    # mixer layers
    for i in range(num_layers):
        prefix = f"{base}.mixer_block.layers.{i}"

        def load_mixer(mixer_name: str):
            mixer_path = f"{prefix}.{mixer_name}"

            gamma, beta = preprocess_layernorm(state_dict, f"{mixer_path}.norm", device=device)
            parameters[f"{mixer_path}.norm.norm.weight"] = gamma
            parameters[f"{mixer_path}.norm.norm.bias"] = beta

            w1, b1 = preprocess_linear(state_dict, f"{mixer_path}.mlp.fc1", device=device)
            w2, b2 = preprocess_linear(state_dict, f"{mixer_path}.mlp.fc2", device=device)
            parameters[f"{mixer_path}.mlp.fc1.weight"] = w1
            parameters[f"{mixer_path}.mlp.fc1.bias"] = b1
            parameters[f"{mixer_path}.mlp.fc2.weight"] = w2
            parameters[f"{mixer_path}.mlp.fc2.bias"] = b2

            if use_gated_attn:
                gw, gb = preprocess_gated_attention(state_dict, f"{mixer_path}.gate", device=device)
                parameters[f"{mixer_path}.gate.attn_layer.weight"] = gw
                parameters[f"{mixer_path}.gate.attn_layer.bias"] = gb

        load_mixer("patch_mixer")
        load_mixer("feature_mixer")

    # head
    hw_tt, hb_tt = preprocess_forecast_head(state_dict, f"{base}.head", device=device)
    parameters[f"{base}.head.proj.weight"] = hw_tt
    parameters[f"{base}.head.proj.bias"] = hb_tt

    # ---- TT model ----
    tt_model = TtPatchTSMixerModelForForecasting(
        device=device,
        base_address=base,
        parameters=parameters,
        context_length=L,
        prediction_length=H,
        patch_length=patch_len,
        patch_stride=stride,
        num_channels=C,
        d_model=D,
        num_layers=num_layers,
        mode=mode,
        expansion=expansion,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    )

    # ---- run parity ----
    past_values = torch.randn(B, L, C)
    torch_out = torch_model(past_values)  # (B, H, C)

    tt_out = tt_model(past_values, dtype=ttnn.bfloat16)  # (B, H, 1, C)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(2)  # (B, H, C)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("use_gated_attn", [False, True])
def test_patchtsmixer_model_for_forecasting(device, reset_seeds, use_gated_attn):
    torch.manual_seed(42)

    # ---- config ----
    B = 2
    L = 96
    C = 7
    patch_len = 16
    stride = 8
    D = 32
    H = 16
    num_layers = 2
    mode = "common_channel"
    expansion = 2

    # ---- torch model ----
    torch_model = PatchTSMixerModelForForecasting(
        context_length=L,
        prediction_length=H,
        patch_length=patch_len,
        patch_stride=stride,
        num_channels=C,
        d_model=D,
        num_layers=num_layers,
        mode=mode,
        expansion=expansion,
        dropout=0.0,
        use_gated_attn=use_gated_attn,
        head_dropout=0.0,
        eps=1e-5,
    ).eval()

    base = "model"
    sd = torch_model.state_dict()

    # ---- fake nested torch-style state_dict (manual) ----
    state_dict = {}

    # (A) embedding proj
    state_dict[f"{base}.patch_embed.proj.weight"] = sd["patch_embed.proj.weight"]
    state_dict[f"{base}.patch_embed.proj.bias"] = sd["patch_embed.proj.bias"]

    # (B) positional encoding (align with your design)
    # If you store pe as buffer or parameter, add it here.
    # Commonly in your port you’ll preprocess it from torch model object,
    # but since you said "already processed", we just place it in state_dict.
    state_dict[f"{base}.pos_enc.pe"] = sd["pos_enc.pe"]

    # (C) mixer_block layers
    for i in range(num_layers):
        prefix = f"{base}.mixer_block.layers.{i}"

        # patch_mixer LN + MLP
        state_dict[f"{prefix}.patch_mixer.norm.norm.weight"] = sd[
            f"mixer_block.layers.{i}.patch_mixer.norm.norm.weight"
        ]
        state_dict[f"{prefix}.patch_mixer.norm.norm.bias"] = sd[f"mixer_block.layers.{i}.patch_mixer.norm.norm.bias"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc1.weight"] = sd[f"mixer_block.layers.{i}.patch_mixer.mlp.fc1.weight"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc1.bias"] = sd[f"mixer_block.layers.{i}.patch_mixer.mlp.fc1.bias"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc2.weight"] = sd[f"mixer_block.layers.{i}.patch_mixer.mlp.fc2.weight"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc2.bias"] = sd[f"mixer_block.layers.{i}.patch_mixer.mlp.fc2.bias"]

        # feature_mixer LN + MLP
        state_dict[f"{prefix}.feature_mixer.norm.norm.weight"] = sd[
            f"mixer_block.layers.{i}.feature_mixer.norm.norm.weight"
        ]
        state_dict[f"{prefix}.feature_mixer.norm.norm.bias"] = sd[
            f"mixer_block.layers.{i}.feature_mixer.norm.norm.bias"
        ]
        state_dict[f"{prefix}.feature_mixer.mlp.fc1.weight"] = sd[
            f"mixer_block.layers.{i}.feature_mixer.mlp.fc1.weight"
        ]
        state_dict[f"{prefix}.feature_mixer.mlp.fc1.bias"] = sd[f"mixer_block.layers.{i}.feature_mixer.mlp.fc1.bias"]
        state_dict[f"{prefix}.feature_mixer.mlp.fc2.weight"] = sd[
            f"mixer_block.layers.{i}.feature_mixer.mlp.fc2.weight"
        ]
        state_dict[f"{prefix}.feature_mixer.mlp.fc2.bias"] = sd[f"mixer_block.layers.{i}.feature_mixer.mlp.fc2.bias"]

        if use_gated_attn:
            state_dict[f"{prefix}.patch_mixer.gate.attn_layer.weight"] = sd[
                f"mixer_block.layers.{i}.patch_mixer.gate.attn_layer.weight"
            ]
            state_dict[f"{prefix}.patch_mixer.gate.attn_layer.bias"] = sd[
                f"mixer_block.layers.{i}.patch_mixer.gate.attn_layer.bias"
            ]
            state_dict[f"{prefix}.feature_mixer.gate.attn_layer.weight"] = sd[
                f"mixer_block.layers.{i}.feature_mixer.gate.attn_layer.weight"
            ]
            state_dict[f"{prefix}.feature_mixer.gate.attn_layer.bias"] = sd[
                f"mixer_block.layers.{i}.feature_mixer.gate.attn_layer.bias"
            ]

    # (D) head proj
    state_dict[f"{base}.head.proj.weight"] = sd["head.proj.weight"]
    state_dict[f"{base}.head.proj.bias"] = sd["head.proj.bias"]

    # ---- preprocess into TT parameters dict ----
    parameters = {}

    # embedding
    w_tt, b_tt = preprocess_embedding_proj(state_dict, f"{base}.patch_embed", device=device)
    parameters[f"{base}.patch_embed.proj.weight"] = w_tt
    parameters[f"{base}.patch_embed.proj.bias"] = b_tt

    tt_pe = preprocess_positional_encoding(state_dict, f"{base}.pos_enc", device=device)
    parameters[f"{base}.pos_enc.pe"] = tt_pe

    # mixer layers
    for i in range(num_layers):
        prefix = f"{base}.mixer_block.layers.{i}"

        def load_mixer(mixer_name: str):
            mixer_path = f"{prefix}.{mixer_name}"

            gamma, beta = preprocess_layernorm(state_dict, f"{mixer_path}.norm", device=device)
            parameters[f"{mixer_path}.norm.norm.weight"] = gamma
            parameters[f"{mixer_path}.norm.norm.bias"] = beta

            w1, b1 = preprocess_linear(state_dict, f"{mixer_path}.mlp.fc1", device=device)
            w2, b2 = preprocess_linear(state_dict, f"{mixer_path}.mlp.fc2", device=device)
            parameters[f"{mixer_path}.mlp.fc1.weight"] = w1
            parameters[f"{mixer_path}.mlp.fc1.bias"] = b1
            parameters[f"{mixer_path}.mlp.fc2.weight"] = w2
            parameters[f"{mixer_path}.mlp.fc2.bias"] = b2

            if use_gated_attn:
                gw, gb = preprocess_gated_attention(state_dict, f"{mixer_path}.gate", device=device)
                parameters[f"{mixer_path}.gate.attn_layer.weight"] = gw
                parameters[f"{mixer_path}.gate.attn_layer.bias"] = gb

        load_mixer("patch_mixer")
        load_mixer("feature_mixer")

    # head
    hw_tt, hb_tt = preprocess_forecast_head(state_dict, f"{base}.head", device=device)
    parameters[f"{base}.head.proj.weight"] = hw_tt
    parameters[f"{base}.head.proj.bias"] = hb_tt

    # ---- TT model ----
    tt_model = TtPatchTSMixerModelForForecasting(
        device=device,
        base_address=base,
        parameters=parameters,
        context_length=L,
        prediction_length=H,
        patch_length=patch_len,
        patch_stride=stride,
        num_channels=C,
        d_model=D,
        num_layers=num_layers,
        mode=mode,
        expansion=expansion,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    )

    # ---- run parity ----
    past_values = torch.randn(B, L, C)
    torch_out = torch_model(past_values)  # (B, H, C)

    tt_out = tt_model(past_values, dtype=ttnn.bfloat16)  # (B, H, 1, C)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(2)  # (B, H, C)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)
