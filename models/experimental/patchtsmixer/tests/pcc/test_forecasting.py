# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import (
    PatchTSMixerModelForForecasting,
    from_hf_config,
    load_hf_checkpoint,
)
from models.experimental.patchtsmixer.tt.model_processing import (
    preprocess_block,
    preprocess_embedding_proj,
    preprocess_forecast_head,
    preprocess_model,
    preprocess_positional_encoding,
)
from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerModelForForecasting
from tt_py_test_utils_common.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("use_gated_attn", [False, True])
@pytest.mark.parametrize("scaling", [None, "std"])
def test_patchtsmixer_model_for_forecasting(device, reset_seeds, use_gated_attn, scaling):
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
        scaling=scaling,
    ).eval()

    base = "model"

    # ---- Preprocess to TTNN parameters ----
    parameters = {}

    # Helper to extract sub-state_dict
    def extract_substate(prefix):
        sd = torch_model.state_dict()
        if prefix:
            return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}
        return sd

    # Embedding
    embed_sd = extract_substate("patch_embed.")
    parameters.update(preprocess_embedding_proj(embed_sd, f"{base}.patch_embed", device=device))

    # Positional encoding
    pos_enc_sd = extract_substate("pos_enc.")
    parameters.update(preprocess_positional_encoding(pos_enc_sd, f"{base}.pos_enc", device=device))

    # Mixer block (all layers)
    mixer_sd = extract_substate("mixer_block.")
    parameters.update(
        preprocess_block(
            mixer_sd, f"{base}.mixer_block", device, num_layers=num_layers, mode=mode, use_gated_attn=use_gated_attn
        )
    )

    # Forecast head
    head_sd = extract_substate("head.")
    parameters.update(preprocess_forecast_head(head_sd, f"{base}.head", device=device))

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
        scaling=scaling,
    )

    # ---- run parity ----
    # With scaling, each channel gets its own offset and spread, which the device must remove and restore
    loc, scale = (torch.linspace(-20, 40, C), torch.linspace(0.5, 8, C)) if scaling else (0.0, 1.0)
    past_values = torch.randn(B, L, C) * scale + loc
    torch_out = torch_model(past_values)  # (B, H, C)

    tt_past_values = ttnn.from_torch(
        past_values, device=device, dtype=ttnn.float32 if scaling else ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    tt_out = tt_model(tt_past_values, dtype=ttnn.bfloat16)  # (B, H, C)
    tt_out_torch = ttnn.to_torch(tt_out)  # (B, H, C)

    # Compared in the units of the standardised series so the channel offsets do not inflate the correlation
    assert_with_pcc((torch_out - loc) / scale, (tt_out_torch - loc) / scale, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_patchtsmixer_hf_checkpoint(device, reset_seeds):
    """IBM's published checkpoint through the TT model against the reference loaded from the same weights."""
    checkpoint = "ibm/patchtsmixer-etth1-forecasting"
    try:
        config, state_dict = load_hf_checkpoint(checkpoint)
    except (ImportError, OSError) as e:
        pytest.skip(f"{checkpoint} unavailable: {e}")
    torch_model = from_hf_config(config, state_dict)

    parameters = preprocess_model(
        torch_model.state_dict(),
        "model",
        device,
        num_layers=config["num_layers"],
        mode=config["mode"],
        use_gated_attn=config["gated_attn"],
    )
    tt_model = TtPatchTSMixerModelForForecasting(
        device=device,
        base_address="model",
        parameters=parameters,
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        patch_length=config["patch_length"],
        patch_stride=config["patch_stride"],
        num_channels=config["num_input_channels"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        mode=config["mode"],
        expansion=config["expansion_factor"],
        use_gated_attn=config["gated_attn"],
        eps=config["norm_eps"],
        scaling=config["scaling"],
        use_positional_encoding=config["use_positional_encoding"],
    )

    B, L, C = 2, config["context_length"], config["num_input_channels"]
    loc, scale = torch.linspace(-20, 40, C), torch.linspace(0.5, 8, C)
    past_values = torch.randn(B, L, C) * scale + loc
    torch_out = torch_model(past_values)  # (B, H, C)

    tt_past_values = ttnn.from_torch(past_values, device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    tt_out_torch = ttnn.to_torch(tt_model(tt_past_values))

    assert_with_pcc((torch_out - loc) / scale, (tt_out_torch - loc) / scale, 0.98)
