import pytest
import torch

import ttnn
from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerModelForForecasting
from models.experimental.patchtsmixer.tt.model_processing import (
    preprocess_block,
    preprocess_embedding_proj,
    preprocess_forecast_head,
    preprocess_positional_encoding,
)
from models.experimental.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerModelForForecasting
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
    )

    # ---- run parity ----
    past_values = torch.randn(B, L, C)
    torch_out = torch_model(past_values)  # (B, H, C)

    tt_past_values = ttnn.from_torch(past_values, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_model(tt_past_values, dtype=ttnn.bfloat16)  # (B, H, 1, C)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(2)  # (B, H, C)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)
