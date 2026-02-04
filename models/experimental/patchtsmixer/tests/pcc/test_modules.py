import pytest
import torch

import ttnn
from models.experimental.patchtsmixer.reference.pytorch_patchtsmixer import (
    FeatureMixerBlock,
    PatchMixerBlock,
    PatchTSMixerBlock,
    PatchTSMixerChannelFeatureMixerBlock,
    PatchTSMixerEmbedding,
    PatchTSMixerForecastHead,
    PatchTSMixerGatedAttention,
    PatchTSMixerLayer,
    PatchTSMixerMLP,
    PatchTSMixerNormLayer,
    PatchTSMixerPatchify,
    PatchTSMixerPositionalEncoding,
)
from models.experimental.patchtsmixer.tt.model_processing import (
    preprocess_embedding_proj,
    preprocess_feature_mixer_block,
    preprocess_forecast_head,
    preprocess_gated_attention,
    preprocess_layernorm,
    preprocess_mlp,
    preprocess_norm_layer_batchnorm,
    preprocess_positional_encoding,
)
from models.experimental.patchtsmixer.tt.patchtsmixer import (
    TtFeatureMixerBlock,
    TtPatchMixerBlock,
    TtPatchTSMixerBatchNorm,
    TtPatchTSMixerBlock,
    TtPatchTSMixerChannelFeatureMixerBlock,
    TtPatchTSMixerEmbedding,
    TtPatchTSMixerForecastHead,
    TtPatchTSMixerGatedAttention,
    TtPatchTSMixerLayer,
    TtPatchTSMixerLayerNorm,
    TtPatchTSMixerMLP,
    TtPatchTSMixerPatchify,
    TtPatchTSMixerPositionalEncoding,
)
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_patchtsmixer_gated_attention(device, reset_seeds):
    torch.manual_seed(42)

    d_model = 32
    torch_gate = PatchTSMixerGatedAttention(d_model)

    # -- Build a fake full-model-like state_dict with a nested path --
    base = "mixer_block.layers.0.feature_mixer.gate"

    # --- Run preprocessor for that gate path ---
    parameters = preprocess_gated_attention(torch_gate.state_dict(), base, device=device)

    # --- Build TTNN gated attention module ---
    tt_gate = TtPatchTSMixerGatedAttention(device=device, base_address=base, parameters=parameters)

    # --- Compare outputs on same input
    x = torch.randn(1, 2, 5, d_model)  # last dim = d_model
    torch_out = torch_gate(x)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_out = tt_gate(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("pe_type", ["sincos", "random"])
def test_patchtsmixer_positional_encoding(device, reset_seeds, pe_type):
    torch.manual_seed(42)

    B, C = 2, 3
    N_p, D = 16, 32

    torch_pe = PatchTSMixerPositionalEncoding(
        num_patches=N_p,
        d_model=D,
        use_pe=True,
        pe_type=pe_type,
    ).eval()

    # input (B, C, N_p, D)
    x = torch.randn(B, C, N_p, D)
    torch_out = torch_pe(x)

    base = "pos_enc"

    parameters = preprocess_positional_encoding(torch_pe.state_dict(), base, device=device)

    tt_pos = TtPatchTSMixerPositionalEncoding(
        device=device,
        base_address=base,
        parameters=parameters,
        num_patches=N_p,
        d_model=D,
    )

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_pos(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_patchtsmixer_norm_layer(device, reset_seeds):
    torch.manual_seed(42)

    B, C, N_p, D = 1, 2, 32, 32
    torch_norm = PatchTSMixerNormLayer(d_model=D, norm_type="LayerNorm", eps=1e-5).eval()

    base = "mixer_block.layers.0.patch_mixer"  # fake base

    parameters = preprocess_layernorm(torch_norm.state_dict(), base, device=device)

    tt_norm = TtPatchTSMixerLayerNorm(device=device, base_address=base, parameters=parameters)

    x = torch.randn(B, C, N_p, D)
    torch_out = torch_norm(x)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_norm(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_patchtsmixer_norm_layer_batchnorm(device, reset_seeds):
    torch.manual_seed(42)

    B, C, N_p, D = 1, 2, 32, 32
    torch_norm = PatchTSMixerNormLayer(d_model=D, norm_type="BatchNorm", eps=1e-5)

    base = "mixer_block.layers.0.patch_mixer"
    parameters = preprocess_norm_layer_batchnorm(torch_norm.state_dict(), base, device=device)

    tt_norm = TtPatchTSMixerBatchNorm(device=device, base_address=base, parameters=parameters)

    x = torch.randn(B, C, N_p, D)
    torch_out = torch_norm(x)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_norm(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    # BatchNorm is more sensitive to precision, use slightly relaxed threshold
    assert_with_pcc(torch_out, tt_out_torch, 0.98)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_patchtsmixer_mlp(device, reset_seeds):
    torch.manual_seed(42)

    in_features = 32
    out_features = 32
    expansion = 2

    torch_mlp = PatchTSMixerMLP(in_features, out_features, expansion=expansion, dropout=0.1).eval()

    base = "mixer_block.layers.0.feature_mixer.mlp"  # Fake path

    parameters = preprocess_mlp(torch_mlp.state_dict(), base, device=device)

    tt_mlp = TtPatchTSMixerMLP(device=device, base_address=base, parameters=parameters)

    # input rank-4 last dim = in_features
    B, C, Np = 1, 2, 64
    x = torch.randn(B, C, Np, in_features)

    torch_out = torch_mlp(x)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_mlp(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("use_gated_attn", [False, True])
def test_patchtsmixer_feature_mixer_block(device, reset_seeds, use_gated_attn):
    torch.manual_seed(42)

    D = 32
    expansion = 2
    norm_type = "LayerNorm"

    torch_block = FeatureMixerBlock(
        d_model=D,
        expansion=expansion,
        dropout=0.0,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    ).eval()

    base = "mixer_block.layers.0.feature_mixer"

    # --- Preprocess to TTNN parameters ---
    parameters = preprocess_feature_mixer_block(
        torch_block.state_dict(), base, device, norm_type=norm_type, use_gated_attn=use_gated_attn
    )

    # -- TT block --
    tt_block = TtFeatureMixerBlock(
        device=device,
        base_address=base,
        parameters=parameters,
        d_model=D,
        norm_type=norm_type,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    )

    #  --- Compare outputs ---
    B, C, N_p = 1, 2, 32
    x = torch.randn(B, C, N_p, D)

    torch_out = torch_block(x)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_block(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("use_gated_attn", [False, True])
def test_patchtsmixer_patch_mixer_block(device, reset_seeds, use_gated_attn):
    torch.manual_seed(42)

    N_p = 32
    D = 32
    expansion = 2
    norm_type = "LayerNorm"

    torch_block = PatchMixerBlock(
        num_patches=N_p,
        d_model=D,
        expansion=expansion,
        dropout=0.0,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    ).eval()

    base = "mixer_block.layers.0.patch_mixer"

    # --- Preprocess to TTNN parameters ---
    from models.experimental.patchtsmixer.tt.model_processing import preprocess_feature_mixer_block

    parameters = preprocess_feature_mixer_block(
        torch_block.state_dict(), base, device, norm_type=norm_type, use_gated_attn=use_gated_attn
    )

    # -- TT block ---
    tt_block = TtPatchMixerBlock(
        device=device,
        base_address=base,
        parameters=parameters,
        norm_type=norm_type,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    )

    # --- Compare outputs ---
    B, C = 1, 2
    x = torch.randn(B, C, N_p, D)

    torch_out = torch_block(x)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_block(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("use_gated_attn", [False, True])
def test_patchtsmixer_channel_feature_mixer_block(device, reset_seeds, use_gated_attn):
    torch.manual_seed(42)

    # Tile friendly shape and also C == D to avoid LN shape mismatch in the pytorch code
    B, C, N_p, D = 1, 32, 32, 32
    expansion = 2
    norm_type = "LayerNorm"

    torch_block = PatchTSMixerChannelFeatureMixerBlock(
        num_channels=C,
        d_model=C,
        norm_type=norm_type,
        expansion=expansion,
        dropout=0.0,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    ).eval()

    base = "mixer_block.layers.0.channel_mixer"

    # --- Preprocess to TTNN parameters ---
    parameters = preprocess_feature_mixer_block(
        torch_block.state_dict(), base, device, norm_type=norm_type, use_gated_attn=use_gated_attn
    )

    # ---- TT block ----
    tt_block = TtPatchTSMixerChannelFeatureMixerBlock(
        device=device,
        base_address=base,
        parameters=parameters,
        d_model=C,
        num_channels=C,
        expansion=expansion,
        norm_type=norm_type,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    )

    # ---- Compare outputs ----
    x = torch.randn(B, C, N_p, D)
    torch_out = torch_block(x)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_block(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("use_gated_attn", [False, True])
@pytest.mark.parametrize("channel", ["common_channel", "mix_channel"])
def test_patchtsmixer_layer(device, reset_seeds, use_gated_attn, channel):
    torch.manual_seed(42)

    B, C = 1, 2
    N_p = 32
    D = 32
    expansion = 2
    mode = channel
    norm_type = "LayerNorm"

    torch_layer = PatchTSMixerLayer(
        num_patches=N_p,
        d_model=D,
        num_channels=C,
        mode=mode,
        expansion=expansion,
        dropout=0.0,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    ).eval()

    base = "mixer_block.layers.0"  # this layer lives at layers.0 in the full model

    # ---- Preprocess to TTNN parameters ----
    from models.experimental.patchtsmixer.tt.model_processing import preprocess_layer

    parameters = preprocess_layer(
        torch_layer.state_dict(), base, device, mode=mode, norm_type=norm_type, use_gated_attn=use_gated_attn
    )

    # ---- TT layer ----
    tt_layer = TtPatchTSMixerLayer(
        device=device,
        base_address=base,
        parameters=parameters,
        num_patches=N_p,
        d_model=D,
        num_channels=C,
        mode=mode,
        norm_type=norm_type,
        expansion=expansion,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    )

    # ---- Compare outputs ----
    x = torch.randn(B, C, N_p, D)
    torch_out = torch_layer(x)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_layer(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize("use_gated_attn", [False, True])
def test_patchtsmixer_block(device, reset_seeds, use_gated_attn):
    torch.manual_seed(42)

    B, C, N_p, D = 1, 2, 32, 32
    num_layers = 2
    expansion = 2
    mode = "common_channel"
    norm_type = "LayerNorm"

    layer_kwargs = dict(
        num_patches=N_p,
        d_model=D,
        num_channels=C,
        mode=mode,
        expansion=expansion,
        dropout=0.0,
        use_gated_attn=use_gated_attn,
        eps=1e-5,
    )

    torch_block = PatchTSMixerBlock(num_layers=num_layers, layer_kwargs=layer_kwargs).eval()

    base = "mixer_block"  # fake full-model path

    # ---- Preprocess to TTNN parameters ----
    from models.experimental.patchtsmixer.tt.model_processing import preprocess_block

    parameters = preprocess_block(
        torch_block.state_dict(),
        base,
        device,
        num_layers=num_layers,
        mode=mode,
        norm_type=norm_type,
        use_gated_attn=use_gated_attn,
    )

    # ---- TT block ----
    tt_block = TtPatchTSMixerBlock(
        device=device,
        base_address=base,
        parameters=parameters,
        num_layers=num_layers,
        layer_kwargs=dict(
            num_patches=N_p,
            d_model=D,
            num_channels=C,
            mode=mode,
            expansion=expansion,
            use_gated_attn=use_gated_attn,
            eps=1e-5,
        ),
        norm_type=norm_type,
    )

    # ---- Compare outputs ----
    x = torch.randn(B, C, N_p, D)
    torch_out, _ = torch_block(x, output_hidden_states=False)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out, _ = tt_block(tt_x, output_hidden_states=False)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_patchtsmixer_forecast_head(device, reset_seeds):
    torch.manual_seed(42)

    B, C, N_p, D = 1, 2, 32, 32
    H = 16

    torch_head = PatchTSMixerForecastHead(num_patches=N_p, d_model=D, prediction_length=H, head_dropout=0.0).eval()
    base = "head"

    # preprocess
    parameters = preprocess_forecast_head(torch_head.state_dict(), base, device=device)

    tt_head = TtPatchTSMixerForecastHead(device=device, base_address=base, parameters=parameters, prediction_length=H)

    x = torch.randn(B, C, N_p, D)
    torch_out = torch_head(x)  # (B, H, C)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_y = tt_head(tt_x)  # (B, H, 1, C)
    tt_out = ttnn.to_torch(tt_y).squeeze(2)  # (B, H, C)

    assert_with_pcc(torch_out, tt_out, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_patchtsmixer_patchify(device, reset_seeds):
    torch.manual_seed(42)

    B = 2
    L = 96
    C = 7
    patch_length = 16
    patch_stride = 8

    torch_patchify = PatchTSMixerPatchify(context_length=L, patch_length=patch_length, patch_stride=patch_stride).eval()
    tt_patchify = TtPatchTSMixerPatchify(
        device=device, context_length=L, patch_length=patch_length, patch_stride=patch_stride
    )

    x = torch.randn(B, L, C)

    torch_out = torch_patchify(x)  # (B, C, Np, patch_len)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_patchify(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_patchtsmixer_embedding(device, reset_seeds):
    torch.manual_seed(42)

    B, C, L = 2, 7, 96
    patch_len = 16
    stride = 8
    d_model = 32

    torch_embed = PatchTSMixerEmbedding(
        context_length=L,
        patch_length=patch_len,
        patch_stride=stride,
        d_model=d_model,
    ).eval()

    base = "patch_embed"

    # preprocess proj params
    parameters = preprocess_embedding_proj(torch_embed.state_dict(), base, device=device)

    tt_embed = TtPatchTSMixerEmbedding(
        device=device,
        base_address=base,
        parameters=parameters,
        context_length=L,
        patch_length=patch_len,
        patch_stride=stride,
        d_model=d_model,
    )

    x = torch.randn(B, C, L)

    torch_out = torch_embed(x)  # (B,C,Np,d_model)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_out = tt_embed(tt_x, dtype=ttnn.bfloat16)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)
