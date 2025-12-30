import pytest
import torch

import ttnn
from models.demos.patchtsmixer.reference.pytorch_patchtsmixer import (
    FeatureMixerBlock,
    PatchMixerBlock,
    PatchTSMixerBlock,
    PatchTSMixerChannelFeatureMixerBlock,
    PatchTSMixerForecastHead,
    PatchTSMixerGatedAttention,
    PatchTSMixerLayer,
    PatchTSMixerMLP,
    PatchTSMixerNormLayer,
    PatchTSMixerPositionalEncoding,
)
from models.demos.patchtsmixer.tt.model_processing import (
    preprocess_feature_mixer_block,
    preprocess_forecast_head,
    preprocess_gated_attention,
    preprocess_layernorm,
    preprocess_linear,
    preprocess_norm_layer_batchnorm,
    preprocess_positional_encoding,
)
from models.demos.patchtsmixer.tt.patchtsmixer import (
    TtFeatureMixerBlock,
    TtPatchMixerBlock,
    TtPatchTSMixerBatchNorm,
    TtPatchTSMixerBlock,
    TtPatchTSMixerChannelFeatureMixerBlock,
    TtPatchTSMixerForecastHead,
    TtPatchTSMixerGatedAttention,
    TtPatchTSMixerLayer,
    TtPatchTSMixerLayerNorm,
    TtPatchTSMixerMLP,
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
    sd = torch_gate.state_dict()
    state_dict = {
        f"{base}.attn_layer.weight": sd["attn_layer.weight"],
        f"{base}.attn_layer.bias": sd["attn_layer.bias"],
    }

    # --- Run preprocessor for that gate path ---
    w_ttnn, b_ttnn = preprocess_gated_attention(state_dict, base, device=device)
    parameters = {
        f"{base}.attn_layer.weight": w_ttnn,
        f"{base}.attn_layer.bias": b_ttnn,
    }

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
    sd = torch_pe.state_dict()
    state_dict = {
        f"{base}.pe": sd["pe"],  # works for both buffer + parameter
    }

    tt_pe = preprocess_positional_encoding(state_dict, base, device=device)
    parameters = {f"{base}.pe": tt_pe}

    tt_pos = TtPatchTSMixerPositionalEncoding(
        device=device,
        base_address=base,
        parameters=parameters,
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
    sd = torch_norm.state_dict()
    state_dict = {
        f"{base}.norm.weight": sd["norm.weight"],
        f"{base}.norm.bias": sd["norm.bias"],
    }

    gamma, beta = preprocess_layernorm(state_dict, base, device=device)
    parameters = {
        f"{base}.norm.weight": gamma,
        f"{base}.norm.bias": beta,
    }

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
    sd = torch_norm.state_dict()
    state_dict = {
        f"{base}.norm.weight": sd["norm.weight"],
        f"{base}.norm.bias": sd["norm.bias"],
        f"{base}.norm.running_mean": sd["norm.running_mean"],
        f"{base}.norm.running_var": sd["norm.running_var"],
    }

    w, b, m, v = preprocess_norm_layer_batchnorm(state_dict, base, device=device)
    parameters = {
        f"{base}.norm.weight": w,
        f"{base}.norm.bias": b,
        f"{base}.norm.running_mean": m,
        f"{base}.norm.running_var": v,
    }

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
    hidden = in_features * expansion

    torch_mlp = PatchTSMixerMLP(in_features, out_features, expansion=expansion, dropout=0.1).eval()

    base = "mixer_block.layers.0.feature_mixer.mlp"  # Fake path

    sd = torch_mlp.state_dict()

    state_dict = {
        f"{base}.fc1.weight": sd["fc1.weight"],
        f"{base}.fc1.bias": sd["fc1.bias"],
        f"{base}.fc2.weight": sd["fc2.weight"],
        f"{base}.fc2.bias": sd["fc2.bias"],
    }

    w1, b1 = preprocess_linear(state_dict, f"{base}.fc1", device=device)
    w2, b2 = preprocess_linear(state_dict, f"{base}.fc2", device=device)

    parameters = {
        f"{base}.fc1.weight": w1,
        f"{base}.fc1.bias": b1,
        f"{base}.fc2.weight": w2,
        f"{base}.fc2.bias": b2,
    }

    tt_mlp = TtPatchTSMixerMLP(device=device, base_address=base, parameters=parameters)

    # input rank-4 last dim = in_features
    B, C, Np = 1, 2, 32
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

    # --- Build fake nested state_dict (keys look like full model) ---
    sd = torch_block.state_dict()
    state_dict = {
        # Norm layer
        f"{base}.norm.norm.weight": sd["norm.norm.weight"],
        f"{base}.norm.norm.bias": sd["norm.norm.bias"],
        # MLP
        f"{base}.mlp.fc1.weight": sd["mlp.fc1.weight"],
        f"{base}.mlp.fc1.bias": sd["mlp.fc1.bias"],
        f"{base}.mlp.fc2.weight": sd["mlp.fc2.weight"],
        f"{base}.mlp.fc2.bias": sd["mlp.fc2.bias"],
    }

    if use_gated_attn:
        state_dict[f"{base}.gate.attn_layer.weight"] = sd["gate.attn_layer.weight"]
        state_dict[f"{base}.gate.attn_layer.bias"] = sd["gate.attn_layer.bias"]

    # -- preprocess into TTNN parameters dict ---
    parameters = {}
    preprocess_feature_mixer_block(parameters, state_dict, base, device, norm_type=norm_type)

    if use_gated_attn:
        gw, gb = preprocess_gated_attention(state_dict, f"{base}.gate", device=device)
        parameters[f"{base}.gate.attn_layer.weight"] = gw
        parameters[f"{base}.gate.attn_layer.bias"] = gb

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

    # --- Fake nested state_dict ---
    sd = torch_block.state_dict()
    state_dict = {
        # NormLayer's internal LN: norm.norm.*
        f"{base}.norm.norm.weight": sd["norm.norm.weight"],
        f"{base}.norm.norm.bias": sd["norm.norm.bias"],
        # MLP mixes patches, so fc1 weight is (hidden, Np) in Pytorch, etc.
        f"{base}.mlp.fc1.weight": sd["mlp.fc1.weight"],
        f"{base}.mlp.fc1.bias": sd["mlp.fc1.bias"],
        f"{base}.mlp.fc2.weight": sd["mlp.fc2.weight"],
        f"{base}.mlp.fc2.bias": sd["mlp.fc2.bias"],
    }

    if use_gated_attn:
        state_dict[f"{base}.gate.attn_layer.weight"] = sd["gate.attn_layer.weight"]
        state_dict[f"{base}.gate.attn_layer.bias"] = sd["gate.attn_layer.bias"]

    # --- Preprocess to TTNN parameters ---

    parameters = {}

    # LayerNorm params
    gamma, beta = preprocess_layernorm(state_dict, f"{base}.norm", device=device)
    parameters[f"{base}.norm.norm.weight"] = gamma
    parameters[f"{base}.norm.norm.bias"] = beta

    # MLP linears
    w1, b1 = preprocess_linear(state_dict, f"{base}.mlp.fc1", device=device)
    w2, b2 = preprocess_linear(state_dict, f"{base}.mlp.fc2", device=device)

    parameters[f"{base}.mlp.fc1.weight"] = w1
    parameters[f"{base}.mlp.fc1.bias"] = b1
    parameters[f"{base}.mlp.fc2.weight"] = w2
    parameters[f"{base}.mlp.fc2.bias"] = b2

    # Gate (optional)
    if use_gated_attn:
        gw, gb = preprocess_gated_attention(state_dict, f"{base}.gate", device=device)
        parameters[f"{base}.gate.attn_layer.weight"] = gw
        parameters[f"{base}.gate.attn_layer.bias"] = gb

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

    # --- Fake nested state_dict ---
    sd = torch_block.state_dict()
    state_dict = {
        # norm is a PatchTSMixerNormLayer, whoe intern LN is norm.*
        f"{base}.norm.norm.weight": sd["norm.norm.weight"],
        f"{base}.norm.norm.bias": sd["norm.norm.bias"],
        # MLP (mix over channels => in_features = C, out_features = C)
        f"{base}.mlp.fc1.weight": sd["mlp.fc1.weight"],
        f"{base}.mlp.fc1.bias": sd["mlp.fc1.bias"],
        f"{base}.mlp.fc2.weight": sd["mlp.fc2.weight"],
        f"{base}.mlp.fc2.bias": sd["mlp.fc2.bias"],
    }

    if use_gated_attn:
        state_dict[f"{base}.gate.attn_layer.weight"] = sd["gate.attn_layer.weight"]
        state_dict[f"{base}.gate.attn_layer.bias"] = sd["gate.attn_layer.bias"]

    # --- preprocess into TTNN parameters dict ---
    parameters = {}

    # LN params
    gamma, beta = preprocess_layernorm(state_dict, f"{base}.norm", device=device)
    parameters[f"{base}.norm.norm.weight"] = gamma
    parameters[f"{base}.norm.norm.bias"] = beta

    # MLP parameters
    w1, b1 = preprocess_linear(state_dict, f"{base}.mlp.fc1", device=device)
    w2, b2 = preprocess_linear(state_dict, f"{base}.mlp.fc2", device=device)
    parameters[f"{base}.mlp.fc1.weight"] = w1
    parameters[f"{base}.mlp.fc1.bias"] = b1
    parameters[f"{base}.mlp.fc2.weight"] = w2
    parameters[f"{base}.mlp.fc2.bias"] = b2

    # Gate params (optional)
    if use_gated_attn:
        gw, gb = preprocess_gated_attention(state_dict, f"{base}.gate", device=device)
        parameters[f"{base}.gate.attn_layer.weight"] = gw
        parameters[f"{base}.gate.attn_layer.bias"] = gb

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

    sd = torch_layer.state_dict()

    # ---- Build fake nested state_dict ----
    state_dict = {
        f"{base}.patch_mixer.norm.norm.weight": sd["patch_mixer.norm.norm.weight"],
        f"{base}.patch_mixer.norm.norm.bias": sd["patch_mixer.norm.norm.bias"],
        f"{base}.patch_mixer.mlp.fc1.weight": sd["patch_mixer.mlp.fc1.weight"],
        f"{base}.patch_mixer.mlp.fc1.bias": sd["patch_mixer.mlp.fc1.bias"],
        f"{base}.patch_mixer.mlp.fc2.weight": sd["patch_mixer.mlp.fc2.weight"],
        f"{base}.patch_mixer.mlp.fc2.bias": sd["patch_mixer.mlp.fc2.bias"],
        # feature_mixer keys
        f"{base}.feature_mixer.norm.norm.weight": sd["feature_mixer.norm.norm.weight"],
        f"{base}.feature_mixer.norm.norm.bias": sd["feature_mixer.norm.norm.bias"],
        f"{base}.feature_mixer.mlp.fc1.weight": sd["feature_mixer.mlp.fc1.weight"],
        f"{base}.feature_mixer.mlp.fc1.bias": sd["feature_mixer.mlp.fc1.bias"],
        f"{base}.feature_mixer.mlp.fc2.weight": sd["feature_mixer.mlp.fc2.weight"],
        f"{base}.feature_mixer.mlp.fc2.bias": sd["feature_mixer.mlp.fc2.bias"],
    }

    if mode == "mix_channel":
        state_dict.update(
            {
                f"{base}.channel_mixer.norm.norm.weight": sd["channel_mixer.norm.norm.weight"],
                f"{base}.channel_mixer.norm.norm.bias": sd["channel_mixer.norm.norm.bias"],
                f"{base}.channel_mixer.mlp.fc1.weight": sd["channel_mixer.mlp.fc1.weight"],
                f"{base}.channel_mixer.mlp.fc1.bias": sd["channel_mixer.mlp.fc1.bias"],
                f"{base}.channel_mixer.mlp.fc2.weight": sd["channel_mixer.mlp.fc2.weight"],
                f"{base}.channel_mixer.mlp.fc2.bias": sd["channel_mixer.mlp.fc2.bias"],
            }
        )

        if use_gated_attn:
            state_dict[f"{base}.channel_mixer.gate.attn_layer.weight"] = sd["channel_mixer.gate.attn_layer.weight"]
            state_dict[f"{base}.channel_mixer.gate.attn_layer.bias"] = sd["channel_mixer.gate.attn_layer.bias"]

    if use_gated_attn:
        # patch_mixer.gate
        state_dict[f"{base}.patch_mixer.gate.attn_layer.weight"] = sd["patch_mixer.gate.attn_layer.weight"]
        state_dict[f"{base}.patch_mixer.gate.attn_layer.bias"] = sd["patch_mixer.gate.attn_layer.bias"]
        # feature_mixer.gate
        state_dict[f"{base}.feature_mixer.gate.attn_layer.weight"] = sd["feature_mixer.gate.attn_layer.weight"]
        state_dict[f"{base}.feature_mixer.gate.attn_layer.bias"] = sd["feature_mixer.gate.attn_layer.bias"]

    # ---- Preprocess to TTNN parameters dict ----
    parameters = {}

    # helper for a mixer (patch_mixer or feature_mixer)
    def load_mixer_params(mixer_path: str):
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

    load_mixer_params(f"{base}.patch_mixer")
    load_mixer_params(f"{base}.feature_mixer")
    if mode == "mix_channel":
        load_mixer_params(f"{base}.channel_mixer")

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

    sd = torch_block.state_dict()

    # ---- Fake nested torch-style state_dict ----
    state_dict = {}
    for i in range(num_layers):
        # Torch ModuleList key names: layers.{i}.<submodule>...
        prefix = f"{base}.layers.{i}"

        # patch_mixer
        state_dict[f"{prefix}.patch_mixer.norm.norm.weight"] = sd[f"layers.{i}.patch_mixer.norm.norm.weight"]
        state_dict[f"{prefix}.patch_mixer.norm.norm.bias"] = sd[f"layers.{i}.patch_mixer.norm.norm.bias"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc1.weight"] = sd[f"layers.{i}.patch_mixer.mlp.fc1.weight"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc1.bias"] = sd[f"layers.{i}.patch_mixer.mlp.fc1.bias"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc2.weight"] = sd[f"layers.{i}.patch_mixer.mlp.fc2.weight"]
        state_dict[f"{prefix}.patch_mixer.mlp.fc2.bias"] = sd[f"layers.{i}.patch_mixer.mlp.fc2.bias"]

        # feature_mixer
        state_dict[f"{prefix}.feature_mixer.norm.norm.weight"] = sd[f"layers.{i}.feature_mixer.norm.norm.weight"]
        state_dict[f"{prefix}.feature_mixer.norm.norm.bias"] = sd[f"layers.{i}.feature_mixer.norm.norm.bias"]
        state_dict[f"{prefix}.feature_mixer.mlp.fc1.weight"] = sd[f"layers.{i}.feature_mixer.mlp.fc1.weight"]
        state_dict[f"{prefix}.feature_mixer.mlp.fc1.bias"] = sd[f"layers.{i}.feature_mixer.mlp.fc1.bias"]
        state_dict[f"{prefix}.feature_mixer.mlp.fc2.weight"] = sd[f"layers.{i}.feature_mixer.mlp.fc2.weight"]
        state_dict[f"{prefix}.feature_mixer.mlp.fc2.bias"] = sd[f"layers.{i}.feature_mixer.mlp.fc2.bias"]

        if use_gated_attn:
            # patch_mixer gate
            state_dict[f"{prefix}.patch_mixer.gate.attn_layer.weight"] = sd[
                f"layers.{i}.patch_mixer.gate.attn_layer.weight"
            ]
            state_dict[f"{prefix}.patch_mixer.gate.attn_layer.bias"] = sd[
                f"layers.{i}.patch_mixer.gate.attn_layer.bias"
            ]
            # feature_mixer gate
            state_dict[f"{prefix}.feature_mixer.gate.attn_layer.weight"] = sd[
                f"layers.{i}.feature_mixer.gate.attn_layer.weight"
            ]
            state_dict[f"{prefix}.feature_mixer.gate.attn_layer.bias"] = sd[
                f"layers.{i}.feature_mixer.gate.attn_layer.bias"
            ]

    # ---- Build TTNN parameters dict (TT naming!) ----
    parameters = {}
    for i in range(num_layers):
        prefix = f"{base}.layers.{i}"

        def load_mixer(mixer_name: str):
            mixer_path = f"{prefix}.{mixer_name}"

            # LN reads torch keys at f"{mixer_path}.norm.norm.*"
            gamma, beta = preprocess_layernorm(state_dict, f"{mixer_path}.norm", device=device)

            # TT expects: f"{mixer_path}.norm.norm.weight/bias" (matching PyTorch structure)
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

    sd = torch_head.state_dict()

    # fake full-model-like torch state_dict
    state_dict = {
        f"{base}.proj.weight": sd["proj.weight"],
        f"{base}.proj.bias": sd["proj.bias"],
    }

    # preprocess
    w_tt, b_tt = preprocess_forecast_head(state_dict, base, device=device)
    parameters = {
        f"{base}.proj.weight": w_tt,
        f"{base}.proj.bias": b_tt,
    }

    tt_head = TtPatchTSMixerForecastHead(device=device, base_address=base, parameters=parameters, prediction_length=H)

    x = torch.randn(B, C, N_p, D)
    torch_out = torch_head(x)  # (B, H, C)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    tt_y = tt_head(tt_x, B=B, C=C, Np=N_p, D=D)  # (B, H, 1, C)
    tt_out = ttnn.to_torch(tt_y).squeeze(2)  # (B, H, C)

    assert_with_pcc(torch_out, tt_out, 0.99)
