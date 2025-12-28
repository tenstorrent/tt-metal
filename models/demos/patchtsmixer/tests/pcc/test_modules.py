import pytest
import torch

import ttnn
from models.demos.patchtsmixer.reference.pytorch_patchtsmixer import (
    PatchTSMixerGatedAttention,
    PatchTSMixerMLP,
    PatchTSMixerNormLayer,
    PatchTSMixerPositionalEncoding,
)
from models.demos.patchtsmixer.tt.model_processing import (
    preprocess_gated_attention,
    preprocess_layernorm,
    preprocess_linear,
    preprocess_norm_layer_batchnorm,
    preprocess_positional_encoding,
)
from models.demos.patchtsmixer.tt.patchtsmixer import (
    TtPatchTSMixerBatchNorm,
    TtPatchTSMixerGatedAttention,
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
