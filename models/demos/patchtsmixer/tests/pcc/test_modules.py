import pytest
import torch

import ttnn
from models.demos.patchtsmixer.reference.pytorch_patchtsmixer import PatchTSMixerGatedAttention
from models.demos.patchtsmixer.tt.model_processing import preprocess_gated_attention
from models.demos.patchtsmixer.tt.patchtsmixer import TtPatchTSMixerGatedAttention
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
    x = torch.randn(2, 5, d_model)  # last dim = d_model
    torch_out = torch_gate(x)

    tt_x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_out = tt_gate(tt_x)
    tt_out_torch = ttnn.to_torch(tt_out)

    assert_with_pcc(torch_out, tt_out_torch, 0.99)
