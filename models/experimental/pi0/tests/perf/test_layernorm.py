"""SigLIP LayerNorm — device kernel parity vs torch fp32.

LN1 of layer 0: real π0.5 weights (D=1152, bf16 gamma + beta).
Activation: same patch_embed + pos_embed real distribution as QKV golden.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden_fc1 import make_real_activation, pcc  # noqa: E402

PI05_WEIGHTS = "/storage/sdawle/pi05_weights/pi05_base/model.safetensors"
VP = "paligemma_with_expert.paligemma.model.vision_tower."
M, D = 256, 1152
EPS = 1e-6


def load_layer0_ln1() -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (gamma [D], beta [D]) for layer-0 layer_norm1, in bf16."""
    sd = load_file(PI05_WEIGHTS)
    gamma = sd[f"{VP}vision_model.encoder.layers.0.layer_norm1.weight"].to(torch.bfloat16)
    beta = sd[f"{VP}vision_model.encoder.layers.0.layer_norm1.bias"].to(torch.bfloat16)
    return gamma, beta


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_device_kernel_pcc(device):
    from layernorm_op import SigLIPLayerNormOp, build_tensors_for_ln_test

    gamma_torch, beta_torch = load_layer0_ln1()
    x_torch = make_real_activation(seed=42)  # (256, 1152) bf16

    # Torch golden: F.layer_norm in fp32, then back to bf16 for comparison.
    y_golden = F.layer_norm(
        x_torch.float(),
        normalized_shape=(D,),
        weight=gamma_torch.float(),
        bias=beta_torch.float(),
        eps=EPS,
    ).to(torch.bfloat16)

    (
        activation_tt,
        gamma_tt,
        beta_tt,
        scaler_tt,
        ones_tt,
        accum_tt,
        xmm_tt,
        xmm2_tt,
        mean_tt,
        var_tt,
        ivar_tt,
        output_tt,
    ) = build_tensors_for_ln_test(device, gamma_torch, beta_torch, x_torch, num_cores=8)

    SigLIPLayerNormOp.op(
        activation_tt,
        gamma_tt,
        beta_tt,
        scaler_tt,
        ones_tt,
        accum_tt,
        xmm_tt,
        xmm2_tt,
        mean_tt,
        var_tt,
        ivar_tt,
        output_tt,
        num_cores=8,
        eps=EPS,
    )

    import ttnn as _ttnn

    y_device = _ttnn.to_torch(output_tt)
    p = pcc(y_golden, y_device)
    print(f"\nPCC (LN1, kernel vs torch) = {p:.6f}")
    assert p >= 0.999, f"PCC {p} below 0.999 gate"
