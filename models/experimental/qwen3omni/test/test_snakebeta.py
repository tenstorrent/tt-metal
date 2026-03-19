import torch
import pytest
import ttnn
from safetensors.torch import safe_open

from models.common.utility_functions import comp_pcc
from models.experimental.tt_symbiote.core.tensor import TorchTTNNTensor
from models.experimental.qwen3omni.tt.snakebeta import TTNNSnakeBeta
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import SnakeBeta


def load_weights():
    weights = {}
    path = "models/experimental/qwen3omni/checkpoints/model-00015-of-00015.safetensors"

    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)

    return weights


@pytest.mark.parametrize(
    "prefix",
    [
        "code2wav.decoder.1.block.0",
        "code2wav.decoder.1.block.2.act1",
        "code2wav.decoder.1.block.2.act2",
    ],
)
@pytest.mark.parametrize("B, T", [(2, 64)])
def test_snakebeta_with_safetensors(prefix, B, T, device):
    weights = load_weights()

    # ----------------------
    # Extract alpha, beta
    # ----------------------
    alpha = weights[f"{prefix}.alpha"].float()
    beta = weights[f"{prefix}.beta"].float()

    C = alpha.shape[0]

    print(f"\nTesting: {prefix}")
    print("Alpha shape:", alpha.shape)

    # ----------------------
    # PyTorch model
    # ----------------------
    pt_model = SnakeBeta(in_features=C)
    pt_model.eval()

    # Load pretrained weights
    pt_model.alpha.data = alpha.clone()
    pt_model.beta.data = beta.clone()

    # Input
    x = torch.randn(B, C, T)

    pt_out = pt_model(x)

    # ----------------------
    # TTNN model
    # ----------------------
    tt_model = TTNNSnakeBeta(
        device=device,
        alpha=alpha,
        beta=beta,
    )

    tt_input = TorchTTNNTensor(x)

    tt_out = tt_model(tt_input)

    tt_out_torch = tt_out.to_torch if isinstance(tt_out, TorchTTNNTensor) else ttnn.to_torch(tt_out)

    # ----------------------
    # Compare
    # ----------------------
    passed, pcc = comp_pcc(pt_out, tt_out_torch)

    tt_torch = tt_out_torch.float() if tt_out_torch.dtype != pt_out.dtype else tt_out_torch
    max_abs = (pt_out.float() - tt_torch).abs().max().item()

    print("PCC:", pcc, "passed:", passed)
    print("shapes pt", tuple(pt_out.shape), "tt", tuple(tt_out_torch.shape), "max_abs", max_abs)

    assert passed, f"PCC={pcc} max_abs={max_abs} pt_shape={tuple(pt_out.shape)} tt_shape={tuple(tt_out_torch.shape)}"
