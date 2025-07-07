import pytest, torch

from models.experimental.boltz1.tenstorrent_moritz_SDPA_jun18 import (
    filter_dict,
    DiffusionTransformerModule,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
import time

from models.experimental.boltz1.reference.model.modules.diffusion import (
    DiffusionTransformer as DiffusionTransformerTorch,
)

torch.set_grad_enabled(False)
torch.manual_seed(893)

state_dict = torch.load("models/experimental/boltz1/boltz1_conf_dict.pth")


@pytest.mark.parametrize("n_heads", [16])
@pytest.mark.parametrize("n_layer", [2, 24])
@pytest.mark.parametrize("seq_len", [128, 512, 768, 1024])
def test_diffusion_transformer(device, seq_len, n_layer, n_heads):
    token_transformer = DiffusionTransformerModule(
        device,
        n_layers=n_layer,
        dim=768,
        n_heads=n_heads,
    )
    token_transformer_torch = DiffusionTransformerTorch(
        depth=n_layer, heads=n_heads, dim=768, dim_single_cond=768, dim_pairwise=128
    ).eval()

    token_transformer_state_dict = filter_dict(state_dict, "structure_module.score_model.token_transformer")
    token_transformer.load_state_dict(
        token_transformer_state_dict,
        strict=False,
    )

    token_transformer_torch.load_state_dict(token_transformer_state_dict, strict=False)

    a = 3 + 5 * torch.randn(1, seq_len, 768)
    s = -2 + 42 * torch.randn(1, seq_len, 768)
    z = 10 * torch.randn(1, seq_len, seq_len, 128)
    mask = torch.ones(1, seq_len)

    a_torch = token_transformer_torch(
        a,
        s,
        z,
        mask,
    )

    start = time.time()
    a_tt = token_transformer(
        a,
        s,
        z,
        mask,
    )
    end = time.time()

    print(f"$$$YF: DiffusionTransformer time: {end - start:.4f} seconds")

    assert_with_pcc(a_tt, a_torch, 0.99)
