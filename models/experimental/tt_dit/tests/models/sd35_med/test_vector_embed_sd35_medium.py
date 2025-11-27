import torch
import torch.nn as nn
import pytest
import ttnn
from loguru import logger

from models.experimental.tt_dit.models.transformers.vector_embed_sd35_medium import VectorEmbedder


# PyTorch Reference Implementation
class VectorEmbedderRef(nn.Module):
    def __init__(self, input_dim, hidden_size, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, dtype=dtype),
        )

    def forward(self, x):
        return self.mlp(x)


# TTNN Test
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "input_dim, hidden_size, batch_size",
    [
        (128, 512, 2),
        (256, 512, 4),
        (128, 1024, 8),
        (768, 512, 1),
    ],
    ids=["v128h512b2", "v256h512b4", "v128h1024b8", "v768h512b1"],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_vector_embedder(device, dtype, input_dim, hidden_size, batch_size, reset_seeds):
    torch.manual_seed(42)

    ref = VectorEmbedderRef(input_dim, hidden_size)
    ref.eval()

    tt_model = VectorEmbedder(input_dim, hidden_size, mesh_device=device)

    # Copy weights
    with torch.no_grad():
        tt_model.linear1.load_torch_state_dict(ref.mlp[0].state_dict())
        tt_model.linear2.load_torch_state_dict(ref.mlp[2].state_dict())

    # Input (random vector)
    x = torch.randn(batch_size, input_dim, dtype=torch.bfloat16)

    ref_out = ref(x)

    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    tt_out = tt_model(tt_input)
    tt_out_torch = ttnn.to_torch(tt_out)

    # Compute PCC
    numerator = ((ref_out - ref_out.mean()) * (tt_out_torch - tt_out_torch.mean())).sum()
    denominator = torch.sqrt(((ref_out - ref_out.mean()) ** 2).sum()) * torch.sqrt(
        ((tt_out_torch - tt_out_torch.mean()) ** 2).sum()
    )
    pcc = (numerator / denominator).item()

    logger.info(f"VectorEmbed PCC: {pcc}")
    assert pcc > 0.99, f"FAILED PCC={pcc}"

    logger.info("VectorEmbedder test PASSED")
