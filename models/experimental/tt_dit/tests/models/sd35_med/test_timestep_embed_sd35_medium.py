# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import ttnn
import pytest
from loguru import logger
import torch.nn as nn

from models.experimental.tt_dit.models.transformers.sd35_med.timestep_embed_sd35_medium import TimestepEmbedder


# Reference PyTorch implementation
class TimestepEmbedderRef(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256, dtype=torch.bfloat16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, dtype=dtype),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, dtype=dtype),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.dtype = dtype

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=None):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        # Convert to specified dtype if provided, otherwise use input dtype
        if dtype is not None:
            return embedding.to(dtype)
        return embedding.to(t.dtype)

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, dtype=self.dtype)
        return self.mlp(t_freq)


# Tests
@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize(
    "hidden_size, freq_size, batch_size",
    [
        (512, 256, 1),  # smallest batch
        (512, 256, 4),  # larger batch
        (1024, 256, 2),  # larger hidden dimension
        (512, 128, 2),  # smaller frequency embedding
    ],
    ids=[
        "HS512_FS256_B1",
        "HS512_FS256_B4",
        "HS1024_FS256_B2",
        "HS512_FS128_B2",
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 79104}], indirect=True)
def test_timestep_embedder(device, dtype, hidden_size, freq_size, batch_size, reset_seeds):
    torch.manual_seed(42)

    ref = TimestepEmbedderRef(hidden_size, freq_size)
    ref.eval()

    # Create TTNN model
    tt_model = TimestepEmbedder(hidden_size, freq_size, mesh_device=device)

    # Copy ref weights into TTNN model
    with torch.no_grad():
        tt_model.linear1.load_torch_state_dict(ref.mlp[0].state_dict())
        tt_model.linear2.load_torch_state_dict(ref.mlp[2].state_dict())

    # Input timesteps
    t = torch.randint(0, 1000, (batch_size,), dtype=torch.bfloat16)

    with torch.no_grad():
        ref_out = ref(t)

    tt_input = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_out = tt_model(tt_input)
    tt_out_torch = ttnn.to_torch(tt_out)

    # Compare outputs using PCC
    numerator = ((ref_out - ref_out.mean()) * (tt_out_torch - tt_out_torch.mean())).sum()
    denominator = torch.sqrt(((ref_out - ref_out.mean()) ** 2).sum()) * torch.sqrt(
        ((tt_out_torch - tt_out_torch.mean()) ** 2).sum()
    )
    pcc = (numerator / denominator).item()

    logger.info(f"PCC: {pcc}")
    assert pcc > 0.99, f"FAILED PCC={pcc}"

    logger.info("TimestepEmbedder test PASSED")
