import torch
import pytest
import numpy as np
import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.qwen3omni.tt.pos_embed import TTNNSinusoidalPositionEmbedding


# -----------------------------
# PyTorch Reference Implementation
# -----------------------------
class SinusoidsPositionEmbedding(torch.nn.Module):
    def __init__(self, length, channels, max_timescale=10000):
        super().__init__()
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale

        if channels % 2 != 0:
            raise ValueError("Channels must be even")

        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)

        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(channels // 2).float())

        scaled_time = torch.arange(length)[:, None] * inv_timescales[None, :]

        self.register_buffer(
            "positional_embedding",
            torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1),
            persistent=False,
        )

    def forward(self, seqlen: int):
        return self.positional_embedding[:seqlen, :]


# -----------------------------
# Test
# -----------------------------
@pytest.mark.parametrize("device_params", [{"l1_small_size": 245760}], indirect=True)
@pytest.mark.parametrize("length, channels", [(32, 64), (128, 128)])
def test_ttnn_sinusoidal_embedding_pcc(device, length, channels):
    # PyTorch reference
    torch_model = SinusoidsPositionEmbedding(length, channels)
    torch_out = torch_model(length)  # (L, D)

    # TTNN input (timesteps)
    timesteps = torch.arange(length, dtype=torch.float32)

    tt_timesteps = ttnn.from_torch(
        timesteps,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # TTNN model
    tt_model = TTNNSinusoidalPositionEmbedding(
        device=device,
        num_channels=channels,
        max_timescale=10000,
    )

    tt_out = tt_model(tt_timesteps)

    # Convert TT → Torch
    tt_out_torch = ttnn.to_torch(tt_out)

    # -----------------------------
    # PCC Comparison (comp_pcc returns passing, pcc_value)
    # -----------------------------
    passing, pcc_value = comp_pcc(torch_out, tt_out_torch, pcc=0.999)

    print(f"PCC: {pcc_value}")

    assert passing, f"PCC too low: {pcc_value} (required >= 0.999)"
