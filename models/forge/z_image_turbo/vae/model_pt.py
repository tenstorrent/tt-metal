"""
PyTorch CPU golden reference for the VAE decoder from Tongyi-MAI/Z-Image-Turbo.

Loads the AutoencoderKL decoder and runs inference on the CPU for PCC comparison.
"""

import torch
from diffusers import AutoencoderKL

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
SCALING_FACTOR = 0.3611
SHIFT_FACTOR = 0.1159
LATENT_CHANNELS = 16
LATENT_H = 64
LATENT_W = 64


class VaeDecoderPT:
    """PyTorch CPU VAE decoder for golden comparison."""

    def __init__(self, dtype=torch.float32):
        vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype)
        vae.config.force_upcast = False
        vae.eval()
        self.decoder = vae.decoder
        self.state_dict = vae.decoder.state_dict()

    @torch.no_grad()
    def forward(self, raw_latent):
        """Run VAE decoder on a raw latent tensor.

        Args:
            raw_latent: [1, 16, 64, 64] float32 tensor (before denormalization)

        Returns:
            [1, 3, 512, 512] float32 tensor
        """
        z = (raw_latent.float() / SCALING_FACTOR) + SHIFT_FACTOR
        return self.decoder(z).float()


def get_input():
    """Return a deterministic random latent tensor for testing."""
    torch.manual_seed(42)
    return torch.randn(1, LATENT_CHANNELS, LATENT_H, LATENT_W, dtype=torch.float32)
