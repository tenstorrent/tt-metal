# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# PyTorch reference for the HunyuanImage-3.0 TimestepEmbedder.
# Extracted verbatim from:
#   HunyuanImage-3.0/hunyuan_image_3/modeling_hunyuan_image_3.py
#     timestep_embedding()  (lines 151-178)
#     TimestepEmbedder      (lines 562-592)
#
# Used as the golden reference for TT-Metal numeric validation.
#
# The model instantiates this embedder several times with identical structure:
#   timestep_emb, time_embed, time_embed_2  (all hidden_size=4096), plus the
#   optional guidance_emb / timestep_r_emb. Each maps a 1-D tensor of scalar
#   timesteps -> [N, hidden_size] embeddings via:
#     t_freq = timestep_embedding(t, frequency_embedding_size)   # sinusoidal
#     t_emb  = Linear(freq, hidden) -> GELU -> Linear(hidden, out)

import math

import torch
import torch.nn as nn


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        device=t.device
    )
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    Forward signature:
        t: [N]  1-D tensor of (possibly fractional) timesteps
        returns [N, out_size]  (out_size defaults to hidden_size)

    Notes for TT-Metal port:
    - `timestep_embedding` is a data-independent sinusoidal featurization of the
      scalar timesteps. It is cheap and exact in fp32, so the TT port computes it
      on host and only runs the two-layer MLP on device.
    - act_layer defaults to nn.GELU (exact, erf-based) -> ttnn.gelu with
      fast_and_approximate_mode=False.
    """

    def __init__(
        self,
        hidden_size,
        act_layer=nn.GELU,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True, **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size, self.max_period).type(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


# ---------------------------------------------------------------------------
# Quick numeric smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from models.experimental.hunyuan_image_3_0.ref.model_config import HIDDEN_SIZE

    torch.manual_seed(42)
    H = HIDDEN_SIZE
    emb = TimestepEmbedder(H).eval()
    t = torch.rand(4)  # 4 fractional timesteps in [0, 1)
    with torch.no_grad():
        out = emb(t)
    print(f"input  shape: {t.shape}  dtype: {t.dtype}")
    print(f"output shape: {out.shape}  dtype: {out.dtype}")
    print(f"output mean={out.float().mean():.6f}  std={out.float().std():.6f}")
