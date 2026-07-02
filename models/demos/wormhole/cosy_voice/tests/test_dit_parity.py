# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.wormhole.cosy_voice.tt.flow.dit import TtDiTBlock

# --- Reference Implementation (Torch) ---


class TorchAdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        # Broadcast over seq dimension
        norm_x = self.norm(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        return norm_x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class TorchFeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(approximate="tanh"), nn.Linear(inner_dim, dim))

    def forward(self, x):
        return self.ff(x)


class TorchDiTAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim)])

    def forward(self, x, mask=None, rope=None):
        b, s, d = x.shape
        q = self.to_q(x).view(b, s, self.heads, self.dim_head).transpose(1, 2)
        k = self.to_k(x).view(b, s, self.heads, self.dim_head).transpose(1, 2)
        v = self.to_v(x).view(b, s, self.heads, self.dim_head).transpose(1, 2)

        # Skip RoPE for basic parity test (can add later if needed)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn_out = attn_out.transpose(1, 2).reshape(b, s, self.inner_dim)
        return self.to_out[0](attn_out)


class TorchDiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4):
        super().__init__()
        self.attn_norm = TorchAdaLayerNormZero(dim)
        self.attn = TorchDiTAttention(dim, heads, dim_head)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = TorchFeedForward(dim, ff_mult)

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output

        ff_norm = self.ff_norm(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x


# --- Test Execution ---


def test_dit_block_parity():
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))

    dim = 1024
    heads = 16
    dim_head = 64
    ff_mult = 2
    batch = 1
    seq = 128

    # Load sample state dict
    sd_path = "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B/flow.pt"
    logger.info(f"Loading weights from {sd_path}")
    full_sd = torch.load(sd_path, map_location="cpu")

    # Extract weights for block 0
    prefix = "decoder.estimator.transformer_blocks.0"
    block_sd = {k[len(prefix) + 1 :]: v for k, v in full_sd.items() if k.startswith(prefix)}
    logger.info(f"Block keys: {list(block_sd.keys())}")

    # Initialize blocks
    torch_block = TorchDiTBlock(dim, heads, dim_head, ff_mult)
    # Map Tt names to Torch names for loading
    remap = {
        "attn_norm.linear.weight": "attn_norm.linear.weight",
        "attn_norm.linear.bias": "attn_norm.linear.bias",
        "attn.to_q.weight": "attn.to_q.weight",
        "attn.to_q.bias": "attn.to_q.bias",
        "attn.to_k.weight": "attn.to_k.weight",
        "attn.to_k.bias": "attn.to_k.bias",
        "attn.to_v.weight": "attn.to_v.weight",
        "attn.to_v.bias": "attn.to_v.bias",
        "attn.to_out.0.weight": "attn.to_out.0.weight",
        "attn.to_out.0.bias": "attn.to_out.0.bias",
        "ff.ff.0.0.weight": "ff.ff.0.weight",
        "ff.ff.0.0.bias": "ff.ff.0.bias",
        "ff.ff.2.weight": "ff.ff.2.weight",
        "ff.ff.2.bias": "ff.ff.2.bias",
    }
    torch_sd = {}
    for tt_k, torch_k in remap.items():
        torch_sd[torch_k] = block_sd[tt_k]
    torch_block.load_state_dict(torch_sd, strict=False)
    torch_block.eval()

    tt_block = TtDiTBlock(dim, heads, dim_head, ff_mult, device, full_sd, prefix)

    # Inputs
    x = torch.randn(batch, seq, dim)
    t = torch.randn(batch, dim)

    # Reference
    with torch.no_grad():
        ref_out = torch_block(x, t)

    # TTNN
    x_tt = ttnn.from_torch(x.unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    t_tt = ttnn.from_torch(t.unsqueeze(0).unsqueeze(0), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)

    tt_out = tt_block(x_tt, t_tt)
    tt_out = ttnn.to_torch(tt_out).squeeze(0).squeeze(0)  # (batch, seq, dim)

    # Comparison
    pcc_res = comp_pcc(ref_out, tt_out)
    logger.info(f"DiT Block Parity: {pcc_res}")

    ttnn.close_mesh_device(device)


if __name__ == "__main__":
    test_dit_block_parity()
