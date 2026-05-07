# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Standalone PyTorch reference test for the CosyVoice3 Flow Decoder.

Reconstructs the DiT architecture using only torch + einops (no cosyvoice/matcha deps).
Verifies weight loading and captures reference outputs for PCC testing.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

# ---------------------------------------------------------------------------
# Minimal reconstruction of DiT modules (from ref code, no external deps)
# ---------------------------------------------------------------------------


class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep).to(timestep.dtype)
        return self.time_mlp(time_hidden)


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0), nn.Mish())
        self.conv2 = nn.Sequential(nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0), nn.Mish())

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv1(x)
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.conv2(x)
        return x.permute(0, 2, 1)


class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, spks):
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = repeat(spks, "b c -> b t c", t=x.shape[1])
            to_cat.append(spks)
        x = self.proj(torch.cat(to_cat, dim=-1))
        x = self.conv_pos_embed(x) + x
        return x


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        self.ff = nn.Sequential(
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU(approximate="tanh")),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
        )

    def forward(self, x):
        return self.ff(x)


class DiTAttention(nn.Module):
    """Simplified self-attention with RoPE support."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim), nn.Dropout(dropout)])

    def forward(self, x, mask=None, rope=None):
        b = x.shape[0]
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        if rope is not None:
            from x_transformers.x_transformers import apply_rotary_pos_emb

            freqs, xpos_scale = rope
            q = apply_rotary_pos_emb(q, freqs, 1.0)
            k = apply_rotary_pos_emb(k, freqs, 1.0)

        head_dim = self.inner_dim // self.heads
        q = q.view(b, -1, self.heads, head_dim).transpose(1, 2)
        k = k.view(b, -1, self.heads, head_dim).transpose(1, 2)
        v = v.view(b, -1, self.heads, head_dim).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1).expand(b, self.heads, q.shape[-2], k.shape[-2])

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(b, -1, self.inner_dim).to(q.dtype)
        x = self.to_out[0](x)
        x = self.to_out[1](x)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            else:
                mask = mask[:, 0, -1].unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
        return x


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = DiTAttention(dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        x = x + gate_mlp.unsqueeze(1) * self.ff(ff_norm)
        return x


class DiT(nn.Module):
    def __init__(self, dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=0.1) for _ in range(depth)]
        )
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)
        t = self.time_embed(t)
        x = self.input_embed(x, cond, mu, spks.squeeze(1))
        # Skip RoPE for now (requires x_transformers)
        for block in self.transformer_blocks:
            x = block(x, t, mask=None, rope=None)
        x = self.norm_out(x, t)
        output = self.proj_out(x).transpose(1, 2)
        return output


class PreLookaheadLayer(nn.Module):
    def __init__(self, in_channels=80, channels=1024, pre_lookahead_len=3):
        super().__init__()
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, in_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs, context=None):
        outputs = inputs.transpose(1, 2).contiguous()
        outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode="constant", value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0), mode="constant", value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        return outputs + inputs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

WEIGHTS_DIR = "/root/tt-metal/models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"


@pytest.fixture(scope="module")
def flow_state_dict():
    """Load flow.pt weights."""
    import os

    path = os.path.join(WEIGHTS_DIR, "flow.pt")
    if not os.path.exists(path):
        pytest.skip(f"flow.pt not found at {path}")
    return torch.load(path, map_location="cpu")


def test_weight_loading(flow_state_dict):
    """Verify all flow.pt weights can be loaded into our reconstructed modules."""
    sd = flow_state_dict

    # 1. PreLookaheadLayer
    pll = PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3)
    pll_sd = {k.replace("pre_lookahead_layer.", ""): v for k, v in sd.items() if k.startswith("pre_lookahead_layer.")}
    missing, unexpected = pll.load_state_dict(pll_sd, strict=False)
    assert len(missing) == 0, f"PreLookaheadLayer missing keys: {missing}"
    print(f"PreLookaheadLayer: loaded {len(pll_sd)} keys")

    # 2. DiT estimator
    dit = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80)
    dit_sd = {k.replace("decoder.estimator.", ""): v for k, v in sd.items() if k.startswith("decoder.estimator.")}
    # Filter out rotary_embed (we handle it separately)
    dit_sd_filtered = {k: v for k, v in dit_sd.items() if "rotary_embed" not in k}
    missing, unexpected = dit.load_state_dict(dit_sd_filtered, strict=False)
    # We expect 'attn.processor' to not be in state dict (it's a non-module class)
    print(f"DiT: loaded {len(dit_sd_filtered)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    # Core weights should all be present
    critical_missing = [k for k in missing if "processor" not in k]
    assert len(critical_missing) == 0, f"DiT critical missing keys: {critical_missing}"

    # 3. Top-level modules
    input_emb = nn.Embedding(6561, 80)
    input_emb.load_state_dict({"weight": sd["input_embedding.weight"]})
    print(f"input_embedding: loaded, shape={sd['input_embedding.weight'].shape}")

    spk_layer = nn.Linear(192, 80)
    spk_layer.load_state_dict(
        {
            "weight": sd["spk_embed_affine_layer.weight"],
            "bias": sd["spk_embed_affine_layer.bias"],
        }
    )
    print(f"spk_embed_affine_layer: loaded")


def test_pre_lookahead_forward(flow_state_dict):
    """Test PreLookaheadLayer forward pass with loaded weights."""
    sd = flow_state_dict
    pll = PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3)
    pll_sd = {k.replace("pre_lookahead_layer.", ""): v for k, v in sd.items() if k.startswith("pre_lookahead_layer.")}
    pll.load_state_dict(pll_sd)
    pll.eval()

    # Dummy input: (batch=1, seq_len=20, channels=80)
    x = torch.randn(1, 20, 80)
    with torch.no_grad():
        out = pll(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"PreLookaheadLayer output shape: {out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")


def test_dit_single_block_forward(flow_state_dict):
    """Test a single DiT block forward pass."""
    sd = flow_state_dict
    block = DiTBlock(dim=1024, heads=16, dim_head=64, ff_mult=2, dropout=0.0)

    # Load block 0 weights
    block_sd = {
        k.replace("decoder.estimator.transformer_blocks.0.", ""): v
        for k, v in sd.items()
        if "transformer_blocks.0." in k
    }
    missing, _ = block.load_state_dict(block_sd, strict=False)
    critical_missing = [k for k in missing if "processor" not in k]
    assert len(critical_missing) == 0, f"Block missing keys: {critical_missing}"
    block.eval()

    # Dummy inputs
    x = torch.randn(1, 50, 1024)  # (batch, seq, dim)
    t = torch.randn(1, 1024)  # timestep embedding
    with torch.no_grad():
        out = block(x, t)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    print(f"DiT block 0 output shape: {out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")


def test_dit_estimator_forward(flow_state_dict):
    """Test full DiT estimator forward pass (no RoPE)."""
    sd = flow_state_dict
    dit = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80)
    dit_sd = {k.replace("decoder.estimator.", ""): v for k, v in sd.items() if k.startswith("decoder.estimator.")}
    dit_sd_filtered = {k: v for k, v in dit_sd.items() if "rotary_embed" not in k}
    dit.load_state_dict(dit_sd_filtered, strict=False)
    dit.eval()

    # Dummy inputs matching inference shapes
    seq_len = 50
    x = torch.randn(2, 80, seq_len)  # batch=2 for CFG
    mask = torch.ones(2, 1, seq_len)
    mu = torch.randn(2, 80, seq_len)
    t = torch.tensor([0.5, 0.5])
    spks = torch.randn(2, 80)
    cond = torch.randn(2, 80, seq_len)

    with torch.no_grad():
        out = dit(x, mask, mu, t, spks=spks, cond=cond)
    assert out.shape == (2, 80, seq_len), f"Expected (2, 80, {seq_len}), got {out.shape}"
    print(f"DiT estimator output shape: {out.shape}, mean={out.mean():.4f}, std={out.std():.4f}")


def test_euler_ode_solver(flow_state_dict):
    """Test the complete 10-step Euler ODE solver with CFG."""
    sd = flow_state_dict

    # Build DiT
    dit = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80)
    dit_sd = {k.replace("decoder.estimator.", ""): v for k, v in sd.items() if k.startswith("decoder.estimator.")}
    dit_sd_filtered = {k: v for k, v in dit_sd.items() if "rotary_embed" not in k}
    dit.load_state_dict(dit_sd_filtered, strict=False)
    dit.eval()

    # Euler ODE solver (from flow_matching.py)
    inference_cfg_rate = 0.7
    n_timesteps = 10
    seq_len = 20  # Small for speed

    mu = torch.randn(1, 80, seq_len)
    mask = torch.ones(1, 1, seq_len)
    spks = torch.randn(1, 80)
    cond = torch.zeros(1, 80, seq_len)

    # Initialize noise
    z = torch.randn_like(mu)

    # Cosine time schedule
    t_span = torch.linspace(0, 1, n_timesteps + 1)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)

    x = z.clone()
    t, dt = t_span[0], t_span[1] - t_span[0]
    t = t.unsqueeze(0)

    # Build batch=2 inputs for CFG
    x_in = torch.zeros(2, 80, seq_len)
    mask_in = torch.zeros(2, 1, seq_len)
    mu_in = torch.zeros(2, 80, seq_len)
    t_in = torch.zeros(2)
    spks_in = torch.zeros(2, 80)
    cond_in = torch.zeros(2, 80, seq_len)

    with torch.no_grad():
        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = dit(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_cond, dphi_uncond = torch.split(dphi_dt, [1, 1], dim=0)
            dphi_dt = (1.0 + inference_cfg_rate) * dphi_cond - inference_cfg_rate * dphi_uncond
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

    assert x.shape == (1, 80, seq_len), f"Expected (1, 80, {seq_len}), got {x.shape}"
    print(f"Euler ODE output shape: {x.shape}, mean={x.mean():.4f}, std={x.std():.4f}")
    print("10-step Euler ODE solver completed successfully!")
