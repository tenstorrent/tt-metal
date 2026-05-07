# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for TTNN DiT flow decoder components.

Tests individual DiT blocks and the full estimator against PyTorch reference
outputs, measuring PCC (Pearson Correlation Coefficient) for numerical parity.
"""

import os

import pytest
import torch

import ttnn
from models.demos.wormhole.cosy_voice.tt.flow.dit import TtDiT
from models.demos.wormhole.cosy_voice.tt.flow.dit_modules import TtDiTBlock

WEIGHTS_DIR = "/root/tt-metal/models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"


@pytest.fixture(scope="module")
def flow_state_dict():
    path = os.path.join(WEIGHTS_DIR, "flow.pt")
    if not os.path.exists(path):
        pytest.skip(f"flow.pt not found at {path}")
    return torch.load(path, map_location="cpu")


@pytest.fixture(scope="module")
def device():
    """Single device for flow decoder tests (no multi-device needed)."""
    dev = ttnn.open_device(device_id=0)
    yield dev
    ttnn.close_device(dev)


def compute_pcc(ref, test):
    """Compute Pearson Correlation Coefficient between two tensors."""
    ref_flat = ref.flatten().float()
    test_flat = test.flatten().float()
    return torch.corrcoef(torch.stack([ref_flat, test_flat]))[0, 1].item()


def test_dit_block_pcc(flow_state_dict, device):
    """Test a single TtDiTBlock against PyTorch reference (block 0)."""
    sd = flow_state_dict
    prefix = "decoder.estimator.transformer_blocks.0"

    # Build TTNN block
    tt_block = TtDiTBlock(
        dim=1024,
        heads=16,
        dim_head=64,
        ff_mult=2,
        device=device,
        state_dict=sd,
        prefix=prefix,
        dtype=ttnn.bfloat16,
    )

    # Build PyTorch reference block (from test_flow_reference.py)
    from models.demos.wormhole.cosy_voice.tests.test_flow_reference import DiTBlock as RefDiTBlock

    ref_block = RefDiTBlock(dim=1024, heads=16, dim_head=64, ff_mult=2, dropout=0.0)
    ref_sd = {k.replace(f"{prefix}.", ""): v for k, v in sd.items() if k.startswith(f"{prefix}.")}
    ref_block.load_state_dict(ref_sd, strict=False)
    ref_block.eval()

    # Dummy inputs
    torch.manual_seed(42)
    x_torch = torch.randn(1, 32, 1024)  # (batch=1, seq=32, dim=1024)
    t_torch = torch.randn(1, 1024)  # timestep embedding

    # PyTorch reference
    with torch.no_grad():
        ref_out = ref_block(x_torch, t_torch)

    # TTNN
    x_tt = ttnn.from_torch(
        x_torch.unsqueeze(0),  # (1, 1, 32, 1024)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    t_tt = ttnn.from_torch(
        t_torch.unsqueeze(0).unsqueeze(0),  # (1, 1, 1, 1024)
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_out = tt_block(x_tt, t_tt)
    tt_out_torch = ttnn.to_torch(tt_out).squeeze(0).float()

    pcc = compute_pcc(ref_out, tt_out_torch)
    print(f"DiT Block 0 PCC: {pcc:.6f}")
    print(f"  ref  mean={ref_out.mean():.4f}, std={ref_out.std():.4f}")
    print(f"  ttnn mean={tt_out_torch.mean():.4f}, std={tt_out_torch.std():.4f}")
    assert pcc >= 0.95, f"PCC {pcc:.6f} below threshold 0.95"


def test_dit_estimator_pcc(flow_state_dict, device):
    """Test full TtDiT estimator (22 blocks) against PyTorch reference."""
    sd = flow_state_dict

    # Build TTNN estimator
    tt_dit = TtDiT(device, sd, dtype=ttnn.bfloat16)

    # Build PyTorch reference
    from models.demos.wormhole.cosy_voice.tests.test_flow_reference import DiT as RefDiT

    ref_dit = RefDiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80)
    dit_sd = {k.replace("decoder.estimator.", ""): v for k, v in sd.items() if k.startswith("decoder.estimator.")}
    dit_sd_filtered = {k: v for k, v in dit_sd.items() if "rotary_embed" not in k}
    ref_dit.load_state_dict(dit_sd_filtered, strict=False)
    ref_dit.eval()

    # Dummy inputs: batch=2 (for CFG), seq=32, mel_dim=80
    torch.manual_seed(42)
    seq_len = 32
    x = torch.randn(2, 80, seq_len)
    mask = torch.ones(2, 1, seq_len)
    mu = torch.randn(2, 80, seq_len)
    t = torch.tensor([0.5, 0.5])
    spks = torch.randn(2, 80)
    cond = torch.randn(2, 80, seq_len)

    # PyTorch reference
    with torch.no_grad():
        ref_out = ref_dit(x, mask, mu, t, spks=spks, cond=cond)

    # TTNN
    tt_out = tt_dit(x, mask, mu, t, spks=spks, cond=cond)

    pcc = compute_pcc(ref_out, tt_out)
    print(f"DiT Estimator PCC: {pcc:.6f}")
    print(f"  ref  mean={ref_out.mean():.4f}, std={ref_out.std():.4f}")
    print(f"  ttnn mean={tt_out.mean():.4f}, std={tt_out.std():.4f}")
    assert pcc >= 0.90, f"PCC {pcc:.6f} below threshold 0.90"


def test_flow_decoder_e2e(flow_state_dict, device):
    """Test full flow decoder: token embedding + PreLookahead + 10-step ODE + CFG."""
    sd = flow_state_dict

    # Build TTNN flow decoder
    from models.demos.wormhole.cosy_voice.tt.flow.flow import TtCausalMaskedDiffWithDiT

    tt_flow = TtCausalMaskedDiffWithDiT(device, sd, dtype=ttnn.bfloat16)

    # Build PyTorch reference (minimal — just ODE solver with reference DiT)
    import math

    import torch.nn as nn
    import torch.nn.functional as F

    from models.demos.wormhole.cosy_voice.tests.test_flow_reference import DiT as RefDiT
    from models.demos.wormhole.cosy_voice.tests.test_flow_reference import PreLookaheadLayer as RefPLL

    # Load reference modules
    ref_dit = RefDiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80)
    dit_sd = {k.replace("decoder.estimator.", ""): v for k, v in sd.items() if k.startswith("decoder.estimator.")}
    ref_dit.load_state_dict({k: v for k, v in dit_sd.items() if "rotary_embed" not in k}, strict=False)
    ref_dit.eval()

    ref_pll = RefPLL(in_channels=80, channels=1024, pre_lookahead_len=3)
    pll_sd = {k.replace("pre_lookahead_layer.", ""): v for k, v in sd.items() if k.startswith("pre_lookahead_layer.")}
    ref_pll.load_state_dict(pll_sd)
    ref_pll.eval()

    ref_input_emb = nn.Embedding(6561, 80)
    ref_input_emb.load_state_dict({"weight": sd["input_embedding.weight"]})

    ref_spk_layer = nn.Linear(192, 80)
    ref_spk_layer.load_state_dict(
        {
            "weight": sd["spk_embed_affine_layer.weight"],
            "bias": sd["spk_embed_affine_layer.bias"],
        }
    )

    # Dummy inputs
    torch.manual_seed(42)
    prompt_len = 5
    target_len = 10
    prompt_token = torch.randint(0, 6561, (1, prompt_len))
    token = torch.randint(0, 6561, (1, target_len))
    prompt_feat = torch.randn(1, prompt_len * 2, 80)  # token_mel_ratio=2
    embedding = torch.randn(1, 192)

    # --- PyTorch reference ---
    with torch.no_grad():
        spk = F.normalize(embedding, dim=1)
        spk = ref_spk_layer(spk)

        all_tokens = torch.cat([prompt_token, token], dim=1)
        mask = torch.ones(1, 1, all_tokens.shape[1])
        token_emb = ref_input_emb(torch.clamp(all_tokens, min=0)) * mask.transpose(1, 2)

        h = ref_pll(token_emb)
        h = h.repeat_interleave(2, dim=1)
        mel_len1 = prompt_feat.shape[1]
        mel_len2 = h.shape[1] - mel_len1

        conds = torch.zeros(1, mel_len1 + mel_len2, 80)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mu = h.transpose(1, 2).contiguous()
        mask = torch.ones(1, 1, mel_len1 + mel_len2)

        # Fixed noise (same as TtCausalMaskedDiffWithDiT)
        torch.manual_seed(0)
        rand_noise = torch.randn(1, 80, 50 * 300)
        z = rand_noise[:, :, : mu.size(2)]

        # Euler ODE
        n_timesteps = 10
        cfg_rate = 0.7
        t_span = torch.linspace(0, 1, n_timesteps + 1)
        t_span = 1 - torch.cos(t_span * 0.5 * math.pi)

        x = z.clone()
        t, dt = t_span[0].unsqueeze(0), t_span[1] - t_span[0]
        seq_len = x.size(2)
        x_in = torch.zeros(2, 80, seq_len)
        mask_in = torch.zeros(2, 1, seq_len)
        mu_in = torch.zeros(2, 80, seq_len)
        t_in = torch.zeros(2)
        spks_in = torch.zeros(2, 80)
        cond_in = torch.zeros(2, 80, seq_len)

        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t
            spks_in[0] = spk
            cond_in[0] = conds
            dphi_dt = ref_dit(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            dphi_c, dphi_u = torch.split(dphi_dt, [1, 1], dim=0)
            dphi_dt = (1.0 + cfg_rate) * dphi_c - cfg_rate * dphi_u
            x = x + dt * dphi_dt
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t_span[step]

        ref_feat = x[:, :, mel_len1:].float()

    # --- TTNN flow decoder ---
    tt_feat, _ = tt_flow.inference(
        token,
        torch.tensor([target_len]),
        prompt_token,
        torch.tensor([prompt_len]),
        prompt_feat,
        torch.tensor([prompt_feat.shape[1]]),
        embedding,
        streaming=False,
        finalize=True,
    )

    pcc = compute_pcc(ref_feat, tt_feat)
    print(f"Flow Decoder E2E PCC: {pcc:.6f}")
    print(f"  ref  shape={ref_feat.shape}, mean={ref_feat.mean():.4f}, std={ref_feat.std():.4f}")
    print(f"  ttnn shape={tt_feat.shape}, mean={tt_feat.mean():.4f}, std={tt_feat.std():.4f}")
    assert pcc >= 0.85, f"PCC {pcc:.6f} below threshold 0.85"
