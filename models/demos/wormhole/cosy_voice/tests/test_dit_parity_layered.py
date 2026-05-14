# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
DiT parity test: compare reference PyTorch DiT vs TT DiT at each stage.
Feeds identical inputs and reports where divergence first exceeds tolerance.
"""
import os
import sys

import soundfile as sf


def custom_torchaudio_load(filepath, **kwargs):
    audio, sr = sf.read(filepath)
    if len(audio.shape) == 1:
        tensor = torch.tensor(audio).unsqueeze(0).float()
    else:
        tensor = torch.tensor(audio).transpose(0, 1).float()
    return tensor, sr


def custom_torchaudio_save(filepath, tensor, sample_rate, **kwargs):
    audio = tensor.transpose(0, 1).cpu().numpy()
    sf.write(filepath, audio, sample_rate)


import torchaudio

torchaudio.load = custom_torchaudio_load
torchaudio.save = custom_torchaudio_save

import torch
import torch.nn.functional as F

sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice")
sys.path.insert(0, "models/demos/wormhole/cosy_voice/ref/CosyVoice/third_party/Matcha-TTS")


def compare(name, ref, tt, atol=0.05):
    """Compare two tensors and report statistics."""
    if ref.shape != tt.shape:
        print(f"  ❌ {name}: SHAPE MISMATCH ref={ref.shape} tt={tt.shape}")
        return False
    diff = (ref - tt).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_norm = ref.abs().mean().item()
    rel_err = mean_diff / (ref_norm + 1e-8)
    status = "✅" if max_diff < atol else "❌"
    print(
        f"  {status} {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_err={rel_err:.4f}, ref_range=[{ref.min():.4f}, {ref.max():.4f}]"
    )
    return max_diff < atol


def main():
    weights_dir = "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"

    # ── Load reference flow decoder with weights ──
    print("Loading reference flow decoder...")
    from hyperpyyaml import load_hyperpyyaml

    with open(os.path.join(weights_dir, "cosyvoice3.yaml"), "r") as f:
        configs = load_hyperpyyaml(
            f,
            overrides={
                "llm": None,
                "hift": None,
                "qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN"),
            },
        )
    ref_flow = configs["flow"]
    flow_sd = torch.load(os.path.join(weights_dir, "flow.pt"), map_location="cpu", weights_only=True)
    ref_flow.load_state_dict(flow_sd)
    ref_flow.eval()

    # ── Create test inputs ──
    print("\nCreating test inputs...")
    batch = 2  # CFG uses batch=2
    seq_len = 160  # reasonable mel length
    mel_dim = 80
    spk_dim = 80

    torch.manual_seed(42)
    x = torch.randn(batch, mel_dim, seq_len)
    mu = torch.randn(batch, mel_dim, seq_len)
    cond = torch.randn(batch, mel_dim, seq_len)
    mask = torch.ones(batch, 1, seq_len)
    t = torch.tensor([0.1, 0.1])
    spks = torch.randn(batch, spk_dim)

    # ── Stage 1: Compare TimestepEmbedding ──
    print("\n=== Stage 1: TimestepEmbedding ===")
    ref_estimator = ref_flow.decoder.estimator
    ref_t_emb = ref_estimator.time_embed(t)

    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        from models.demos.wormhole.cosy_voice.tt.flow.dit import TtDiT

        tt_dit = TtDiT(device, flow_sd, dtype=ttnn.bfloat16)

        tt_t_emb = tt_dit._timestep_embedding(t)
        compare("TimestepEmbedding", ref_t_emb, tt_t_emb)

        # ── Stage 2: Compare InputEmbedding ──
        print("\n=== Stage 2: InputEmbedding ===")
        # Reference: x, cond, mu are (batch, seq, mel) after transpose in DiT.forward
        x_t = x.transpose(1, 2)  # (batch, seq, mel)
        mu_t = mu.transpose(1, 2)
        cond_t = cond.transpose(1, 2)
        spks_sq = spks.unsqueeze(1)  # (batch, 1, spk_dim)

        ref_input_emb = ref_estimator.input_embed(x_t, cond_t, mu_t, spks_sq.squeeze(1))
        tt_input_emb = tt_dit._input_embedding(x_t, cond_t, mu_t, spks_sq.squeeze(1))
        compare("InputEmbedding (proj+conv_pos)", ref_input_emb, tt_input_emb)

        # ── Stage 2a: Compare just the projection (before conv_pos) ──
        print("\n=== Stage 2a: InputEmbedding projection only ===")
        from einops import repeat

        spks_expanded = repeat(spks, "b c -> b t c", t=seq_len)
        inp = torch.cat([x_t, cond_t, mu_t, spks_expanded], dim=-1)
        ref_proj = ref_estimator.input_embed.proj(inp)
        tt_proj = F.linear(inp, tt_dit.input_proj_w, tt_dit.input_proj_b)
        compare("Linear proj", ref_proj, tt_proj)

        # ── Stage 2b: Compare conv_pos ──
        print("\n=== Stage 2b: CausalConvPositionEmbedding ===")
        ref_conv_pos = ref_estimator.input_embed.conv_pos_embed(ref_proj)
        # TT conv_pos (manual)
        h = ref_proj.permute(0, 2, 1)
        k = tt_dit.conv_pos_conv1_w.shape[2]
        h = F.pad(h, (k - 1, 0))
        h = F.conv1d(h, tt_dit.conv_pos_conv1_w, tt_dit.conv_pos_conv1_b, groups=16)
        h = F.mish(h)
        h = F.pad(h, (k - 1, 0))
        h = F.conv1d(h, tt_dit.conv_pos_conv2_w, tt_dit.conv_pos_conv2_b, groups=16)
        h = F.mish(h)
        tt_conv_pos = h.permute(0, 2, 1)
        compare("ConvPosEmbed output", ref_conv_pos, tt_conv_pos)

        # ── Stage 3: Compare full DiT forward (1 step) ──
        print("\n=== Stage 3: Full DiT forward (single evaluation) ===")
        with torch.no_grad():
            ref_vel = ref_estimator(x, mask, mu, t, spks, cond)

        # For TT, we need to call the full forward
        with torch.no_grad():
            tt_vel = tt_dit(x, mask, mu, t, spks, cond)

        compare("Full DiT velocity", ref_vel, tt_vel, atol=0.5)

        # ── Stage 4: If full forward diverges, test block-by-block ──
        ref_vel_f = ref_vel.float()
        tt_vel_f = tt_vel.float()
        full_diff = (ref_vel_f - tt_vel_f).abs().max().item()
        if full_diff > 0.5:
            print(f"\n=== Stage 4: Block 0 sub-component parity (full diff={full_diff:.4f}) ===")

            ref_x = ref_input_emb.clone()

            from x_transformers.x_transformers import RotaryEmbedding

            rope_embed = RotaryEmbedding(64)
            rope = rope_embed.forward_from_seq_len(seq_len)

            ref_attn_mask = mask.bool().expand(-1, seq_len, -1).unsqueeze(1)
            tt_attn_mask = ref_attn_mask.clone()

            ref_block = ref_estimator.transformer_blocks[0]
            tt_block = tt_dit.blocks[0]

            with torch.no_grad():
                # ── 4a. AdaLayerNormZero ──
                print("\n  --- AdaLayerNormZero ---")
                ref_norm, ref_gate_msa, ref_shift_mlp, ref_scale_mlp, ref_gate_mlp = ref_block.attn_norm(
                    ref_x, emb=ref_t_emb
                )

                # TT AdaLN
                tt_x_4d = ttnn.from_torch(
                    ref_x.unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                tt_t_4d = ttnn.from_torch(
                    ref_t_emb.unsqueeze(0).unsqueeze(0),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                )
                tt_norm_4d, tt_gate_msa_4d, tt_shift_mlp_4d, tt_scale_mlp_4d, tt_gate_mlp_4d = tt_block.attn_norm(
                    tt_x_4d, tt_t_4d
                )
                tt_norm = ttnn.to_torch(tt_norm_4d).squeeze(0).float()
                tt_gate_msa = ttnn.to_torch(tt_gate_msa_4d).squeeze(0).squeeze(1).float()
                tt_gate_mlp = ttnn.to_torch(tt_gate_mlp_4d).squeeze(0).squeeze(1).float()

                compare("AdaLN norm_x", ref_norm, tt_norm)
                compare("AdaLN gate_msa", ref_gate_msa, tt_gate_msa)
                compare("AdaLN gate_mlp", ref_gate_mlp, tt_gate_mlp)

                # ── 4b. Attention ──
                print("\n  --- Attention ---")
                ref_attn_out = ref_block.attn(x=ref_norm, mask=ref_attn_mask.bool(), rope=rope)
                tt_attn_out_4d = tt_block.attn(tt_norm_4d, mask=tt_attn_mask, rope=rope)
                tt_attn_out = ttnn.to_torch(tt_attn_out_4d).squeeze(0).float()
                compare("Attention output", ref_attn_out, tt_attn_out)

                # ── 4c. Gated residual (MSA) ──
                print("\n  --- Gated residual (MSA) ---")
                ref_x_after_attn = ref_x + ref_gate_msa.unsqueeze(1) * ref_attn_out
                # For TT: gate_msa is (1, batch, 1, dim), attn_out is (1, batch, seq, dim)
                tt_x_after_attn_4d = ttnn.add(tt_x_4d, ttnn.multiply(tt_gate_msa_4d, tt_attn_out_4d))
                tt_x_after_attn = ttnn.to_torch(tt_x_after_attn_4d).squeeze(0).float()
                compare("After MSA residual", ref_x_after_attn, tt_x_after_attn)

                # ── 4d. FF LayerNorm + modulation ──
                print("\n  --- FF LayerNorm + modulation ---")
                ref_ff_norm = ref_block.ff_norm(ref_x_after_attn)
                ref_ff_norm = ref_ff_norm * (1 + ref_scale_mlp[:, None]) + ref_shift_mlp[:, None]

                tt_ff_norm_4d = ttnn.layer_norm(tt_x_after_attn_4d, epsilon=1e-6)
                tt_ff_norm_raw = ttnn.to_torch(tt_ff_norm_4d).squeeze(0).float()
                compare("FF LayerNorm (raw)", F.layer_norm(ref_x_after_attn, [1024], eps=1e-6), tt_ff_norm_raw)

                tt_ff_norm_4d = tt_ff_norm_4d * (ttnn.add(tt_scale_mlp_4d, 1.0)) + tt_shift_mlp_4d
                tt_ff_norm = ttnn.to_torch(tt_ff_norm_4d).squeeze(0).float()
                compare("FF norm+modulation", ref_ff_norm, tt_ff_norm)

                # ── 4e. FeedForward ──
                print("\n  --- FeedForward ---")
                ref_ff_out = ref_block.ff(ref_ff_norm)
                tt_ff_out_4d = tt_block.ff(tt_ff_norm_4d)
                tt_ff_out = ttnn.to_torch(tt_ff_out_4d).squeeze(0).float()
                compare("FeedForward output", ref_ff_out, tt_ff_out)

                # ── 4f. Final gated residual ──
                print("\n  --- Final gated residual ---")
                ref_final = ref_x_after_attn + ref_gate_mlp.unsqueeze(1) * ref_ff_out
                tt_final_4d = ttnn.add(tt_x_after_attn_4d, ttnn.multiply(tt_gate_mlp_4d, tt_ff_out_4d))
                tt_final = ttnn.to_torch(tt_final_4d).squeeze(0).float()
                compare("Block 0 final output", ref_final, tt_final)

    finally:
        ttnn.close_device(device)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
