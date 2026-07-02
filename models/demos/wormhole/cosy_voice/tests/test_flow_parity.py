"""
Compare the ENTIRE flow decoder pipeline (outside DiT) stage by stage.
Since DiT blocks are now float32 on host, divergence must be in preprocessing or ODE solver.
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


def compare(name, ref, tt, atol=0.01):
    if ref.shape != tt.shape:
        print(f"  ❌ {name}: SHAPE MISMATCH ref={ref.shape} tt={tt.shape}")
        return False
    diff = (ref.float() - tt.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ref_norm = ref.float().abs().mean().item()
    rel_err = mean_diff / (ref_norm + 1e-8)
    status = "✅" if max_diff < atol else "❌"
    print(
        f"  {status} {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}, rel_err={rel_err:.4f}, ref_mean={ref.float().mean():.4f}"
    )
    return max_diff < atol


def main():
    weights_dir = "models/demos/wormhole/cosy_voice/pretrained_models/Fun-CosyVoice3-0.5B"

    # Load reference flow decoder
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
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

    # Load frontend
    with open(os.path.join(weights_dir, "cosyvoice3.yaml"), "r") as f:
        fe_configs = load_hyperpyyaml(
            f,
            overrides={
                "llm": None,
                "flow": None,
                "hift": None,
                "qwen_pretrain_path": os.path.join(weights_dir, "CosyVoice-BlankEN"),
            },
        )
    frontend = CosyVoiceFrontEnd(
        fe_configs["get_tokenizer"],
        fe_configs["feat_extractor"],
        os.path.join(weights_dir, "campplus.onnx"),
        os.path.join(weights_dir, "speech_tokenizer_v3.onnx"),
        os.path.join(weights_dir, "spk2info.pt"),
        fe_configs["allowed_special"],
    )
    model_input = frontend.frontend_zero_shot(
        "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。",
        "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
        "models/demos/wormhole/cosy_voice/ref/CosyVoice/asset/zero_shot_prompt.wav",
        16000,
        "",
    )

    prompt_speech_token = model_input["llm_prompt_speech_token"].to(torch.int32)
    prompt_feat = model_input["prompt_speech_feat"]
    embedding = model_input["llm_embedding"]

    # Use SAME tokens for both (prompt tokens as target)
    target_token = prompt_speech_token.clone()

    # ── Load TT flow decoder ──
    import ttnn

    device = ttnn.open_device(device_id=0)
    try:
        from models.demos.wormhole.cosy_voice.tt.flow.flow import TtCausalMaskedDiffWithDiT

        tt_flow = TtCausalMaskedDiffWithDiT(device, flow_sd, dtype=ttnn.bfloat16)

        # ── Stage-by-stage comparison ──
        print("\n=== Stage 1: Speaker embedding ===")
        ref_spk = F.normalize(embedding, dim=1)
        ref_spk = ref_flow.spk_embed_affine_layer(ref_spk)
        tt_spk = F.normalize(embedding, dim=1)
        tt_spk = tt_flow.spk_embed_affine_layer(tt_spk)
        compare("Speaker embedding", ref_spk, tt_spk)

        print("\n=== Stage 2: Token embedding ===")
        all_tokens = torch.cat([prompt_speech_token, target_token], dim=1)
        from cosyvoice.utils.mask import make_pad_mask

        all_token_len = torch.tensor([prompt_speech_token.shape[1]]) + torch.tensor([target_token.shape[1]])
        ref_mask = (~make_pad_mask(all_token_len)).unsqueeze(-1).to(ref_spk)
        ref_token_emb = ref_flow.input_embedding(torch.clamp(all_tokens, min=0)) * ref_mask
        tt_mask = torch.ones(1, 1, all_tokens.shape[1]).to(tt_spk)
        tt_token_emb = tt_flow.input_embedding(torch.clamp(all_tokens, min=0)) * tt_mask.transpose(1, 2)
        compare("Token embedding", ref_token_emb, tt_token_emb)

        print("\n=== Stage 3: PreLookaheadLayer ===")
        with torch.no_grad():
            ref_h = ref_flow.pre_lookahead_layer(ref_token_emb)
        tt_h = tt_flow._pre_lookahead_forward(tt_token_emb)
        compare("PreLookaheadLayer output", ref_h, tt_h)

        print("\n=== Stage 4: Repeat interleave ===")
        ref_h = ref_h.repeat_interleave(2, dim=1)
        tt_h = tt_h.repeat_interleave(2, dim=1)
        compare("After repeat_interleave", ref_h, tt_h)

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = ref_h.shape[1] - mel_len1
        print(f"  mel_len1={mel_len1}, mel_len2={mel_len2}")

        print("\n=== Stage 5: Conditioning ===")
        ref_conds = torch.zeros(1, mel_len1 + mel_len2, 80, dtype=ref_h.dtype)
        ref_conds[:, :mel_len1] = prompt_feat
        ref_conds = ref_conds.transpose(1, 2)
        tt_conds = torch.zeros(1, mel_len1 + mel_len2, 80, dtype=tt_h.dtype)
        tt_conds[:, :mel_len1] = prompt_feat
        tt_conds = tt_conds.transpose(1, 2)
        compare("Conditioning", ref_conds, tt_conds)

        print("\n=== Stage 6: mu and mask ===")
        ref_mu = ref_h.transpose(1, 2).contiguous()
        tt_mu = tt_h.transpose(1, 2).contiguous()
        compare("mu", ref_mu, tt_mu)

        ref_mask_euler = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(ref_h)
        ref_mask_euler = ref_mask_euler.unsqueeze(1)
        tt_mask_euler = torch.ones(1, 1, mel_len1 + mel_len2).to(tt_h)
        compare("mask", ref_mask_euler, tt_mask_euler)

        print("\n=== Stage 7: Noise initialization ===")
        # Reference noise
        ref_z = ref_flow.decoder.rand_noise[:, :, : ref_mu.size(2)].to(ref_mu.dtype)
        tt_z = tt_flow.rand_noise[:, :, : tt_mu.size(2)].to(tt_mu.dtype)
        compare("rand_noise z", ref_z, tt_z)

        print("\n=== Stage 8: ODE solver - Step 1 velocity ===")
        import math

        n = 10
        t_span = torch.linspace(0, 1, n + 1, dtype=ref_mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * math.pi)
        t = t_span[0].unsqueeze(0)

        # Reference single step
        with torch.no_grad():
            ref_vel = ref_flow.decoder.estimator(
                ref_z, ref_mask_euler, ref_mu, t.expand(1), ref_spk, ref_conds, streaming=False
            )
        print(f"  Reference velocity: min={ref_vel.min():.4f}, max={ref_vel.max():.4f}, mean={ref_vel.mean():.4f}")

        # TT single step (uses CFG batch=2)
        with torch.no_grad():
            seq_len = ref_z.size(2)

            # TT batch=1 execution
            tt_vel_cond_b1 = tt_flow.dit(ref_z, tt_mask_euler, ref_mu, t.expand(1), ref_spk, ref_conds)

            x_in = torch.zeros(2, 80, seq_len, dtype=ref_z.dtype)
            mask_in = torch.zeros(2, 1, seq_len, dtype=ref_z.dtype)
            mu_in = torch.zeros(2, 80, seq_len, dtype=ref_z.dtype)
            t_in = torch.zeros(2, dtype=ref_z.dtype)
            spks_in = torch.zeros(2, 80, dtype=ref_z.dtype)
            cond_in = torch.zeros(2, 80, seq_len, dtype=ref_z.dtype)

            x_in[:] = ref_z
            mask_in[:] = tt_mask_euler
            mu_in[0] = ref_mu
            t_in[:] = t
            spks_in[0] = ref_spk.squeeze(0)
            cond_in[0] = ref_conds

            tt_vel = tt_flow.dit(x_in, mask_in, mu_in, t_in, spks_in, cond_in)
            tt_vel_cond, tt_vel_uncond = torch.split(tt_vel, [1, 1], dim=0)
            tt_vel_cfg = (1.0 + 0.7) * tt_vel_cond - 0.7 * tt_vel_uncond
        print(
            f"  TT velocity (CFG): min={tt_vel_cfg.min():.4f}, max={tt_vel_cfg.max():.4f}, mean={tt_vel_cfg.mean():.4f}"
        )

        # Compare conditional velocity (no CFG) to reference
        compare("Step 1 velocity (cond only batch=1)", ref_vel, tt_vel_cond_b1)
        compare("Step 1 velocity (cond only batch=2)", ref_vel, tt_vel_cond)

        # Does the reference also use CFG?
        print("\n  --- Does the reference use CFG? ---")
        print(f"  ref_flow.decoder.estimator type: {type(ref_flow.decoder.estimator)}")
        # Check inference_cfg_rate
        print(f"  ref CFM t_scheduler: {ref_flow.decoder.t_scheduler}")
        print(f"  ref CFM inference_cfg_rate: {ref_flow.decoder.inference_cfg_rate}")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
