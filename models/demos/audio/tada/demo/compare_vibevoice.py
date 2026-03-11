#!/usr/bin/env python3
"""
Compare VibeVoice ODE output: TTNN hidden states + reference VibeVoice on CPU.
Loads saved hidden states from the TTNN run, runs reference VibeVoice, decodes to audio.
"""
import math
import os
import sys

import numpy as np
import torch
from loguru import logger
from safetensors.torch import load_file as safetensors_load_file
from scipy.io import wavfile

TADA_MODEL_PATH = os.environ.get("TADA_MODEL_PATH", "/home/ttuser/atupe/tada/tada-1b")
TADA_CODEC_PATH = os.environ.get("TADA_CODEC_PATH", "/home/ttuser/atupe/tada/tada-codec")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
SAMPLE_RATE = 24000

# TADA 1B constants
ACOUSTIC_DIM = 512
HIDDEN_SIZE = 2048
NUM_TIME_BITS = 8
TIME_DIM = 2 * NUM_TIME_BITS  # 16
LATENT_SIZE = ACOUSTIC_DIM + TIME_DIM  # 528
SHIFT_ACOUSTIC = 5
ACOUSTIC_STD = 1.5
ACOUSTIC_MEAN = 0.0
NUM_TIME_CLASSES = 256


def _gray_code_to_int(gray):
    binary = gray
    shift = 1
    while shift < 32:
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary


def decode_gray_code_to_time(gray_bits, num_bits):
    gray_bits_binary = ((gray_bits + 1.0) / 2.0).round().long()
    gray_code = torch.zeros(*gray_bits_binary.shape[:-1], dtype=torch.long)
    for i in range(num_bits):
        gray_code += gray_bits_binary[..., num_bits - 1 - i] << i
    return _gray_code_to_int(gray_code)


def build_time_schedule(num_steps, schedule):
    if schedule == "logsnr":
        log_snr = torch.linspace(5.0, -5.0, num_steps + 1)
        t_span = torch.sigmoid(-log_snr / 2)
        t_span[0] = 0.0
        t_span[-1] = 1.0
        return t_span
    return torch.linspace(0, 1, num_steps + 1)


def scheduled_cfg(base_scale, t, schedule):
    if schedule == "constant" or base_scale == 1.0:
        return base_scale
    if schedule == "cosine":
        return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + math.cos(math.pi * t))
    return base_scale


def load_vibevoice_cpu(model_path):
    """Load VibeVoice diffusion head for CPU execution."""
    sys.path.insert(0, "/home/ttuser/atupe/tada/tada")
    from tada.nn.vibevoice import VibeVoiceDiffusionHead, VibeVoiceDiffusionHeadConfig

    weights = safetensors_load_file(os.path.join(model_path, "model.safetensors"))

    # Extract VibeVoice config from model
    import json

    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    # Build VibeVoice head
    vv_config = VibeVoiceDiffusionHeadConfig(
        hidden_size=HIDDEN_SIZE,
        latent_size=LATENT_SIZE,
        head_layers=config.get("prediction_head_num_layers", 6),
        head_ffn_ratio=config.get("prediction_head_ffn_ratio", 4.0),
    )
    vv = VibeVoiceDiffusionHead(vv_config)

    # Load weights
    vv_state = {}
    for k, v in weights.items():
        if k.startswith("prediction_head."):
            vv_state[k[len("prediction_head.") :]] = v.float()
    missing, unexpected = vv.load_state_dict(vv_state, strict=False)
    if missing:
        logger.warning(f"Missing VibeVoice keys: {missing}")
    if unexpected:
        logger.warning(f"Unexpected VibeVoice keys: {unexpected}")

    vv.eval()
    return vv


def solve_flow_matching_cpu(
    vv_head, cond, neg_cond, num_steps=20, noise_temp=0.9, acoustic_cfg_scale=1.6, duration_cfg_scale=1.0
):
    """Run ODE solver on CPU with reference VibeVoice."""
    B = cond.shape[0]
    speech = torch.randn(B, LATENT_SIZE) * noise_temp
    t_span = build_time_schedule(num_steps, "logsnr")
    t_curr = t_span[0]

    for i in range(1, len(t_span)):
        dt = t_span[i] - t_curr
        t_val = t_curr.item()
        a_cfg = scheduled_cfg(acoustic_cfg_scale, t_val, "cosine")
        d_cfg = scheduled_cfg(duration_cfg_scale, t_val, "cosine")
        t_torch = t_curr.expand(B)

        if acoustic_cfg_scale != 1.0:
            speech_doubled = torch.cat([speech, speech], dim=0)
            t_doubled = t_torch.repeat(2)
            cond_combined = torch.cat([cond, neg_cond], dim=0)

            velocity_combined = vv_head(speech_doubled, t_doubled, condition=cond_combined)
            vel_pos = velocity_combined[:B]
            vel_neg = velocity_combined[B:]

            velocity = torch.cat(
                [
                    (vel_neg + a_cfg * (vel_pos - vel_neg))[..., :ACOUSTIC_DIM],
                    (vel_neg + d_cfg * (vel_pos - vel_neg))[..., ACOUSTIC_DIM:],
                ],
                dim=-1,
            )
        else:
            velocity = vv_head(speech, t_torch, condition=cond)

        speech = speech + dt * velocity
        t_curr = t_span[i]

    return speech


def save_audio(filepath, audio, sample_rate=SAMPLE_RATE):
    if audio.dim() == 3:
        audio = audio.squeeze(0).squeeze(0)
    elif audio.dim() == 2:
        audio = audio.squeeze(0)
    audio = audio.clamp(-1.0, 1.0)
    audio_np = (audio.float().cpu().numpy() * 32767).astype("int16")
    wavfile.write(filepath, sample_rate, audio_np)
    logger.info(f"Saved {filepath} ({audio_np.shape[0] / sample_rate:.2f}s, std={np.std(audio_np):.0f})")


def main():
    # Load saved hidden states from TTNN run
    debug_path = os.path.join(OUTPUT_DIR, "debug_hidden_states.pt")
    if not os.path.exists(debug_path):
        logger.error(f"No hidden states found at {debug_path}. Run the demo first.")
        return

    data = torch.load(debug_path, weights_only=False)
    hidden_states = data["hidden_states"]  # List of (B, 1, 2048) tensors
    input_ids = data["input_ids"]

    logger.info(f"Loaded {len(hidden_states)} hidden states, input_ids shape={input_ids.shape}")

    # Load reference VibeVoice
    logger.info("Loading reference VibeVoice on CPU...")
    vv_head = load_vibevoice_cpu(TADA_MODEL_PATH)
    logger.info("VibeVoice loaded.")

    # Run VibeVoice ODE on CPU for each step using TTNN hidden states
    # This tests: are the acoustic features from CPU VibeVoice + TTNN hidden states
    # different from the TTNN VibeVoice + TTNN hidden states?
    prefix_len = 8
    n_prefix_acoustic = max(0, prefix_len - SHIFT_ACOUSTIC)

    all_acoustic = []
    all_time_before = []

    # For negative conditioning: use zeros as a simple approximation
    # (the proper neg_cond would need a separate Llama pass, not available here)
    logger.info("Running reference VibeVoice ODE solver with TTNN hidden states...")
    for step in range(len(hidden_states)):
        h = hidden_states[step].float()  # (B, 1, 2048)
        cond = h.squeeze(1)  # (B, 2048)
        neg_cond = torch.zeros_like(cond)  # Simple zero neg cond

        with torch.no_grad():
            speech = solve_flow_matching_cpu(vv_head, cond, neg_cond)

        # Extract time
        time_gray = speech[..., -TIME_DIM:]
        t_before = decode_gray_code_to_time(time_gray[..., :NUM_TIME_BITS], NUM_TIME_BITS)
        t_after = decode_gray_code_to_time(time_gray[..., NUM_TIME_BITS:], NUM_TIME_BITS)

        if step >= SHIFT_ACOUSTIC:
            all_acoustic.append(speech[..., :ACOUSTIC_DIM])
            all_time_before.append(t_before)

        tok_id = input_ids[0, step].item() if step < input_ids.shape[1] else -1
        if step < 15 or step % 5 == 0:
            logger.info(
                f"  Step {step}: speech_norm={speech.norm():.3f}, "
                f"t_before={t_before.tolist()}, t_after={t_after.tolist()}"
            )

    if not all_acoustic:
        logger.error("No acoustic features collected!")
        return

    # Stack and denormalize
    acoustic_cat = torch.stack(all_acoustic, dim=1)  # (B, N, 512)
    acoustic_cat = acoustic_cat * ACOUSTIC_STD + ACOUSTIC_MEAN

    time_cat = torch.stack(all_time_before, dim=1)  # (B, N)
    # Add trailing time
    time_cat = torch.cat([time_cat, time_cat[:, -1:]], dim=1)

    logger.info(f"Acoustic features: shape={acoustic_cat.shape}, norm={acoustic_cat.norm():.3f}")
    logger.info(f"Time before: {time_cat[0].tolist()}")

    # Trim prefix
    if n_prefix_acoustic > 0:
        acoustic_cat = acoustic_cat[:, n_prefix_acoustic:, :]
        time_cat = time_cat[:, n_prefix_acoustic:]
        logger.info(f"After prefix trim ({n_prefix_acoustic}): {acoustic_cat.shape}")

    logger.info(f"Time before (trimmed): {time_cat[0].tolist()}")

    # Decode with reference decoder
    sys.path.insert(0, "/home/ttuser/atupe/tada/tada")
    try:
        from tada.modules.decoder import Decoder, DecoderConfig

        decoder = Decoder(DecoderConfig(), codec_model_path=TADA_CODEC_PATH)
        decoder.eval()

        # Expand
        encoded = acoustic_cat[0]
        time_before = time_cat[0]
        time_before = time_before[: encoded.shape[0] + 1]

        encoded_expanded = []
        for pos in range(encoded.shape[0]):
            gap_len = max(0, int(time_before[pos].item()) - 1)
            if gap_len > 0:
                encoded_expanded.append(torch.zeros(gap_len, encoded.shape[-1]))
            encoded_expanded.append(encoded[pos].unsqueeze(0))
        trail_len = int(time_before[-1].item())
        if trail_len > 0:
            encoded_expanded.append(torch.zeros(trail_len, encoded.shape[-1]))
        encoded_expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)
        token_masks = (torch.norm(encoded_expanded, dim=-1) != 0).long()

        logger.info(f"Expanded: {encoded_expanded.shape}, masks sum={token_masks.sum()}")

        with torch.no_grad():
            wav = decoder.generate(encoded_expanded.float(), token_masks)

        # Trim leading silence
        leading = int(time_before[0].item() * 480)
        if leading > 0 and wav.shape[-1] > leading:
            wav = wav[..., leading:]

        save_audio(os.path.join(OUTPUT_DIR, "cpu_vibevoice_output.wav"), wav)
        logger.info(f"CPU VibeVoice output: shape={wav.shape}, norm={wav.norm():.4f}")
    except Exception as e:
        logger.error(f"Decoder failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
