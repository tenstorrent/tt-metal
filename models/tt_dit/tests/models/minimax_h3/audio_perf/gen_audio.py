"""Decode one fixed latent to a WAV, timed -- run once per code state to A/B pre-fix vs post-fix.

Usage:  python gen_audio.py <tag>

Writes into OUT_DIR:
    <tag>.wav     the device decode, 32 kHz stereo
    <tag>.pt      {wav, seconds} for scoring by compare_audio.py
    source.wav    the synthetic input signal          (written once)
    reference.wav the CPU/diffusers decode of the same latent (written once, the ground truth)
    latents.pt    the exact latent both runs decode   (written once, then reused)

The input is a *musical* signal rather than the `torch.randn` the tests use: tones, a sweep and an
envelope make codec artefacts audible, where broadband noise mostly hides them. The latent is
computed once and cached so both code states decode bit-identical input.
"""

import os
import sys
import time

import numpy as np
import soundfile as sf
import torch

import ttnn

SR = 32000
HOP = 800
NUM_LATENT_FRAMES = 207  # ~5.2 s
OUT_DIR = "/data/rshirvani/audio_compare"
WEIGHTS = os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "/data/cglagovich/MiniMax-H3-diffusers")


def source_signal(num_samples: int) -> torch.Tensor:
    """Stereo test signal: a triad with vibrato, a log sweep, and plucked transients."""
    t = torch.arange(num_samples, dtype=torch.float64) / SR
    env = torch.exp(-1.5 * (t % 1.0))  # re-plucked every second: sharp transients

    left = torch.zeros_like(t)
    for f, a in ((220.0, 0.5), (277.18, 0.34), (329.63, 0.26), (440.0, 0.16)):
        vib = 1.0 + 0.004 * torch.sin(2 * np.pi * 5.5 * t)
        left += a * torch.sin(2 * np.pi * f * vib * t)
    left *= env

    # right: log sweep 120 Hz -> 6 kHz, which exposes band-dependent error
    f0, f1 = 120.0, 6000.0
    dur = t[-1].item()
    phase = 2 * np.pi * f0 * dur / np.log(f1 / f0) * (torch.pow(f1 / f0, t / dur) - 1.0)
    right = 0.6 * torch.sin(phase) * (0.35 + 0.65 * env)

    wav = torch.stack([left, right]).to(torch.float32)
    return (0.85 * wav / wav.abs().max()).unsqueeze(1)  # (2, 1, N)


def write_wav(path: str, wav: torch.Tensor) -> None:
    """wav is (2, 1, N) -> interleaved stereo float32."""
    data = wav.detach().float().squeeze(1).transpose(0, 1).contiguous().numpy()
    sf.write(path, np.clip(data, -1.0, 1.0), SR, subtype="FLOAT")


def main():
    tag = sys.argv[1]
    os.makedirs(OUT_DIR, exist_ok=True)
    latents_path = os.path.join(OUT_DIR, "latents.pt")

    import json

    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
    from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder

    audio_dir = os.path.join(WEIGHTS, "audio_vae")
    with open(os.path.join(audio_dir, "config.json")) as fh:
        config = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(audio_dir, "diffusion_pytorch_model.safetensors")))

    if os.path.exists(latents_path):
        latents = torch.load(latents_path)
        print(f"reusing cached latents {tuple(latents.shape)}")
    else:
        src = source_signal(NUM_LATENT_FRAMES * HOP)
        write_wav(os.path.join(OUT_DIR, "source.wav"), src)
        with torch.no_grad():
            latents = reference.encode(src).latent_dist.mode()[..., :NUM_LATENT_FRAMES]
        torch.save(latents, latents_path)
        print(f"encoded latents {tuple(latents.shape)}")

    ref_path = os.path.join(OUT_DIR, "reference.wav")
    if not os.path.exists(ref_path):
        t0 = time.perf_counter()
        with torch.no_grad():
            expected = reference.decode(latents).sample
        cpu_secs = time.perf_counter() - t0
        write_wav(ref_path, expected)
        torch.save({"wav": expected, "seconds": cpu_secs}, os.path.join(OUT_DIR, "reference.pt"))
        print(f"CPU reference decode: {cpu_secs:.3f} s")

    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        decoder = MiniMaxH3AudioDecoder(
            latent_channels=config["latent_channels"],
            latent_dim=config["latent_dim"],
            decoder_dim=config["decoder_dim"],
            decoder_rates=tuple(config["decoder_rates"]),
            decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
            resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
            mesh_device=device,
        )
        decoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)

        decoder(latents)  # warm: exclude JIT compile and weight upload from the timing
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            actual = decoder(latents)
            runs.append(time.perf_counter() - t0)
        secs = min(runs)

        write_wav(os.path.join(OUT_DIR, f"{tag}.wav"), actual)
        torch.save({"wav": actual, "seconds": secs, "runs": runs}, os.path.join(OUT_DIR, f"{tag}.pt"))
        print(f"{tag}: decode {secs:.3f} s  (runs {', '.join(f'{r:.3f}' for r in runs)})")
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
