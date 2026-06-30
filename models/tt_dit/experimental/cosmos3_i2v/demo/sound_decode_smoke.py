"""CPU smoke test for the vendored Cosmos3 audio VAE.

Loads `nvidia/Cosmos3-Super/sound_tokenizer/` (the only public Cosmos3 checkpoint
that ships audio weights — Cosmos3-Super-Image2Video is sound_gen=False) and
decodes a random latent to a waveform on CPU. Writes the result to a .wav so
listening / spectrogram-inspecting confirms the decoder produces non-trivial audio.

Run on a dev box that has torch + HF auth for the gated repo:
    HF_HOME=/proj_sw/user_dev/$USER/hf_cache uv run python \\
        models/tt_dit/experimental/cosmos3_i2v/demo/sound_decode_smoke.py

This intentionally bypasses the full Cosmos3OmniPipeline — it's only validating
the vendored autoencoder loads and runs. End-to-end audio (transformer +
sound_tokenizer) is a follow-up smoke.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import torch

from models.tt_dit.experimental.cosmos3_i2v.reference.autoencoder_cosmos3_audio import Cosmos3AVAEAudioTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-repo", default="nvidia/Cosmos3-Super")
    parser.add_argument("--subfolder", default="sound_tokenizer")
    parser.add_argument(
        "--latent-frames", type=int, default=200, help="Number of latent timesteps T. 200 ≈ 8s at 48kHz / hop 1920."
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/cosmos3_audio_smoke.wav"))
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = parser.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]

    print(f"Loading {args.hf_repo}/{args.subfolder} ...", flush=True)
    tokenizer: Cosmos3AVAEAudioTokenizer = Cosmos3AVAEAudioTokenizer.from_pretrained(
        args.hf_repo, subfolder=args.subfolder, torch_dtype=dtype
    )
    tokenizer.eval()

    cfg = tokenizer.config
    print(
        f"  sampling_rate={cfg.sampling_rate}  vocoder_input_dim={cfg.vocoder_input_dim}  "
        f"hop_size={tokenizer._hop_size}  out_channels={cfg.dec_out_channels}",
        flush=True,
    )

    torch.manual_seed(0)
    latent = torch.randn(1, cfg.vocoder_input_dim, args.latent_frames, dtype=dtype)
    print(f"Decoding latent of shape {tuple(latent.shape)} ...", flush=True)
    with torch.no_grad():
        waveform = tokenizer.decode(latent)
    print(
        f"  waveform shape = {tuple(waveform.shape)}  dtype={waveform.dtype}  "
        f"min={waveform.float().min().item():.3f}  max={waveform.float().max().item():.3f}",
        flush=True,
    )

    pcm = (waveform.float().clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16).cpu()
    pcm = pcm.squeeze(0).T.contiguous().numpy()  # [N, channels] interleaved

    with wave.open(str(args.out), "wb") as wf:
        wf.setnchannels(cfg.dec_out_channels)
        wf.setsampwidth(2)
        wf.setframerate(cfg.sampling_rate)
        wf.writeframes(pcm.tobytes())

    duration_s = pcm.shape[0] / cfg.sampling_rate
    print(
        f"Wrote {args.out} — {duration_s:.2f}s of {cfg.sampling_rate} Hz " f"{cfg.dec_out_channels}-ch audio.",
        flush=True,
    )


if __name__ == "__main__":
    main()
