#!/usr/bin/env python
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only controls that justify the stage-06 log-mel spectral-distance bar (no device).

Runs the vendored torch vocoder on the golden latents (``~/mm3-bringup/reference/chunks.pt``) and measures
the log-mel distance (``tt/audio_metrics.log_mel_distance``) of the stitched wav against the golden
``audio.wav`` for:

1. ``fp32``            - the vendored fp32 vocoder itself (should be at the 16-bit quantization floor);
2. ``bf16``            - the same vocoder in bf16 (the "torch-vs-torch bf16 control" the stage prompt asks for);
3. ``perturbed_<pcc>`` - fp32 vocoder on golden latents with Gaussian noise added until their PCC vs the golden
                         latents equals the given value (0.9996 = the stage-05 per-chunk DiT PCC, 0.98 = the
                         latent PCC bar), i.e. what a DiT error of that size alone does to the spectrum;
4. ``unrelated_noise_latents`` - fp32 vocoder on pure noise latents (no denoising) as an "unrelated audio"
                         reference point for the scale of the metric.

Writes ``doc/pipeline/pcc/vocoder_control.json``. Run with the ttnn python (``$MM3_PY``); takes a few minutes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import soundfile as sf
import torch

from models.autoports.minimaxai_minimax_music3.reference import vocoder_ref as V
from models.autoports.minimaxai_minimax_music3.reference.hf_llm import reference_dir
from models.autoports.minimaxai_minimax_music3.tt.audio_metrics import audio_stats, log_mel_distance
from models.common.utility_functions import comp_pcc

MODEL_DIR = Path(__file__).resolve().parents[1]


def perturb_to_pcc(latents, target_pcc: float, generator: torch.Generator):
    """Add white noise to every chunk so that PCC(golden, perturbed) == target (bisection on the noise scale)."""
    noises = [torch.randn(t.shape, generator=generator) for t in latents]
    cat = torch.cat([t.reshape(-1) for t in latents])
    ncat = torch.cat([n.reshape(-1) for n in noises])

    def pcc_for(scale):
        return float(comp_pcc(cat, cat + scale * ncat, 0.0)[1])

    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if pcc_for(mid) > target_pcc:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    return [t + scale * n for t, n in zip(latents, noises)], scale, pcc_for(scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MODEL_DIR / "doc" / "pipeline" / "pcc" / "vocoder_control.json"))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--pccs", type=float, nargs="*", default=[0.9996, 0.99, 0.98])
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    root = reference_dir()
    chunks = torch.load(root / "chunks.pt")
    latents = [t.float() for t in chunks["latents"]]
    golden_np, sr = sf.read(root / "audio.wav", dtype="float32")
    golden = torch.from_numpy(golden_np.T).unsqueeze(0)  # [1, 2, S]

    results = {
        "_meta": {
            "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threads": args.threads,
            "torch": torch.__version__,
        }
    }
    voc32 = V.load_vocoder(dtype=torch.float32)

    def run(name, wav, note):
        d = log_mel_distance(wav, golden, sr)
        _, pcc = comp_pcc(golden, wav[..., : golden.shape[-1]], 0.0)
        results[name] = {**d, "wav_pcc_vs_golden": float(pcc), "note": note}
        print(
            f"{name:>18}: log-mel rms {d['rms_db']:.3f} dB, mean abs {d['mean_abs_db']:.3f} dB, wav PCC {float(pcc):.6f}  ({note})",
            flush=True,
        )

    t0 = time.time()
    wav32 = V.decode_latent_chunks(voc32, latents)
    results["_meta"]["fp32_decode_seconds"] = time.time() - t0
    run("fp32", wav32, "vendored fp32 vocoder on the golden latents vs golden audio.wav (16-bit PCM)")
    results["fp32"]["max_abs_sample_err"] = float((wav32 - golden).abs().max())

    voc16 = V.load_vocoder(dtype=torch.bfloat16)
    t0 = time.time()
    wav16 = V.decode_latent_chunks(voc16, latents)
    results["_meta"]["bf16_decode_seconds"] = time.time() - t0
    run("bf16", wav16, "vendored vocoder in bf16 on the golden latents (torch-vs-torch bf16 control)")
    results["bf16"]["wav_pcc_vs_fp32"] = float(comp_pcc(wav32, wav16, 0.0)[1])

    gen = torch.Generator().manual_seed(1234)
    for target in args.pccs:
        pert, scale, got = perturb_to_pcc(latents, target, gen)
        wav = V.decode_latent_chunks(voc32, pert)
        run(
            f"perturbed_{target}",
            wav,
            f"fp32 vocoder on golden latents + white noise (scale {scale:.4f}) at latent PCC {got:.5f}",
        )
        results[f"perturbed_{target}"]["latent_pcc"] = got
        results[f"perturbed_{target}"]["noise_scale"] = scale

    other = [torch.randn(t.shape, generator=gen) for t in latents]
    wav = V.decode_latent_chunks(voc32, other)
    run("unrelated_noise_latents", wav, "fp32 vocoder on pure noise latents (unrelated audio; scale of the metric)")

    results["golden_stats"] = audio_stats(golden, sr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
