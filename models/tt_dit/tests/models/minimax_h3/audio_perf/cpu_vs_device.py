"""CPU reference vs device decode, fp32, over several real clips including speech.

For each clip: resample to 32 kHz, take one production window (207 latents = 5.17 s), encode with the
torch/diffusers reference, then decode twice -- once on CPU (the ground truth) and once on device --
and score the device output against the CPU one. Every WAV is written so the difference can be heard
rather than only read off a table.

Batch is 2 throughout, matching the shipping working point ("stereo" is carried as batch 2; the
autoencoder itself is mono). Mono clips are duplicated across the two batch slots, so the device time
here is directly comparable to `decode_bench.py`'s medians.

The decoder is built once and reused across clips, so the reported time is steady-state decode and
excludes model construction, weight upload and JIT.
"""

import json
import os
import time

import librosa
import numpy as np
import soundfile as sf
import torch

import ttnn

SR = 32000
HOP = 800
NUM_LATENT_FRAMES = 207
OUT_DIR = os.environ.get("CVD_OUT_DIR", "/data/rshirvani/audio_compare/clips")
# `MINIMAX_H3_MODEL_PATH` is what the test suite uses (`common.weights_subdir`); the older
# `MINIMAX_H3_DIFFUSERS_DIR` is still accepted so existing shells keep working.
WEIGHTS = os.environ.get("MINIMAX_H3_MODEL_PATH") or os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "")
if not WEIGHTS:
    raise SystemExit("set MINIMAX_H3_MODEL_PATH to a MiniMax-H3 diffusers snapshot")

# Mesh / sharding / trace, all env-gated so the default stays exactly the single-device run this
# script has always been. The acceptance criterion is PSNR **against the CPU reference**, and the
# T-parallel test only ever scores sharded against unsharded -- a different, looser bar (40 dB). So the
# shipping configuration has to be scored here, not there.
#
#   CVD_MESH=4x8 CVD_T_FACTOR=8 CVD_MESH_AXIS=1 CVD_TRACED=1 python cpu_vs_device.py
#
# The T-shard factor must equal the mesh axis length: on a 4x8 mesh only (4, axis 0) and (8, axis 1)
# work; anything else dies in `_partition_t` on a non-tile-aligned slice.
MESH = tuple(int(v) for v in os.environ.get("CVD_MESH", "1x1").split("x"))
T_FACTOR = int(os.environ.get("CVD_T_FACTOR", "1"))
MESH_AXIS = int(os.environ.get("CVD_MESH_AXIS", "1"))
TRACED = os.environ.get("CVD_TRACED", "0") == "1"
IS_DEFAULT = MESH == (1, 1) and T_FACTOR == 1 and not TRACED
# Non-default runs get their config in the filename. Overwriting `{label}_2_device.wav` with a
# differently-configured decode is how the stale `*_3_device_prefix.wav` confusion started.
TAG = "" if IS_DEFAULT else f"_{MESH[0]}x{MESH[1]}_f{T_FACTOR}ax{MESH_AXIS}{'_traced' if TRACED else ''}"

# (label, librosa example key, seconds to skip -- past leading silence / into a busy passage)
CLIPS = [
    ("voice_libri1", "libri1", 0.5),
    ("voice_libri2", "libri2", 0.5),
    ("music_trumpet", "trumpet", 0.5),
    ("music_brahms", "brahms", 8.0),
]


def load_clip(key: str, offset: float, num_samples: int) -> torch.Tensor:
    """-> (2, 1, num_samples) at SR, peak-normalised, mono duplicated across the batch slots."""
    y, _ = librosa.load(librosa.ex(key), sr=SR, mono=True, offset=offset, duration=num_samples / SR + 1.0)
    if len(y) < num_samples:
        y = np.pad(y, (0, num_samples - len(y)))
    y = y[:num_samples]
    peak = float(np.abs(y).max()) or 1.0
    mono = torch.from_numpy(0.85 * y / peak).float()
    return mono.unsqueeze(0).unsqueeze(0).repeat(2, 1, 1)


def write_wav(path: str, wav: torch.Tensor) -> None:
    data = wav.detach().float()[0, 0].numpy()  # both batch slots are identical; write one
    sf.write(path, np.clip(data, -1.0, 1.0), SR, subtype="FLOAT")


def psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    mse = torch.mean((ref.float() - test.float()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(ref.abs().max().item()) - 10.0 * np.log10(mse)


def rel_rmse(ref: torch.Tensor, test: torch.Tensor) -> float:
    return float((test.double() - ref.double()).pow(2).mean().sqrt() / ref.double().std())


def log_spec_distance(a: torch.Tensor, b: torch.Tensor, n_fft: int = 1024, hop: int = 256) -> float:
    window = torch.hann_window(n_fft)
    spec = []
    for sig in (a, b):
        flat = sig.reshape(-1, sig.shape[-1]).float()
        spec.append(
            torch.log(
                torch.stft(flat, n_fft=n_fft, hop_length=hop, window=window, return_complex=True).abs().clamp_min(1e-5)
            )
        )
    return (spec[0] - spec[1]).abs().mean().item()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import convert_minimax_h3_audio_state_dict
    from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder

    audio_dir = os.path.join(WEIGHTS, "audio_vae")
    with open(os.path.join(audio_dir, "config.json")) as fh:
        config = {k: v for k, v in json.load(fh).items() if not k.startswith("_")}
    reference = AutoencoderKLMiniMaxH3Audio(**config).eval()
    reference.load_state_dict(load_file(os.path.join(audio_dir, "diffusion_pytorch_model.safetensors")))

    num_samples = NUM_LATENT_FRAMES * HOP
    sharded = T_FACTOR > 1
    if MESH != (1, 1):
        # open_mesh_device takes no fabric_config; conftest sets it separately, before opening.
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        ttnn.MeshShape(*MESH),
        l1_small_size=65536,
        **({"trace_region_size": 450_000_000} if TRACED else {}),
    )
    rows = []
    try:
        print(
            f"config: mesh {MESH[0]}x{MESH[1]} ({device.get_num_devices()} chips), t_factor={T_FACTOR} "
            f"axis={MESH_AXIS}, traced={TRACED}",
            flush=True,
        )
        parallel_config = None
        ccl_manager = None
        if sharded:
            from models.tt_dit.parallel.config import ParallelFactor
            from models.tt_dit.parallel.manager import CCLManager

            parallel_config = ParallelFactor(factor=T_FACTOR, mesh_axis=MESH_AXIS)
            ccl_manager = CCLManager(device, num_links=1, topology=ttnn.Topology.Linear)

        decoder = MiniMaxH3AudioDecoder(
            latent_channels=config["latent_channels"],
            latent_dim=config["latent_dim"],
            decoder_dim=config["decoder_dim"],
            decoder_rates=tuple(config["decoder_rates"]),
            decoder_kernel_sizes=tuple(config["decoder_kernel_sizes"]),
            resblock_kernel_sizes=tuple(config["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(d) for d in config["resblock_dilation_sizes"]),
            mesh_device=device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        decoder.load_torch_state_dict(convert_minimax_h3_audio_state_dict(dict(reference.state_dict())), strict=False)

        for label, key, offset in CLIPS:
            src = load_clip(key, offset, num_samples)
            write_wav(os.path.join(OUT_DIR, f"{label}_0_source.wav"), src)

            with torch.no_grad():
                latents = reference.encode(src).latent_dist.mode()[..., :NUM_LATENT_FRAMES]
                t0 = time.perf_counter()
                cpu_out = reference.decode(latents).sample
                cpu_secs = time.perf_counter() - t0
            write_wav(os.path.join(OUT_DIR, f"{label}_1_cpu.wav"), cpu_out)

            # Warm with the same `traced` flag the timed runs use, so the trace is captured outside the
            # measurement rather than inside the first one.
            decoder(latents, traced=TRACED)
            runs = []
            for _ in range(3):
                t0 = time.perf_counter()
                dev_out = decoder(latents, traced=TRACED)
                runs.append(time.perf_counter() - t0)
            dev_secs = min(runs)
            write_wav(os.path.join(OUT_DIR, f"{label}_2_device{TAG}.wav"), dev_out)

            rows.append(
                (
                    label,
                    cpu_secs,
                    dev_secs,
                    psnr(cpu_out, dev_out),
                    rel_rmse(cpu_out, dev_out),
                    log_spec_distance(cpu_out, dev_out),
                )
            )
            print(f"done {label}: cpu {cpu_secs:.3f}s  device {dev_secs:.3f}s  psnr {rows[-1][3]:.2f} dB", flush=True)
    finally:
        if TRACED:
            try:
                decoder.release_trace()
            except Exception:  # nothing to release if the run died before capture
                pass
        ttnn.close_mesh_device(device)
        if MESH != (1, 1):
            # Leaving fabric enabled after close can wedge the next job's init.
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)

    print(f"\n=== fp32 decode, {num_samples / SR:.2f} s of 32 kHz audio per clip, batch 2 ===")
    print(f"config: mesh {MESH[0]}x{MESH[1]}, t_factor={T_FACTOR} axis={MESH_AXIS}, traced={TRACED}")
    print(f"{'clip':<15} {'CPU s':>7} {'device s':>9} {'speedup':>8} {'PSNR dB':>9} {'rel_rmse':>11} {'log-spec':>9}")
    print("-" * 74)
    for label, c, d, p, r, ls in rows:
        print(f"{label:<15} {c:>7.3f} {d:>9.3f} {c / d:>7.2f}x {p:>9.2f} {r:>11.3e} {ls:>9.4f}")
    if rows:
        print("-" * 74)
        n = len(rows)
        print(
            f"{'mean':<15} {sum(r[1] for r in rows) / n:>7.3f} {sum(r[2] for r in rows) / n:>9.3f} "
            f"{sum(r[1] for r in rows) / sum(r[2] for r in rows):>7.2f}x {sum(r[3] for r in rows) / n:>9.2f}"
        )
        mean_psnr = sum(r[3] for r in rows) / n
        mean_secs = sum(r[2] for r in rows) / n
        # The accuracy criterion is "no worse than the single-device path at the same levers", not a
        # fixed number: the decoder's constructed defaults are accurate mode (split_mode='full',
        # tap_matmul, prefer_mac), which scores far above the ~49.45 dB the fast defaults used to give.
        # So the baseline has to be *measured* on this branch -- run once without the CVD_* variables and
        # pass the mean back in via CVD_BASELINE_PSNR. Unset, this reports and asserts nothing on
        # accuracy, because a hard-coded bar from another configuration is worse than no bar.
        baseline_env = os.environ.get("CVD_BASELINE_PSNR")
        if baseline_env:
            baseline = float(baseline_env)
            drop = baseline - mean_psnr
            accuracy = (
                f"psnr {mean_psnr:.2f} dB vs {baseline:.2f} baseline "
                f"({'no degradation' if drop <= 0.05 else f'DOWN {drop:.2f} dB'}): "
                f"{'PASS' if drop <= 0.05 else 'FAIL'}"
            )
        else:
            accuracy = (
                f"psnr {mean_psnr:.2f} dB (no baseline set -- run single-device first, then pass CVD_BASELINE_PSNR)"
            )
        print(
            f"\nACCEPTANCE  {accuracy}"
            f"   |   latency {mean_secs * 1e3:.1f} ms vs 300 ms: {'PASS' if mean_secs <= 0.300 else 'FAIL'}"
            f" (222 ms stretch: {'PASS' if mean_secs <= 0.222 else 'FAIL'})"
        )
    print(f"\nWAVs in {OUT_DIR} (per clip: _0_source, _1_cpu, _2_device{TAG})")


if __name__ == "__main__":
    main()
