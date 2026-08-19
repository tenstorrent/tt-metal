# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU reference vs device decode, fp32, scored on real clips.

Per clip: resample to 32 kHz, take one production window (207 latents = 5.17 s), encode with the
torch/diffusers reference, decode on CPU (ground truth) and on device, score device against CPU. WAVs
are written so the difference can be heard. Batch 2 throughout, matching the shipping working point,
so times are comparable to `decode_bench.py`. The decoder is built once, so times are steady-state.

Usage -- single device, shipping defaults:

    export MINIMAX_H3_MODEL_PATH=/path/to/MiniMax-H3-diffusers
    python models/tt_dit/tests/models/minimax_h3/cpu_vs_device.py

8-way T-shard on a 4x8 mesh with trace, all-fast levers (the sub-300 ms configuration):

    env CVD_MESH=4x8 CVD_T_FACTOR=8 CVD_MESH_AXIS=1 CVD_TRACED=1 \
        CVD_SPLIT_MODE=off CVD_TAP_MATMUL=0 CVD_PREFER_MAC=0 python .../cpu_vs_device.py

Env: CVD_MESH (default 1x1), CVD_T_FACTOR, CVD_MESH_AXIS, CVD_TRACED, CVD_SPLIT_MODE (off|weight|full),
CVD_TAP_MATMUL, CVD_PREFER_MAC, CVD_MAX_C_IN_BLOCK, CVD_OUT_DIR, CVD_BASELINE_PSNR.

Accuracy is scored against the CPU reference, i.e. an absolute number; the T-parallel test only scores
sharded against unsharded, which is a looser bar. Unset levers keep the constructor default (accurate
mode), so `=0` is not the same as omitting. Read each run's `levers:` line rather than the filename.
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
# Under `generated/` rather than any developer's filesystem, so this runs on CI and a fresh clone.
OUT_DIR = os.environ.get("CVD_OUT_DIR") or os.path.join(
    os.environ.get("TT_METAL_HOME", os.getcwd()), "generated", "minimax_h3_audio", "clips"
)
# `MINIMAX_H3_MODEL_PATH` matches the test suite; the older var is still accepted.
WEIGHTS = os.environ.get("MINIMAX_H3_MODEL_PATH") or os.environ.get("MINIMAX_H3_DIFFUSERS_DIR", "")
if not WEIGHTS:
    raise SystemExit("set MINIMAX_H3_MODEL_PATH to a MiniMax-H3 diffusers snapshot")

# T_FACTOR must equal the length of the axis it shards: on a 4x8 mesh only (4, axis 0) and
# (8, axis 1) work, anything else dies in `_partition_t` on a non-tile-aligned slice.
MESH = tuple(int(v) for v in os.environ.get("CVD_MESH", "1x1").split("x"))
T_FACTOR = int(os.environ.get("CVD_T_FACTOR", "1"))
MESH_AXIS = int(os.environ.get("CVD_MESH_AXIS", "1"))
TRACED = os.environ.get("CVD_TRACED", "0") == "1"


# The precision levers select a different operator set, not just a different speed: all-fast
# measures 292 ms / 45.80 dB where `prefer_mac` alone is 841 ms / 49.45 dB. Quoting one row's latency
# with another's PSNR is what produced the retired "281.6 ms at 49.45 dB" claim. `None` means "leave
# the constructor default alone".
def _flag(name):
    raw = os.environ.get(name)
    return None if raw is None else raw not in ("0", "false", "False", "")


SPLIT_MODE = os.environ.get("CVD_SPLIT_MODE")
TAP_MATMUL = _flag("CVD_TAP_MATMUL")
PREFER_MAC = _flag("CVD_PREFER_MAC")
# conv3d's C_in_block cap. Moves accuracy on its own -- conv_pre's error falls as the block grows
# (2.40e-03 at 32, 1.86e-03 at 128) -- but 256 buys 0.02 dB and 512 exceeds L1.
MAX_C_IN_BLOCK = os.environ.get("CVD_MAX_C_IN_BLOCK")
LEVERS = {
    k: v
    for k, v in (
        ("split_mode", SPLIT_MODE),
        ("tap_matmul", TAP_MATMUL),
        ("prefer_mac", PREFER_MAC),
        ("max_c_in_block", int(MAX_C_IN_BLOCK) if MAX_C_IN_BLOCK else None),
    )
    if v is not None
}

IS_DEFAULT = MESH == (1, 1) and T_FACTOR == 1 and not TRACED and not LEVERS
# Non-default runs get their config in the filename. Overwriting `{label}_2_device.wav` with a
# differently-configured decode is how the stale `*_3_device_prefix.wav` confusion started.
_lever_tag = "".join(
    f"_{k}-{v}"
    for k, v in (("sm", SPLIT_MODE), ("tap", TAP_MATMUL), ("mac", PREFER_MAC), ("cin", MAX_C_IN_BLOCK))
    if v is not None
)
TAG = "" if IS_DEFAULT else f"_{MESH[0]}x{MESH[1]}_f{T_FACTOR}ax{MESH_AXIS}{'_traced' if TRACED else ''}{_lever_tag}"

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

        if LEVERS:
            print(f"levers: {LEVERS} (unset ones keep the constructor default)", flush=True)
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
            **LEVERS,
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
        # tap_matmul, prefer_mac) and score ~67 dB, where all-fast scores ~45.8 dB. A bar that does not
        # name its lever set is meaningless across that 21 dB spread.
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
