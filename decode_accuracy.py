"""Does the L1 routing change (K1) cost accuracy on the *default* decode path?

K1 lets shapes reach `ttnn.conv1d` that previously failed the DRAM slice search and fell back to the
exact MAC form, so it can trade accuracy for speed without saying so. There is no torch reference
available in this environment (the installed diffusers has no `AutoencoderKLMiniMaxH3Audio`), so the
golden here is the decoder's own `MINIMAX_H3_AUDIO_ACCURATE=1` output, which STATE.md am. 113 measures
at 0.45 % rel RMSE / 67.5 dB PSNR against torch -- roughly 25x better than the default path's 10.46 %,
so it resolves a change in the default path even though it is not exact itself.

Each configuration runs in its own subprocess: the audio env knobs are read at call time but the conv
blockings and cached prepared weights are process-global, and a config change mid-process would be
scored against the previous config's cache.
"""

import json
import os
import subprocess
import sys

import torch

CONFIGS = {
    # label: env overrides on top of the default path
    "golden_accurate": {"MINIMAX_H3_AUDIO_ACCURATE": "1"},
    "default_noK1": {"MINIMAX_H3_AUDIO_CONV1D_L1": "off"},
    # `default_K1` was captured when `auto` meant "L1 first, fall back" -- i.e. today's `aggressive`.
    "default_K1": {"MINIMAX_H3_AUDIO_CONV1D_L1": "aggressive"},
    "l1_safe": {"MINIMAX_H3_AUDIO_CONV1D_L1": "safe"},
    # Splitting is deliberately not scored end to end: measured per shape it plateaus ~6e-04 against
    # MAC's ~5e-08 while costing 2.5-6x, so it cannot substitute for the SFPU fix.
}

OUT_DIR = "/home/rshirvani/.claude/jobs/00644216/tmp/acc"
FRAMES = int(os.environ.get("ACC_FRAMES", "207"))


def run_one(label: str) -> None:
    """Child mode: build the decoder, decode a fixed latent, save the waveform."""
    import time

    import ttnn
    from safetensors.torch import load_file

    from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import (
        convert_minimax_h3_audio_state_dict,
    )
    from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder

    weights_dir = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
    config = {
        k: v for k, v in json.load(open(os.path.join(weights_dir, "config.json"))).items() if not k.startswith("_")
    }

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
        decoder.load_torch_state_dict(
            convert_minimax_h3_audio_state_dict(
                load_file(os.path.join(weights_dir, "diffusion_pytorch_model.safetensors"))
            ),
            strict=False,
        )
        torch.manual_seed(2)
        latents = torch.randn(2, config["latent_channels"], FRAMES) * 0.1

        out = decoder(latents)  # warm: compile + prepared-weight cache
        ttnn.synchronize_device(device)
        start = time.perf_counter()
        out = decoder(latents)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - start

        wav = out if isinstance(out, torch.Tensor) else torch.as_tensor(out)
        os.makedirs(OUT_DIR, exist_ok=True)
        torch.save({"wav": wav.float().cpu(), "seconds": elapsed}, os.path.join(OUT_DIR, f"{label}.pt"))
        print(f"{label}: {elapsed:.3f} s, shape {tuple(wav.shape)}")
    finally:
        ttnn.close_mesh_device(device)


def psnr(ref: torch.Tensor, test: torch.Tensor) -> float:
    peak = float(ref.abs().max())
    mse = float((ref - test).pow(2).mean())
    if mse == 0:
        return float("inf")
    return 20.0 * torch.log10(torch.tensor(peak)).item() - 10.0 * torch.log10(torch.tensor(mse)).item()


def main() -> None:
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
        return

    for label, env in CONFIGS.items():
        if os.path.exists(os.path.join(OUT_DIR, f"{label}.pt")) and os.environ.get("ACC_REUSE", "1") == "1":
            print(f"--- {label}: reusing cached run", flush=True)
            continue
        child_env = dict(os.environ)
        # Clear every knob first so one config cannot inherit another's setting.
        for key in (
            "MINIMAX_H3_AUDIO_ACCURATE",
            "MINIMAX_H3_AUDIO_CONV1D_L1",
            "MINIMAX_H3_AUDIO_DEPTHWISE_SPLIT",
            "MINIMAX_H3_AUDIO_DEPTHWISE_MAC",
            "MINIMAX_H3_AUDIO_CONV_SPLIT",
            "MINIMAX_H3_AUDIO_TAP_MATMUL",
        ):
            child_env.pop(key, None)
        child_env.update(env)
        print(f"--- {label}: {env}", flush=True)
        result = subprocess.run(
            [sys.executable, __file__, label], env=child_env, capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            print(f"{label} FAILED rc={result.returncode}")
            print(result.stdout[-1500:])
            print(result.stderr[-1500:])

    golden_path = os.path.join(OUT_DIR, "golden_accurate.pt")
    if not os.path.exists(golden_path):
        print("no golden produced; cannot score")
        return
    golden = torch.load(golden_path)
    ref = golden["wav"].double()

    print()
    print(f"{'config':<18} {'seconds':>8} {'rel_rmse':>11} {'PSNR dB':>9}   (vs MINIMAX_H3_AUDIO_ACCURATE=1)")
    print("-" * 74)
    print(f"{'golden_accurate':<18} {golden['seconds']:>8.3f} {'-':>11} {'-':>9}")
    for label in CONFIGS:
        if label == "golden_accurate":
            continue
        path = os.path.join(OUT_DIR, f"{label}.pt")
        if not os.path.exists(path):
            print(f"{label:<18} {'-':>8} {'MISSING':>11}")
            continue
        blob = torch.load(path)
        got = blob["wav"].double()
        err = float((got - ref).pow(2).mean().sqrt() / ref.std())
        print(f"{label:<18} {blob['seconds']:>8.3f} {err:>11.3e} {psnr(ref, got):>9.2f}")


if __name__ == "__main__":
    main()
