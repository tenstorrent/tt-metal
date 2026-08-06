"""What does the fp32 mandate actually cost, now that precision is fixed?

The whole audio decode runs fp32 because the vocoder had a precision problem. That problem is now
fixed at its source (SFPU tap accumulation + UnpackToDestFp32 operands make conv1d bit-equal to the
exact elementwise form), and the default path sits at 42.86 dB against a 28 dB gate. That is a large
unspent precision budget, and this stage is bandwidth-bound -- bf16 halves every byte moved.

Decodes the real 207-latent clip at each dtype, scores against the fp32 `MINIMAX_H3_AUDIO_ACCURATE=1`
output, and reports seconds. Each config runs in its own subprocess because the conv blockings and
prepared weights are process-global.
"""

import json
import os
import subprocess
import sys

import torch

OUT = "/home/rshirvani/.claude/jobs/00644216/tmp/dtype"
FRAMES = int(os.environ.get("DT_FRAMES", "207"))

CONFIGS = {
    "golden_fp32_accurate": ("float32", {"MINIMAX_H3_AUDIO_ACCURATE": "1"}),
    "fp32_default": ("float32", {}),
    "bf16_default": ("bfloat16", {}),
}


def run_one(label: str) -> None:
    import time

    import ttnn
    from safetensors.torch import load_file

    from models.tt_dit.models.audio_vae.minimax_h3.convert_minimax_h3_audio import (
        convert_minimax_h3_audio_state_dict,
    )
    from models.tt_dit.models.audio_vae.minimax_h3.decoder_minimax_h3_audio import MiniMaxH3AudioDecoder

    dtype_name, _ = CONFIGS[label]
    dtype = {"float32": ttnn.float32, "bfloat16": ttnn.bfloat16}[dtype_name]

    wd = os.path.join(os.environ["MINIMAX_H3_DIFFUSERS_DIR"], "audio_vae")
    cfg = {k: v for k, v in json.load(open(os.path.join(wd, "config.json"))).items() if not k.startswith("_")}
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), l1_small_size=65536)
    try:
        dec = MiniMaxH3AudioDecoder(
            latent_channels=cfg["latent_channels"],
            latent_dim=cfg["latent_dim"],
            decoder_dim=cfg["decoder_dim"],
            decoder_rates=tuple(cfg["decoder_rates"]),
            decoder_kernel_sizes=tuple(cfg["decoder_kernel_sizes"]),
            resblock_kernel_sizes=tuple(cfg["resblock_kernel_sizes"]),
            resblock_dilation_sizes=tuple(tuple(d) for d in cfg["resblock_dilation_sizes"]),
            mesh_device=device,
            dtype=dtype,
        )
        dec.load_torch_state_dict(
            convert_minimax_h3_audio_state_dict(load_file(os.path.join(wd, "diffusion_pytorch_model.safetensors"))),
            strict=False,
        )
        torch.manual_seed(2)
        latents = torch.randn(2, cfg["latent_channels"], FRAMES) * 0.1

        dec(latents)
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = dec(latents)
        ttnn.synchronize_device(device)
        elapsed = time.perf_counter() - t0

        os.makedirs(OUT, exist_ok=True)
        torch.save({"wav": out.float().cpu(), "seconds": elapsed}, os.path.join(OUT, f"{label}.pt"))
        print(f"{label}: {elapsed:.3f} s")
    finally:
        ttnn.close_mesh_device(device)


def psnr(ref, test):
    mse = float((ref - test).pow(2).mean())
    return (
        float("inf")
        if mse == 0
        else 20 * torch.log10(ref.abs().max()).item() - 10 * torch.log10(torch.tensor(mse)).item()
    )


def main():
    if len(sys.argv) > 1:
        run_one(sys.argv[1])
        return
    for label, (_, env) in CONFIGS.items():
        e = dict(os.environ)
        for k in (
            "MINIMAX_H3_AUDIO_ACCURATE",
            "MINIMAX_H3_AUDIO_CONV1D_L1",
            "MINIMAX_H3_AUDIO_DEPTHWISE_MAC",
            "MINIMAX_H3_AUDIO_CONV_SPLIT",
            "MINIMAX_H3_AUDIO_TAP_MATMUL",
            "MINIMAX_H3_AUDIO_FUSE_BAND",
            "MINIMAX_H3_AUDIO_DEPTHWISE_SPLIT",
        ):
            e.pop(k, None)
        e.update(env)
        print(f"--- {label} {env}", flush=True)
        r = subprocess.run([sys.executable, __file__, label], env=e, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(f"{label} FAILED rc={r.returncode}")
            print(r.stdout[-800:])
            print(r.stderr[-1500:])

    gp = os.path.join(OUT, "golden_fp32_accurate.pt")
    if not os.path.exists(gp):
        print("no golden; cannot score")
        return
    ref = torch.load(gp)["wav"].double()
    print(f"\n{'config':<24} {'seconds':>8} {'rel_rmse':>11} {'PSNR dB':>9}")
    print("-" * 56)
    for label in CONFIGS:
        p = os.path.join(OUT, f"{label}.pt")
        if not os.path.exists(p):
            print(f"{label:<24} {'MISSING':>8}")
            continue
        b = torch.load(p)
        got = b["wav"].double()
        if got.shape != ref.shape:
            print(f"{label:<24} {b['seconds']:>8.3f}   shape {tuple(got.shape)} != {tuple(ref.shape)}")
            continue
        err = float((got - ref).pow(2).mean().sqrt() / ref.std())
        print(f"{label:<24} {b['seconds']:>8.3f} {err:>11.3e} {psnr(ref, got):>9.2f}")


if __name__ == "__main__":
    main()
