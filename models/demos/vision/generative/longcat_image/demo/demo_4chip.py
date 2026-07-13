# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LongCat-Image interactive server on a 4-chip QB2 — ALL weights resident, nothing reloads.

This is the 4-chip generalization of demo_server.py's warm 2-chip server. The one (1,4)
FABRIC_1D_RING mesh is carved into two non-overlapping submeshes, one per role:

    chip 3        text encoder (fp32, ~28GB) — RESIDENT ((1,1) submesh)
    chips 0,1,2   DiT tensor-parallel tp=3 + VAE — RESIDENT + traced ((1,3) submesh)

Why this split (and not tp=4): the Qwen2.5-VL text encoder runs fp32 weights (~28GB — bf16
caps its PCC at ~0.82 over the 28-layer stack) and, on its own, fills a ~32GB chip to the
edge. It cannot co-reside with a resident DiT shard, so it gets a chip to itself and the DiT
tensor-parallels across the other three (tp=3 divides cleanly: 24 heads/3 = 8, FFN 12288/3 =
4096). The pipeline auto-detects tp from the DiT submesh (get_num_devices) — no caller flag.

After warmup() + the first request (which uploads the resident encoder once), EVERY stage is
resident: a request pays only compute (encode forward + traced tp=3 denoise + VAE), no weight
reloads. Measured warm e2e ~17.8s / 512px-50step image (encode ~1.5s + denoise ~15.9s [318
ms/step] + VAE ~0.3s).

Run:
    python -m models.demos.vision.generative.longcat_image.demo.demo_4chip \
        --steps 50 --size 512 --max_length 512 --cq 2
"""

from __future__ import annotations

import argparse
import os
import time

import torch

import ttnn
from models.demos.vision.generative.longcat_image.demo._demo_common import save_png as _save_png
from models.demos.vision.generative.longcat_image.tt import pipeline as P

HF_MODEL_ID = "meituan-longcat/LongCat-Image"
OUTPUT_DIR = "outputs"
WARMUP_PROMPT = "a cat sitting on a mat"


def main():
    ap = argparse.ArgumentParser(description="LongCat-Image interactive server (all-resident, 4 chips)")
    ap.add_argument("--steps", type=int, default=50)  # HF reference default
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--cq",
        type=int,
        default=2,
        choices=[1, 2],
        help="command queues per chip; 2 runs the resident DiT trace under trace+2CQ",
    )
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        import readline  # noqa: F401  -- enables history/editing inside input()
    except ImportError:
        pass

    from diffusers import LongCatImagePipeline

    print(f"[server-4chip] loading {HF_MODEL_ID} (bf16) ...", flush=True)
    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.tokenizer_max_length = args.max_length

    if ttnn.get_num_devices() < 4:
        raise SystemExit(f"demo_4chip needs 4 chips; found {ttnn.get_num_devices()} (use demo_server.py for 2)")

    bar = "=" * 72
    # One coherent 4-chip ring mesh, carved into two non-overlapping submeshes (see module
    # docstring): a 1x1 for the resident fp32 encoder, a 1x3 for the tensor-parallel DiT + VAE.
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.RELAXED_INIT)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=24576,
        num_command_queues=args.cq,
        trace_region_size=209715200,  # 200 MB for the resident denoise trace
    )
    text_encoder_device = dit_device = None
    try:
        text_encoder_device = mesh.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 0))
        dit_device = mesh.create_submesh(ttnn.MeshShape(1, 3), offset=ttnn.MeshCoordinate(0, 1))
    except Exception:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        raise
    print(
        f"[server-4chip] text_encoder chip={text_encoder_device.get_device_ids()}  "
        f"dit+vae tp={dit_device.get_num_devices()} chips={dit_device.get_device_ids()}",
        flush=True,
    )

    ttp = None
    try:
        ttp = P.LongCatImagePipelineTT(
            dit_device, pipe, text_encoder_device=text_encoder_device, num_cqs=args.cq, profile=True
        )

        print(bar, flush=True)
        print(
            "Warming up (chips 0-2: DiT tp=3 + VAE resident + traced; chip 3: text encoder, "
            "resident after the first request) — one-time setup, not paid per request.",
            flush=True,
        )
        print(bar, flush=True)
        t0 = time.time()
        ttp.warmup(max_length=args.max_length, height=args.size, width=args.size, guidance_scale=args.guidance)
        print(f"  warmup() took {time.time() - t0:.1f}s\n", flush=True)

        def generate(prompt, idx):
            t0 = time.time()
            result = ttp.run_text_to_image(
                prompt=prompt,
                height=args.size,
                width=args.size,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                seed=args.seed + idx,
                max_length=args.max_length,
            )
            path = os.path.join(OUTPUT_DIR, f"out_{idx}.png")
            _save_png(result["image_denorm"], path)
            elapsed = time.time() - t0
            print(f"  END-TO-END: {elapsed:.2f}s  |  {os.path.abspath(path)}", flush=True)
            if ttp.profile.enabled:
                for label, secs in sorted(ttp.profile.timings.items(), key=lambda kv: -kv[1]):
                    print(f"    {label:24s} {secs * 1000:9.1f} ms", flush=True)

        print(bar, flush=True)
        print(
            f"Generating a warmup image for {WARMUP_PROMPT!r} (first request uploads the resident encoder) ...",
            flush=True,
        )
        generate(WARMUP_PROMPT, 0)

        print(bar, flush=True)
        print("Ready. Type a prompt and press ENTER to generate (all stages warm/resident now).")
        print("Ctrl-D or Ctrl-C to exit.")
        print(bar, flush=True)

        idx = 1
        while True:
            try:
                prompt = input(f"\nprompt [{idx}]> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break
            if not prompt:
                continue
            try:
                generate(prompt, idx)
                idx += 1
            except KeyboardInterrupt:
                print("\n  Interrupted during generation; shutting down.")
                break
            except Exception as exc:  # noqa: BLE001
                print(f"\n  ERROR: {exc}\n", flush=True)
    finally:
        if ttp is not None:
            ttp.close()
        # Close submeshes before the parent mesh they were carved from.
        if dit_device is not None:
            ttnn.close_mesh_device(dit_device)
        if text_encoder_device is not None:
            ttnn.close_mesh_device(text_encoder_device)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
