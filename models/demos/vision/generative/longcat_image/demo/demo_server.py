# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LongCat-Image interactive server — REPL for generating images from text prompts,
keeping the DiT + VAE resident and traced across requests (Phase 0 of the QB2 porting
plan: warm-server throughput, not the one-shot-per-process pattern of demo_text_to_image.py).

Needs TWO chips: measured on real hardware, a resident (warm) DiT (~12.5GB) plus a full
text-encoder pass (~26-28GB fp32) do not fit together in one chip's ~34GB DRAM — not a
close call, the encoder was still partially loaded when the chip ran out of space. So the
DiT + VAE stay resident + warmed on one chip (LongCatImagePipelineTT.warmup()), and the
text encoder gets the SECOND chip to itself — where it, too, now stays resident across
requests (its ~26-28GB fp32 weights upload once on the first request and are reused for
every later request's pos AND neg branch, instead of the old build+upload+free per branch
per request; see LongCatImagePipelineTT._acquire_text_encoder). So after the first request
BOTH stages are warm: only per-request compute is paid, no weight reloads.

On a QB2 (multi-chip, ethernet-fabric-connected Blackhole) box this pair of chips is opened
as a genuine 1x2 `ttnn.MeshDevice` (`ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 2),
physical_device_ids=[device_id, text_encoder_device_id])`), then split into two independently
addressable 1x1 submeshes via `MeshDevice.create_submesh` — one per role. This is a
heterogeneous split (each submesh runs a DIFFERENT stage of the SAME pipeline, not a
replicated/sharded copy of one model), unlike this codebase's other mesh usage
(data/tensor-parallel replication across a mesh). We moved off two independent
`ttnn.open_device()` calls (the original Phase 0 shape of this file) because opening
individual chips out of a fabric-connected QB2 cluster one at a time is explicitly flagged
by tt-metal itself ("Opening subset of mmio devices slows down UMD read/write to remote
chips... consider using CreateDevices API") and was observed hanging on real hardware;
`open_mesh_device` opens the whole local cluster coherently in one call. `_tt_text_encode`
still just hands off a plain host tensor to the denoise stage — no CCL/tensor-mapper
machinery runs across the two submeshes, only independent single-chip compute on each.

Run:
    python -m models.demos.vision.generative.longcat_image.demo.demo_server \
        --steps 24 --size 512 --max_length 512
#   --cq 2  runs the resident DiT trace (and any cold-path fallback) under trace+2CQ
#           (queue-1-staged per-step inputs); honest note: ~0 win over trace+1CQ for this
#           workload (see _run_traced_steps_2cq's docstring in tt/pipeline.py) — wired for
#           parity with the other demos' --cq flag, not for an expected speedup.
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
    ap = argparse.ArgumentParser(description="LongCat-Image interactive server (warm DiT+VAE, 2 chips)")
    ap.add_argument("--steps", type=int, default=50)  # HF reference default
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--guidance", type=float, default=4.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--text_encoder_device_id",
        type=int,
        default=1,
        help="chip id for the (cold, per-request) text encoder; must differ from --device_id",
    )
    ap.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="chip id for the resident (warm) DiT + VAE",
    )
    ap.add_argument(
        "--cq",
        type=int,
        default=1,
        choices=[1, 2],
        help="command queues per chip; 2 runs the resident trace (and cold fallback) under "
        "trace+2CQ (mirrors demo_text_to_image.py's --cq)",
    )
    args = ap.parse_args()
    if args.text_encoder_device_id == args.device_id:
        raise SystemExit("--text_encoder_device_id must differ from --device_id (see module docstring)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        import readline  # noqa: F401  -- enables history/editing inside input()
    except ImportError:
        pass

    from diffusers import LongCatImagePipeline

    print(f"[server] loading {HF_MODEL_ID} (bf16) ...", flush=True)
    pipe = LongCatImagePipeline.from_pretrained(HF_MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    pipe.set_progress_bar_config(disable=True)
    pipe.tokenizer_max_length = args.max_length

    bar = "=" * 72
    # Open the two chips as ONE coherent 1x2 mesh (not two independent ttnn.open_device() calls —
    # see the module docstring for why: opening individual chips out of a fabric-connected QB2
    # cluster one at a time is flagged by tt-metal as slow/fragile and was observed hanging on
    # real hardware), then carve it into two independently-addressable 1x1 submeshes, one per role.
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 2),
        l1_small_size=24576,
        num_command_queues=args.cq,
        physical_device_ids=[args.device_id, args.text_encoder_device_id],
    )
    try:
        device = mesh.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 0))
        text_encoder_device = mesh.create_submesh(ttnn.MeshShape(1, 1), offset=ttnn.MeshCoordinate(0, 1))
    except Exception:
        ttnn.close_mesh_device(mesh)
        raise
    print(
        f"[server] mesh submesh device ids: dit_vae(chip {args.device_id})={device.get_device_ids()} "
        f"text_encoder(chip {args.text_encoder_device_id})={text_encoder_device.get_device_ids()}",
        flush=True,
    )
    ttp = None
    try:
        ttp = P.LongCatImagePipelineTT(
            device, pipe, text_encoder_device=text_encoder_device, num_cqs=args.cq, profile=True
        )

        print(bar, flush=True)
        print(
            f"Warming up (chip {args.device_id}: DiT+VAE resident; chip {args.text_encoder_device_id}: "
            f"text encoder, resident after the first request) — one-time setup, not paid per request.",
            flush=True,
        )
        print(bar, flush=True)
        t0 = time.time()
        ttp.warmup(max_length=args.max_length, height=args.size, width=args.size, guidance_scale=args.guidance)
        print(f"  warmup() took {time.time() - t0:.1f}s\n", flush=True)

        print(bar, flush=True)
        print(f"Generating a warmup image for {WARMUP_PROMPT!r} ...", flush=True)
        t0 = time.time()
        result = ttp.run_text_to_image(
            prompt=WARMUP_PROMPT,
            height=args.size,
            width=args.size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            max_length=args.max_length,
        )
        path = os.path.join(OUTPUT_DIR, "out_0.png")
        _save_png(result["image_denorm"], path)
        print(f"  END-TO-END: {time.time() - t0:.2f}s  |  {os.path.abspath(path)}\n", flush=True)

        print(bar, flush=True)
        print("Ready. Type a prompt and press ENTER to generate.")
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
        ttnn.close_mesh_device(text_encoder_device)
        ttnn.close_mesh_device(device)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
