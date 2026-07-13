# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""LongCat-Image interactive server on a 4-chip QB2 — ALL weights resident at tp=4, nothing reloads.

Every stage is tensor-parallel across ALL 4 chips of one (1,4) FABRIC_1D_RING mesh, and every
stage stays resident — no dedicated chips, no weight reloads between requests:

    chips 0-3   text encoder tp=4 (RESIDENT) + DiT tp=4 (RESIDENT + traced) + VAE

This became possible once the Qwen2.5-VL text encoder was made tensor-parallel: its fp32 weights
(~28GB — bf16 caps its PCC at ~0.82) shard to ~7GB/chip, so it co-fits with the ~1.5GB/chip DiT
shard on every chip (a resident fp32 encoder on ONE chip fills it and can't co-reside with a tp=4
DiT shard — that's why the earlier 2-chip server, and the dedicated-encoder-chip tp=3 variant,
kept the encoder off the DiT's chips). Encoder TP numerically verified: last_hidden_state PCC
tp4-vs-tp1 = 0.9986.

Ordering: the resident encoder is uploaded BEFORE warmup() captures the DiT trace, so the trace
planner reserves around the resident encoder weights (allocating persistent buffers after trace
capture can corrupt the trace — same reason warmup warms the VAE pre-capture). The pipeline
auto-detects tp from get_num_devices on each device; resident_text_encoder=True keeps the encoder
resident even though it shares self.device with the DiT.

Measured warm e2e ~15.4s / 512px-50step image: encode ~0.57s + neg ~0.57s (resident) + tp=4 denoise
~14.0s (279 ms/step) + VAE ~0.31s. (~13% faster than the tp=3 dedicated-encoder-chip variant's
17.75s, because the DiT gets all 4 chips.)

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
from models.demos.vision.generative.longcat_image.tt.pipeline import build_text_input_ids

HF_MODEL_ID = "meituan-longcat/LongCat-Image"
OUTPUT_DIR = "outputs"
WARMUP_PROMPT = "a cat sitting on a mat"


def main():
    ap = argparse.ArgumentParser(description="LongCat-Image interactive server (tp=4 all-resident, 4 chips)")
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

    if ttnn.get_num_devices() != 4:
        raise SystemExit(f"demo_4chip needs exactly 4 chips; found {ttnn.get_num_devices()} (use demo_server.py for 2)")

    bar = "=" * 72
    # ONE coherent 4-chip ring mesh; encoder + DiT + VAE all tensor-parallel across it (tp=4).
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, ttnn.FabricReliabilityMode.RELAXED_INIT)
    mesh = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 4),
        l1_small_size=24576,
        num_command_queues=args.cq,
        trace_region_size=209715200,  # 200 MB for the resident denoise trace
    )
    ttp = None
    try:
        ttp = P.LongCatImagePipelineTT(
            mesh, pipe, num_cqs=args.cq, text_encoder_device=mesh, resident_text_encoder=True, profile=True
        )
        print(
            f"[server-4chip] DiT tp={ttp._device_tp()}  encoder tp={ttp._text_encoder_tp()}  "
            f"chips={mesh.get_device_ids()}",
            flush=True,
        )

        print(bar, flush=True)
        print(
            "Warming up (all 4 chips: text encoder tp=4 + DiT tp=4 + VAE resident) — one-time setup, "
            "not paid per request.",
            flush=True,
        )
        print(bar, flush=True)
        t0 = time.time()
        # Upload the RESIDENT tp=4 encoder FIRST (one throwaway forward) so warmup()'s DiT trace
        # capture reserves around the resident encoder weights rather than colliding with them.
        ids, mask, pre, suf = build_text_input_ids(pipe, WARMUP_PROMPT, args.max_length)
        te, _owned = ttp._acquire_text_encoder()
        ttp._tt_text_encode(ids, mask, pre, suf, stub=te)
        ttp.warmup(max_length=args.max_length, height=args.size, width=args.size, guidance_scale=args.guidance)
        print(f"  warmup() (encoder + DiT + VAE resident) took {time.time() - t0:.1f}s\n", flush=True)

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
        print(f"Generating a warmup image for {WARMUP_PROMPT!r} ...", flush=True)
        generate(WARMUP_PROMPT, 0)

        print(bar, flush=True)
        print("Ready. Type a prompt and press ENTER to generate (all stages warm/resident, tp=4).")
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
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
