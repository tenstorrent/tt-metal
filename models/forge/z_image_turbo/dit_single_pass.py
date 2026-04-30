#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Single DIT forward pass — no tracing, for perf tuning.

Loads the text encoder to produce real caption features, then runs the DIT
forward pass repeatedly, printing per-iteration latency.

Usage (from z_image_turbo directory):
    python dit_single_pass.py
    python dit_single_pass.py --runs 10
    python dit_single_pass.py --prompt "a robot painting"
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from dit.model_ttnn import ZImageTransformerTTNN
from text_encoder.model_ttnn import TextEncoderTTNN
from transformers import AutoTokenizer

import ttnn

HERE = os.path.dirname(os.path.abspath(__file__))
REF_OUTPUT_PATH = os.path.join(HERE, "reference_output.pt")


def compute_pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def compute_metrics(a, b):
    a_f = a.float()
    b_f = b.float()
    diff = (a_f - b_f).abs()
    rel = diff / (b_f.abs() + 1e-10)
    return {
        "pcc": compute_pcc(a, b),
        "max_abs_diff": diff.max().item(),
        "max_rel_diff": rel.max().item(),
        "mean_abs_diff": diff.mean().item(),
    }


MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CAP_TOKENS = 128
IMG_LATENT_H = 64
IMG_LATENT_W = 64
LATENT_CHANNELS = 16

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)


def open_mesh_device():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 4)),
        l1_small_size=1 << 15,
        trace_region_size=0,
    )
    device.enable_program_cache()
    return device


def _to_device_bf16(pt, mesh_device):
    return ttnn.from_torch(
        pt.bfloat16(),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _to_device_int32(pt, mesh_device):
    return ttnn.from_torch(
        pt.to(torch.int32),
        dtype=ttnn.DataType.INT32,
        layout=ttnn.Layout.ROW_MAJOR,
        device=mesh_device,
        memory_config=DRAM_RM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _tt_to_torch(tt_tensor, mesh_device):
    host = ttnn.to_torch(
        ttnn.from_device(tt_tensor),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )
    return host[: host.shape[0] // 4].float()


def encode_prompt(te, tokenizer, mesh_device, prompt):
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(formatted, padding="max_length", truncation=True, max_length=CAP_TOKENS, return_tensors="pt")[
        "input_ids"
    ]
    tt_ids = _to_device_int32(input_ids, mesh_device)
    tt_out = te(tt_ids)
    cap_cpu = _tt_to_torch(tt_out, mesh_device)[:CAP_TOKENS].bfloat16()
    ttnn.deallocate(tt_ids, force=True)
    ttnn.deallocate(tt_out, force=True)
    return cap_cpu.unsqueeze(0)  # [1, CAP_TOKENS, 2560]


def main():
    parser = argparse.ArgumentParser(description="DIT single-pass perf benchmark (no trace)")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed iterations")
    parser.add_argument("--prompt", type=str, default="a beautiful sunset over the ocean")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Opening mesh device ...")
    mesh_device = open_mesh_device()

    print("Loading text encoder ...")
    te = TextEncoderTTNN(mesh_device, seq_len=CAP_TOKENS)

    print("Loading DIT ...")
    dit = ZImageTransformerTTNN(mesh_device)

    # Encode a real prompt for caption features
    print("Encoding prompt ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
    cap_cpu = encode_prompt(te, tokenizer, mesh_device, args.prompt)
    dit.set_cap_feats(cap_cpu)

    # Prepare latent + timestep inputs
    torch.manual_seed(args.seed)
    latent_pt = torch.randn(1, LATENT_CHANNELS, IMG_LATENT_H, IMG_LATENT_W)
    lat_pt = latent_pt.squeeze(0).unsqueeze(1).bfloat16()  # [16, 1, 64, 64]
    t_norm = 0.5
    ts_pt = torch.tensor([t_norm], dtype=torch.bfloat16)  # [1]

    # ── Compile run ───────────────────────────────────────────────────────────
    print("\n[1/3] Compile run ...")
    tt_lat = _to_device_bf16(lat_pt, mesh_device)
    tt_ts = _to_device_bf16(ts_pt, mesh_device)
    t0 = time.time()
    out = dit([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    print(f"  Compile: {(time.time() - t0) * 1000:.0f} ms")
    for t in out:
        ttnn.deallocate(t, force=True)
    ttnn.deallocate(tt_lat, force=True)
    ttnn.deallocate(tt_ts, force=True)

    # ── Warm run (programs cached) ────────────────────────────────────────────
    print("[2/3] Warm run ...")
    tt_lat = _to_device_bf16(lat_pt, mesh_device)
    tt_ts = _to_device_bf16(ts_pt, mesh_device)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    out = dit([tt_lat], tt_ts)
    ttnn.synchronize_device(mesh_device)
    warm_ms = (time.perf_counter() - t0) * 1000
    print(f"  Warm: {warm_ms:.1f} ms")
    for t in out:
        ttnn.deallocate(t, force=True)
    ttnn.deallocate(tt_lat, force=True)
    ttnn.deallocate(tt_ts, force=True)

    # ── Timed runs ────────────────────────────────────────────────────────────
    print(f"[3/3] Running {args.runs} timed iterations ...")
    timings = []
    last_out = None
    for i in range(args.runs):
        tt_lat = _to_device_bf16(lat_pt, mesh_device)
        tt_ts = _to_device_bf16(ts_pt, mesh_device)
        ttnn.synchronize_device(mesh_device)

        t0 = time.perf_counter()
        out = dit([tt_lat], tt_ts)
        ttnn.synchronize_device(mesh_device)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        timings.append(elapsed_ms)
        print(f"  Run {i + 1}: {elapsed_ms:.1f} ms")

        if i < args.runs - 1:
            for t in out:
                ttnn.deallocate(t, force=True)
        else:
            last_out = out
        ttnn.deallocate(tt_lat, force=True)
        ttnn.deallocate(tt_ts, force=True)

    # ── PCC check ─────────────────────────────────────────────────────────────
    if last_out is not None:
        out_torch = _tt_to_torch(last_out[0], mesh_device)
        for t in last_out:
            ttnn.deallocate(t, force=True)

        if not os.path.exists(REF_OUTPUT_PATH):
            torch.save(out_torch, REF_OUTPUT_PATH)
            print(f"\nSaved reference output to {REF_OUTPUT_PATH}")
            print(f"PCC: 1.0000 (reference)")
        else:
            ref = torch.load(REF_OUTPUT_PATH, weights_only=True)
            m = compute_metrics(out_torch, ref)
            print(f"\nPCC: {m['pcc']:.6f}")
            print(f"Max abs diff: {m['max_abs_diff']:.6f}")
            print(f"Max rel diff: {m['max_rel_diff']:.6f}")
            print(f"Mean abs diff: {m['mean_abs_diff']:.6f}")

    # Summary
    avg = sum(timings) / len(timings)
    best = min(timings)
    worst = max(timings)
    print(f"\nSummary ({args.runs} runs):")
    print(f"  Warm:  {warm_ms:.1f} ms")
    print(f"  Avg:   {avg:.1f} ms")
    print(f"  Best:  {best:.1f} ms")
    print(f"  Worst: {worst:.1f} ms")

    ttnn.close_mesh_device(mesh_device)
    print("\nDone.")


if __name__ == "__main__":
    main()
