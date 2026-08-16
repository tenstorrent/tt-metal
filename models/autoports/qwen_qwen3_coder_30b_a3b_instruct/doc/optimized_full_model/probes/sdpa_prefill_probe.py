# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does the *prefill* SDPA have the same missing-program-config problem?

``attention_prefill`` calls ``ttnn.transformer.scaled_dot_product_attention(q,
k, v, is_causal=True)`` with no program config
(``tt/functional_decoder.py:441``), exactly the shape of the decode-path gap
stage 06 found. This probe answers it rather than assuming, on the shipped
per-die prefill shapes (TP=4: 8 Q heads, 1 KV head, head_dim 128, bfloat16), and
it deliberately includes **non-tile-aligned sequence lengths**, because the
stage contract requires arbitrary prompt lengths to keep working and a chunked
program config is exactly the kind of thing that would quietly demand alignment.

    python sdpa_prefill_probe.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

import ttnn

HERE = Path(__file__).resolve().parent
N_Q_HEADS, N_KV_HEADS, HEAD_DIM = 8, 1, 128
SEQS = [128, 1024, 4096, 16384]
NON_ALIGNED = [100, 1000, 4095]
CHUNKS = [(128, 128), (256, 256), (512, 512), (128, 512), (512, 128), (32, 512)]


def mk(mesh, seq, seed=3):
    torch.manual_seed(seed)
    mk1 = lambda shape: ttnn.from_torch(  # noqa: E731
        torch.randn(shape).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return mk1((1, N_Q_HEADS, seq, HEAD_DIM)), mk1((1, N_KV_HEADS, seq, HEAD_DIM)), mk1((1, N_KV_HEADS, seq, HEAD_DIM))


def call(q, k, v, pc):
    return ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, program_config=pc)


def bench(mesh, q, k, v, pc, iters=10, blocks=3):
    o = call(q, k, v, pc)
    ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    out = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        for _ in range(iters):
            o = call(q, k, v, pc)
            ttnn.deallocate(o)
        ttnn.synchronize_device(mesh)
        out.append(1e6 * (time.perf_counter() - t0) / iters)
    return statistics.median(out)


def ref_pcc(mesh, q, k, v, pc):
    got = ttnn.to_torch(ttnn.get_device_tensors(call(q, k, v, pc))[0]).float()
    qt = ttnn.to_torch(ttnn.get_device_tensors(q)[0]).float()[0]
    kt = ttnn.to_torch(ttnn.get_device_tensors(k)[0]).float()[0]
    vt = ttnn.to_torch(ttnn.get_device_tensors(v)[0]).float()[0]
    rep = N_Q_HEADS // N_KV_HEADS
    kt, vt = kt.repeat_interleave(rep, 0), vt.repeat_interleave(rep, 0)
    s = qt @ kt.transpose(-1, -2) * (HEAD_DIM**-0.5)
    n = s.shape[-1]
    s = s + torch.triu(torch.full((n, n), float("-inf")), 1)
    exp = s.softmax(-1) @ vt
    a, b = got.reshape(-1), exp.reshape(-1)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    grid = mesh.compute_with_storage_grid_size()
    results = []
    try:
        for seq in SEQS:
            q, k, v = mk(mesh, seq)
            legs = [("None", None)] + [
                (
                    f"q{a}/k{b}",
                    ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=a, k_chunk_size=b),
                )
                for a, b in CHUNKS
            ]
            base = None
            print(f"\n=== prefill seq {seq} ===", flush=True)
            for name, pc in legs:
                try:
                    us = bench(mesh, q, k, v, pc)
                    p = ref_pcc(mesh, q, k, v, pc)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {name:<12} FAILED: {str(exc).splitlines()[0][:100]}", flush=True)
                    results.append({"seq": seq, "cfg": name, "error": str(exc).splitlines()[0][:200]})
                    continue
                if base is None:
                    base = us
                print(f"  {name:<12} {us:10.2f} us  {base/us:6.2f}x  PCC {p:.6f}", flush=True)
                results.append({"seq": seq, "cfg": name, "us": us, "speedup": base / us, "pcc": p})
            for t in (q, k, v):
                ttnn.deallocate(t)

        print("\n=== non-tile-aligned sequence lengths ===", flush=True)
        for seq in NON_ALIGNED:
            for name, pc in (
                ("None", None),
                (
                    "q512/k512",
                    ttnn.SDPAProgramConfig(compute_with_storage_grid_size=grid, q_chunk_size=512, k_chunk_size=512),
                ),
            ):
                try:
                    q, k, v = mk(mesh, seq)
                    p = ref_pcc(mesh, q, k, v, pc)
                    print(f"  seq {seq:<6} {name:<12} OK   PCC {p:.6f}", flush=True)
                    results.append({"seq": seq, "cfg": name, "aligned": False, "pcc": p})
                    for t in (q, k, v):
                        ttnn.deallocate(t)
                except Exception as exc:  # noqa: BLE001
                    print(f"  seq {seq:<6} {name:<12} FAILED: {str(exc).splitlines()[0][:110]}", flush=True)
                    results.append({"seq": seq, "cfg": name, "aligned": False, "error": str(exc).splitlines()[0][:200]})
    finally:
        ttnn.close_mesh_device(mesh)
    (HERE / "sdpa_prefill_probe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
