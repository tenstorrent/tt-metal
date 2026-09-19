# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Does `sdpa_decode` survive `begin_trace_capture`?

`probe_sdpa_wiring.py` localised the model's accuracy loss precisely:

    A  untraced fused vs untraced explicit    0.9995602176
    B  traced fused   vs untraced fused       0.9176136440
    C  traced explicit vs untraced explicit   1.0000000000

The arithmetic is right (A) and the explicit path traces bit-exactly (C). What fails
is the combination. `sdpa_decode` is a multi-core flash kernel: it splits the key axis
across cores, each computes a partial softmax, and a reducer core combines them through
inter-core semaphores. Semaphores and per-core work division are exactly the kind of
state a replayed trace can get wrong -- and nothing about it raises.

Three questions, in order:

  1. Is the **first** replay right and later ones wrong? That is leftover state.
  2. Is **every** replay wrong the same way? That is a capture-time work division.
  3. Does forcing **one core per head-batch** fix it? If the cross-core reduction is
     the problem, removing the reduction removes it -- at some cost in speed, which
     is worth measuring rather than assuming fatal.

    python3 models/demos/cosyvoice/scripts/probe_sdpa_trace.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

import ttnn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

H, DK, W, VALID = 16, 64, 384, 210
REPLAYS = 4
LAYERS = 14
REPS = 5


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def main() -> int:
    device = ttnn.open_device(device_id=0, l1_small_size=32768, trace_region_size=402653184)
    cc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    scale = DK**-0.5
    grid = device.compute_with_storage_grid_size()

    def dev(v):
        return ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch.manual_seed(0)
    qu_t = torch.randn(1, H, 1, DK) * 0.1
    k_t = torch.randn(1, H, W, DK) * 0.1
    v_t = torch.randn(1, H, W, DK) * 0.1
    bd_t = torch.randn(1, H, 1, W) * 0.1
    mask_t = torch.zeros(1, 1, 1, W)
    mask_t[..., : W - VALID] = -1e9
    gold = (torch.softmax((qu_t @ k_t.transpose(-1, -2) + bd_t) * scale + mask_t, dim=-1) @ v_t).reshape(H, DK)

    k, v = dev(k_t), dev(v_t)
    q4 = dev(qu_t.permute(0, 2, 1, 3).contiguous())
    bias = dev((bd_t + mask_t).permute(0, 2, 1, 3).contiguous())

    configs = {
        "default grid": dict(compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=128),
        "1 core / head-batch": dict(
            compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=128, max_cores_per_head_batch=1
        ),
        "1x1 grid": dict(compute_with_storage_grid_size=ttnn.CoreCoord(1, 1), q_chunk_size=32, k_chunk_size=128),
    }

    for name, kw in configs.items():
        try:
            prog = ttnn.SDPAProgramConfig(exp_approx_mode=False, **kw)
        except Exception as exc:  # noqa: BLE001
            print(f"\n  {name}: config rejected -- {str(exc)[:90]}")
            continue

        def one():
            return ttnn.transformer.scaled_dot_product_attention_decode(
                q4,
                k,
                v,
                is_causal=False,
                attn_mask=bias,
                scale=scale,
                program_config=prog,
                compute_kernel_config=cc,
            )

        try:
            ref = one()
        except Exception as exc:  # noqa: BLE001
            print(f"\n  {name}: call failed -- {str(exc)[:90]}")
            continue
        p_untraced = pcc(ttnn.to_torch(ref).float().reshape(H, DK), gold)
        ttnn.deallocate(ref)

        # A persistent output buffer, because reading a trace's own output after
        # replay is fragile -- the same discipline the KV cache uses.
        out_buf = dev(torch.zeros(1, 1, H, DK))

        def capture_body():
            held = one()
            ttnn.copy(held, out_buf)
            ttnn.deallocate(held)

        # Warm the *whole* body, copy included -- warming only the op leaves the copy
        # uncompiled and capture dies at `mesh_workload.cpp:153`.
        for _ in range(2):
            capture_body()
        ttnn.synchronize_device(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        try:
            capture_body()
        finally:
            ttnn.end_trace_capture(device, tid, cq_id=0)

        replays = []
        for _ in range(REPLAYS):
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            replays.append(pcc(ttnn.to_torch(out_buf).float().reshape(H, DK), gold))

        # Timing, so a config that fixes accuracy can be priced immediately.
        def body():
            out = None
            for _ in range(LAYERS):
                nxt = one()
                if out is not None:
                    ttnn.deallocate(out)
                out = nxt
            return out

        ttnn.deallocate(body())
        ttnn.synchronize_device(device)
        tid2 = ttnn.begin_trace_capture(device, cq_id=0)
        try:
            held2 = body()
        finally:
            ttnn.end_trace_capture(device, tid2, cq_id=0)
        best = None
        for _ in range(REPS):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid2, cq_id=0, blocking=True)
            best = min(best or 1e9, time.perf_counter() - t0)
        ttnn.release_trace(device, tid2)
        ttnn.deallocate(held2)
        ttnn.release_trace(device, tid)
        ttnn.deallocate(out_buf)

        print(f"\n  {name}")
        print(f"    untraced          PCC {p_untraced:.10f}")
        for i, p in enumerate(replays):
            print(f"    trace replay {i}    PCC {p:.10f}")
        print(f"    14 layers traced  {best * 1e3:.3f} ms")

    print("\n  replay 0 good, later ones bad -> leftover per-core state between replays.")
    print("  every replay bad the same way -> the work division bakes at capture.")
    ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
