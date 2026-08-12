# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Crossover sweep: ``ttnn.linear`` vs ``ttnn.experimental.minimal_matmul``.

``_dense()`` picks between the two on row count alone, so the threshold has to
be measured, not guessed — and it has to be measured *per projection*, because
the crossover is not the same for a 4608-wide output and a 19968-wide one.

Both sides run the **shipped** compute-kernel config (HiFi2,
``fp32_dest_acc_en=False``, ``packer_l1_acc=True``) — the same policy
``ttnn.linear`` selects by default for BF16 — so this is a pure kernel
comparison.  ``minimal_matmul``'s own default is also reported at the largest M
for reference; it is more accurate and slower, and choosing it is the
optimized-decoder stage's call.

Output: ``doc/fused_decoder/logs/minimal_matmul_sweep.log``.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.common.utility_functions import comp_pcc

#: (label, K, N) for every dense projection in the layer.
PROJECTIONS = [
    ("wqkv", 6656, 4608),
    ("attn_gate", 6656, 4096),
    ("o_proj", 4096, 6656),
    ("mlp_gate_up", 6656, 19968),
    ("mlp_down", 19968, 6656),
]
#: Row counts.  32 is a decode step; 512..8192 brackets every prefill chunk the
#: layer can produce (the internal chunk size is 8192 and a short prompt is
#: whatever it is), and the band above 1024 is where the wide MLP projections
#: actually cross over.
ROWS = (32, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
ROUNDS = 3


def main():
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        shipped_ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for label, k_dim, n_dim in PROJECTIONS:
            weight = (torch.randn(1, 1, k_dim, n_dim) * 0.02).to(torch.bfloat16)
            tw = ttnn.from_torch(
                weight, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            for rows in ROWS:
                torch.manual_seed(0)
                x = (torch.randn(1, 1, rows, k_dim) * 0.1).to(torch.bfloat16)
                tx = ttnn.from_torch(
                    x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                ref = x.float().reshape(rows, k_dim) @ weight.float().reshape(k_dim, n_dim)
                candidates = [
                    ("linear", lambda: ttnn.linear(tx, tw, memory_config=ttnn.DRAM_MEMORY_CONFIG)),
                    (
                        "minimal",
                        lambda: ttnn.experimental.minimal_matmul(
                            tx, tw, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=shipped_ck
                        ),
                    ),
                ]
                if rows == ROWS[-1]:
                    candidates.append(
                        (
                            "minimal_opdefault",
                            lambda: ttnn.experimental.minimal_matmul(tx, tw, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                        )
                    )
                out_line = []
                for name, fn in candidates:
                    try:
                        out = fn()
                    except Exception as exc:  # noqa: BLE001
                        out_line.append(f"{name}=FAILED({type(exc).__name__}: {str(exc)[:200]})")
                        continue
                    pcc = comp_pcc(ref, ttnn.to_torch(out).reshape(rows, n_dim).float(), 0.99)[1]
                    ttnn.deallocate(out)
                    iters = 3 if rows > 2000 else 20
                    best = None
                    for _ in range(ROUNDS):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(iters):
                            ttnn.deallocate(fn())
                        ttnn.synchronize_device(mesh)
                        dt = (time.perf_counter() - start) / iters * 1e3
                        best = dt if best is None else min(best, dt)
                    out_line.append(f"{name}={best:8.3f}ms(pcc={pcc:.6f})")
                print(f"MM2 {label:12s} M={rows:5d}  " + "  ".join(out_line), flush=True)
                ttnn.deallocate(tx)
            ttnn.deallocate(tw)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
