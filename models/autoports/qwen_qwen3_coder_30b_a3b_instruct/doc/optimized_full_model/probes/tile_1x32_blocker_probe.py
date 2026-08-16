# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone probe: does stage 02's ``output_tile=Tile([1, 32])`` blocker still hold?

**A measurement, not a change.** Nothing in ``tt/`` is touched; this builds the
shipped decode sparse-matmul shapes from random data and asks the *downstream*
ops whether they can read a 1x32-tile result yet.

The ledger entry being re-tested (``doc/optimized_decoder/README.md``, and
``tt/optimized_decoder.py``'s "Rejected, with measurements"): at decode M=1 the
expert matmuls pad M to a full 32-row tile, so ``gate_up`` writes 12 MB and
``down`` writes 16 MB where 0.4/0.5 MB is real, and the reshapes that compact it
away cost 31 + 33 + 46 us. ``output_tile=ttnn.Tile([1, 32])`` removes the
padding at the source and was **1.07x faster end to end** -- but was rejected
because no downstream op consumed the result: ``slice`` rejected it, ``sum`` and
``reshape`` raised ``MeshBuffer must be large enough``, ``untilize`` returned
wrong data *silently*, and ``fast_reduce_nc`` returned zeros.

Stage 06 re-tests every one of those consumers, plus a value check against the
same matmul with the shipped ``Tile([32, 32])``, because "untilize returns wrong
data without erroring" means a pass/fail on exceptions alone is not enough.

    python tile_1x32_blocker_probe.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch

import ttnn

sys.path.insert(0, str(Path(__file__).resolve().parents[6]))
from models.autoports.qwen_qwen3_coder_30b_a3b_instruct.tt.optimized_decoder import (  # noqa: E402
    EXPERT_IN0_BLOCK_W_GATE_UP,
    _tuned_sparse_matmul_config,
)

MESH_SHAPE = (1, 4)
BATCH = 1
EXPERTS = 32
HIDDEN = 2048
INTER = 768


def build(mesh, tile):
    torch.manual_seed(0)
    x = ttnn.from_torch(
        torch.randn((1, BATCH, 1, HIDDEN)).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    w = ttnn.from_torch(
        torch.randn((1, EXPERTS, HIDDEN, 2 * INTER)).float() * 0.02,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    routing = torch.zeros((1, 1, BATCH, EXPERTS))
    routing[..., :2] = 0.5  # two active experts, the decode average at top-8 over 4 dies
    sparsity = ttnn.from_torch(
        routing,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    out = ttnn.sparse_matmul(
        x,
        w,
        sparsity=sparsity,
        nnz=None,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        # the shipped decode program config; ``program_config`` is required on
        # this build's ``sparse_matmul`` binding, so it is not optional here.
        program_config=_tuned_sparse_matmul_config(1, 2 * INTER, HIDDEN, EXPERT_IN0_BLOCK_W_GATE_UP),
        output_tile=tile,
        dtype=ttnn.bfloat16,
    )
    return x, w, sparsity, out


def attempt(name, fn):
    try:
        value = fn()
        print(f"  {name:<34} OK    {value}")
        return True
    except Exception as exc:  # noqa: BLE001 -- the whole point is which ones raise
        first = str(exc).strip().splitlines()[0][:130]
        print(f"  {name:<34} RAISE {type(exc).__name__}: {first}")
        return False


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
    try:
        ref = None
        for tile_h in (32, 1):
            tile = ttnn.Tile([tile_h, 32])
            print(f"\n=== output_tile = Tile([{tile_h}, 32]) ===")
            try:
                x, w, sparsity, out = build(mesh, tile)
            except Exception:  # noqa: BLE001
                print("  sparse_matmul itself raised:")
                traceback.print_exc()
                continue
            print(f"  produced {out.shape} {out.dtype} {out.layout}")

            attempt("ttnn.slice(.., 0, half)", lambda: ttnn.slice(out, [0, 0, 0, 0], [1, BATCH, 1, INTER]).shape)
            attempt("ttnn.reshape -> (B,E,W)", lambda: ttnn.reshape(out, (BATCH, EXPERTS, 2 * INTER)).shape)
            attempt("ttnn.sum(dim=1)", lambda: ttnn.sum(out, dim=1).shape)
            attempt("ttnn.untilize", lambda: ttnn.untilize(out).shape)
            attempt("ttnn.to_layout(ROW_MAJOR)", lambda: ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT).shape)
            attempt("ttnn.silu (eltwise)", lambda: ttnn.silu(out).shape)

            # The value check: "untilize returns wrong data without erroring" is
            # why exception-freedom is not the test.
            try:
                host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:1].float()
                live = host.reshape(-1)[: 2 * INTER]
                if tile_h == 32:
                    ref = live
                    print(f"  value: reference row captured, |mean| {live.abs().mean():.5f}")
                else:
                    pcc = torch.corrcoef(torch.stack([ref, live]))[0, 1].item()
                    print(f"  value: PCC vs Tile([32,32]) = {pcc:.6f}, max|diff| {(ref - live).abs().max():.3e}")
            except Exception as exc:  # noqa: BLE001
                print(f"  value: readback RAISE {type(exc).__name__}: {str(exc).splitlines()[0][:120]}")

            for t in (x, w, sparsity, out):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
