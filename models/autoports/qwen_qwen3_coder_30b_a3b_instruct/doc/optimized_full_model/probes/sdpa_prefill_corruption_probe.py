# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The prefill program config at ``S < q_chunk_size`` corrupts *other* tensors.

``test_multichip_decode_batch`` (prompt 32, 128-deep paged cache) returned PCC
**-0.0686** against HF once ``attention_prefill`` was given ``q128/k128`` --
while ``sdpa_prefill_probe.py`` measured PCC 0.9998 for that exact config at
S = 1, 3, 31, 33. Both are true: the op's **output** is correct at S < chunk,
and something else in DRAM is not. The suite's prefill-only tests pass and only
the *decode* that reads the KV cache afterwards fails, which points at the cache.

This probe holds a KV-cache-shaped tensor next to the SDPA call, runs the call,
and reads the tensor back. Nothing but the SDPA op touches it.

    python sdpa_prefill_corruption_probe.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import ttnn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sdpa_prefill_probe import HEAD_DIM, N_KV_HEADS, mk  # noqa: E402

SEQS = [8, 32, 64, 100, 127, 128, 129, 255, 256, 384, 512, 1000, 1024]
CHUNKS = [None, 128, 256]


def main():
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    grid = mesh.compute_with_storage_grid_size()
    results = []
    try:
        for seq in SEQS:
            row = f"  S {seq:6d}  "
            for chunk in CHUNKS:
                pc = (
                    None
                    if chunk is None
                    else ttnn.SDPAProgramConfig(
                        compute_with_storage_grid_size=grid, q_chunk_size=chunk, k_chunk_size=chunk
                    )
                )
                # a bystander laid out like the paged KV cache: 4 pages x 1 head x 32 x 128
                witness_t = torch.randn((4, N_KV_HEADS, 32, HEAD_DIM)).float()
                witness = ttnn.from_torch(
                    witness_t,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                before = ttnn.to_torch(ttnn.get_device_tensors(witness)[0]).float()
                q, k, v = mk(mesh, seq)
                try:
                    out = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, program_config=pc)
                    ttnn.synchronize_device(mesh)
                    ttnn.deallocate(out)
                    after = ttnn.to_torch(ttnn.get_device_tensors(witness)[0]).float()
                    delta = (after - before).abs().max().item()
                    row += f"{'None' if chunk is None else 'q'+str(chunk)}: d={delta:.2e}  "
                    results.append({"seq": seq, "chunk": chunk, "witness_max_abs_delta": delta})
                except Exception as exc:  # noqa: BLE001
                    row += f"{'None' if chunk is None else 'q'+str(chunk)}: RAISE  "
                    results.append({"seq": seq, "chunk": chunk, "error": str(exc).splitlines()[0][:200]})
                for t in (q, k, v, witness):
                    ttnn.deallocate(t)
            print(row, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    (HERE / "sdpa_prefill_corruption_probe.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
