# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep the *paged* prefill SDPA's q/k chunk size.

`bench/sdpa_chunk_sweep.py` retunes `ttnn.transformer.scaled_dot_product_attention`,
which is what a `sliding` layer uses for every chunk and what a `full` layer uses
for chunk 0.  Every *later* chunk of a `full` (NoPE) prefill instead calls
``ttnn.transformer.chunked_scaled_dot_product_attention``, reading the whole
prefix back out of the paged KV cache — a different op with its own program
config, and the dominant op of any prefill longer than one internal chunk.

This sweeps that op at the offsets a real 8192-chunked prefill produces, against
a PyTorch masked-softmax reference over the same permuted paged cache.

Output: ``doc/fused_decoder/logs/chunked_sdpa_sweep.log``.
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.functional_decoder import _as_float32
from models.common.utility_functions import comp_pcc

NUM_Q_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
BLOCK = 64
#: chunked_scaled_dot_product_attention declares `scale` as
#: `std::optional<float>` with `.noconvert()`, so a Python double that is not
#: exactly float32-representable raises a signature TypeError (functional-stage
#: limitation 2).  Round it the same way the layer does.
SCALE = _as_float32(0.342063)
CHUNK = 8192
#: (chunk_start_idx, total prefix length) — chunk 1 and chunk 4 of a long prompt.
OFFSETS = [(8192, 16384), (32768, 40960)]
CHUNKS = (128, 256, 320, 512)
ITERS = 3
ROUNDS = 2


def main() -> None:
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        grid = mesh.compute_with_storage_grid_size()
        ck = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        for start, total in OFFSETS:
            torch.manual_seed(3)
            blocks = total // BLOCK
            keys = torch.randn(1, NUM_KV_HEADS, total, HEAD_DIM) / 3
            values = torch.randn(1, NUM_KV_HEADS, total, HEAD_DIM) / 3
            query = torch.randn(1, NUM_Q_HEADS, CHUNK, HEAD_DIM) / 3
            permutation = torch.randperm(blocks)

            def paged(source):
                out = torch.zeros(blocks, NUM_KV_HEADS, BLOCK, HEAD_DIM)
                for logical in range(blocks):
                    out[permutation[logical]] = source[0, :, logical * BLOCK : (logical + 1) * BLOCK, :]
                return out

            def dev(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
                return ttnn.from_torch(
                    t, device=mesh, layout=layout, dtype=dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )

            tq = dev(query.to(torch.bfloat16))
            tk = dev(paged(keys).to(torch.bfloat16))
            tv = dev(paged(values).to(torch.bfloat16))
            pt = dev(permutation.reshape(1, blocks).to(torch.int32), ttnn.ROW_MAJOR_LAYOUT, ttnn.int32)

            grouped_k = keys.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1).float()
            grouped_v = values.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1).float()
            q_idx = torch.arange(CHUNK) + start
            mask = q_idx[:, None] >= torch.arange(total)[None, :]
            scores = (query.float() @ grouped_k.transpose(-1, -2)) * SCALE
            ref = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=-1) @ grouped_v

            for chunk in CHUNKS:
                if start % chunk:
                    print(
                        f"CSDPA start={start:6d} total={total:6d} chunk={chunk:4d}  SKIPPED "
                        f"(chunk_start_idx must divide by q_chunk_size)",
                        flush=True,
                    )
                    continue
                pc = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, grid.y),
                    q_chunk_size=chunk,
                    k_chunk_size=chunk,
                    exp_approx_mode=False,
                )
                try:
                    out = ttnn.transformer.chunked_scaled_dot_product_attention(
                        tq, tk, tv, pt, start, scale=SCALE, program_config=pc, compute_kernel_config=ck
                    )
                    pcc = comp_pcc(ref, ttnn.to_torch(out).float(), 0.99)[1]
                    ttnn.deallocate(out)
                    best = None
                    for _ in range(ROUNDS):
                        ttnn.synchronize_device(mesh)
                        t0 = time.perf_counter()
                        for _ in range(ITERS):
                            o = ttnn.transformer.chunked_scaled_dot_product_attention(
                                tq, tk, tv, pt, start, scale=SCALE, program_config=pc, compute_kernel_config=ck
                            )
                            ttnn.deallocate(o)
                        ttnn.synchronize_device(mesh)
                        dt = (time.perf_counter() - t0) / ITERS * 1e3
                        best = dt if best is None else min(best, dt)
                    print(
                        f"CSDPA start={start:6d} total={total:6d} chunk={chunk:4d}  min {best:8.3f} ms  " f"PCC={pcc}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    detail = [l.strip() for l in str(exc).splitlines() if l.strip()]
                    info = next(
                        (l for l in detail if "beyond max L1" in l or "must" in l or "info:" in l),
                        detail[0] if detail else type(exc).__name__,
                    )
                    print(
                        f"CSDPA start={start:6d} total={total:6d} chunk={chunk:4d}  FAILED "
                        f"{type(exc).__name__}: {info[:170]}",
                        flush=True,
                    )
            for t in (tq, tk, tv, pt):
                ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
