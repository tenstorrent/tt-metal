# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Accuracy and latency of paged decode attention against an FP64 oracle (one chip).

Paths: the stock op with default settings ("stock*") and with SDPAProgramConfig.fp32_accumulation
("fp32*", FP32 scores and online-softmax state, accurate exponentials), which the model uses beyond the
stock decode bound ("fp32_k256": k_chunk 256, up to 64 cores per head, HiFi4).

Fixtures use the per-chip TP4 geometry (8 Q heads, 2 KV heads, head_dim 128, BFP8 paged
cache, page 32). The oracle reads back the stored BFP8 cache, as the stage-11 comparator.

  stage11:  seed 570129, B32, capacity 8192 (the bring-up's stage-11 control case)
  long:     B1 at long positions, uniform-ish (q_scale 0.5) and peaked (q_scale 2.0) attention
"""

import argparse
import json
import time

import torch

import ttnn


def metrics(actual, expected):
    a, b = actual.double().reshape(-1), expected.double().reshape(-1)
    ac, bc = a - a.mean(), b - b.mean()
    return {
        "relative_l2": float((a - b).norm() / b.norm()),
        "pcc": float((ac @ bc) / (ac.norm() * bc.norm())),
        "max_abs": float((a - b).abs().max()),
    }


def reference(q, keys, values, table, positions):
    out = torch.empty(q.shape, dtype=torch.float64)
    for row, position in enumerate(positions.tolist()):
        pages = table[row, : (position + 32) // 32].long()
        k = keys[pages].permute(1, 0, 2, 3).reshape(2, -1, 128)[:, : position + 1].double()
        v = values[pages].permute(1, 0, 2, 3).reshape(2, -1, 128)[:, : position + 1].double()
        k, v = k.repeat_interleave(4, dim=0), v.repeat_interleave(4, dim=0)
        scores = torch.einsum("hd,hsd->hs", q[0, row].double(), k) / (128**0.5)
        out[0, row] = torch.einsum("hs,hsd->hd", scores.softmax(-1), v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fixtures", nargs="+", default=["stage11", "long"])
    p.add_argument(
        "--batched",
        nargs="+",
        default=["32:4096", "32:8191", "16:16383", "15:32767", "8:65535", "4:131071"],
        help="batch:position cases for the 'batched' fixture",
    )
    p.add_argument("--long-positions", nargs="+", type=int, default=[65663, 131199, 262143, 524287])
    p.add_argument("--paths", nargs="+", default=["stock16", "stock64_k256", "fp32_k256", "fp32_16"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output")
    a = p.parse_args()

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[a.device])
    up = lambda x, dt, lay: ttnn.from_torch(
        x.contiguous(), device=mesh, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    hifi = {
        f: ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, f),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        for f in ("HiFi2", "HiFi4")
    }
    records = []

    def run_case(name, q, table, kt, vt, keys, values, positions, capacity):
        batch = q.shape[1]
        qt = up(q, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        pt = up(table[:, : capacity // 32], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        pos = up(positions, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        expected = reference(q, keys, values, table, positions)

        def sdpa(cores, k_chunk=128, fidelity="HiFi2", fp32_accumulation=False):
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(11, 10),
                q_chunk_size=0,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
                max_cores_per_head_batch=cores,
                fp32_accumulation=fp32_accumulation,
            )
            return lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(
                qt,
                kt,
                vt,
                page_table_tensor=pt,
                cur_pos_tensor=pos,
                compute_kernel_config=hifi[fidelity],
                program_config=cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        fns = {
            "stock16": sdpa(16),
            "stock16_hifi4": sdpa(16, fidelity="HiFi4"),
            "stock64_k256": sdpa(64, 256, "HiFi4"),
            "fp32_k256": sdpa(64, 256, "HiFi4", fp32_accumulation=True),
            "fp32_16": sdpa(16, 128, "HiFi4", fp32_accumulation=True),
            "fp32_55": sdpa(55, 128, "HiFi4", fp32_accumulation=True),
        }
        for path in a.paths:
            fn = fns[path]
            out = fn()
            actual = ttnn.to_torch(out)[:, :, :8, :].float()
            out.deallocate(True)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(a.iters):
                fn().deallocate(True)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / a.iters * 1e3
            row = dict(
                fixture=name,
                path=path,
                batch=batch,
                positions=sorted(set(positions.tolist())),
                capacity=capacity,
                ms=round(ms, 3),
                **metrics(actual, expected),
            )
            print(json.dumps(row), flush=True)
            records.append(row)
        for t in (qt, pt, pos):
            t.deallocate(True)

    if "stage11" in a.fixtures:
        torch.manual_seed(570129)
        batch, capacity, pages = 32, 8192, 256
        table = torch.randperm(batch * pages, dtype=torch.int32).reshape(batch, pages)
        q = (torch.randn(1, batch, 8, 128) * 0.5).bfloat16()
        keys = torch.randn(batch * pages, 2, 32, 128).bfloat16()
        values = torch.randn_like(keys)
        kt, vt = up(keys, ttnn.bfloat8_b, ttnn.TILE_LAYOUT), up(values, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        keys, values = ttnn.to_torch(kt).float(), ttnn.to_torch(vt).float()
        for position in (4096, 8191):
            run_case(
                "stage11", q, table, kt, vt, keys, values, torch.full((batch,), position, dtype=torch.int32), capacity
            )
        kt.deallocate(True)
        vt.deallocate(True)

    if "long" in a.fixtures:
        torch.manual_seed(20260930)
        capacity = 1 << max(a.long_positions).bit_length()
        pages = capacity // 32
        table = torch.randperm(pages, dtype=torch.int32).reshape(1, pages)
        keys = torch.randn(pages, 2, 32, 128).bfloat16()
        values = torch.randn_like(keys)
        kt, vt = up(keys, ttnn.bfloat8_b, ttnn.TILE_LAYOUT), up(values, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        keys, values = ttnn.to_torch(kt).float(), ttnn.to_torch(vt).float()
        for q_scale in (0.5, 2.0):
            q = (torch.randn(1, 1, 8, 128) * q_scale).bfloat16()
            for position in a.long_positions:
                cap = max(4096, 1 << position.bit_length())
                run_case(
                    f"long_q{q_scale}", q, table, kt, vt, keys, values, torch.tensor([position], dtype=torch.int32), cap
                )
    if "batched" in a.fixtures:
        cases = [tuple(map(int, c.split(":"))) for c in a.batched]
        torch.manual_seed(1234)
        batch_max = max(b for b, _ in cases)
        capacity = max(1 << pos.bit_length() for _, pos in cases)
        pages = capacity // 32
        # Distinct per-request pages drawn from one pool large enough for every case.
        pool = max(b * (1 << pos.bit_length()) // 32 for b, pos in cases)
        keys = torch.randn(pool, 2, 32, 128).bfloat16()
        values = torch.randn_like(keys)
        kt, vt = up(keys, ttnn.bfloat8_b, ttnn.TILE_LAYOUT), up(values, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
        keys, values = ttnn.to_torch(kt).float(), ttnn.to_torch(vt).float()
        for batch, position in cases:
            cap = 1 << position.bit_length()
            table = torch.randperm(pool, dtype=torch.int32)[: batch * cap // 32].reshape(batch, cap // 32)
            q = (torch.randn(1, batch, 8, 128) * 0.5).bfloat16()
            positions = torch.full((batch,), position, dtype=torch.int32) - torch.arange(batch, dtype=torch.int32) * 7
            run_case("batched", q, table, kt, vt, keys, values, positions, cap)
    if a.output:
        open(a.output, "w").write(json.dumps(records, indent=1) + "\n")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
