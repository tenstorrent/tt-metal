"""Accuracy and latency of one 4096-token prefill chunk's attention at long start positions (one chip).

Per-chip TP4 geometry: 8 Q heads, 2 KV heads, head_dim 128, BFP8 paged cache (page 32). The FP32
oracle uses the stored BFP8 cache and a fixed sample of query rows (default 64, spread over the
chunk, always including its last row).
"""

import argparse
import json
import time

import torch

import ttnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--starts", nargs="+", type=int, default=[61440, 126976, 258048, 520192])
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--paths", nargs="+", default=["stock_k256", "acc_8x8_k128", "acc_11x10_k128", "acc_11x10_k256"])
    p.add_argument("--q-scale", type=float, default=1.0)
    p.add_argument("--sample-rows", type=int, default=64)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output")
    a = p.parse_args()

    from models.demos.k2_horizon_7b_qb2.tt.accurate_attention import accurate_attention

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[a.device])
    up = lambda x, dt, lay: ttnn.from_torch(
        x.contiguous(), device=mesh, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    compute = {
        f: ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=getattr(ttnn.MathFidelity, f),
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        for f in ("HiFi2", "HiFi4")
    }
    torch.manual_seed(20261001)
    capacity = 1 << (max(a.starts) + a.chunk - 1).bit_length()
    pages = capacity // 32
    table = torch.randperm(pages, dtype=torch.int32).reshape(1, pages)
    keys = torch.randn(pages, 2, 32, 128).bfloat16()
    values = torch.randn_like(keys)
    kt, vt = up(keys, ttnn.bfloat8_b, ttnn.TILE_LAYOUT), up(values, ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    keys, values = ttnn.to_torch(kt).float(), ttnn.to_torch(vt).float()
    pt = up(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
    k_lin = keys[table[0].long()].permute(1, 0, 2, 3).reshape(2, -1, 128)
    v_lin = values[table[0].long()].permute(1, 0, 2, 3).reshape(2, -1, 128)
    rows = torch.unique(torch.cat([torch.linspace(0, a.chunk - 1, a.sample_rows).long(), torch.tensor([a.chunk - 1])]))
    records = []
    for start in a.starts:
        q = (torch.randn(1, 8, a.chunk, 128) * a.q_scale).bfloat16()
        qt = up(q, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        expected = torch.empty(8, len(rows), 128, dtype=torch.float64)
        for i, r in enumerate(rows.tolist()):
            end = start + r + 1
            kk = k_lin[:, :end].repeat_interleave(4, 0)
            vv = v_lin[:, :end].repeat_interleave(4, 0)
            s = torch.einsum("hd,hsd->hs", q[0, :, r].float(), kk) / (128**0.5)
            expected[:, i] = torch.einsum("hs,hsd->hd", s.double().softmax(-1), vv.double())

        def stock(k_chunk, q_chunk=128):
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(11, 10),
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                exp_approx_mode=False,
            )
            return lambda: ttnn.transformer.chunked_scaled_dot_product_attention(
                qt, kt, vt, pt, chunk_start_idx=start, compute_kernel_config=compute["HiFi2"], program_config=cfg
            )

        def acc(grid, q_chunk, k_chunk, fidelity="HiFi4"):
            return lambda: accurate_attention(
                qt,
                kt,
                vt,
                pt,
                chunk_start_idx=start,
                q_chunk_size=q_chunk,
                k_chunk_size=k_chunk,
                grid=grid,
                math_fidelity=fidelity,
            )

        fns = {
            "stock_k256": stock(256),
            "stock_q64_k256": stock(256, 64),
            "acc_8x8_k128": acc((8, 8), 128, 128),
            "acc_11x10_k128": acc((11, 10), 128, 128),
            "acc_11x10_k256": acc((11, 10), 128, 256),
            "acc_11x10_q256_k256": acc((11, 10), 256, 256),
            "acc_11x10_k512": acc((11, 10), 128, 512),
            "acc_4x4_k128": acc((4, 4), 128, 128),
            "acc_8x8_k512": acc((8, 8), 128, 512),
            "acc_11x10_k1024": acc((11, 10), 128, 1024),
            "acc_11x10_q64_k512": acc((11, 10), 64, 512),
            "acc_11x10_q64_k1024": acc((11, 10), 64, 1024),
            "acc_11x10_q64_k1024_hifi2": acc((11, 10), 64, 1024, "HiFi2"),
            "acc_11x10_q64_k1024_hifi3": acc((11, 10), 64, 1024, "HiFi3"),
            "acc_11x10_q64_k1024_lofi": acc((11, 10), 64, 1024, "LoFi"),
            "acc_11x10_q64_k256": acc((11, 10), 64, 256),
            "acc_11x10_q32_k512": acc((11, 10), 32, 512),
            "acc_11x10_q32_k1024": acc((11, 10), 32, 1024),
        }
        for path in a.paths:
            fn = fns[path]
            out = fn()
            actual = ttnn.to_torch(out)[0][:, rows].double()
            out.deallocate(True)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(a.iters):
                fn().deallocate(True)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / a.iters * 1e3
            d = (actual - expected).reshape(-1)
            row = dict(
                path=path,
                start=start,
                chunk=a.chunk,
                q_scale=a.q_scale,
                ms=round(ms, 2),
                relative_l2=float(d.norm() / expected.norm()),
                max_abs=float(d.abs().max()),
            )
            print(json.dumps(row), flush=True)
            records.append(row)
        qt.deallocate(True)
    if a.output:
        open(a.output, "w").write(json.dumps(records, indent=1) + "\n")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
