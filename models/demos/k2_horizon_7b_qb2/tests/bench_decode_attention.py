"""Microbenchmark B-user decode attention paths at long context on one Blackhole chip.

Per-chip local geometry of the TP4 model: 8 Q heads, 2 KV heads, head_dim 128,
BFP8 paged cache, page 32. Reports device wall time per call (synchronized loop).
"""

import argparse
import json
import time

import torch

import ttnn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--positions", nargs="+", type=int, default=[32767, 65663, 131199, 262143, 524287])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--paths", nargs="+", default=["stock16", "stock55", "accurate"])
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--output")
    a = p.parse_args()

    from models.demos.k2_horizon_7b_qb2.tt.accurate_attention import accurate_attention

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[a.device])
    torch.manual_seed(0)
    capacity = 1 << (max(a.positions)).bit_length()
    pages = capacity // 32
    b = a.batch
    table = torch.randperm(b * pages, dtype=torch.int32).reshape(b, pages)
    up = lambda x, dt, lay: ttnn.from_torch(x, device=mesh, dtype=dt, layout=lay, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    kt = up(torch.randn(b * pages, 2, 32, 128).bfloat16(), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    vt = up(torch.randn(b * pages, 2, 32, 128).bfloat16(), ttnn.bfloat8_b, ttnn.TILE_LAYOUT)
    q = up((torch.randn(1, b, 8, 128) * 0.5).bfloat16(), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    hifi2 = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    results = []
    for position in a.positions:
        cap = max(4096, 1 << position.bit_length())
        pt = up(table[:, : cap // 32].contiguous(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        pos = up(torch.full((b,), position, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        def stock(cores):
            cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(11, 10),
                q_chunk_size=0,
                k_chunk_size=128,
                exp_approx_mode=False,
                max_cores_per_head_batch=cores,
            )
            return lambda: ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q, kt, vt, page_table_tensor=pt, cur_pos_tensor=pos, compute_kernel_config=hifi2, program_config=cfg
            )

        def accurate():
            assert b == 1
            qq = ttnn.to_memory_config(q, ttnn.DRAM_MEMORY_CONFIG)
            p_ = ttnn.to_layout(ttnn.reshape(pos[0:1], (1, 1, 1, 1)), ttnn.TILE_LAYOUT)
            p_ = ttnn.maximum(p_, 0)
            offset = ttnn.reshape(ttnn.to_layout(ttnn.bitwise_and(p_, -32), ttnn.ROW_MAJOR_LAYOUT), (1,))
            index = ttnn.repeat(ttnn.typecast(ttnn.bitwise_and(p_, 31), ttnn.uint32), (1, 8, 1, 128))
            query = ttnn.repeat(ttnn.permute(qq[:, 0:1, :, :], (0, 2, 1, 3)), (1, 1, 32, 1))
            tab = pt
            pad = (-tab.shape[1]) % 8
            if pad:
                tab = ttnn.concat([tab, ttnn.repeat(tab[:, -1:], (1, pad))], dim=1)
            att = accurate_attention(
                query, kt, vt, tab, chunk_start_idx_tensor=offset, q_chunk_size=32, k_chunk_size=128
            )
            return ttnn.permute(ttnn.gather(att, 2, index), (0, 2, 1, 3))

        fns = {"stock16": stock(16), "stock32": stock(32), "stock55": stock(55), "accurate": accurate}
        for name in a.paths:
            fn = fns[name]
            out = fn()
            ttnn.synchronize_device(mesh)
            out.deallocate(True)
            t0 = time.perf_counter()
            for _ in range(a.iters):
                out = fn()
                out.deallocate(True)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / a.iters * 1e3
            row = dict(path=name, batch=b, position=position, capacity=cap, ms=round(ms, 3))
            print(json.dumps(row), flush=True)
            results.append(row)
        pt.deallocate(True)
        pos.deallocate(True)
    if a.output:
        open(a.output, "w").write(json.dumps(results, indent=1) + "\n")
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
