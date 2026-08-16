import json
import sys
import time

import torch

import ttnn

P = "/home/raahem/tt-metal/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/optimized_full_model/probes"
sys.path.insert(0, P)
from sdpa_depth_probe import BATCH, HEAD_DIM, MESH_SHAPE, N_KV_HEADS, N_Q_HEADS, PAGE


def bench(mesh, depth, cur_pos, pc, iters=30):
    pages = depth // PAGE
    caches = [
        ttnn.from_torch(
            torch.randn((pages, N_KV_HEADS, PAGE, HEAD_DIM)).float(),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for _ in range(2)
    ]
    q = ttnn.from_torch(
        torch.randn((1, BATCH, N_Q_HEADS, HEAD_DIM)).float(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    pt = ttnn.from_torch(
        torch.arange(BATCH * pages, dtype=torch.int32).reshape(BATCH, pages),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    pos = ttnn.from_torch(
        torch.tensor([cur_pos] * BATCH, dtype=torch.int32),
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def call():
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            caches[0],
            caches[1],
            page_table_tensor=pt,
            cur_pos_tensor=pos,
            scale=HEAD_DIM**-0.5,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=pc,
        )
        ttnn.deallocate(o)

    call()
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(iters):
        call()
    ttnn.synchronize_device(mesh)
    ms = 1e3 * (time.perf_counter() - t) / iters
    for x in (q, pt, pos, *caches):
        ttnn.deallocate(x)
    return ms


mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
grid = mesh.compute_with_storage_grid_size()
print("grid", grid)
out = []
try:
    for cur in (1024, 8192):
        for name, pc in [("None (shipped paged path)", None)] + [
            (
                f"k_chunk={k},max_cores={c}",
                ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=k, max_cores_per_head_batch=c
                ),
            )
            for k in (32, 128, 512)
            for c in (32, 64)
        ]:
            try:
                ms = bench(mesh, 16384, cur, pc)
                print(f"cur_pos {cur:6d}  {name:<34} {ms*1000:9.2f} us", flush=True)
                out.append((cur, name, ms))
            except Exception as e:
                print(f"cur_pos {cur:6d}  {name:<34} FAILED: {str(e).splitlines()[0][:110]}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
json.dump(out, open(P + "/sdpa_progcfg_probe.json", "w"), indent=2)
