import sys
import time

import torch

import ttnn

P = "/home/raahem/tt-metal/models/autoports/qwen_qwen3_coder_30b_a3b_instruct/doc/optimized_full_model/probes"
sys.path.insert(0, P)
from sdpa_depth_probe import BATCH, HEAD_DIM, MESH_SHAPE, N_KV_HEADS, N_Q_HEADS, PAGE


def mk(mesh, depth, cur_pos, seed=0):
    torch.manual_seed(seed)
    pages = depth // PAGE
    kt, vt = [torch.randn((pages, N_KV_HEADS, PAGE, HEAD_DIM)).float() for _ in range(2)]
    caches = [
        ttnn.from_torch(
            t,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        for t in (kt, vt)
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
    return q, caches, pt, pos


def run(q, caches, pt, pos, pc):
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q,
        caches[0],
        caches[1],
        page_table_tensor=pt,
        cur_pos_tensor=pos,
        scale=HEAD_DIM**-0.5,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
    )


def bench(mesh, q, caches, pt, pos, pc, iters=30):
    o = run(q, caches, pt, pos, pc)
    ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(iters):
        o = run(q, caches, pt, pos, pc)
        ttnn.deallocate(o)
    ttnn.synchronize_device(mesh)
    return 1e3 * (time.perf_counter() - t) / iters


mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*MESH_SHAPE))
grid = mesh.compute_with_storage_grid_size()
best = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=512, max_cores_per_head_batch=32
)
try:
    print("--- crossover sweep, depth 16384 ---")
    for cur in (127, 131, 255, 511, 1023, 2047, 4095):
        q, c, pt, pos = mk(mesh, 16384, cur)
        a = bench(mesh, q, c, pt, pos, None)
        b = bench(mesh, q, c, pt, pos, best)
        print(f"cur_pos {cur:6d}   None {a*1000:8.2f} us   k512/c32 {b*1000:8.2f} us   {a/b:5.2f}x", flush=True)
        for x in (q, pt, pos, *c):
            ttnn.deallocate(x)
    print("--- correctness: same output? depth 16384 ---")
    for cur in (131, 1023, 8191):
        q, c, pt, pos = mk(mesh, 16384, cur, seed=7)
        ra = ttnn.to_torch(run(q, c, pt, pos, None), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        rb = ttnn.to_torch(run(q, c, pt, pos, best), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        a = ra.flatten().float()
        b = rb.flatten().float()
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        print(f"cur_pos {cur:6d}   PCC {pcc:.8f}   max|diff| {(a-b).abs().max().item():.3e}", flush=True)
        for x in (q, pt, pos, *c):
            ttnn.deallocate(x)
finally:
    ttnn.close_mesh_device(mesh)
