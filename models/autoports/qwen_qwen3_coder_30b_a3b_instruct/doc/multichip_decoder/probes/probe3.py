import statistics
import time

import torch

import ttnn

# --- part 1: single-device SDPA / paged cache with 1 local KV head (TP=4) ---
dev = ttnn.open_device(device_id=0, trace_region_size=50_000_000, l1_small_size=32768)
B, HD, CTX, BLK = 1, 128, 128, 32
for nq, nkv, tag in [(32, 4, "single-die 32Q/4KV"), (8, 1, "TP4 8Q/1KV")]:
    try:
        npages = (CTX // BLK) * B
        kc = ttnn.from_torch(
            torch.randn(npages, nkv, BLK, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        vc = ttnn.from_torch(
            torch.randn(npages, nkv, BLK, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pt = ttnn.from_torch(torch.arange(npages).reshape(B, CTX // BLK).int(), dtype=ttnn.int32, device=dev)
        pos = ttnn.from_torch(torch.tensor([CTX - 1] * B).int(), dtype=ttnn.int32, device=dev)
        q = ttnn.from_torch(
            torch.randn(1, B, nq, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q, kc, vc, page_table_tensor=pt, cur_pos_tensor=pos, scale=HD**-0.5, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        print(f"P|{tag} paged SDPA decode OK out={list(o.shape)}")
        # paged_update_cache with 1 kv head
        upd = ttnn.from_torch(
            torch.randn(1, B, nkv, HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.experimental.paged_update_cache(kc, upd, update_idxs_tensor=pos, page_table=pt)
        print(f"P|{tag} paged_update_cache OK")
        for t in (kc, vc, pt, pos, q, o, upd):
            ttnn.deallocate(t)
    except Exception as e:
        print(f"P|{tag} ERR {str(e)[:250]}")

# prefill create-qkv-heads with 1 kv head
for nq, nkv, tag in [(32, 4, "single-die"), (8, 1, "TP4")]:
    try:
        S = 512
        x = ttnn.from_torch(
            torch.randn(1, 1, S, (nq + 2 * nkv) * HD),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            x, num_heads=nq, num_kv_heads=nkv, transpose_k_heads=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        print(f"P|{tag} prefill nlp_create_qkv_heads OK q{list(q.shape)} k{list(k.shape)}")
        o = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=HD**-0.5)
        print(f"P|{tag} prefill SDPA OK out={list(o.shape)}")
        for t in (x, q, k, v, o):
            ttnn.deallocate(t)
    except Exception as e:
        print(f"P|{tag} prefill ERR {str(e)[:250]}")
ttnn.close_device(dev)

# --- part 2: 2-device vs 4-device collective latency (for the 2x2 rejection) ---
ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=100_000_000, l1_small_size=32768)


def bench(md, n, rows, width):
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    sems = [ttnn.create_global_semaphore(md, cores, 0) for _ in range(4)]
    bar = ttnn.create_global_semaphore(md, cores, 0)
    x = ttnn.from_torch(
        torch.randn(1, 1, rows, width // n),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=md,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )
    topo = ttnn.Topology.Ring if n > 2 else ttnn.Topology.Linear
    f = lambda: ttnn.experimental.all_gather_async(
        x,
        dim=3,
        multi_device_global_semaphore=sems[0:2],
        barrier_semaphore=bar,
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topo,
    )

    def cap(r):
        o = f()
        ttnn.synchronize_device(md)
        ttnn.deallocate(o)
        tid = ttnn.begin_trace_capture(md, cq_id=0)
        for _ in range(r):
            ttnn.deallocate(f())
        ttnn.end_trace_capture(md, tid, cq_id=0)
        ttnn.synchronize_device(md)
        ts = []
        for _ in range(25):
            t0 = time.perf_counter()
            ttnn.execute_trace(md, tid, cq_id=0, blocking=True)
            ts.append((time.perf_counter() - t0) * 1e6)
        ttnn.release_trace(md, tid)
        return statistics.median(ts)

    a, b = cap(1), cap(17)
    ttnn.deallocate(x)
    return (b - a) / 16


try:
    print(f"P|AG 4-dev ring   b32 h2048 = {bench(mesh, 4, 32, 2048):.2f}us")
except Exception as e:
    print("P|AG4 ERR", str(e)[:200])
try:
    sm = mesh.create_submesh(ttnn.MeshShape(1, 2))
    print(f"P|AG 2-dev linear b32 h2048 = {bench(sm, 2, 32, 2048):.2f}us")
except Exception as e:
    print("P|AG2 ERR", str(e)[:200])
ttnn.close_mesh_device(mesh)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
