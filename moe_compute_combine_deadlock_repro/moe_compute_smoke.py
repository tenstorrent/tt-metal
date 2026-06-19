"""Standalone smoke test for the moe_compute fused MoE path with GLM-4.7 shapes.

Uses RANDOM weights/activations (no HF load) so it iterates in ~1-2 min instead of
~6 min, isolating moe_compute plumbing (weight packing, dispatch_metadata, moe_compute,
epilogue) from the rest of the model. Validates the path RUNS and produces [16, 5120].

Run inside docker:
  cd .../graph_0 && python moe_compute_smoke.py
"""
import math
import torch
import ttnn
from ttnn.experimental.moe_compute_utils import (
    get_weight_core_shard_maps,
    get_weight_mem_configs,
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
)

MESH_SHAPE = (4, 8)
H = 5120          # hidden
N = 1536          # moe intermediate
EXPERTS = 160
K = 8             # experts per token
TOKENS_PER_DEV = 16        # batch 64 / 4 dispatch devices
CLUSTER_AXIS = 0
L1 = 1 << 15

NUM_DEV = MESH_SHAPE[0] * MESH_SHAPE[1]            # 32
NUM_DISPATCH = MESH_SHAPE[CLUSTER_AXIS]            # 4
NUM_REPLICATED = NUM_DEV // NUM_DISPATCH           # 8
EXPERTS_PER_DEV = EXPERTS // NUM_DEV               # 5
EXPERTS_PER_CLUSTER = EXPERTS // NUM_REPLICATED    # 20
TOTAL_TOKENS = TOKENS_PER_DEV * NUM_DISPATCH       # 64
DRAIN_CORE = ttnn.CoreCoord(6, 9)
OUTPUT_HEIGHT_SHARD_DIM = 4

# --- Full-model device footprint (the smoke<->full-model distinguisher) ---------
# The standalone smoke passes; the full model hangs in moe_compute's fused combine.
# Everything config/version-level is ruled out, so the remaining difference is the
# RESIDENT NON-MoE WEIGHT FOOTPRINT (smoke only allocates the bf4 experts). This
# block allocates dummy tensors matching the full model's dominant device tensors
# (real GLM-4.7 dims + the exact mesh placement from params.py/consteval.py/main.py)
# and holds them resident across moe_compute, to test the memory-pressure hypothesis.
#   SMOKE_FULL_FOOTPRINT=1            -> allocate ALL of: embed, lmhead, kv, attn, dense
#   SMOKE_FP=embed,kv                 -> allocate only the named subset (for bisection)
# Dominant per-device consumer: embed_tokens REPLICATED [151552,5120] bf16 ~1.55 GB/dev.
VOCAB = 151552
NUM_ATTN_HEADS = 96
NUM_KV_HEADS = 8
HEAD_DIM = 128
KV_BATCH = 64
KV_SEQ = 128            # INPUT_SEQUENCE_LENGTH (StaticCache max_cache_len)
NUM_LAYERS = 4          # 4-layer test model (L0-2 dense, L3 MoE)
DENSE_INTERMEDIATE = 12288


def _fp_parts():
    import os
    sel = os.environ.get("SMOKE_FP", "")
    if sel:
        return set(p.strip() for p in sel.split(",") if p.strip())
    if os.environ.get("SMOKE_FULL_FOOTPRINT") == "1":
        return {"embed", "lmhead", "kv", "attn", "dense"}
    return set()


def build_full_footprint(device):
    """Allocate dummy versions of the full model's dominant resident device tensors,
    matching params.py/consteval.py/main.py placement. Returns a hold-list (kept alive
    so the allocations stay resident across moe_compute)."""
    parts = _fp_parts()
    if not parts:
        return []
    hold = []
    repl = ttnn.ReplicateTensorToMesh(device)
    # SHARD_DIM0 in params.py = ShardTensor2dMesh (None, 0): tensor dim0 over the 8 cols.
    shard0 = ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (None, 0))
    dram = ttnn.DRAM_MEMORY_CONFIG

    def alloc(name, t, dtype, layout, mapper):
        x = ttnn.from_torch(t, device=device, layout=layout, dtype=dtype,
                            memory_config=dram, mesh_mapper=mapper)
        hold.append(x)
        print(f"  footprint+ {name} {tuple(t.shape)} {dtype}", flush=True)

    if "embed" in parts:
        # consteval: embed_tokens.weight.device, REPLICATED, DRAM. The 1.55 GB/dev one.
        alloc("embed_tokens(repl)", torch.zeros(VOCAB, H, dtype=torch.bfloat16),
              ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, repl)
    if "lmhead" in parts:
        # lm_head.weight SHARD_DIM0 (vocab over 8 cols) then consteval bf8b-transposed.
        alloc("lm_head(bf8b,shard0)", torch.zeros(VOCAB, H, dtype=torch.bfloat16),
              ttnn.bfloat8_b, ttnn.TILE_LAYOUT, shard0)
    if "kv" in parts:
        # main.py KV caches: per layer keys+values, head-sharded (0,1), TILE bf16, DRAM.
        kv_map = ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (0, 1))
        for L in range(NUM_LAYERS):
            for nm in ("keys", "values"):
                alloc(f"L{L}.{nm}",
                      torch.zeros(KV_BATCH, NUM_KV_HEADS, KV_SEQ, HEAD_DIM, dtype=torch.bfloat16),
                      ttnn.bfloat16, ttnn.TILE_LAYOUT, kv_map)
    if "attn" in parts:
        # q/k/v/o per layer, SHARD_DIM0 (q/k/v) and SHARD_DIM1 (o). bf16 DRAM.
        shard1 = ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (None, 1))
        q_out = NUM_ATTN_HEADS * HEAD_DIM
        kv_out = NUM_KV_HEADS * HEAD_DIM
        for L in range(NUM_LAYERS):
            alloc(f"L{L}.q_proj", torch.zeros(q_out, H, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard0)
            alloc(f"L{L}.k_proj", torch.zeros(kv_out, H, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard0)
            alloc(f"L{L}.v_proj", torch.zeros(kv_out, H, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard0)
            alloc(f"L{L}.o_proj", torch.zeros(H, q_out, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard1)
    if "dense" in parts:
        # dense MLP gate/up (SHARD_DIM0) + down (SHARD_DIM1) for layers 0-2. bf16 DRAM.
        shard1 = ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (None, 1))
        for L in range(3):
            alloc(f"L{L}.gate", torch.zeros(DENSE_INTERMEDIATE, H, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard0)
            alloc(f"L{L}.up", torch.zeros(DENSE_INTERMEDIATE, H, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard0)
            alloc(f"L{L}.down", torch.zeros(H, DENSE_INTERMEDIATE, dtype=torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT, shard1)
    print(f"FULL FOOTPRINT allocated: parts={sorted(parts)}, {len(hold)} tensors held", flush=True)
    return hold


def linearized_coord(e):
    cluster_id = e // EXPERTS_PER_CLUSTER
    eic = e % EXPERTS_PER_CLUSTER
    dev_in_cluster = eic // EXPERTS_PER_DEV
    return dev_in_cluster * NUM_REPLICATED + cluster_id


def build_one_hot_expert_mapping(device):
    # column-major device_of_expert == linearized_coord (verified). One-hot [1,1,E,D].
    dev_of_e = torch.tensor([linearized_coord(e) for e in range(EXPERTS)], dtype=torch.int64)
    m = torch.zeros(1, 1, EXPERTS, NUM_DEV, dtype=torch.int32)
    m[0, 0, torch.arange(EXPERTS), dev_of_e] = 1
    t = ttnn.from_torch(m, dtype=ttnn.uint16, layout=ttnn.ROW_MAJOR_LAYOUT,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(device))
    return ttnn.to_device(t, device, ttnn.DRAM_MEMORY_CONFIG)


def build_linearized_expert_mapping(device):
    m = torch.zeros(1, EXPERTS, dtype=torch.int64)
    for e in range(EXPERTS):
        m[0, e] = linearized_coord(e)
    m = m.repeat(NUM_DEV, 1).to(torch.int32)  # [32,160]
    return ttnn.from_torch(m, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(None, None), mesh_shape=MESH_SHAPE))


def build_weights(device):
    # raw HF orientation: gate/up [E, N, H], down [E, H, N]
    gate = torch.randn(EXPERTS, N, H, dtype=torch.float32) * 0.02
    up = torch.randn(EXPERTS, N, H, dtype=torch.float32) * 0.02
    down = torch.randn(EXPERTS, H, N, dtype=torch.float32) * 0.02
    # transpose to prepare orientation: w0/w1 (H,N), w2 (N,H)
    gate_t = gate.transpose(-1, -2).contiguous()   # [E,H,N]
    up_t = up.transpose(-1, -2).contiguous()       # [E,H,N]
    down_t = down.transpose(-1, -2).contiguous()   # [E,N,H]

    w0w1_map, w2_map, dram_crs = get_weight_core_shard_maps(device, H, N)
    w0w1_per_dev = [None] * NUM_DEV
    w2_per_dev = [None] * NUM_DEV
    for e in range(0, EXPERTS, EXPERTS_PER_DEV):
        w0 = torch.cat([gate_t[e + j].view(1, 1, H, N) for j in range(EXPERTS_PER_DEV)], dim=1)
        w1 = torch.cat([up_t[e + j].view(1, 1, H, N) for j in range(EXPERTS_PER_DEV)], dim=1)
        w2 = torch.cat([down_t[e + j].view(1, 1, N, H) for j in range(EXPERTS_PER_DEV)], dim=1)
        w0w1_r = prepare_w0_w1_tensor_for_moe_compute(w0, w1, 1, EXPERTS_PER_DEV, H, N, w0w1_map)
        w2_r = prepare_w2_tensor_for_moe_compute(w2, 1, EXPERTS_PER_DEV, N, H, w2_map, w0w1_map)
        d = linearized_coord(e)
        w0w1_per_dev[d] = w0w1_r
        w2_per_dev[d] = w2_r
    torch_w0w1 = torch.cat(w0w1_per_dev, dim=0)
    torch_w2 = torch.cat(w2_per_dev, dim=0)
    print("packed w0w1", tuple(torch_w0w1.shape), "w2", tuple(torch_w2.shape))
    w0w1_mem, w2_mem, _, _ = get_weight_mem_configs(1, EXPERTS_PER_DEV, H, N, w0w1_map, w2_map, dram_crs)
    tt_w0w1 = ttnn.from_torch(torch_w0w1, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b,
                              memory_config=w0w1_mem, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))
    tt_w2 = ttnn.from_torch(torch_w2, device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b,
                            memory_config=w2_mem, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))
    return tt_w0w1, tt_w2


def build_dispatch_prealloc(device):
    shard_dims = (CLUSTER_AXIS, None) if CLUSTER_AXIS == 0 else (None, CLUSTER_AXIS)
    sparse = ttnn.from_torch(torch.zeros(NUM_DISPATCH, TOTAL_TOKENS, H, dtype=torch.bfloat16),
                             device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG,
                             mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=MESH_SHAPE))
    shard_spec = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(DRAIN_CORE, DRAIN_CORE)}),
                                [TOTAL_TOKENS, K], ttnn.ShardOrientation.ROW_MAJOR)
    idx_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    idx = ttnn.from_torch(torch.zeros(NUM_DISPATCH, TOTAL_TOKENS, K, dtype=torch.int32),
                          device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16,
                          memory_config=idx_mem,
                          mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=MESH_SHAPE))
    scr = ttnn.from_torch(torch.zeros(NUM_DISPATCH, TOTAL_TOKENS, K, dtype=torch.float32),
                          device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                          memory_config=idx_mem,
                          mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=shard_dims, mesh_shape=MESH_SHAPE))
    return (sparse, idx, scr)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    # moe_compute hardwires tilize cores at the (5-6, 8-9) grid corner; with the
    # default ROW dispatch axis the DRAM matmul-core assignment spans the whole
    # 8x9 grid and overlaps them ("tilize and matmul bounding boxes cannot
    # overlap"). COL dispatch axis (what the deepseek TG reference uses) reshapes
    # the usable grid so they don't collide.
    dispatch_cfg = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), l1_small_size=L1,
                                   dispatch_core_config=dispatch_cfg)
    try:
        dram = ttnn.DRAM_MEMORY_CONFIG
        one_hot = build_one_hot_expert_mapping(device)
        lin_map = build_linearized_expert_mapping(device)
        tt_w0w1, tt_w2 = build_weights(device)
        prealloc = build_dispatch_prealloc(device)

        grid = device.compute_with_storage_grid_size()
        worker_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                         ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        dispatch_sem = ttnn.create_global_semaphore(device, worker_cores, 0)
        combine_sem = ttnn.create_global_semaphore(device, worker_cores, 0)
        mux_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(4, 7))])

        # REPRO PROBE: mimic the full model's persistent KV-cache L1 shards on
        # (0,0)-(3,3), which overlap moe_compute's mux cores (x=3, y=0-3). If
        # holding this alloc across moe_compute makes the smoke hang, core
        # contention is the full-model deadlock cause. Gated on SMOKE_KV_HOLD=1.
        import os as _os
        _kv_hold = None
        if _os.environ.get("SMOKE_KV_HOLD") == "1":
            _kv_mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))]),
                    [32, 128], ttnn.ShardOrientation.ROW_MAJOR))
            _kv_hold = ttnn.from_torch(
                torch.zeros(16 * 32, 128, dtype=torch.bfloat16),
                device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                memory_config=_kv_mem,
                mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (None, None)))
            print("SMOKE_KV_HOLD: persistent L1 KV-like tensor on (0,0)-(3,3)", flush=True)

        # Full-model device footprint probe (resident across moe_compute). If this
        # flips the smoke from PASS to HANG, the resident footprint is the
        # smoke<->full-model trigger; SMOKE_FP=<subset> then bisects which tensor.
        _footprint_hold = build_full_footprint(device)
        # Decisive control: allocate the footprint, then FREE it before moe_compute.
        # If freeing makes the smoke PASS again, it's the RESIDENT device occupancy
        # (memory map) that breaks the post-moe_compute CCL -- not a transient
        # upload/host perturbation. SMOKE_FP_FREE=1.
        if _footprint_hold and _os.environ.get("SMOKE_FP_FREE") == "1":
            for _t in _footprint_hold:
                ttnn.deallocate(_t, False)
            _footprint_hold = []
            ttnn.synchronize_device(device)
            print("SMOKE_FP_FREE: footprint deallocated before moe_compute", flush=True)

        # Router output, batch-sharded along cluster_axis=0 like the real model:
        # global TOTAL_TOKENS=64 tokens, dims=(0,None) -> 16/row, replicated cols.
        # (all_to_all_dispatch_metadata REQUIRES the input sharded along cluster
        # axis; a replicated input leaves the fabric in a state that deadlocks the
        # subsequent cross-col CCL.)
        idx_t = torch.stack([torch.randperm(EXPERTS)[:K] for _ in range(TOTAL_TOKENS)]).to(torch.int32)
        idx_t = idx_t.view(TOTAL_TOKENS, 1, 1, K)
        scr_t = torch.rand(TOTAL_TOKENS, 1, 1, K, dtype=torch.float32)
        x_t = (torch.randn(TOTAL_TOKENS, 1, 1, H) * 0.05)

        def to_dev(t, dt):
            return ttnn.from_torch(t, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt,
                                   memory_config=dram,
                                   mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (0, None)))
        x = to_dev(x_t, ttnn.bfloat16)
        idx = to_dev(idx_t, ttnn.uint16)
        scr = to_dev(scr_t, ttnn.bfloat16)

        # NOTE: build-tree dispatch_metadata asserts expert_mapping rank-2
        # [devices, experts] (the linearized form) -- NOT the one-hot [1,1,E,D]
        # the docstring claims. So BOTH ops take lin_map.
        sparse, disp_idx, disp_scr = ttnn.experimental.all_to_all_dispatch_metadata(
            x, idx, scr, lin_map, cluster_axis=CLUSTER_AXIS, num_links=4,
            worker_mode=ttnn.WorkerMode.DIRECT,
            dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
            output_tensors=prealloc, cross_device_semaphore=dispatch_sem)
        print("dispatch ok: sparse", tuple(sparse.shape), "idx", tuple(disp_idx.shape), "scr", tuple(disp_scr.shape))

        # Confirmation test: moe_compute MATMUL path only (compute_only=True, NO fused
        # combine ring). If this + sync PASSES, the device + dispatch_metadata +
        # moe_compute's matmul all work on this build, isolating the hang to the fused
        # selective_reduce_combine (not "everything hangs"). SMOKE_COMPUTE_ONLY=1.
        if _os.environ.get("SMOKE_COMPUTE_ONLY") == "1":
            co = ttnn.experimental.moe_compute(
                sparse, disp_idx, disp_scr, lin_map, tt_w0w1, tt_w2,
                layer_id=0, output_height_shard_dim=OUTPUT_HEIGHT_SHARD_DIM,
                intermediate_size=N, has_bias=False, compute_only=True)
            print(">>> moe_compute(compute_only) enqueued, matmul_output", tuple(co[4].shape), flush=True)
            mm = ttnn.to_memory_config(co[4], memory_config=dram)
            ttnn.synchronize_device(device)
            print(">>> SMOKE SYNC after moe_compute(compute_only) OK", tuple(mm.shape), flush=True)
            print("SMOKE PASSED (compute_only)")
            return

        combine_prealloc = ttnn.moreh_full(shape=[K, TOKENS_PER_DEV, H], fill_value=0, device=device,
                                           layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # The fused combine is selective_reduce_combine internally. The smoke never
        # passed `topology`, so it resolved to Ring (FABRIC_1D_RING default) -- which
        # deadlocks. deepseek uses Topology.Linear for the combine on 4-row ("4U")
        # galaxy (our (4,8) cluster_axis=0 = 4 dispatch devices). SMOKE_COMBINE_TOPO=
        # linear|ring forces it explicitly (default: unset = library default = Ring).
        _topo = {"linear": ttnn.Topology.Linear, "line": ttnn.Topology.Linear,
                 "ring": ttnn.Topology.Ring}.get(_os.environ.get("SMOKE_COMBINE_TOPO", "").lower())
        _topo_kw = {} if _topo is None else dict(topology=_topo, num_links=1)
        if _topo is not None:
            print(f"SMOKE_COMBINE_TOPO: fused moe_compute topology={_topo}", flush=True)
        outs = ttnn.experimental.moe_compute(
            sparse, disp_idx, disp_scr, lin_map, tt_w0w1, tt_w2,
            layer_id=0, output_height_shard_dim=OUTPUT_HEIGHT_SHARD_DIM, intermediate_size=N,
            has_bias=False, cluster_axis=CLUSTER_AXIS, mux_core_range_set=mux_cores,
            optional_output_tensor=combine_prealloc, optional_cross_device_semaphore=combine_sem,
            **_topo_kw)
        combine_output = outs[-1]
        print("moe_compute ok: returned", len(outs), "combine", tuple(combine_output.shape), flush=True)
        # Pinpoint: does the fused combine itself DRAIN, or only enqueue? A sync here
        # that hangs => the combine never completes (full-model failure mode). A sync
        # that passes but ep5 reduce_scatter then hangs => combine completes but
        # poisons the subsequent cross-col CCL (smoke failure mode). SMOKE_PINPOINT=1.
        import os as _os2
        if _os2.environ.get("SMOKE_PINPOINT") == "1":
            ttnn.synchronize_device(device)
            print(">>> SMOKE SYNC after moe_compute combine OK", flush=True)
        # Free moe_compute's L1 metadata/matmul outputs before the cross-col CCL
        # (compute uses almost all of L1; reduce_scatter needs workspace).
        for i in (0, 1, 2, 4):
            try:
                ttnn.deallocate(outs[i], False)
            except Exception as e:
                print("  dealloc out", i, "skip:", repr(e), flush=True)
        print("freed moe_compute aux outputs", flush=True)

        # epilogue: scale by topk scores, sum over k, cross-col all-reduce
        c = ttnn.to_layout(combine_output, ttnn.TILE_LAYOUT, memory_config=dram)
        c = ttnn.unsqueeze(c, dim=1)                                   # [k,1,tokens,H]
        print("  ep1 to_layout+unsqueeze", tuple(c.shape), flush=True)
        scores_perm = ttnn.permute(scr, (3, 1, 0, 2), memory_config=dram)  # [k,1,tokens,1]
        scores_perm = ttnn.to_layout(scores_perm, ttnn.TILE_LAYOUT, memory_config=dram)
        print("  ep2 scores_perm", tuple(scores_perm.shape), flush=True)
        scaled = ttnn.mul(c, scores_perm, memory_config=dram)         # [k,1,tokens,H]
        print("  ep3 mul", tuple(scaled.shape), flush=True)
        summed = ttnn.sum(scaled, dim=0)                              # [1,tokens,H] partial over col experts
        summed = ttnn.reshape(summed, [1, 1, TOKENS_PER_DEV, H], memory_config=dram)
        print("  ep4 sum (col partial)", tuple(summed.shape), flush=True)
        # Cross-col all-reduce, exactly mirroring the working model MoE: rank-4,
        # dim=3, cluster_axis=1, Linear, HiFi4 reduce_scatter then all_gather.
        hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 math_approx_mode=False, fp32_dest_acc_en=True,
                                                 packer_l1_acc=False)
        rs = ttnn.reduce_scatter(input_tensor=summed, dim=3, cluster_axis=1, subdevice_id=None,
                                 memory_config=dram, num_links=None, topology=ttnn.Topology.Linear,
                                 compute_kernel_config=hifi4)
        print("  ep5 reduce_scatter", tuple(rs.shape), flush=True)
        ag = ttnn.all_gather(input_tensor=rs, dim=3, cluster_axis=1, subdevice_id=None,
                             memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
        print("  ep6 all_gather", tuple(ag.shape), flush=True)
        sparse_output = ttnn.reshape(ag, [TOKENS_PER_DEV, H], memory_config=dram)
        print("epilogue ok: sparse_output", tuple(sparse_output.shape))
        ttnn.synchronize_device(device)
        print("SMOKE PASSED")
    finally:
        ttnn.close_mesh_device(device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
