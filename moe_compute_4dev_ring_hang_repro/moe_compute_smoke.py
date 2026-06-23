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
import os
TOTAL_TOKENS = 64          # fixed batch
CLUSTER_AXIS = int(os.environ.get("SMOKE_CLUSTER_AXIS", "0"))  # 0=4-dev ring, 1=8-dev ring
REPLICATE_AXIS = 1 - CLUSTER_AXIS  # epilogue cross-cluster reduce axis
L1 = 1 << 15

NUM_DEV = MESH_SHAPE[0] * MESH_SHAPE[1]            # 32
NUM_DISPATCH = MESH_SHAPE[CLUSTER_AXIS]            # axis0->4, axis1->8
NUM_REPLICATED = NUM_DEV // NUM_DISPATCH           # axis0->8, axis1->4
EXPERTS_PER_DEV = EXPERTS // NUM_DEV               # 5
EXPERTS_PER_CLUSTER = EXPERTS // NUM_REPLICATED    # axis0->20, axis1->40
TOKENS_PER_DEV = TOTAL_TOKENS // NUM_DISPATCH      # axis0->16, axis1->8
DRAIN_CORE = ttnn.CoreCoord(6, 9)
OUTPUT_HEIGHT_SHARD_DIM = 4


def linearized_coord(e):
    # get_linearized_mesh_coord: cluster_axis=0 is column-major within cluster;
    # cluster_axis=1 is row-major identity (device = e // experts_per_device).
    if CLUSTER_AXIS == 0:
        cluster_id = e // EXPERTS_PER_CLUSTER
        eic = e % EXPERTS_PER_CLUSTER
        dev_in_cluster = eic // EXPERTS_PER_DEV
        return dev_in_cluster * NUM_REPLICATED + cluster_id
    return e // EXPERTS_PER_DEV


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
    shard_dims = (0, None) if CLUSTER_AXIS == 0 else (None, 0)
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

        # Router output, batch-sharded along cluster_axis=0 like the real model:
        # global TOTAL_TOKENS=64 tokens, dims=(0,None) -> 16/row, replicated cols.
        # (all_to_all_dispatch_metadata REQUIRES the input sharded along cluster
        # axis; a replicated input leaves the fabric in a state that deadlocks the
        # subsequent cross-col CCL.)
        idx_t = torch.stack([torch.randperm(EXPERTS)[:K] for _ in range(TOTAL_TOKENS)]).to(torch.int32)
        idx_t = idx_t.view(TOTAL_TOKENS, 1, 1, K)
        scr_t = torch.rand(TOTAL_TOKENS, 1, 1, K, dtype=torch.float32)
        x_t = (torch.randn(TOTAL_TOKENS, 1, 1, H) * 0.05)

        in_shard_dims = (0, None) if CLUSTER_AXIS == 0 else (None, 0)

        def to_dev(t, dt):
            return ttnn.from_torch(t, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt,
                                   memory_config=dram,
                                   mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, in_shard_dims))
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

        combine_prealloc = ttnn.moreh_full(shape=[K, TOKENS_PER_DEV, H], fill_value=0, device=device,
                                           layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                                           memory_config=ttnn.DRAM_MEMORY_CONFIG)
        outs = ttnn.experimental.moe_compute(
            sparse, disp_idx, disp_scr, lin_map, tt_w0w1, tt_w2,
            layer_id=0, output_height_shard_dim=OUTPUT_HEIGHT_SHARD_DIM, intermediate_size=N,
            has_bias=False, cluster_axis=CLUSTER_AXIS, mux_core_range_set=mux_cores,
            optional_output_tensor=combine_prealloc, optional_cross_device_semaphore=combine_sem)
        combine_output = outs[-1]
        print("moe_compute ok: returned", len(outs), "combine", tuple(combine_output.shape), flush=True)
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
        rs = ttnn.reduce_scatter(input_tensor=summed, dim=3, cluster_axis=REPLICATE_AXIS, subdevice_id=None,
                                 memory_config=dram, num_links=None, topology=ttnn.Topology.Linear,
                                 compute_kernel_config=hifi4)
        print("  ep5 reduce_scatter", tuple(rs.shape), flush=True)
        ag = ttnn.all_gather(input_tensor=rs, dim=3, cluster_axis=REPLICATE_AXIS, subdevice_id=None,
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
