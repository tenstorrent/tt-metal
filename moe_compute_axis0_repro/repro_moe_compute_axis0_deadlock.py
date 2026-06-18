"""Minimal repro: a cluster_axis=0 CCL (or a bare synchronize_device) deadlocks
after ttnn.experimental.moe_compute on a WH galaxy (4,8) mesh.

moe_compute fuses an all-to-all dispatch + expert matmuls + all-to-all combine
over the `cluster_axis` (here axis 0 = the 4 mesh rows). The combine produces a
correct output tensor, but afterward the cluster_axis=0 fabric is left in a
non-terminating state: any subsequent cluster_axis=0 CCL hangs, and even a bare
ttnn.synchronize_device() hangs. cluster_axis=1 CCLs after moe_compute are fine.

This mirrors a real GLM-4.7 decode model where the lm_head's cluster_axis=0
all_gather (gather batch across the 4 rows) deadlocks right after the MoE layer.

Config: WH galaxy, mesh (4,8), FABRIC_1D_RING, DispatchCoreAxis.COL.
GLM-4.7 layer-3 MoE dims: hidden 5120, intermediate 1536, 160 experts, k=8.

Run (inside the tt-metal venv, galaxy):
  python repro_moe_compute_axis0_deadlock.py [--probe sync|axis0|axis1]
    --probe sync  : after moe_compute, call synchronize_device  (HANGS)
    --probe axis0 : after moe_compute, all_gather(dim=0, cluster_axis=0) (HANGS)
    --probe axis1 : after moe_compute, all_gather(dim=1, cluster_axis=1) (works
                    when the axis was warmed up by a prior axis-1 CCL; see note)
Reset the galaxy after a hang: `tt-smi -glx_reset_auto` on the host.
"""
import argparse
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
TOKENS_PER_DEV = 16
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


def linearized_coord(e):
    cluster_id = e // EXPERTS_PER_CLUSTER
    eic = e % EXPERTS_PER_CLUSTER
    dev_in_cluster = eic // EXPERTS_PER_DEV
    return dev_in_cluster * NUM_REPLICATED + cluster_id


def build_linearized_expert_mapping(device):
    m = torch.zeros(1, EXPERTS, dtype=torch.int64)
    for e in range(EXPERTS):
        m[0, e] = linearized_coord(e)
    m = m.repeat(NUM_DEV, 1).to(torch.int32)  # [32,160], replicated
    return ttnn.from_torch(m, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG,
                           mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (None, None)))


def build_weights(device):
    gate = torch.randn(EXPERTS, N, H) * 0.02
    up = torch.randn(EXPERTS, N, H) * 0.02
    down = torch.randn(EXPERTS, H, N) * 0.02
    gate_t = gate.transpose(-1, -2).contiguous()   # [E,H,N]
    up_t = up.transpose(-1, -2).contiguous()
    down_t = down.transpose(-1, -2).contiguous()   # [E,N,H]
    w0w1_map, w2_map, dram_crs = get_weight_core_shard_maps(device, H, N)
    w0w1_pd = [None] * NUM_DEV
    w2_pd = [None] * NUM_DEV
    for e in range(0, EXPERTS, EXPERTS_PER_DEV):
        w0 = torch.cat([gate_t[e + j].view(1, 1, H, N) for j in range(EXPERTS_PER_DEV)], dim=1)
        w1 = torch.cat([up_t[e + j].view(1, 1, H, N) for j in range(EXPERTS_PER_DEV)], dim=1)
        w2 = torch.cat([down_t[e + j].view(1, 1, N, H) for j in range(EXPERTS_PER_DEV)], dim=1)
        d = linearized_coord(e)
        w0w1_pd[d] = prepare_w0_w1_tensor_for_moe_compute(w0, w1, 1, EXPERTS_PER_DEV, H, N, w0w1_map)
        w2_pd[d] = prepare_w2_tensor_for_moe_compute(w2, 1, EXPERTS_PER_DEV, N, H, w2_map, w0w1_map)
    w0w1_mem, w2_mem, _, _ = get_weight_mem_configs(1, EXPERTS_PER_DEV, H, N, w0w1_map, w2_map, dram_crs)
    tt_w0w1 = ttnn.from_torch(torch.cat(w0w1_pd, dim=0), device=device, layout=ttnn.TILE_LAYOUT,
                              dtype=ttnn.bfloat4_b, memory_config=w0w1_mem,
                              mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))
    tt_w2 = ttnn.from_torch(torch.cat(w2_pd, dim=0), device=device, layout=ttnn.TILE_LAYOUT,
                            dtype=ttnn.bfloat4_b, memory_config=w2_mem,
                            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))
    return tt_w0w1, tt_w2


def build_dispatch_prealloc(device):
    shard_dims = (CLUSTER_AXIS, None)
    dram = ttnn.DRAM_MEMORY_CONFIG
    sparse = ttnn.from_torch(torch.zeros(NUM_DISPATCH, TOTAL_TOKENS, H, dtype=torch.bfloat16),
                             device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                             memory_config=dram,
                             mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, shard_dims))
    spec = ttnn.ShardSpec(ttnn.CoreRangeSet({ttnn.CoreRange(DRAIN_CORE, DRAIN_CORE)}),
                          [TOTAL_TOKENS, K], ttnn.ShardOrientation.ROW_MAJOR)
    l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)
    idx = ttnn.from_torch(torch.zeros(NUM_DISPATCH, TOTAL_TOKENS, K, dtype=torch.int32),
                          device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint16,
                          memory_config=l1, mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, shard_dims))
    scr = ttnn.from_torch(torch.zeros(NUM_DISPATCH, TOTAL_TOKENS, K, dtype=torch.float32),
                          device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                          memory_config=l1, mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, shard_dims))
    return (sparse, idx, scr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["sync", "axis0", "axis1", "model_seq"], default="sync")
    ap.add_argument("--warmup", type=int, default=0,
                    help="rounds of prior cluster_axis=0 all_gather + cluster_axis=1 "
                         "reduce_scatter/all_gather BEFORE dispatch+moe_compute, to mimic the "
                         "dense layers + router axis-0 traffic that precede the MoE in the model")
    args = ap.parse_args()

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    dcfg = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.COL)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(MESH_SHAPE), l1_small_size=L1,
                                   dispatch_core_config=dcfg)
    try:
        dram = ttnn.DRAM_MEMORY_CONFIG
        lin = build_linearized_expert_mapping(device)
        tt_w0w1, tt_w2 = build_weights(device)
        prealloc = build_dispatch_prealloc(device)
        grid = device.compute_with_storage_grid_size()
        worker = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                   ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        dsem = ttnn.create_global_semaphore(device, worker, 0)
        csem = ttnn.create_global_semaphore(device, worker, 0)
        mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(4, 7))])

        idx_t = torch.stack([torch.randperm(EXPERTS)[:K] for _ in range(TOTAL_TOKENS)]).to(torch.int32)
        idx_t = idx_t.view(TOTAL_TOKENS, 1, 1, K)
        scr_t = torch.rand(TOTAL_TOKENS, 1, 1, K)
        x_t = torch.randn(TOTAL_TOKENS, 1, 1, H) * 0.05

        def to_dev(t, dt):
            return ttnn.from_torch(t, device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dt,
                                   memory_config=dram,
                                   mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (0, None)))
        x = to_dev(x_t, ttnn.bfloat16)
        idx = to_dev(idx_t, ttnn.uint16)
        scr = to_dev(scr_t, ttnn.bfloat16)

        # Warmup: mimic the dense layers + router axis-0/axis-1 CCL traffic that runs
        # before the MoE layer in the full model (accumulated fabric/core state).
        for r in range(args.warmup):
            w0 = ttnn.from_torch(torch.randn(NUM_DISPATCH, TOKENS_PER_DEV, H) * 0.05,
                                 device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                 memory_config=dram, mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (0, None)))
            # cluster_axis=0 all_gather (router-like)
            g0 = ttnn.all_gather(input_tensor=w0, dim=0, cluster_axis=0, subdevice_id=None,
                                 memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
            ttnn.deallocate(w0, False)
            ttnn.deallocate(g0, False)
            # cluster_axis=1 reduce_scatter + all_gather (dense-MLP-like)
            w1 = ttnn.from_torch(torch.randn(1, 1, TOKENS_PER_DEV, H) * 0.05,
                                 device=device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16,
                                 memory_config=dram, mesh_mapper=ttnn.ReplicateTensorToMesh(device))
            rs = ttnn.reduce_scatter(input_tensor=w1, dim=3, cluster_axis=1, subdevice_id=None,
                                     memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
            rs = ttnn.reshape(rs, [TOKENS_PER_DEV, H // NUM_REPLICATED], memory_config=dram)
            ag = ttnn.all_gather(input_tensor=rs, dim=1, cluster_axis=1, subdevice_id=None,
                                 memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
            ttnn.deallocate(w1, False)
            ttnn.deallocate(rs, False)
            ttnn.deallocate(ag, False)
        if args.warmup:
            print(f">>> warmup ({args.warmup} rounds) done", flush=True)

        sparse, d_idx, d_scr = ttnn.experimental.all_to_all_dispatch_metadata(
            x, idx, scr, lin, cluster_axis=CLUSTER_AXIS, num_links=4,
            worker_mode=ttnn.WorkerMode.DIRECT,
            dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
            output_tensors=prealloc, cross_device_semaphore=dsem)
        print(">>> dispatch_metadata done", flush=True)

        combine_prealloc = ttnn.moreh_full(shape=[K, TOKENS_PER_DEV, H], fill_value=0, device=device,
                                           layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                                           memory_config=dram)
        outs = ttnn.experimental.moe_compute(
            sparse, d_idx, d_scr, lin, tt_w0w1, tt_w2,
            layer_id=0, output_height_shard_dim=OUTPUT_HEIGHT_SHARD_DIM, intermediate_size=N,
            has_bias=False, cluster_axis=CLUSTER_AXIS, mux_core_range_set=mux,
            optional_output_tensor=combine_prealloc, optional_cross_device_semaphore=csem)
        combine = outs[-1]
        print(">>> moe_compute done; combine", tuple(combine.shape), flush=True)

        if args.probe == "sync":
            print(">>> calling synchronize_device (EXPECT HANG) ...", flush=True)
            ttnn.synchronize_device(device)
            print(">>> synchronize_device RETURNED (no hang)", flush=True)
        elif args.probe == "axis0":
            c = ttnn.to_layout(combine, ttnn.TILE_LAYOUT, memory_config=dram)
            c = ttnn.reshape(c, [K * TOKENS_PER_DEV, H], memory_config=dram)
            print(">>> calling all_gather dim=0 cluster_axis=0 (EXPECT HANG) ...", flush=True)
            g = ttnn.all_gather(input_tensor=c, dim=0, cluster_axis=0, subdevice_id=None,
                                memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
            print(">>> axis-0 all_gather RETURNED (no hang)", tuple(g.shape), flush=True)
        elif args.probe == "axis1":
            c = ttnn.to_layout(combine, ttnn.TILE_LAYOUT, memory_config=dram)
            c = ttnn.reshape(c, [K * TOKENS_PER_DEV, H], memory_config=dram)
            print(">>> calling all_gather dim=1 cluster_axis=1 ...", flush=True)
            g = ttnn.all_gather(input_tensor=c, dim=1, cluster_axis=1, subdevice_id=None,
                                memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
            print(">>> axis-1 all_gather RETURNED", tuple(g.shape), flush=True)
        elif args.probe == "model_seq":
            # Mimic the GLM model: moe_compute -> axis-1 epilogue (reduce_scatter
            # +all_gather) -> lm_head-shaped axis-0 all_gather. This is the exact
            # sequence that deadlocks in the full model.
            hifi4 = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4,
                                                     math_approx_mode=False, fp32_dest_acc_en=True,
                                                     packer_l1_acc=False)
            c = ttnn.to_layout(combine, ttnn.TILE_LAYOUT, memory_config=dram)
            c = ttnn.unsqueeze(c, dim=1)                                  # [8,1,16,5120]
            summed = ttnn.sum(c, [0], False, memory_config=dram)          # [1,16,5120]
            summed = ttnn.reshape(summed, [1, 1, TOKENS_PER_DEV, H], memory_config=dram)
            rs = ttnn.reduce_scatter(input_tensor=summed, dim=3, cluster_axis=1, subdevice_id=None,
                                     memory_config=dram, num_links=None, topology=ttnn.Topology.Linear,
                                     compute_kernel_config=hifi4)
            rs = ttnn.reshape(rs, [TOKENS_PER_DEV, H // NUM_REPLICATED], memory_config=dram)
            ag1 = ttnn.all_gather(input_tensor=rs, dim=1, cluster_axis=1, subdevice_id=None,
                                  memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
            print(">>> axis-1 epilogue done", tuple(ag1.shape), flush=True)
            # lm_head-shaped axis-0 all_gather: [16,1,18944] per device (batch on
            # rows, vocab-shard on cols), gather batch across the 4 rows -> [64,1,18944].
            VOCAB_SHARD = 18944
            logits = ttnn.from_torch(
                torch.zeros(TOTAL_TOKENS, 1, VOCAB_SHARD, dtype=torch.bfloat16),
                device=device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16,
                memory_config=dram, mesh_mapper=ttnn.ShardTensor2dMesh(device, MESH_SHAPE, (0, None)))
            print(">>> calling lm_head-style all_gather dim=0 cluster_axis=0 (EXPECT HANG) ...", flush=True)
            g = ttnn.all_gather(input_tensor=logits, dim=0, cluster_axis=0, subdevice_id=None,
                                memory_config=dram, num_links=None, topology=ttnn.Topology.Linear)
            print(">>> lm_head-style axis-0 all_gather RETURNED", tuple(g.shape), flush=True)
        ttnn.synchronize_device(device)
        print(">>> REPRO COMPLETED WITHOUT HANG", flush=True)
    finally:
        ttnn.close_mesh_device(device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
