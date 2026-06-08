# SPDX-License-Identifier: Apache-2.0
"""De-risking prototype: run ttnn.experimental.moe_compute on the graph_0 (4,8) BH
mesh with synthetic weights, using the SAME manual tail (weighted-k-sum ->
all_reduce_async(axis1) -> mesh_partition) planned for the real moe_test.py
integration, and validate the final per-token MoE output vs a torch golden.

Config mirrors test_optimized_moe_decode_block (cluster_axis=0, hidden=7168,
N=2048, k=8) but for our 4x8 mesh: 4 dispatch devices, 8 replicated, 8 experts/device.
"""
import os, sys, time
from pathlib import Path
import torch

THIS_DIR = Path(__file__).resolve().parent
os.chdir(THIS_DIR)
sys.path.insert(0, str(THIS_DIR))
import ttnn  # noqa
import utils  # noqa
import main as M  # noqa
from ttnn.experimental.moe_compute_utils import (
    get_weight_core_shard_maps,
    get_weight_mem_configs,
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
    get_tilize_drain_core,
)
from models.demos.deepseek_v3.tests.fused_op_unit_tests.moe.test_optimized_moe_decode_block import (
    get_linearized_mesh_coord,
    create_torch_expert_mapping_tensor,
    create_torch_dispatch_input_tensor,
    create_torch_dispatch_input_expert_indices_tensor,
    create_torch_dispatch_input_expert_scores_tensor,
    gen_output_golden,
)
from models.common.utility_functions import comp_pcc
from ttnn.operations.ccl import MoEActivationFunction

torch.manual_seed(2005)
import random

random.seed(2005)

MESH = (4, 8)
cluster_axis = 0
hidden = 7168
N = 2048
k = 8
experts_per_device = 8
batches_per_device = 32
seq = 1

num_devices = MESH[0] * MESH[1]
num_dispatch = MESH[cluster_axis]  # 4
num_replicated = num_devices // num_dispatch  # 8
batch = batches_per_device * num_dispatch  # 128
total_tokens = batch * seq  # 128
tokens_per_device = batch // num_dispatch  # 32
routed_experts = experts_per_device * num_devices  # 256
routed_experts_per_cluster = routed_experts // num_replicated  # 32
shard_dims = (0, None)  # cluster_axis=0: tokens on axis0, replicated axis1

dev = utils.DeviceGetter.get_device(MESH)
drain = get_tilize_drain_core()
print(f"drain core = {drain}")


def t2t(dt):
    return {ttnn.bfloat16: torch.bfloat16, ttnn.uint16: torch.uint16, ttnn.float32: torch.float32}[dt]


# ---- expert mapping (canonical [num_devices, experts]) ----
torch_em = create_torch_expert_mapping_tensor(
    num_devices,
    num_replicated,
    cluster_axis,
    routed_experts,
    routed_experts_per_cluster,
    experts_per_device,
    ttnn.uint16,
)
tt_em = ttnn.from_torch(
    torch_em,
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.uint16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(None, None), mesh_shape=MESH),
)


# ---- weights (synthetic), per-expert, then per-device prepare + bf4 ----
def w_list(L, E, a, b):
    return [torch.rand((L, 1, a, b), dtype=torch.bfloat16) - 0.5 for _ in range(E)]


w0s = w_list(1, routed_experts, hidden, N)
w1s = w_list(1, routed_experts, hidden, N)
w2s = w_list(1, routed_experts, N, hidden)

w0w1_map, w2_map, dram_crs = get_weight_core_shard_maps(dev, hidden, N)
w0w1_reord = [None] * num_devices
w2_reord = [None] * num_devices
for e in range(0, routed_experts, experts_per_device):
    tw0 = torch.cat([w0s[e + i] for i in range(experts_per_device)], dim=1)
    tw1 = torch.cat([w1s[e + i] for i in range(experts_per_device)], dim=1)
    tw2 = torch.cat([w2s[e + i] for i in range(experts_per_device)], dim=1)
    r0 = prepare_w0_w1_tensor_for_moe_compute(tw0, tw1, 1, experts_per_device, hidden, N, w0w1_map)
    r2 = prepare_w2_tensor_for_moe_compute(tw2, 1, experts_per_device, N, hidden, w2_map, w0w1_map)
    coord = get_linearized_mesh_coord(num_replicated, cluster_axis, e, routed_experts_per_cluster, experts_per_device)
    w0w1_reord[coord] = r0
    w2_reord[coord] = r2
w0w1_cat = torch.cat(w0w1_reord, dim=0)
w2_cat = torch.cat(w2_reord, dim=0)
w0w1_mc, w2_mc, _, _ = get_weight_mem_configs(1, experts_per_device, hidden, N, w0w1_map, w2_map, dram_crs)
tt_w0w1 = ttnn.from_torch(
    w0w1_cat,
    device=dev,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat4_b,
    memory_config=w0w1_mc,
    mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=0),
)
tt_w2 = ttnn.from_torch(
    w2_cat,
    device=dev,
    layout=ttnn.TILE_LAYOUT,
    dtype=ttnn.bfloat4_b,
    memory_config=w2_mc,
    mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=0),
)
print("weights uploaded")

# ---- semaphores + dispatch output buffers ----
grid = dev.compute_with_storage_grid_size()
worker_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
disp_sem = ttnn.create_global_semaphore(dev, worker_cores, 0)
comb_sem = ttnn.create_global_semaphore(dev, worker_cores, 0)
mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])

sparse_buf = ttnn.from_torch(
    torch.zeros([num_dispatch, total_tokens, hidden], dtype=torch.bfloat16),
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH),
)
out_shard = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(drain, drain)}), [total_tokens, k], ttnn.ShardOrientation.ROW_MAJOR
)
out_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard)
out_idx = ttnn.from_torch(
    torch.zeros([num_dispatch, total_tokens, k], dtype=torch.uint16),
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.uint16,
    memory_config=out_l1,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH),
)
out_scr = ttnn.from_torch(
    torch.zeros([num_dispatch, total_tokens, k], dtype=torch.bfloat16),
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=out_l1,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH),
)
disp_out = (sparse_buf, out_idx, out_scr)

# dispatch input mem configs
ncy = min(8, tokens_per_device)
ncx = (tokens_per_device + ncy - 1) // ncy
in_shard = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(ncx - 1, ncy - 1))}),
    [1, seq * k],
    ttnn.ShardOrientation.ROW_MAJOR,
)
in_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, in_shard)

# ---- inputs + golden ----
tx = create_torch_dispatch_input_tensor(batch, seq, hidden, ttnn.bfloat16)
tidx = create_torch_dispatch_input_expert_indices_tensor(
    "random_sequential_experts",
    num_devices,
    routed_experts,
    total_tokens,
    experts_per_device,
    batches_per_device,
    batch,
    seq,
    k,
    ttnn.uint16,
)
tscr = create_torch_dispatch_input_expert_scores_tensor(batch, seq, k, ttnn.bfloat16)
golden = gen_output_golden(tx, tidx, tscr, w0s, w1s, w2s, batch, hidden, k)  # [batch,1,1,hidden]

tt_x = ttnn.from_torch(
    tx,
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH),
)
tt_idx = ttnn.from_torch(
    tidx,
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.uint16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH),
)
tt_scr = ttnn.from_torch(
    tscr,
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH),
)

combine_out = ttnn.from_torch(
    torch.zeros([k, tokens_per_device, hidden], dtype=torch.bfloat16),
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
)

DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
pools = M.main_const_eval_all_reduce_semaphores(dev)


def run():
    x = ttnn.to_memory_config(tt_x, ttnn.L1_MEMORY_CONFIG)
    idx = ttnn.to_memory_config(tt_idx, in_l1)
    scr = ttnn.to_memory_config(tt_scr, in_l1)
    d_sparse, d_idx, d_scr = ttnn.experimental.all_to_all_dispatch_metadata(
        x,
        idx,
        scr,
        tt_em,
        cluster_axis=cluster_axis,
        num_links=None,
        worker_mode=ttnn.WorkerMode.DIRECT,
        dispatch_algorithm=ttnn.DispatchAlgorithm.SPARSE_MCAST_SHORTEST_PATH,
        output_tensors=disp_out,
        cross_device_semaphore=disp_sem,
    )
    ttnn.deallocate(x)
    ttnn.deallocate(scr)

    outs = ttnn.experimental.moe_compute(
        d_sparse,
        d_idx,
        d_scr,
        tt_em,
        tt_w0w1,
        tt_w2,
        layer_id=0,
        output_height_shard_dim=4,
        intermediate_size=N,
        has_bias=False,
        cluster_axis=cluster_axis,
        mux_core_range_set=mux,
        optional_output_tensor=combine_out,
        optional_cross_device_semaphore=comb_sem,
        activation_type=MoEActivationFunction.SILU,
    )
    tt_combine = outs[5]  # [k, tokens_per_device, hidden]

    # ---- manual tail: weighted-k-sum -> all_reduce(axis1) -> mesh_partition ----
    c = ttnn.to_layout(tt_combine, ttnn.Layout.TILE, None, memory_config=DRAM)
    c = ttnn.typecast(c, ttnn.DataType.FLOAT32, memory_config=DRAM)
    # scores aligned to combine k-order: input scr [tokens,1,seq,k] -> per-device [32,8]; need [k,tokens,1]
    sc = ttnn.reshape(tt_scr, [tokens_per_device, k], memory_config=DRAM)  # NOTE tt_scr is DRAM bf16
    sc = ttnn.to_layout(sc, ttnn.Layout.TILE, None, memory_config=DRAM)
    sc = ttnn.typecast(sc, ttnn.DataType.FLOAT32, memory_config=DRAM)
    sc = ttnn.permute(sc, [1, 0], memory_config=DRAM)  # [k, tokens]
    sc = ttnn.reshape(sc, [k, tokens_per_device, 1], memory_config=DRAM)
    weighted = ttnn.multiply(c, sc, dtype=ttnn.DataType.FLOAT32, memory_config=DRAM)
    ttnn.deallocate(c)
    ttnn.deallocate(sc)
    summed = ttnn.sum(weighted, [0], False, memory_config=DRAM)  # [tokens, hidden] partial
    ttnn.deallocate(weighted)
    summed = ttnn.reshape(summed, [1, 1, tokens_per_device, hidden], memory_config=DRAM)
    ar = ttnn.experimental.all_reduce_async(
        summed,
        cluster_axis=1,
        mesh_device=dev,
        barrier_semaphores=pools[0][0],
        rs_global_semaphores=pools[1][0],
        ag_global_semaphores=pools[2][0],
        math_op=ttnn.ReduceType.Sum,
        memory_config=DRAM,
    )
    ttnn.deallocate(summed)
    part = ttnn.mesh_partition(input_tensor=ar, dim=3, cluster_axis=1, memory_config=DRAM)  # [1,1,32,896]
    ttnn.deallocate(ar)
    return part


t = time.perf_counter()
out = run()
ttnn.synchronize_device(dev)
print(f"pipeline ran in {time.perf_counter()-t:.3f}s, out shape (per-dev)={tuple(out.shape)}")

# gather: [1,1,tokens_per_device,hidden/repl] per device -> [1,1,batch,hidden]
host = ttnn.to_torch(
    out, dtype=torch.bfloat16, mesh_composer=ttnn.ConcatMesh2dToTensor(dev, mesh_shape=MESH, dims=(-2, -1))
)
host = host.reshape(host.shape[-2], 1, 1, host.shape[-1])  # [batch,1,1,hidden]
ok, pcc = comp_pcc(host, golden, pcc=0.97)
print(f"FINAL PCC vs torch golden: {pcc}  -> {'PASS' if ok else 'FAIL'}")

if utils.DeviceGetter._instance is not None:
    ttnn.close_mesh_device(utils.DeviceGetter._instance)
    utils.DeviceGetter._instance = None
