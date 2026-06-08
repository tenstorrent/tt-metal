# SPDX-License-Identifier: Apache-2.0
"""Minimal standalone repro: ttnn.experimental.moe_compute raises a mux-L1-overlap TT_FATAL
during program creation on a Blackhole 4x8 Galaxy (cluster_axis=0) for a DeepSeek-V3-scale MoE
shape: experts_per_device=8, hidden_size=7168, intermediate_size=2048, tokens_per_device=32.

The combine stage (selective_reduce_combine) sizes its fabric-mux L1 region with a Blackhole
buffer count (`num_buffers=14`) calibrated for experts_per_device=2 (deepseek_v3 on WH 6U). At
epd=8 the per-core compute L1 tensor sits ~8 KB below the mux end, tripping:
    TT_FATAL ... Mux L1 memory [base=..,end=..] overlaps with L1 tensor .. (assert.hpp)
in selective_reduce_combine_program_factory.cpp::launch_mux_workers.

All inputs are synthetic (no model data). The op FATALs at program-creation time, before any
device execution, so this is deterministic and does not hang.

Run (on UNMODIFIED tt-metal):
    source <tt-metal>/python_env/bin/activate
    export TT_METAL_HOME=<tt-metal> PYTHONPATH=<tt-metal> ARCH_NAME=blackhole
    timeout 300 python3 repro_moe_compute_combine_l1_overlap.py
Expected: RuntimeError with "Mux L1 memory [...] overlaps with L1 tensor [...]".
"""
import torch
import ttnn
from ttnn.operations.ccl import MoEActivationFunction
from ttnn.experimental.moe_compute_utils import get_tilize_drain_core

MESH = (4, 8)
CLUSTER_AXIS = 0
EPD = 8  # experts per device  (vs the epd=2 BH calibration) -> the L1 driver
HIDDEN = 7168
N = 2048  # intermediate_size
K = 8  # selected experts per token
TOKENS_PER_DEV = 32

num_devices = MESH[0] * MESH[1]  # 32
num_dispatch = MESH[CLUSTER_AXIS]  # 4
experts = EPD * num_devices  # 256 -> op infers experts_per_device = 256/32 = 8
total_tokens = TOKENS_PER_DEV * num_dispatch  # 128
shard_dims = (0, None)  # cluster_axis=0: tokens sharded on mesh axis 0
DRAM = ttnn.DRAM_MEMORY_CONFIG

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
dev = ttnn.open_mesh_device(ttnn.MeshShape(MESH), l1_small_size=32768)
print(f"opened {dev}", flush=True)
drain = get_tilize_drain_core()

# ---- expert_mapping [num_devices, experts] uint16, replicated (contiguous: expert e -> device e//EPD) ----
em_t = (torch.arange(experts, dtype=torch.int32) // EPD).reshape(1, experts).repeat(num_devices, 1)
tt_em = ttnn.from_torch(
    em_t.to(torch.int32),
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.uint16,
    memory_config=DRAM,
    mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
)


# ---- weights: synthetic, REPLICATED (each device holds EPD=8 experts), C++ prepare -> bfloat4_b ----
def _up_rm(t):  # ROW_MAJOR bf16 replicated (the layout the C++ prepare_* helpers expect)
    return ttnn.from_torch(
        t,
        device=dev,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
    )


w0 = _up_rm(torch.rand(1, EPD, HIDDEN, N, dtype=torch.bfloat16) - 0.5)
w1 = _up_rm(torch.rand(1, EPD, HIDDEN, N, dtype=torch.bfloat16) - 0.5)
w2 = _up_rm(torch.rand(1, EPD, N, HIDDEN, dtype=torch.bfloat16) - 0.5)
mc = ttnn.experimental.get_weight_mem_configs(
    dev, num_layers=1, experts_per_device=EPD, hidden_size=HIDDEN, intermediate_size=N, has_bias=False
)
w0w1_p = ttnn.experimental.prepare_w0_w1_tensor_for_moe_compute(w0, w1, L=1, E=EPD, K=HIDDEN, N=N)
tt_w0w1 = ttnn.experimental.quantize_weights_via_host(w0w1_p, dtype=ttnn.bfloat4_b, memory_config=mc.w0_w1)
w2_p = ttnn.experimental.prepare_w2_tensor_for_moe_compute(w2, L=1, E=EPD, N=N, K=HIDDEN)
tt_w2 = ttnn.experimental.quantize_weights_via_host(w2_p, dtype=ttnn.bfloat4_b, memory_config=mc.w2)
print("weights prepared (bfloat4_b)", flush=True)

# ---- moe_compute inputs (the dispatch outputs), synthetic zeros with the expected shapes/mem-configs ----
sparse = ttnn.from_torch(
    torch.zeros(num_dispatch, total_tokens, HIDDEN, dtype=torch.bfloat16),
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=DRAM,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH),
)
out_shard = ttnn.ShardSpec(
    ttnn.CoreRangeSet({ttnn.CoreRange(drain, drain)}), [total_tokens, K], ttnn.ShardOrientation.ROW_MAJOR
)
out_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, out_shard)
oi = ttnn.from_torch(
    torch.zeros(num_dispatch, total_tokens, K, dtype=torch.uint16),
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.uint16,
    memory_config=out_l1,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH),
)
os_ = ttnn.from_torch(
    torch.zeros(num_dispatch, total_tokens, K, dtype=torch.bfloat16),
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=out_l1,
    mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=shard_dims, mesh_shape=MESH),
)
combine_out = ttnn.from_torch(
    torch.zeros(K, TOKENS_PER_DEV, HIDDEN, dtype=torch.bfloat16),
    device=dev,
    layout=ttnn.ROW_MAJOR_LAYOUT,
    dtype=ttnn.bfloat16,
    memory_config=DRAM,
    mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
)
worker = ttnn.CoreRangeSet(
    {
        ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(dev.compute_with_storage_grid_size().x - 1, dev.compute_with_storage_grid_size().y - 1),
        )
    }
)
comb_sem = ttnn.create_global_semaphore(dev, worker, 0)
mux = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))])

print("calling moe_compute (expect mux-L1-overlap TT_FATAL during program creation)...", flush=True)
try:
    outs = ttnn.experimental.moe_compute(
        sparse,
        oi,
        os_,
        tt_em,
        tt_w0w1,
        tt_w2,
        layer_id=0,
        output_height_shard_dim=4,
        intermediate_size=N,
        has_bias=False,
        cluster_axis=CLUSTER_AXIS,
        mux_core_range_set=mux,
        optional_output_tensor=combine_out,
        optional_cross_device_semaphore=comb_sem,
        activation_type=MoEActivationFunction.SILU,
    )
    ttnn.synchronize_device(dev)
    print("UNEXPECTED: moe_compute did NOT raise (no L1 overlap on this build/shape)", flush=True)
except RuntimeError as e:
    msg = str(e)
    hit = "Mux L1 memory" in msg and "overlaps with L1 tensor" in msg
    print(f"REPRO {'CONFIRMED' if hit else 'raised (different error)'}:", flush=True)
    print(msg.splitlines()[0] if msg else "", flush=True)
    for line in msg.splitlines():
        if "Mux L1 memory" in line:
            print(line, flush=True)

ttnn.close_mesh_device(dev)
