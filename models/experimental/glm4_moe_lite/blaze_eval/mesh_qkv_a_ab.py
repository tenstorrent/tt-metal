# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Does the Blaze q_kv_a program run on the 32-chip Galaxy mesh, and is it still correct?

EVERY number in this evaluation so far was taken on ttnn.open_device(device_id=0) -- ONE chip.
The model runs on a 4x8 mesh of 32. So "1.29x" is a single-chip result that has never been shown
to hold on the Galaxy, and this script is the first test of that.

Three things it answers, in order:
  1. Does a FusedProgram build and dispatch across a 32-device MeshDevice at all?
  2. Is the result still PCC-correct, checked on a real device shard rather than device 0's copy?
  3. What does it cost per chip, against the same ttnn op on the same mesh?

Mesh setup mirrors blaze's own GLM-5 real-weights harness
(tests/blaze/glm5_1/glm5_moe_real_weights.py:155-175): open the 4x8 parent, then create_submesh
for the shape a stage actually wants. That submesh call is the mechanism behind "each stage runs
on a specific number of chips".

    BLAZE_DSM_WORKERS_PER_BANK=4 python mesh_qkv_a_ab.py [rows] [cols]
"""

import sys

sys.path.insert(0, "/home/ttuser/sdawle/skills/blaze-vs-ttnn-bench/scripts")
import ab_harness

ab_harness.set_profiler_env()  # must precede ttnn

import os  # noqa: E402

import torch  # noqa: E402
import ttnn  # noqa: E402
from blaze.models.blaze_tests_namespace import register_blaze_tests_namespace  # noqa: E402

register_blaze_tests_namespace()
from blaze.blaze_op import Risc  # noqa: E402
from blaze.fused_program import FusedProgram  # noqa: E402
from blaze.ops.dram_streaming_matmul import DRAMStreamingMatmul  # noqa: E402
from blaze.ops.dram_streaming_matmul.common import dram_bank_worker_cores  # noqa: E402
from blaze.ops.gather_row_to_dram import GatherRowToDRAM  # noqa: E402
from blaze.ops.tile_row_replicate import TileRowReplicate  # noqa: E402
from blaze_tests.micro_ops.common.test_dram_streaming_matmul import (  # noqa: E402
    _make_weights_tensor,
    _pad_to_dram_banks,
)
from models.demos.deepseek_v3.utils.config_helpers import (  # noqa: E402
    dram_sharded_weight_config,
    get_activation_sharding_core_counts_for_dram_matmul,
    get_dram_sharded_matmul_config,
)

ROWS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
COLS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
K, NQ, NKV = 2048, 768, 576
NCAT = NQ + NKV  # 1344 -- the model's concatenated q_kv_a weight
W = int(os.environ.get("BLAZE_DSM_WORKERS_PER_BANK", "1"))

avail = ttnn.get_num_devices()
print(f"RESULT devices available = {avail}, requesting {ROWS}x{COLS}", flush=True)
parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(4, 8) if avail >= 32 else ttnn.MeshShape(avail, 1))
print(f"RESULT parent mesh = {parent.shape}, n={parent.get_num_devices()}", flush=True)
mesh = parent.create_submesh(ttnn.MeshShape(ROWS, COLS))
print(f"RESULT submesh = {mesh.shape}, n={mesh.get_num_devices()}", flush=True)

banks = mesh.dram_grid_size().x
ncat = _pad_to_dram_banks(NCAT, 32, 32 * banks * W)
torch.manual_seed(47)
row = torch.randn(1, 1, 1, K).bfloat16().float()
act_t = torch.zeros(1, 1, 32, K, dtype=torch.bfloat16).float()
act_t[..., :1, :] = row
w_cat = torch.randn(1, 1, K, ncat).bfloat16().float()

rep = ttnn.ReplicateTensorToMesh(mesh)
act = ttnn.from_torch(
    act_t,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=rep,
)
out = ttnn.from_torch(
    torch.zeros(1, 1, 1, ncat),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=mesh,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
    mesh_mapper=rep,
)
b_w = _make_weights_tensor(mesh, w_cat, k=K, n_padded=ncat, tile_w=32, num_banks=banks, weight_dtype=ttnn.bfloat8_b)
worker_list, worker_grid = dram_bank_worker_cores(mesh)
# Gather receiver: first core that is not a DRAM-bank worker, same rule GLMQKVAProjection uses.
_bank_xy = {(c.x, c.y) for c in worker_list}
_g = mesh.compute_with_storage_grid_size()
RECEIVER = next(ttnn.CoreCoord(x, y) for y in range(_g.y) for x in range(_g.x) if (x, y) not in _bank_xy)


def build():
    f = FusedProgram(
        kernel=None,
        device=mesh,
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        name="mesh_qkv_a",
    )
    stream = TileRowReplicate.emit(f, act, prefix="qkv_a_in", cores=worker_grid, num_tile_cols=K // 32, row=0)
    mm = DRAMStreamingMatmul.emit(
        f,
        stream,
        b_w,
        index=None,
        bias=None,
        out=None,
        prefix="qkv_a_mm",
        fp32_dest_acc_en=True,
        subblock_k=2,
        fused_activation=None,
        index_offset=0,
        wait_for_out=False,
        pop_index=False,
        pop_act=True,
    )
    GatherRowToDRAM.emit(f, mm, out, prefix="qkv_a_out", receiver=RECEIVER)
    return f


_g2 = mesh.compute_with_storage_grid_size()
_mx = _g2.x * _g2.y
_in_c = max(get_activation_sharding_core_counts_for_dram_matmul(K, _mx))
_out_c = max(get_activation_sharding_core_counts_for_dram_matmul(ncat, _mx))
t_w = ttnn.from_torch(w_cat, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, mesh_mapper=rep).to(
    mesh, dram_sharded_weight_config(K, ncat, mesh.dram_grid_size())
)
t_pc = get_dram_sharded_matmul_config(m=32, k=K, n=ncat, input_num_shards=_in_c, output_num_shards=_out_c)
t_ckc = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
)
_act_mc = ttnn.create_sharded_memory_config_(
    shape=(32, K // _in_c),
    core_grid=ttnn.num_cores_to_corerangeset(_in_c, ttnn.CoreCoord(_g2.x, _g2.y), row_wise=True),
    strategy=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    tile_layout=True,
    use_height_and_width_as_shard_shape=True,
)


def ttnn_fn():
    a = ttnn.to_memory_config(act, _act_mc)
    o = ttnn.linear(
        a, t_w, program_config=t_pc, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, compute_kernel_config=t_ckc
    )
    return ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


try:
    prog = build()
    print("RESULT BUILD OK on mesh", flush=True)
    prog.run()
    print("RESULT RUN OK on mesh", flush=True)
    # Check a real shard, and check EVERY device -- a program that only lands on device 0 would
    # pass a device-0-only check while silently leaving 31 chips wrong.
    ref = (row[0, 0, 0] @ w_cat[0, 0]).float()
    shards = ttnn.get_device_tensors(out)
    pccs = [_pcc(ref[:NCAT], ttnn.to_torch(s)[..., :1, :NCAT]) for s in shards]
    print(f"RESULT shards checked = {len(shards)}, min PCC = {min(pccs):.6f}, max = {max(pccs):.6f}", flush=True)
    if min(pccs) < 0.99:
        bad = [i for i, p in enumerate(pccs) if p < 0.99]
        print(f"RESULT GATE FAILED on {len(bad)} shard(s), first few: {bad[:8]}", flush=True)
    else:
        n = mesh.get_num_devices()
        # ab_harness._measure sums kernel durations across every device in the mesh (its loop is
        # over device ids), so on a 32-chip mesh the raw figure is 32x a per-chip cost. Divide.
        b_sum, _ = ab_harness._measure(mesh, lambda: prog.run(), warmup=2, iters=5)
        t_sum, _ = ab_harness._measure(mesh, ttnn_fn, warmup=2, iters=5)
        print(f"RESULT blaze q_kv_a: {b_sum:.1f} us summed / {n} chips = {b_sum/n:.1f} us per chip (W={W})", flush=True)
        print(f"RESULT ttnn  q_kv_a: {t_sum:.1f} us summed / {n} chips = {t_sum/n:.1f} us per chip", flush=True)
        print(f"RESULT speedup on {n} chips = {t_sum/b_sum:.2f}x", flush=True)
except Exception as e:
    print("RESULT FAIL:", str(e).split(chr(10))[0][:220], flush=True)

for s in parent.get_submeshes():
    ttnn.close_mesh_device(s)
ttnn.close_mesh_device(parent)
