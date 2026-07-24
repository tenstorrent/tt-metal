# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Hand-tuning unit tests for the GLM-5.2 MLA matmuls under chunked prefill (seq_len_local=640).

Covers all 9 matmuls in GLM's sparse-MLA forward: the base 6 (q_a_proj, q_b_proj, wkv_b1,
kv_a_proj_with_mqa, wkv_b2, o_proj) plus the 3 indexer linears (indexer.wq_b, indexer.wk,
indexer.weights_proj). Each is run in isolation with a hand-tuned program_config and PCC-checked
against torch; profile with tracy to read DEVICE KERNEL DURATION and iterate the program_config.

  test_glm_mla_mm       -- the current best config per matmul (what gets wired into mla_config.py)
  test_glm_mla_mm_sweep -- candidate variants for a single tracy pass; pick the fastest per matmul

GLM-5.2 geometry (reference/glm_5_2_config.py): hidden=6144, heads=64, q_lora=2048, kv_lora=512,
qk_nope=192, qk_rope=64, v_head=256, index_n_heads=32, index_head_dim=128.

Adapted to an 8-device loudbox: mesh (2,4) with sp_axis=0 (size 2), tp_axis=1 (size 4). Global
seq is 2*640=1280 so each chip sees seq_len_local=640 (M_t=20) — the production 8x4 per-chip shape
(chunk 5120 / sp 8 = 640, tp 4). Compute grid capped at 11x10 (di/dt + throttling headroom).

Note: q_b_proj and indexer.wq_b are the IDENTICAL per-chip matmul (640x2048x4096) — one config
serves both.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

PCC_REQUIRED = 0.99

# Available core grid is 12x10, but due to di/dt and throttling problems, use 11x10.
GRID = (11, 10)

# GLM-5.2 geometry
HIDDEN = 6144
NUM_HEADS = 64
Q_LORA = 2048
KV_LORA = 512
QK_NOPE = 192
QK_ROPE = 64
QK_HEAD = QK_NOPE + QK_ROPE  # 256
V_HEAD = 256
IDX_HEADS = 32
IDX_HEAD_DIM = 128

# Mesh: sp on axis 0 (size 2), tp on axis 1 (size 4). Global seq 1280 -> per-chip 640.
SP = 2
TP = 4
SEQ_LOCAL = 640
SEQ_GLOBAL = SP * SEQ_LOCAL  # 1280

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
BF16 = ttnn.bfloat16
BF8 = ttnn.bfloat8_b


def _mc2d(in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


def _reuse(in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N):
    return ttnn.MatmulMultiCoreReuseProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
    )


def _mc1d(in0_block_w, out_subblock_h, out_subblock_w, per_core_M, per_core_N):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=GRID,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fuse_batch=False,
        mcast_in0=False,
    )


# Per-matmul activation (in0) dtype — matches the GLM-5.2 Galaxy run (ops_perf_results). All are
# BF16 except o_proj, which consumes the BF8 SDPA/concat-heads output. Weights (in1) are BF8 for all
# 9 (the 3 indexer weights are loaded BF16 in indexer.py today but tuned/targeted at BF8 to match
# the base MLA weights — the indexer.py load switches to bfloat8_b alongside this).
IN0_DTYPE = {"o_proj": BF8}  # default BF16

# name -> (in0_shape(global), in0_tp_sharded, in0_tp_dim, in1_shape, in1_tp_sharded, in1_tp_dim,
#          default_out_dtype, tp_out_mode). in0 is ALWAYS sp-sharded on dim2 (seq).
# tp_out_mode in {sum, shard_n, shard_heads, replicated}.
SHAPES = {
    "q_a_proj": ((1, 1, SEQ_GLOBAL, HIDDEN), True, 3, (1, 1, HIDDEN, Q_LORA), True, 2, BF16, "sum"),
    "q_b_proj": (
        (1, 1, SEQ_GLOBAL, Q_LORA),
        False,
        None,
        (1, 1, Q_LORA, NUM_HEADS * QK_HEAD),
        True,
        3,
        BF16,
        "shard_n",
    ),
    "wkv_b1": (
        (1, NUM_HEADS, SEQ_GLOBAL, QK_NOPE),
        True,
        1,
        (1, NUM_HEADS, QK_NOPE, KV_LORA),
        True,
        1,
        BF16,
        "shard_heads",
    ),
    "kv_a_proj_with_mqa": (
        (1, 1, SEQ_GLOBAL, HIDDEN),
        True,
        3,
        (1, 1, HIDDEN, KV_LORA + QK_ROPE),
        True,
        2,
        BF16,
        "sum",
    ),
    "wkv_b2": (
        (1, NUM_HEADS, SEQ_GLOBAL, KV_LORA),
        True,
        1,
        (1, NUM_HEADS, KV_LORA, V_HEAD),
        True,
        1,
        BF8,
        "shard_heads",
    ),
    "o_proj": (
        (1, 1, SEQ_GLOBAL, NUM_HEADS * V_HEAD),
        True,
        3,
        (1, 1, NUM_HEADS * V_HEAD, HIDDEN),
        True,
        2,
        BF16,
        "sum",
    ),
    "indexer.wq_b": (
        (1, 1, SEQ_GLOBAL, Q_LORA),
        False,
        None,
        (1, 1, Q_LORA, IDX_HEADS * IDX_HEAD_DIM),
        False,
        None,
        BF16,
        "replicated",
    ),
    "indexer.wk": ((1, 1, SEQ_GLOBAL, HIDDEN), True, 3, (1, 1, HIDDEN, IDX_HEAD_DIM), True, 2, BF16, "sum"),
    "indexer.weights_proj": ((1, 1, SEQ_GLOBAL, HIDDEN), True, 3, (1, 1, HIDDEN, IDX_HEADS), True, 2, BF16, "sum"),
}

# Current best config per matmul: name -> (prog_config, act_mem, out_mem, out_dtype).
BEST = {
    "q_a_proj": (_mc2d(8, 1, 6, 2, 6), DRAM, L1, BF16),  # 110c, 22.8us, 58%
    "q_b_proj": (_mc2d(8, 1, 6, 2, 12), L1, L1, BF16),  # 110c, 49.1us, 72%
    "wkv_b1": (_reuse(6, 2, 4, 4, 16), L1, L1, BF16),  # 80c (max: 320 B*M_t), 50.0us, DM-bound
    "kv_a_proj_with_mqa": (_mc2d(8, 1, 2, 2, 2), L1, L1, BF16),  # 90c, 11.2us, 41%
    "wkv_b2": (_reuse(2, 4, 2, 4, 8), L1, L1, BF8),  # 80c, 51.9us, DM-bound
    "o_proj": (_mc2d(16, 1, 6, 2, 18), DRAM, L1, BF16),  # 110c, 135.4us, 78%
    "indexer.wq_b": (_mc2d(8, 1, 6, 2, 12), L1, L1, BF16),  # == q_b_proj (BF16 act/BF8 wt), 49.1us, 72%
    "indexer.wk": (_mc2d(8, 1, 1, 2, 1), DRAM, DRAM, BF16),  # 40c (N_t=4), 13.9us, tiny
    "indexer.weights_proj": (_mc2d(8, 1, 1, 2, 1), DRAM, DRAM, BF16),  # 10c (N_t=1), 10.2us, tiny
}

# Sweep variants for a single tracy pass: (variant_id, base_shape_name, prog, act, out, out_dtype).
SWEEP = [
    # Round 2: batched wkv_b1/wkv_b2 were config-invariant at 80c (pcm=4) in round 1 — test whether
    # more cores (pcm=3 -> ceil(16*20/3)=107c) moves them, to distinguish core-bound from DM-bound.
    ("wkv_b1__pcm3_h1w8", "wkv_b1", _reuse(6, 1, 8, 3, 16), L1, L1, BF16),
    ("wkv_b1__pcm3_h3w2", "wkv_b1", _reuse(6, 3, 2, 3, 16), L1, L1, BF16),
    ("wkv_b1__pcm2_h1w8", "wkv_b1", _reuse(6, 1, 8, 2, 16), L1, L1, BF16),
    ("wkv_b2__pcm3_h1w8", "wkv_b2", _reuse(2, 1, 8, 3, 8), L1, L1, BF8),
    ("wkv_b2__pcm3_h3w2", "wkv_b2", _reuse(2, 3, 2, 3, 8), L1, L1, BF8),
    ("wkv_b2__pcm2_h1w8", "wkv_b2", _reuse(2, 1, 8, 2, 8), L1, L1, BF8),
    # confirm the round-1 o_proj winner (ib16, out L1) reproduces.
    ("o_proj__ib16_outL1", "o_proj", _mc2d(16, 1, 6, 2, 18), DRAM, L1, BF16),
]


def _reconstruct(tt_output, mesh_device, tp_out_mode):
    """Rebuild the global (unsharded) torch tensor from the (SP, TP) mesh output. Devices are stacked
    device-major (row-major id = sp*TP + tp), reshaped to [SP, TP, Y, SEQ, N], TP shards reduced per
    mode, then SP shards concatenated on the sequence axis."""
    stacked = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    _, y, m, n = stacked.shape
    stacked = stacked.reshape(SP, TP, y, m, n)
    if tp_out_mode == "sum":
        tp_red = stacked.to(torch.float32).sum(dim=1)
    elif tp_out_mode == "replicated":
        tp_red = stacked[:, 0]
    elif tp_out_mode == "shard_n":
        tp_red = torch.cat([stacked[:, t] for t in range(TP)], dim=-1)
    elif tp_out_mode == "shard_heads":
        tp_red = torch.cat([stacked[:, t] for t in range(TP)], dim=1)
    else:
        raise ValueError(tp_out_mode)
    out = torch.cat([tp_red[s] for s in range(SP)], dim=1)
    return out.unsqueeze(0)


def _run(mesh_device, name, prog_config, act_mem_config, out_mem_config, out_dtype):
    (in0_shape, in0_tp_sharded, in0_tp_dim, in1_shape, in1_tp_sharded, in1_tp_dim, _dd, tp_out_mode) = SHAPES[name]
    torch.manual_seed(42)
    hidden_states = torch.randn(*in0_shape, dtype=torch.bfloat16)
    weight = torch.randn(*in1_shape, dtype=torch.bfloat16) * 0.02

    sp_axis, tp_axis = 0, 1
    in0_shard_dims = [None, None]
    in0_shard_dims[sp_axis] = 2
    if in0_tp_sharded:
        in0_shard_dims[tp_axis] = in0_tp_dim
    tt_input = ttnn.from_torch(
        hidden_states,
        device=mesh_device,
        dtype=IN0_DTYPE.get(name, BF16),
        layout=ttnn.TILE_LAYOUT,
        memory_config=act_mem_config,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=in0_shard_dims),
    )

    if in1_tp_sharded:
        in1_shard_dims = [None, None]
        in1_shard_dims[tp_axis] = in1_tp_dim
        in1_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=in1_shard_dims)
    else:
        in1_mapper = ttnn.ReplicateTensorToMesh(mesh_device)
    tt_weight = ttnn.from_torch(
        weight, device=mesh_device, dtype=BF8, layout=ttnn.TILE_LAYOUT, memory_config=DRAM, mesh_mapper=in1_mapper
    )

    compute_kernel_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    tt_output = ttnn.linear(
        tt_input,
        tt_weight,
        memory_config=out_mem_config,
        compute_kernel_config=compute_kernel_config,
        dtype=out_dtype,
        program_config=prog_config,
    )
    ttnn.synchronize_device(mesh_device)

    reference = torch.matmul(hidden_states, weight)
    got = _reconstruct(tt_output, mesh_device, tp_out_mode)
    passing, pcc = comp_pcc(reference, got, PCC_REQUIRED)
    logger.info(f"[{name}] PCC: {pcc} (required {PCC_REQUIRED}), out shape {tuple(got.shape)}")
    assert passing, f"{name} matmul PCC {pcc} < {PCC_REQUIRED}"


@pytest.mark.parametrize("mesh_device", [(SP, TP)], ids=[f"{SP}x{TP}"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("name", list(SHAPES.keys()), ids=list(SHAPES.keys()))
def test_glm_mla_mm(mesh_device, name):
    prog_config, act_mem_config, out_mem_config, out_dtype = BEST[name]
    _run(mesh_device, name, prog_config, act_mem_config, out_mem_config, out_dtype)


@pytest.mark.parametrize("mesh_device", [(SP, TP)], ids=[f"{SP}x{TP}"], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize(
    "variant_id, base_name, prog_config, act_mem_config, out_mem_config, out_dtype",
    SWEEP,
    ids=[s[0] for s in SWEEP],
)
def test_glm_mla_mm_sweep(mesh_device, variant_id, base_name, prog_config, act_mem_config, out_mem_config, out_dtype):
    _run(mesh_device, base_name, prog_config, act_mem_config, out_mem_config, out_dtype)
