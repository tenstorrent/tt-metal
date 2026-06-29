# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LOCAL microtest (do not commit): validate the DRAM-sharded decode matmul
contract for the Qwen3.6 MLP shapes against the current interleaved path.

Compares, per matmul shape, the DRAM-interleaved ttnn.linear vs the
DRAM-sharded ttnn.linear (width-sharded activation + width-sharded DRAM weight
+ L1_WIDTH_SHARDED output). Asserts PCC ~1.0 between them (pure memory-layout
change => identical math) and reports timing.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_dram_sharded_mm.py -v -s
"""

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp, replicate_to_device
from models.demos.blackhole.qwen36.tt import tp_common as tpc

# Qwen3.6-27B dims, TP=4: dim=5120, hidden_dim=17408 -> hidden//tp=4352
DIM = 5120
HIDDEN_TP = 4352

# (label, K, N, weight_dtype)  — the three MLP matmuls (per device)
SHAPES = [
    ("w1_gate", DIM, HIDDEN_TP, ttnn.bfloat4_b),
    ("w3_up", DIM, HIDDEN_TP, ttnn.bfloat4_b),
    ("w2_down", HIDDEN_TP, DIM, ttnn.bfloat8_b),
]


def _interleaved_weight(mesh_device, w_torch, dtype):
    return ttnn.as_tensor(
        w_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _dram_sharded_weight(mesh_device, w_torch, dtype, k, n):
    memcfg = tpc.create_dram_sharded_mem_config(k, n)
    return ttnn.as_tensor(
        w_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=memcfg,
    )


@torch.no_grad()
@parametrize_mesh_tp()
def test_dram_sharded_mm(mesh_device, reset_seeds, ensure_gc):
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
    )
    T = 32  # decode: B=1 padded to one tile
    results = []

    for label, K, N, dtype in SHAPES:
        logger.info(f"### START {label} K={K} N={N} {dtype}")
        w_torch = torch.randn(K, N, dtype=torch.bfloat16)  # ttnn.linear convention [in, out]
        x_torch = torch.randn(1, 1, T, K, dtype=torch.bfloat16)
        ref = x_torch.to(torch.float32)[0, 0] @ w_torch.to(torch.float32)  # [T, N]

        x_tt = replicate_to_device(mesh_device, x_torch)

        # ---- baseline: DRAM-interleaved ----
        logger.info(f"[{label}] interleaved linear...")
        w_int = _interleaved_weight(mesh_device, w_torch, dtype)
        out_int = ttnn.linear(x_tt, w_int, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"[{label}] interleaved OK")

        # ---- candidate: DRAM-sharded ----
        logger.info(f"[{label}] building dram-sharded weight + act shard + progcfg...")
        w_ds = _dram_sharded_weight(mesh_device, w_torch, dtype, K, N)
        act_cfg = tpc.create_activation_shard_config(K)
        progcfg = tpc.create_dram_sharded_matmul_program_config(1, K, N)
        logger.info(f"[{label}] to_memory_config(act shard)...")
        x_ds = ttnn.to_memory_config(x_tt, act_cfg)
        logger.info(f"[{label}] dram-sharded linear...")
        out_ds = ttnn.linear(
            x_ds,
            w_ds,
            compute_kernel_config=ckc,
            program_config=progcfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        ttnn.synchronize_device(mesh_device)
        logger.info(f"[{label}] dram-sharded linear OK")

        out_ds_dram = ttnn.to_memory_config(out_ds, ttnn.DRAM_MEMORY_CONFIG)

        # take device-0 replica
        int_torch = ttnn.to_torch(out_int, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        int_torch = int_torch[0][0] if int_torch.dim() == 4 else int_torch[:T]
        ds_torch = ttnn.to_torch(out_ds_dram, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        ds_torch = ds_torch[0][0] if ds_torch.dim() == 4 else ds_torch[:T]
        int_torch = int_torch[:T, :N].to(torch.float32)
        ds_torch = ds_torch[:T, :N].to(torch.float32)

        _, pcc_ref_int = comp_pcc(ref, int_torch, 0.99)
        _, pcc_ref_ds = comp_pcc(ref, ds_torch, 0.99)
        passing, pcc_int_ds = comp_pcc(int_torch, ds_torch, 0.99)
        logger.info(
            f"### RESULT [{label}] K={K} N={N} {dtype}  "
            f"PCC(ref,int)={pcc_ref_int}  PCC(ref,ds)={pcc_ref_ds}  PCC(int,ds)={pcc_int_ds}"
        )
        results.append((label, passing, pcc_int_ds))

    for label, passing, pcc in results:
        assert passing, f"[{label}] DRAM-sharded diverges from interleaved: PCC={pcc}"
