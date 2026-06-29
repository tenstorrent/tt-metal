# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""LOCAL microtest (do not commit): can a DRAM-WIDTH_SHARDED weight serve the
PREFILL matmul (large M)? Decides whether switching w1/w3 to DRAM-sharded is
safe for prefill (all ISLs). Tries auto-select and an explicit 2D progcfg, both
against the interleaved baseline.

Run:
    MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
      pytest models/demos/blackhole/qwen36/tests/test_dram_sharded_prefill.py -v -s
"""
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp, replicate_to_device
from models.demos.blackhole.qwen36.tt import tp_common as tpc

DIM = 5120
HIDDEN_TP = 4352


@torch.no_grad()
@parametrize_mesh_tp()
def test_dram_sharded_prefill(mesh_device, reset_seeds, ensure_gc):
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
    )
    K, N = DIM, HIDDEN_TP
    w_torch = torch.randn(K, N, dtype=torch.bfloat16)

    w_int = ttnn.as_tensor(
        w_torch,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w_ds = ttnn.as_tensor(
        w_torch,
        dtype=ttnn.bfloat4_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=tpc.create_dram_sharded_mem_config(K, N),
    )

    for M in (512, 2048):
        x_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
        ref = x_torch.to(torch.float32)[0, 0] @ w_torch.to(torch.float32)
        x_tt = replicate_to_device(mesh_device, x_torch)

        out_int = ttnn.linear(x_tt, w_int, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        it = ttnn.to_torch(out_int, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0][0][:M, :N].to(
            torch.float32
        )
        _, pcc_int = comp_pcc(ref, it, 0.97)

        # (a) DRAM-sharded weight + auto-select
        auto_ok, auto_pcc, auto_err = True, None, None
        try:
            out_a = ttnn.linear(x_tt, w_ds, compute_kernel_config=ckc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            at = ttnn.to_torch(out_a, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0][0][:M, :N].to(
                torch.float32
            )
            auto_ok, auto_pcc = comp_pcc(it, at, 0.97)
        except Exception as e:
            auto_ok, auto_err = False, str(e)[:160]

        # (b) DRAM-sharded weight + explicit 2D prefill progcfg
        pc_ok, pc_pcc, pc_err = True, None, None
        try:
            pc = tpc.create_prefill_matmul_program_config(M, K, N)
            out_b = ttnn.linear(
                x_tt, w_ds, compute_kernel_config=ckc, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
            bt = ttnn.to_torch(out_b, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0][0][:M, :N].to(
                torch.float32
            )
            pc_ok, pc_pcc = comp_pcc(it, bt, 0.97)
        except Exception as e:
            pc_ok, pc_err = False, str(e)[:160]

        logger.info(
            f"PREFILL M={M}: PCC(ref,int)={pcc_int} | auto: ok={auto_ok} pcc={auto_pcc} err={auto_err} "
            f"| 2dcfg: ok={pc_ok} pcc={pc_pcc} err={pc_err}"
        )
