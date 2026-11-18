# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Perf and accuracy sweep of the Qwen MLP prefill block on a 4-chip ring.

Run with TT_VISIBLE_DEVICES=0,1,4,5 on an 8 x P150 machine so that the 2x2 mesh is a physical ring.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.tt_ccl import TT_CCL
from models.common.utility_functions import comp_pcc
from tests.ttnn.unit_tests.operations.qwen_mlp.multi_op_mlp import multi_op_mlp

H = 5120
I = 17408
TP = 4
EPS = 1e-6
SEQUENCE_LENGTHS = [4096, 2048, 1024]
WARMUP_ITERATIONS = 3
TIMED_ITERATIONS = 10
PCC_THRESHOLD = 0.99

DRAM_BW = 512e9
COMPUTE_FLOPS = 594e12
ETH_BW = 800e9
BFP4_BYTES_PER_ELEMENT = 576 / 1024
BF16_BYTES = 2

mlp_under_test = multi_op_mlp


def roofline_seconds(S):
    flops = 6 * S * H * I / TP
    dram_bytes = 2 * S * H * BF16_BYTES + 3 * H * I / TP * BFP4_BYTES_PER_ELEMENT + H * BF16_BYTES
    eth_byte_hops = 2 * (TP - 1) * S * H * BF16_BYTES
    return max(flops / COMPUTE_FLOPS, dram_bytes / DRAM_BW, eth_byte_hops / ETH_BW)


def interleave_gate_up(gate, up):
    """[H, I/TP] x2 -> [H, 2*I/TP] with 32-column blocks alternating gate, up."""
    return torch.stack([gate.reshape(H, -1, 32), up.reshape(H, -1, 32)], dim=2).reshape(H, -1)


def deinterleave_gate_up(gate_up):
    blocks = gate_up.reshape(H, -1, 2, 32)
    return blocks[:, :, 0].reshape(H, -1), blocks[:, :, 1].reshape(H, -1)


def reference_mlp(h, w_norm, gate_up_per_chip, down_per_chip):
    h = h.float()
    n = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS) * w_norm.float()
    y = h.clone()
    for gate_up, down in zip(gate_up_per_chip, down_per_chip):
        gate, up = deinterleave_gate_up(gate_up.float())
        y += (torch.nn.functional.silu(n @ gate) * (n @ up)) @ down.float()
    return y


def upload(mesh, tensor, dtype, shard_dim=None):
    return ttnn.from_torch(
        tensor.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=(
            ttnn.ReplicateTensorToMesh(mesh) if shard_dim is None else ttnn.ShardTensorToMesh(mesh, dim=shard_dim)
        ),
    )


def test_qwen_mlp_prefill_perf():
    torch.manual_seed(0)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    try:
        mesh.reshape(ttnn.MeshShape(1, 4))
        ccl = TT_CCL(mesh)

        w_norm_torch = 1 + 0.1 * torch.randn(H)
        gate_up_torch = torch.cat(
            [interleave_gate_up(0.03 * torch.randn(H, I // TP), 0.03 * torch.randn(H, I // TP)) for _ in range(TP)],
            dim=1,
        )
        down_torch = 0.03 * torch.randn(I, H)
        w_norm = upload(mesh, w_norm_torch.reshape(-1), ttnn.bfloat16)
        w_gate_up = upload(mesh, gate_up_torch, ttnn.bfloat4_b, shard_dim=1)
        w_down = upload(mesh, down_torch, ttnn.bfloat4_b, shard_dim=0)

        gate_up_stored = ttnn.to_torch(w_gate_up, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=1)).chunk(TP, dim=1)
        down_stored = ttnn.to_torch(w_down, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).chunk(TP, dim=0)

        results = []
        for S in SEQUENCE_LENGTHS:
            h_torch = torch.randn(S, H).bfloat16()
            h = upload(mesh, h_torch, ttnn.bfloat16)

            for _ in range(WARMUP_ITERATIONS):
                y = mlp_under_test(h, w_norm, w_gate_up, w_down, ccl)
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(TIMED_ITERATIONS):
                y = mlp_under_test(h, w_norm, w_gate_up, w_down, ccl)
            ttnn.synchronize_device(mesh)
            seconds = (time.perf_counter() - start) / TIMED_ITERATIONS

            expected = reference_mlp(h_torch, w_norm_torch.bfloat16(), gate_up_stored, down_stored)
            for chip, y_chip in enumerate(ttnn.get_device_tensors(y)):
                passed, pcc = comp_pcc(expected, ttnn.to_torch(y_chip).reshape(S, H).float(), PCC_THRESHOLD)
                assert passed, f"S={S} chip {chip}: PCC {pcc}"
            results.append((S, seconds, roofline_seconds(S), pcc))

        for S, seconds, roofline, pcc in results:
            logger.info(
                f"S={S}: {seconds * 1e6:.0f} us, roofline {roofline * 1e6:.0f} us, {roofline / seconds:.0%} of roofline, pcc {pcc}"
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
