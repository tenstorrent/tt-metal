# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from tests.ttnn.unit_tests.operations.qwen_mlp.test_qwen_mlp_perf import (
    H,
    I,
    PCC_THRESHOLD,
    SEQUENCE_LENGTHS,
    TIMED_ITERATIONS,
    TP,
    WARMUP_ITERATIONS,
    interleave_gate_up,
    reference_mlp,
    roofline_seconds,
    upload,
)
from ttnn.operations.qwen_mlp_prefill import qwen_mlp_prefill
from ttnn.operations.qwen_mlp_prefill.layout import SEGMENT_BYTES


@pytest.mark.parametrize("sequence_lengths", [SEQUENCE_LENGTHS])
def test_qwen_mlp_prefill_perf(sequence_lengths):
    torch.manual_seed(0)
    router_config = ttnn.FabricRouterConfig()
    router_config.max_packet_payload_size_bytes = SEGMENT_BYTES
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING, router_config=router_config)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))
    try:
        mesh.reshape(ttnn.MeshShape(1, 4))
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

        for S in sequence_lengths:
            h_torch = torch.randn(S, H).bfloat16()
            h = upload(mesh, h_torch, ttnn.bfloat16)
            for _ in range(WARMUP_ITERATIONS):
                y = qwen_mlp_prefill(h, w_norm, w_gate_up, w_down)
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(TIMED_ITERATIONS):
                y = qwen_mlp_prefill(h, w_norm, w_gate_up, w_down)
            ttnn.synchronize_device(mesh)
            seconds = (time.perf_counter() - start) / TIMED_ITERATIONS

            expected = reference_mlp(h_torch, w_norm_torch.bfloat16(), gate_up_stored, down_stored)
            for chip, y_chip in enumerate(ttnn.get_device_tensors(y)):
                passed, pcc = comp_pcc(expected, ttnn.to_torch(y_chip).reshape(S, H).float(), PCC_THRESHOLD)
                assert passed, f"S={S} chip {chip}: PCC {pcc}"
            logger.info(
                f"S={S}: {seconds * 1e6:.0f} us, roofline {roofline_seconds(S) * 1e6:.0f} us, "
                f"{roofline_seconds(S) / seconds:.0%} of roofline, pcc {pcc}"
            )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
