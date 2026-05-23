# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
PCC test for the unified (single-op) routed expert FFN.

Mirrors test_single_routed_expert.py but calls the new
ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn directly
instead of going through TtRoutedExpert.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.reference.tt.moe.expert import TorchExpert
from tests.ttnn.utils_for_testing import comp_pcc

COMPUTE_KERNEL_CONFIG_LOFI = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.LoFi,
    math_approx_mode=False,
    fp32_dest_acc_en=False,
    packer_l1_acc=True,
)


@pytest.mark.parametrize(
    "num_tokens, emb_dim, hidden_dim",
    [
        (2048, 7168, 2048),  # DeepSeek V3 dims, 2K tokens (single chunk)
        (4096, 7168, 2048),  # 2 chunks
        (8192, 7168, 2048),  # 4 chunks
    ],
    ids=["ds-v3-2k", "ds-v3-4k", "ds-v3-8k"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [
        pytest.param(
            1,
            {"fabric_config": ttnn.FabricConfig.DISABLED},
            id="single-chip",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_unified_routed_expert(
    mesh_device,
    device_params,
    num_tokens: int,
    emb_dim: int,
    hidden_dim: int,
):
    """Minimum-viable PCC test for the unified single-op routed expert."""

    torch.manual_seed(42)
    weights = {
        "gate_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "up_proj": torch.randn(hidden_dim, emb_dim, dtype=torch.float32) * 0.02,
        "down_proj": torch.randn(emb_dim, hidden_dim, dtype=torch.float32) * 0.02,
    }
    torch_expert = TorchExpert(emb_dim, hidden_dim, weights)
    torch_input = torch.randn(num_tokens, emb_dim, dtype=torch.float32)

    with torch.no_grad():
        torch_output = torch_expert(torch_input)

    tt_input = ttnn.from_torch(
        torch_input,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat8_b,
    )

    # TTNN weights — match the existing routed-expert layout:
    #   gate/up: (emb, hidden), down: (hidden, emb)
    def _w(t):
        return ttnn.from_torch(
            t.T.contiguous(),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            dtype=ttnn.bfloat4_b,
        )

    tt_gate = _w(weights["gate_proj"])
    tt_up = _w(weights["up_proj"])
    tt_down = _w(weights["down_proj"])

    def _idx_tensor(values):
        return ttnn.from_torch(
            torch.tensor(values, dtype=torch.int32),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            dtype=ttnn.uint32,
        )

    counts = _idx_tensor([num_tokens])
    idx_table = _idx_tensor([0])

    tt_output = ttnn.experimental.deepseek_prefill.unified_routed_expert_ffn(
        tt_input,
        tt_gate,
        tt_up,
        tt_down,
        counts,
        idx_table,
        local_expert_id=0,
        compute_kernel_config=COMPUTE_KERNEL_CONFIG_LOFI,
    )

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )

    print(
        f"[unified] tt out: shape={tt_output_torch.shape}, "
        f"min={tt_output_torch.min().item():.4f}, max={tt_output_torch.max().item():.4f}, "
        f"abs_mean={tt_output_torch.float().abs().mean().item():.4f}, "
        f"nz_frac={(tt_output_torch.float().abs() > 1e-6).float().mean().item():.4f}",
        flush=True,
    )
    print(
        f"[unified] torch out: min={torch_output.min().item():.4f}, max={torch_output.max().item():.4f}, "
        f"abs_mean={torch_output.float().abs().mean().item():.4f}",
        flush=True,
    )
    _, pcc = comp_pcc(torch_output, tt_output_torch)
    logger.info(f"unified routed expert PCC: {pcc:.6f}")
    assert pcc >= 0.95, f"PCC {pcc:.6f} below threshold 0.95"
    assert not torch.isnan(tt_output_torch).any(), "Output has NaN"
    assert not torch.isinf(tt_output_torch).any(), "Output has Inf"
