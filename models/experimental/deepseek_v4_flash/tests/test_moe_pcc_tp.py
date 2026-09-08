# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel PCC for ``DeepSeekV4SparseMoeBlock`` on a 1x4 submesh.

Same gold-reference bundle as ``test_moe_pcc.py`` (full ``I = 2048`` experts). The
ttnn side takes a 1x4 slice of the host 8x4 mesh and column/row-shards every
SwiGLU weight on the intermediate axis (``I/tp`` per chip): routed ``fused_experts``
and the shared expert both emit H-partials that are summed, then one all-reduce
recovers the full residual.

Requires fabric (CCL) and an 8x4 system mesh.
"""

from __future__ import annotations

import types

import pytest
import torch
from loguru import logger
import tracy
import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.deepseek_v4_flash.tests.test_moe_pcc import (
    WEIGHT_DTYPE_PCC,
    _generate_reference,
)
from models.experimental.deepseek_v4_flash.tt.moe import (
    DeepSeekV4PreloadedExperts,
    DeepSeekV4SparseMoeBlock,
)

TP_SIZE = 4
SUBMESH_SHAPE = (1, TP_SIZE)
PARENT_MESH = (8, 4)


def _to_tt_replicated(t: torch.Tensor, device) -> ttnn.Tensor:
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"])
@pytest.mark.parametrize("weight_dtype", tuple(WEIGHT_DTYPE_PCC), ids=lambda d: str(d).split(".")[-1])
@pytest.mark.parametrize("seq_len", (1,))
@pytest.mark.parametrize("batch_size", (1,))
def test_moe_pcc_tp(mesh_device, reset_seeds, tmp_path, batch_size: int, seq_len: int, weight_dtype) -> None:
    if tuple(mesh_device.shape) != PARENT_MESH:
        pytest.skip(f"need an {PARENT_MESH[0]}x{PARENT_MESH[1]} mesh, got {tuple(mesh_device.shape)}")

    submesh = mesh_device.create_submesh(ttnn.MeshShape(*SUBMESH_SHAPE), ttnn.MeshCoordinate(0, 0))
    assert submesh.get_num_devices() == TP_SIZE, f"1x4 submesh has {submesh.get_num_devices()} devices"

    ref_path = tmp_path / "ref_moe_tp.pt"
    if not _generate_reference(ref_path, batch_size, seq_len):
        pytest.skip("could not generate HF reference for moe (cached transformers 5.8.1 unavailable)")

    bundle = torch.load(ref_path, weights_only=False)
    cfg = types.SimpleNamespace(**bundle["config"])
    assert cfg.moe_intermediate_size % TP_SIZE == 0

    pcc_threshold = WEIGHT_DTYPE_PCC[weight_dtype]
    state_dict = bundle["state_dict"]
    stacked_gate_up = state_dict["experts.gate_up_proj"]  # [E, 2I, H]
    stacked_down = state_dict["experts.down_proj"]  # [E, H, I]

    def _provider(e: int):
        return stacked_gate_up[e], stacked_down[e]

    experts = DeepSeekV4PreloadedExperts(cfg, _provider, submesh, dtype=weight_dtype, tp_size=TP_SIZE)
    moe = DeepSeekV4SparseMoeBlock(cfg, state_dict, submesh, experts=experts, tp_size=TP_SIZE)

    hidden_tt = _to_tt_replicated(bundle["hidden"].unsqueeze(2), submesh)
    tracy.signpost("moe.forward.start")
    out_tt = moe.forward(hidden_tt)
    tracy.signpost("moe.forward.end")
    # All-reduce replicates the full residual; read one chip's copy.
    out_torch = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0))
    out_torch = out_torch[0].reshape(bundle["output"].shape).to(torch.float32)

    reference = bundle["output"].to(torch.float32)
    passing, pcc_message = comp_pcc(reference, out_torch, pcc=pcc_threshold)
    logger.info(comp_allclose(reference, out_torch))
    logger.info(f"[moe tp{TP_SIZE}] PCC: {pcc_message}")

    assert (
        passing
    ), f"moe TP{TP_SIZE} PCC < {pcc_threshold} (batch={batch_size}, seq={seq_len}, {weight_dtype}): {pcc_message}"
