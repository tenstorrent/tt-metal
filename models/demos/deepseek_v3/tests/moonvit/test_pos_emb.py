# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Per-submodule test for Learnable2DInterpPosEmb.

Because the interpolation lives entirely on host (and uses the exact
same F.interpolate(mode="bicubic") as the HF reference), the
host-vs-host PCC should be essentially 1.0. We also test:
  - the matching-shape fast path (grid == 64x64),
  - multi-image packing (several (H_i, W_i) concatenated),
  - that adding the host posemb to a device tensor via ttnn.add
    matches HF's `x + pos_embs` end-to-end.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_pos_emb.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.pos_emb import Learnable2DInterpPosEmb

# ----------------------------------------------------------------------
# Host-only tests (no device needed). Use a torch fixture for the ref.


@torch.no_grad()
@pytest.mark.parametrize(
    "grid_hws",
    [
        ([[64, 64]]),  # fast path: matches base 64x64.
        ([[16, 16]]),  # standard 224x224-equivalent grid (interp down).
        ([[32, 24]]),  # asymmetric, smaller.
        ([[16, 16], [32, 24]]),  # multi-image packing.
        ([[80, 80]]),  # interp UP from base — exercises extrapolation direction.
    ],
)
def test_pos_emb_host_matches_hf(model_args, grid_hws):
    """Host-side interp output matches HF reference (PCC ~ 1.0)."""
    pcc_threshold = 0.9999

    # HF reference: invoke forward(zeros, grid_hws) to get the bare posembs.
    ref = model_args.reference_pos_emb()
    assert ref.weight.shape == (model_args.init_pos_emb_height, model_args.init_pos_emb_width, model_args.hidden_size)

    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    total = int(grid_tensor.prod(dim=1).sum().item())
    # `forward(x, grid_hws)` returns x + pos_embs; pass x=0 to recover pos_embs.
    zero_in = torch.zeros(total, model_args.hidden_size, dtype=ref.weight.dtype)
    ref_pos = ref(zero_in, grid_tensor)
    assert ref_pos.shape == (total, model_args.hidden_size)

    # Our wrapper: same call.
    tt_wrapper = Learnable2DInterpPosEmb.from_torch(ref)
    our_pos = tt_wrapper.compute(grid_tensor, dtype=ref.weight.dtype)
    assert our_pos.shape == ref_pos.shape

    passing, pcc_msg = comp_pcc(ref_pos.float(), our_pos.float(), pcc_threshold)
    logger.info(f"[grid_hws={grid_hws}] {comp_allclose(ref_pos, our_pos)} {pcc_msg}")
    assert passing, f"host interp PCC mismatch for grid_hws={grid_hws}: {pcc_msg}"


# ----------------------------------------------------------------------
# Device add test: confirm that staging the host posemb to device and doing
# ttnn.add(patch_embed, pos_embs) matches the HF `x + pos_embs` semantics.


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("grid_hws", [[[16, 16]], [[32, 24]]])
def test_pos_emb_device_add(mesh_device, model_args, grid_hws):
    """End-to-end: host interp -> push to device -> ttnn.add matches torch."""
    pcc_threshold = 0.99

    ref = model_args.reference_pos_emb()
    tt_wrapper = Learnable2DInterpPosEmb.from_torch(ref)

    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    L = int(grid_tensor.prod(dim=1).sum().item())
    D = model_args.hidden_size

    torch.manual_seed(0)
    x_pt = torch.randn(L, D, dtype=torch.bfloat16)
    pos_pt = tt_wrapper.compute(grid_tensor, dtype=torch.bfloat16)
    ref_out = x_pt.float() + pos_pt.float()

    # Push both to device as 4D for ttnn.add.
    is_mesh = type(mesh_device).__name__ == "MeshDevice"

    def to_device(t):
        return ttnn.from_torch(
            t.view(1, 1, *t.shape).contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
        )

    x_tt = to_device(x_pt)
    pos_tt = to_device(pos_pt)
    out_tt = ttnn.add(x_tt, pos_tt, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out_pt = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and out_pt.shape[0] != 1:
        out_pt = out_pt[:1]
    out_pt = out_pt.view(L, D)

    passing, pcc_msg = comp_pcc(ref_out, out_pt, pcc_threshold)
    logger.info(f"[device-add grid_hws={grid_hws}] {comp_allclose(ref_out, out_pt)} {pcc_msg}")
    assert passing, f"device add PCC mismatch for grid_hws={grid_hws}: {pcc_msg}"
