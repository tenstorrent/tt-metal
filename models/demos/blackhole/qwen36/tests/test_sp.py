# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SeqParallel GDN entry states on a 2D (rows, tp) Galaxy mesh, against a host chain of the rows' transforms.

Run (Blackhole Galaxy):
    pytest models/demos/blackhole/qwen36/tests/test_sp.py -v
Random-free and checkpoint-free: each row's transform is S_out = a_r S_in + B_r with a dyadic scalar a_r, so the
expected entry states are hand-checkable.
"""
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt.sp import SeqParallel
from models.ttt_compat.tt.ccl import TT_CCL

NV, DK, DV = 12, 128, 128  # value heads per TP device and GDN head dims (SeqParallel needs Dk == Dv)
LOCAL_ROWS = 32


def _row_transforms(rows):
    """Row r: A_r = a_r I (a_r = 1/2 on even rows, 3/4 on odd rows) and B_r = (h + 1) J per head h (J all ones)."""
    a = torch.tensor([0.5 if r % 2 == 0 else 0.75 for r in range(rows)], dtype=torch.float64)
    eye = torch.eye(DK, dtype=torch.float64).expand(NV, DK, DK)
    heads = torch.arange(1, NV + 1, dtype=torch.float64).view(NV, 1, 1)
    A = torch.stack([a[r] * eye for r in range(rows)])  # [R, Nv, Dk, Dk]
    B = (heads * torch.ones(NV, DK, DV, dtype=torch.float64)).expand(rows, NV, DK, DV)  # [R, Nv, Dk, Dv]
    return A, B


def _expected_entries(A, B):
    """S_0 = 0, S_{r+1} = A_r @ S_r + B_r; row r's entry state is S_r. With a = (1/2, 3/4, ...) and per-head scale
    h + 1: S_1 = 1, S_2 = 3/4 + 1 = 1.75, S_3 = 1.75/2 + 1 = 1.875, ... times (h + 1) J."""
    S = torch.zeros(NV, DK, DV, dtype=torch.float64)
    entries = []
    for r in range(A.shape[0]):
        entries.append(S)
        S = A[r] @ S + B[r]
    return torch.stack(entries)


def _final_state(A, B, fold):
    """What the delta op hands the entry-state chain per row: the fold call's [A | B] [1, Nv, Dk, Dk + Dv], or the
    batch-2 call's [B, A] [2, Nv, Dk, Dv]; stacked over the rows along dim 0."""
    per_row = torch.cat([A, B], dim=3).unsqueeze(1) if fold else torch.stack([B, A], dim=1)
    return per_row.flatten(0, 1).to(torch.float32)


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True, ids=["ring"]
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4"), pytest.param((4, 8), id="4x8")], indirect=True)
@pytest.mark.parametrize("fold", [False, True], ids=["batch2", "fold"])
def test_gdn_entry_state_chains_rows(mesh_device, fold):
    """Every device's entry state is its row's S_r from a nonzero chain of rows; covers both final-state layouts."""
    args = SimpleNamespace(gdn_nv_tp=NV, gdn_dk=DK, gdn_dv=DV, ccl_topology=lambda: ttnn.Topology.Ring)
    sp = SeqParallel(mesh_device, args, TT_CCL(mesh_device))
    rows, cols = tuple(mesh_device.shape)

    A, B = _row_transforms(rows)
    fin = ttnn.from_torch(
        _final_state(A, B, fold),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=sp.rows_mapper(0),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    entry_state = sp.gdn_entry_state_fold if fold else sp.gdn_entry_state
    entry = entry_state(fin, local_rows=LOCAL_ROWS)

    expected = _expected_entries(A, B)
    # E and B cross as exact BF16 values; the chain's E @ S matmul rounds its S operand (the op's own tests gate at
    # NRMSE 0.004-0.005), so allow 2^-8 relative per element. Row 0 enters with the exact zero initial state. A chain
    # that applies (I + A) instead of A misses from row 2 on by more than 50 % (row 2: 2.75 instead of 1.75).
    for index, shard in enumerate(ttnn.get_device_tensors(entry)):
        row = index // cols
        actual = ttnn.to_torch(shard).to(torch.float64).reshape(NV, DK, DV)
        torch.testing.assert_close(
            actual, expected[row], rtol=2**-8, atol=0, msg=lambda m: f"device {index} (row {row}): {m}"
        )
