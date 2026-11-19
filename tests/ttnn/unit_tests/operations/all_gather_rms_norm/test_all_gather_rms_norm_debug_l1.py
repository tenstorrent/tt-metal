# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Debug (keep): where the op's cached global semaphores land in L1 relative to sharded I/O, and the CB budget."""

import pytest
import ttnn

import sys
import ttnn.operations.all_gather_rms_norm  # noqa: F401

op_mod = sys.modules["ttnn.operations.all_gather_rms_norm.all_gather_rms_norm"]
import ttnn.operations.all_gather_rms_norm.all_gather_rms_norm_program_descriptor as pd

from .test_all_gather_rms_norm import _DEVICE_PARAMS, run_case, R, LINE, HS


def _view(mesh, tag):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
    blocks = [(int(b["address"]), int(b["size"]), b["allocated"]) for b in v.block_table]
    print(tag, "total", v.total_bytes_per_bank, "blocks", blocks[:12])
    for key, (_, sems) in op_mod._SEM_CACHE.items():
        print("  sems", [int(ttnn.get_global_semaphore_address(s)) for s in sems])


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], ids=["mesh2x2"], indirect=True)
def test_l1_view(mesh_device):
    _view(mesh_device, "start")
    run_case(mesh_device, cluster_axis=1, shape=(1, 32, 4096), gamma_layout=R, topology=LINE, memory_layout=HS)
    print("budget after first", pd.LAST_INFO["budget"])
    _view(mesh_device, "after first")
    run_case(mesh_device, cluster_axis=1, shape=(1, 1, 64, 128), gamma_layout=R, topology=LINE)
    print("budget interleaved", pd.LAST_INFO["budget"])
    _view(mesh_device, "after second")


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], ids=["mesh2x2"], indirect=True)
def test_l1_view_golden_sequence(mesh_device):
    """Two golden HEIGHT_SHARDED cells back to back: does the first leave L1 allocated?"""
    import gc

    from eval.golden_tests.all_gather_rms_norm.helpers import run_all_gather_rms_norm

    axes = dict(
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        fp32_dest_acc_en=True,
        gamma_mode="gamma",
        gamma_dtype=ttnn.bfloat16,
        gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        cluster_axis=0,
        topology=ttnn.Topology.Linear,
        num_links=1,
        norm_type="layer_norm",
        beta_mode="beta",
    )
    _view(mesh_device, "start")
    run_all_gather_rms_norm(((1, 1, 32, 8192),), mesh_device=mesh_device, **axes)
    print("budget first", pd.LAST_INFO["budget"])
    _view(mesh_device, "after first")
    gc.collect()
    _view(mesh_device, "after gc")
    run_all_gather_rms_norm(((1, 1, 32, 8192),), mesh_device=mesh_device, **dict(axes, cluster_axis=1))
    print("budget second", pd.LAST_INFO["budget"])
    _view(mesh_device, "after second")
    # an interleaved call (no L1 tensors): the semaphore pairs pinned under the freed shards move up
    run_all_gather_rms_norm(
        ((1, 1, 32, 256),), mesh_device=mesh_device, **dict(axes, memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED)
    )
    print("budget third (interleaved)", pd.LAST_INFO["budget"])
    _view(mesh_device, "after third")
    run_all_gather_rms_norm(((1, 1, 32, 8192),), mesh_device=mesh_device, **dict(axes, cluster_axis=1))
    print("budget fourth", pd.LAST_INFO["budget"])


@pytest.mark.parametrize("device_params", _DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], ids=["mesh1x4"], indirect=True)
@pytest.mark.parametrize(
    "case",
    [
        dict(shape=(1, 1, 32, 2048), gamma_layout=R),
        dict(shape=(1, 1, 32, 2048), norm_type="layer_norm", gamma_layout=R, with_beta=True),
        dict(shape=(1, 1, 32, 2048), norm_type="layer_norm", gamma_layout=R, with_beta=True, fp32_dest_acc_en=False),
        dict(
            shape=(1, 1, 32, 1024),
            dtype=ttnn.float32,
            norm_type="layer_norm",
            gamma_layout=R,
            gamma_dtype=ttnn.float32,
            offset=50.0,
        ),
    ],
    ids=["rms_bf16", "ln_bf16", "ln_bf16_acc16", "ln_fp32_offset50"],
)
def test_owner_metrics(mesh_device, case, monkeypatch):
    """Owner path accuracy (prints worst PCC / rel RMS over devices and slices) for the changelog."""
    from . import test_all_gather_rms_norm as acc

    seen = []
    orig = acc._pcc_rel_rms

    def rec(a, e):
        r = orig(a, e)
        seen.append(r)
        return r

    monkeypatch.setattr(acc, "_pcc_rel_rms", rec)
    run_case(mesh_device, cluster_axis=1, topology=ttnn.Topology.Ring, **case)
    print("owner metrics min_pcc", min(p for p, _ in seen), "max_rel_rms", max(r for _, r in seen))
    print("owner info", {k: pd.LAST_INFO[k] for k in ("k", "owner", "rows", "pitch")})
