# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""k2_soak: QWEN38_QSA_OWNPOS=1 (index_tail, main_tail unpaged / paged) and QWEN38_QSA_SELROW2=1 (selection_row unpaged /
paged), 20 calls per trace, 1000 replays; the in-place state (ring, caches, staging) is rewritten every 10 replays.
Temporary: delete with soak_common.py."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import soak_common as sc
from models.demos.blackhole.qwen38_flash_next.tests.test_qsa_tail_q1_device import (
    BF16,
    DEVICE_PARAMS,
    MESH_SHAPE,
    RM,
    TILE,
    IndexCase,
    MainCase,
    SelectionCase,
    bf,
    flags,
)

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires the held mesh")
MESH = pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 64_000_000}], indirect=True)
ROWS = (32, 8)


def state_hosts(mesh, case, kind):
    """Per variant, the host tensors of the case's in-place state (same order as ``state``); variant 0 = the case's own."""
    out = []
    for v in range(sc.VARIANTS):
        g = torch.Generator().manual_seed(777 + v)
        if kind == "index":
            h = (
                case.host
                if v == 0
                else dict(ring=bf(g, *case.host["ring"].shape), cache=bf(g, *case.host["cache"].shape))
            )
            out.append([sc.host_tensor(h["ring"], BF16, TILE, mesh), sc.host_tensor(h["cache"], BF16, TILE, mesh)])
        elif kind == "main":
            st = case.staging_host if v == 0 else bf(g, *case.staging_host.shape)
            ca = case.cache_host if v == 0 else bf(g, *case.cache_host.shape)
            out.append([sc.host_tensor(st, BF16, TILE, mesh), sc.host_tensor(ca, BF16, RM, mesh)])
        else:
            out.append([])
    return out


def run_case(mesh, name, rows, case, kind, flag_kwargs):
    case.prepare_timing()
    state = {"index": lambda: [case._ring, case._cache], "main": lambda: [case._staging, case._cache]}.get(
        kind, lambda: []
    )
    hosts = state_hosts(mesh, case, kind)
    box = {}

    def call(keep):
        out = case.call()
        if keep:
            box["out"] = out
        else:
            ttnn.deallocate(out)

    def prepare(i):
        for h, d in zip(hosts[i % sc.VARIANTS], state()):
            sc.copy_in(h, d)

    def observe():
        return [d for t in [box["out"], *state()] for d in sc.dies(t)]

    with flags(**flag_kwargs):
        sc.soak(mesh, name, rows, call, prepare, observe)


@MESH
@PARAMS
@pytest.mark.parametrize("rows", ROWS)
def test_soak_index_tail_ownpos(mesh_device, rows):
    run_case(
        mesh_device, "index_tail_ownpos", rows, IndexCase(mesh_device, rows, 110 + rows), "index", dict(ownpos="1")
    )


@MESH
@PARAMS
@pytest.mark.parametrize("paged", [False, True], ids=["unpaged", "paged"])
@pytest.mark.parametrize("rows", ROWS)
def test_soak_main_tail_ownpos(mesh_device, rows, paged):
    name = "main_tail_ownpos_" + ("paged" if paged else "unpaged")
    run_case(mesh_device, name, rows, MainCase(mesh_device, rows, paged, 120 + rows), "main", dict(ownpos="1"))


@MESH
@PARAMS
@pytest.mark.parametrize("paged", [False, True], ids=["unpaged", "paged"])
@pytest.mark.parametrize("rows", ROWS)
def test_soak_selection_row_selrow2(mesh_device, rows, paged):
    name = "selection_row_selrow2_" + ("paged" if paged else "unpaged")
    run_case(
        mesh_device, name, rows, SelectionCase(mesh_device, rows, paged, 130 + rows), "selection", dict(selrow2="1")
    )
