# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device test of QWEN38_LANE_PLE_DMAJOR: the device-major persistent host tensor of the lane PLE upload.

For 8 and 32 lanes: the real ``_lane_ple_host_tensor`` builds the d-major buffer; the borrow probe must pass; three random
row sets go buffer.copy_ -> copy_host_to_device_tensor -> to_torch and must equal the rows bit for bit; the fresh
``from_torch`` and the d-major ``copy_`` are timed (``DMAJOR_TIMING`` lines).

Run:  see runs/plan_nov05/h5_dmajor/go.sh (needs the device lock)."""

import statistics
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen38_flash_next.demo.text_demo import (  # noqa: F401  (device_params is a fixture)
    DEVICE_PARAMS,
    device_params,
)
from models.demos.blackhole.qwen38_flash_next.ttnn import model as model_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import MESH_SHAPE, Qwen38MeshContract, TensorPlacement

ROW_ELEMS = 2560
REPEATS = 50


def _stub(mesh_device):
    stub = SimpleNamespace(mesh_device=mesh_device, _lane_ple_hosts={})
    stub._fresh_lane_ple_host = model_module.Qwen38TTNNTextModel._fresh_lane_ple_host
    stub._lane_ple_borrow_proven = lambda b, h, dim=3: model_module.Qwen38TTNNTextModel._lane_ple_borrow_proven(
        stub, b, h, dim
    )
    return stub


def _device_rows(mesh_device, lanes):
    """Like ``prepare_lanes_input``: ``[1,1,lanes,2560]`` BF16 row-major DRAM, sharded on dim 3 over the mesh columns."""

    return ttnn.from_torch(
        torch.zeros(1, 1, lanes, ROW_ELEMS, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3)),
    )


def _read(mesh_device, tensor):
    return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=3))


@run_for_blackhole()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("lanes", [8, 32])
@pytest.mark.timeout(600)
def test_lane_ple_dmajor_upload(mesh_device, lanes, monkeypatch):
    monkeypatch.setattr(model_module, "LANE_PLE_DMAJOR", True)
    columns = MESH_SHAPE[1]
    width = ROW_ELEMS // columns
    device_rows = _device_rows(mesh_device, lanes)
    stub = _stub(mesh_device)
    state = object()
    gen = torch.Generator().manual_seed(1234 + lanes)

    def random_rows():
        return torch.randn(1, 1, lanes, ROW_ELEMS, generator=gen).to(torch.bfloat16)

    # (i) the real helper builds the d-major buffer and the host tensor (the first call also runs the probe)
    rows = random_rows()
    host = model_module.Qwen38TTNNTextModel._lane_ple_host_tensor(stub, state, rows, None)
    entry = stub._lane_ple_hosts[id(state)]
    buffer = entry[1]
    assert tuple(buffer.shape) == (1, columns, lanes, width) and buffer.is_contiguous()
    # (ii) the borrow probe
    assert entry[4] is True and entry[2] is host, "the d-major host tensor does not borrow the torch buffer"

    # (ii-b) the copy gives the device tensor the host tensor's topology: it must equal the lane-major one (dims=(None, 3))
    lane_major = _device_rows(mesh_device, lanes)
    ttnn.copy_host_to_device_tensor(host, device_rows)
    ttnn.synchronize_device(mesh_device)
    want, got = lane_major.tensor_topology(), device_rows.tensor_topology()

    def _pl(t):
        return [(type(p).__name__, getattr(p, "dim", None)) for p in t.placements()]

    assert _pl(got) == _pl(want)
    assert tuple(got.distribution_shape()) == tuple(want.distribution_shape())
    assert tuple(got.mesh_coords()) == tuple(want.mesh_coords())
    assert got.placements()[1].dim == 3
    contract = Qwen38MeshContract(tuple(mesh_device.get_device_ids()))  # the class the model's mesh_contract is
    contract.validate_tensor(device_rows, placement=TensorPlacement.HIDDEN_SHARDED, shard_dim=3)

    # (iii) three random row sets through the in-place buffer
    for _ in range(3):
        rows = random_rows()
        buffer.copy_(rows.reshape(1, lanes, columns, width).permute(0, 2, 1, 3))
        ttnn.copy_host_to_device_tensor(host, device_rows)
        ttnn.synchronize_device(mesh_device)
        back = _read(mesh_device, device_rows)
        assert torch.equal(back, rows), "device rows differ from the rows written through the d-major buffer"

    # the helper's own steady-state path gives the same device bytes
    rows = random_rows()
    again = model_module.Qwen38TTNNTextModel._lane_ple_host_tensor(stub, state, rows, None)
    assert again is host
    ttnn.copy_host_to_device_tensor(again, device_rows)
    ttnn.synchronize_device(mesh_device)
    assert torch.equal(_read(mesh_device, device_rows), rows)

    # (iv) timing: fresh from_torch with dims=(None, 3) vs the d-major copy_
    fresh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, 3))
    fresh_ms, copy_ms = [], []
    for _ in range(REPEATS):
        rows = random_rows()
        t0 = time.perf_counter()
        ttnn.from_torch(rows.contiguous(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=fresh_mapper)
        t1 = time.perf_counter()
        buffer.copy_(rows.reshape(1, lanes, columns, width).permute(0, 2, 1, 3))
        t2 = time.perf_counter()
        fresh_ms.append((t1 - t0) * 1e3)
        copy_ms.append((t2 - t1) * 1e3)
    line = (
        f"DMAJOR_TIMING lanes={lanes} fresh_from_torch_ms={statistics.median(fresh_ms):.4f} "
        f"dmajor_copy_ms={statistics.median(copy_ms):.4f}"
    )
    print(line)
    logger.info(line)
