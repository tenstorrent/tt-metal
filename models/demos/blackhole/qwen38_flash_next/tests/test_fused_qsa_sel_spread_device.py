# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""QWEN38_QSA_SEL_SPREAD=1 on the 1x8 mesh (every die's row compared): the spread selection_row (selection_row_spread.cpp) writes the same 2080-entry
row, bit for bit, as the default program and as the torch reference, for positions around the block, budget and
tail boundaries, for several core counts; rows != 1 keep the default program."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen38_flash_next.tests import qsa_paged_reference as ref
from models.demos.blackhole.qwen38_flash_next.tests.tp_harness import DEVICE_PARAMS
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa
from models.demos.blackhole.qwen38_flash_next.ttnn.fused import qsa_block

pytestmark = pytest.mark.skipif(os.environ.get("QWEN38_FUSED_DEVICE_TEST") != "1", reason="requires a device")
MESH = pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
PARAMS = pytest.mark.parametrize("device_params", [{**DEVICE_PARAMS, "trace_region_size": 2_000_000}], indirect=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
POSITIONS = [0, 1, 5, 31, 32, 33, 127, 128, 2046, 2047, 2048, 2049, 4095, 4096, 8191, 8192, 20000, 65535, 100003]


def up(mesh, tensor):
    return ttnn.from_torch(
        tensor.to(torch.int64).to(torch.uint32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def run(monkeypatch, spread, cores, args):
    monkeypatch.setenv(qsa_block.SEL_SPREAD_ENV, "1" if spread else "0")
    monkeypatch.setenv(qsa_block.SEL_SPREAD_CORES_ENV, str(cores))
    out = qsa_block.selection_row(*args)
    replicas = [ttnn.to_torch(local).to(torch.int64) & 0xFFFFFFFF for local in ttnn.get_device_tensors(out)]
    assert all(torch.equal(replicas[0], other) for other in replicas[1:])
    return replicas[0].reshape(-1, 2080)


@MESH
@PARAMS
@pytest.mark.parametrize("cores", [16, 32, 65])
def test_selection_spread_bitwise(mesh_device, monkeypatch, cores):
    generator = torch.Generator().manual_seed(7)
    constants = qsa.qsa_row_constants()
    offsets = constants["block_offsets"].reshape(1, 2048)
    sentinel = torch.randint(0, 2**32, (1, 32), generator=generator)
    d_off, d_sent = up(mesh_device, offsets.reshape(1, 1, 1, 2048)), up(mesh_device, sentinel.reshape(1, 1, 1, 32))
    for position in POSITIONS:
        inputs = qsa.emulate_qsa_position_inputs(position, allocated_compressed_blocks=65536)
        keep, fill = inputs["row_keep_bits"].reshape(1, 2080), inputs["row_fill"].reshape(1, 2080)
        ids = torch.randint(0, 2**30, (1, 512), generator=generator)
        ids[0, ::7] = position // 4
        args = (
            up(mesh_device, ids.reshape(1, 1, 1, 512)),
            d_sent,
            d_off,
            up(mesh_device, keep.reshape(1, 1, 1, 2080)),
            up(mesh_device, fill.reshape(1, 1, 1, 2080)),
        )
        expected = ref.selection_row_reference(ids, offsets, keep, fill, sentinel)
        base = run(monkeypatch, False, cores, args)
        spread = run(monkeypatch, True, cores, args)
        assert torch.equal(base, expected), position
        assert torch.equal(spread, base), position


@MESH
@PARAMS
def test_selection_spread_rows_unchanged(mesh_device, monkeypatch):
    generator = torch.Generator().manual_seed(11)
    rows = 8
    ids = torch.randint(0, 2**30, (rows, 512), generator=generator)
    offsets = torch.randint(0, 1000, (32, 2048), generator=generator)
    keep = torch.where(torch.rand(32, 2080, generator=generator) < 0.8, 0xFFFFFFFF, 0)
    fill = torch.where(keep == 0, torch.randint(0, 2**32, (32, 2080), generator=generator), torch.zeros_like(keep))
    sentinel = torch.randint(0, 2**32, (1, 32), generator=generator)
    args = (
        up(mesh_device, ids.reshape(1, 1, rows, 512)),
        up(mesh_device, sentinel.reshape(1, 1, 1, 32)),
        up(mesh_device, offsets.reshape(1, 1, 32, 2048)),
        up(mesh_device, keep.reshape(1, 1, 32, 2080)),
        up(mesh_device, fill.reshape(1, 1, 32, 2080)),
    )
    base = run(monkeypatch, False, 32, args)
    assert torch.equal(run(monkeypatch, True, 32, args), base)
