# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures and helpers of the block_keys pre-migration suite.

The tests run under ``--import-mode=importlib`` (no package), so the shared helpers are handed to the tests through
the ``bk`` fixture instead of an import. Everything here is independent of the operation: torch references, input
generators and upload/readback helpers only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import ttnn

SENT = 0xFFFFFFFF
U32 = 0xFFFFFFFF


def _load_route_selection():
    """test_acceptance.py owns the route selection (written by the driver). The golden-cut files call the operation
    through the very same ``block_keys_op`` fixture: test_acceptance.py is loaded by path under a private name and its
    fixture re-exported here (test_acceptance.py's own definition takes precedence in that file)."""
    path = Path(__file__).with_name("test_acceptance.py")
    spec = importlib.util.spec_from_file_location("_block_keys_pre_migration_route", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


block_keys_op = _load_route_selection().block_keys_op


def u32(t: torch.Tensor) -> torch.Tensor:
    """Any integer torch tensor -> its uint32 bit pattern as non-negative int64."""
    return t.to(torch.int64) & U32


def ref_keys(pos: torch.Tensor, ids: torch.Tensor, B: int, width: int) -> torch.Tensor:
    """The golden suite's reference (ttnn/ttnn/bringup/block_select_ttnn/tests/unit/test_block_select.py, ref_keys),
    copied verbatim so the cut does not depend on the op's own test folder: key id ids[pos / B] B + pos % B, SENT where
    pos >= K B (none) or the block id it names is >= width / B."""
    pos, ids = u32(pos), u32(ids)
    K = ids.shape[-1]
    okp = pos < K * B
    blk = torch.gather(ids, -1, (pos.clamp(max=K * B - 1) // B))
    ok = okp & (blk < width // B)
    return torch.where(ok, blk * B + pos % B, torch.full_like(pos, SENT))


def ref_keys_loop(pos: torch.Tensor, ids: torch.Tensor, B: int, width: int) -> torch.Tensor:
    """A second, element-wise reference (small inputs only) that the vectorised one is cross-checked against."""
    shape = pos.shape
    pos, ids = u32(pos).reshape(-1, pos.shape[-1]), u32(ids).reshape(-1, ids.shape[-1])
    K, nbw = ids.shape[-1], width // B
    out = torch.empty_like(pos)
    for r in range(pos.shape[0]):
        for j in range(pos.shape[1]):
            p = int(pos[r, j])
            if p >= K * B:
                out[r, j] = SENT
                continue
            b = int(ids[r, p // B])
            out[r, j] = SENT if b >= nbw else b * B + p % B
    return out.reshape(shape)


def make_inputs(
    lead: tuple,
    K: int,
    KP: int,
    B: int,
    width: int,
    *,
    seed: int = 0,
    specials: bool = True,
    pos_oob: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(pos [*lead, KP], ids [*lead, K]) as int64 uint32 values. ids in [0, width / B); pos in [0, K B). ``specials``
    plants, where the rows exist: SENT and just-out-of-range (width / B) and huge (2^31) block ids, a row of one repeated
    block, a row of SENT positions, the first and the last position of the gathered row. ``pos_oob`` adds positions in
    [K B, SENT) (not none, but past the gathered row)."""
    g = torch.Generator().manual_seed(seed)
    nbw = width // B
    ids = torch.randint(0, nbw, (*lead, K), generator=g, dtype=torch.int64)
    pos = torch.randint(0, K * B, (*lead, KP), generator=g, dtype=torch.int64)
    if specials:
        f_ids = ids.view(-1, K)
        f_pos = pos.view(-1, KP)
        R = f_ids.shape[0]
        f_ids[0, -min(5, K) :] = SENT  # none blocks
        f_ids[R // 2, min(3, K - 1)] = nbw  # just out of range
        f_ids[R - 1, 0] = 1 << 31  # far out of range (sign bit as int32)
        if R > 2:
            f_ids[1, :] = nbw - 1  # one block repeated (the last valid one: the largest key ids)
        if R > 3:
            f_pos[2, : min(7, KP)] = SENT  # none positions
        f_pos[0, 0] = 0
        f_pos[R - 1, KP - 1] = K * B - 1
    if pos_oob:
        f_pos = pos.view(-1, KP)
        R = f_pos.shape[0]
        f_pos[0, : min(4, KP)] = torch.tensor([K * B, K * B + 1, 1 << 31, SENT - 1], dtype=torch.int64)[: min(4, KP)]
        f_pos[R - 1, KP // 2] = K * B
    return pos, ids


def _host_u32(t: torch.Tensor) -> torch.Tensor:
    return u32(t).to(torch.int32)  # int64 -> int32 wraps: the uint32 bit pattern


def up(t: torch.Tensor, device, *, memory_config=None, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32, mesh=False):
    """Upload uint32 values (or any dtype when given) to ``device``; DRAM interleaved ROW_MAJOR by default."""
    host = _host_u32(t) if dtype in (ttnn.uint32, ttnn.int32) else t
    kwargs = {}
    if mesh:
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(
        host.contiguous(),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
        **kwargs,
    )


def down(t) -> torch.Tensor:
    """Read back a single-device uint32 tensor as int64 bit patterns."""
    return u32(ttnn.to_torch(t))


def down_all(t) -> list[torch.Tensor]:
    """Read back every device's shard of a replicated mesh tensor."""
    return [u32(ttnn.to_torch(s)) for s in ttnn.get_device_tensors(t)]


def addr(t) -> int:
    return t.buffer_address()


def assert_dram_rm_u32(out, shape: list):
    assert list(out.shape) == list(shape), (list(out.shape), shape)
    assert list(out.padded_shape) == list(shape), list(out.padded_shape)
    assert out.dtype == ttnn.uint32, out.dtype
    assert out.layout == ttnn.ROW_MAJOR_LAYOUT, out.layout
    mc = out.memory_config()
    assert mc.buffer_type == ttnn.BufferType.DRAM, mc
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED, mc
    assert out.storage_type() == ttnn.StorageType.DEVICE


def assert_exact(got: torch.Tensor, want: torch.Tensor, what=""):
    assert got.shape == want.shape, (what, got.shape, want.shape)
    bad = (got != want).nonzero()
    assert len(bad) == 0, (
        f"{what}: {len(bad)} of {want.numel()} key ids differ; first {bad[:6].tolist()}: "
        f"got {[int(got[tuple(i)]) for i in bad[:6]]} want {[int(want[tuple(i)]) for i in bad[:6]]}"
    )


@pytest.fixture
def bk():
    return SimpleNamespace(
        SENT=SENT,
        u32=u32,
        ref_keys=ref_keys,
        ref_keys_loop=ref_keys_loop,
        make_inputs=make_inputs,
        up=up,
        down=down,
        down_all=down_all,
        addr=addr,
        assert_dram_rm_u32=assert_dram_rm_u32,
        assert_exact=assert_exact,
    )


@pytest.fixture(scope="module")
def system_mesh():
    """The whole system mesh, as the golden suite opens it (without its fabric config: block_keys moves no data
    between chips)."""
    m = ttnn.open_mesh_device()
    yield m
    ttnn.close_mesh_device(m)


@pytest.fixture
def fresh_cache():
    """Start a cache sequence from a known, enabled, empty program cache; returns the entry counter."""

    def start(device):
        ttnn.synchronize_device(device)
        device.enable_program_cache()
        device.clear_program_cache()
        assert device.num_program_cache_entries() == 0
        return device.num_program_cache_entries

    return start
