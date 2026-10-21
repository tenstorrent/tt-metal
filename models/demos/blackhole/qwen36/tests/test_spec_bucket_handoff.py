# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device probe of the bucket hand-off primitives (tt/spec_bucket.py handoff_*): copy the b rows of a bucket output into
rows 0..b-1 of the shared 8-row buffer, rows b..7 keep their previous content, inside a trace capture / replay.

TILE rows: the source is a [1,1,b,d] bf16 TILE tensor whose tile padding rows (b..31) hold NaN, as a matmul output can.
Ids: the source is a [1,1,b,1] uint32 ROW_MAJOR tensor. For each candidate the test checks, eagerly and then on trace replay
(after a warm pass, program cache misses disallowed during the capture): rows < b equal the source bit for bit, rows b..7 equal
their previous content, the destination address is unchanged, and the replay still holds when the destination and source change
between replays. Run one node id per device run:
  pytest models/demos/blackhole/qwen36/tests/test_spec_bucket_handoff.py::test_handoff_rows[...]
"""

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tt import spec_bucket as sb

from .test_factory import parametrize_mesh_tp

B = 8
D = 256
ROW_CANDIDATES = ("slice_write", "concat")
ID_CANDIDATES = ("slice_write", "concat")


def _up(mesh, t, dtype, layout):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _down(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def _nan_pad_src(mesh, vals):
    """[1,1,b,D] bf16 TILE tensor holding `vals` [b, D] whose tile padding rows b..31 hold NaN.

    Most ttnn ops refill the tile padding with zeros, so it is written explicitly: ttnn.fill_implicit_tile_padding(NaN). (The padding
    cannot be read back from Python: to_torch, pad, concat and matmul all hide or refill it; the destination checks below are exact.)
    """
    b = vals.shape[0]
    t = _up(mesh, vals.reshape(1, 1, b, D), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    return ttnn.fill_implicit_tile_padding(t, float("nan"))  # may alias t: t is not freed


def _read_rows(t, n):
    return _down(t).reshape(-1, t.shape[-1])[:n]


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("cand", ROW_CANDIDATES)
@pytest.mark.parametrize("b", [1, 2, 4])
def test_handoff_rows(mesh_device, b, cand, reset_seeds):
    mesh = mesh_device
    fn = {
        "slice_write": lambda s, d: sb.handoff_rows_slice_write(s, d, b),
        "concat": lambda s, d: sb.handoff_rows_concat(s, d, b),
    }[cand]
    prev = torch.randn(B, D).to(torch.bfloat16)
    dst = _up(mesh, prev.reshape(1, 1, B, D), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    addr = dst.buffer_address()
    src = _nan_pad_src(mesh, torch.randn(b, D).to(torch.bfloat16) + 3.0)
    v1 = _read_rows(src, b)
    assert not bool(torch.isnan(v1).any()) and bool((v1 != 0).all())
    assert tuple(src.shape) == (1, 1, b, D)

    def check(label, vals, before):
        got = _read_rows(dst, B)
        assert torch.equal(got[:b], vals), f"{label}: rows < {b} differ from the source"
        assert torch.equal(got[b:], before[b:]), f"{label}: rows {b}..{B - 1} did not keep their content"
        assert not bool(torch.isnan(got).any()), f"{label}: NaN leaked into the destination"
        assert dst.buffer_address() == addr, f"{label}: destination moved"

    fn(src, dst)  # eager = warm pass (compiles every program)
    ttnn.synchronize_device(mesh)
    check("eager", v1, prev)

    # reset the destination, capture, reset again so only the replay can fix it
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            prev.reshape(1, 1, B, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        ),
        dst,
    )
    ttnn.synchronize_device(mesh)
    mesh.set_program_cache_misses_allowed(False)
    try:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        fn(src, dst)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    finally:
        mesh.set_program_cache_misses_allowed(True)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    check("replay 1", v1, prev)

    # new source content and new previous destination content, same buffers: the replay must follow them
    prev2 = torch.randn(B, D).to(torch.bfloat16)
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(
            prev2.reshape(1, 1, B, D),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        ),
        dst,
    )
    src2 = _nan_pad_src(
        mesh, torch.randn(b, D).to(torch.bfloat16) + 3.0
    )  # fresh tensor, copied over the captured source
    v2 = _read_rows(src2, b)
    ttnn.copy(src2, src)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    check("replay 2", v2, prev2)
    ttnn.release_trace(mesh, tid)


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("cand", ID_CANDIDATES)
@pytest.mark.parametrize("b", [1, 2, 4])
def test_handoff_ids(mesh_device, b, cand, reset_seeds):
    mesh = mesh_device
    fn = {
        "slice_write": lambda s, d: sb.handoff_ids_slice_write(s, d, b),
        "concat": lambda s, d: sb.handoff_ids_concat(s, d, b),
    }[cand]

    def u32(t):
        return _up(mesh, t.to(torch.int32).reshape(1, 1, -1, 1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)

    def host(t):
        return ttnn.from_torch(
            t.to(torch.int32).reshape(1, 1, -1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    prev = torch.randint(0, 248320, (B,))
    dst = u32(prev)
    addr = dst.buffer_address()
    v1 = torch.tensor([248319, 1 << 20, 16777217, 5][:b])  # ids above 2**24 too: a float round trip would corrupt them
    src = u32(v1)

    def check(label, vals, before):
        got = _down(dst).reshape(-1)[:B].to(torch.int64)
        assert torch.equal(got[:b], vals.to(torch.int64)), f"{label}: ids < {b} differ: {got.tolist()}"
        assert torch.equal(got[b:], before[b:].to(torch.int64)), f"{label}: ids {b}..{B - 1} did not keep their content"
        assert dst.buffer_address() == addr, f"{label}: destination moved"

    fn(src, dst)
    ttnn.synchronize_device(mesh)
    check("eager", v1, prev)

    ttnn.copy_host_to_device_tensor(host(prev), dst)
    ttnn.synchronize_device(mesh)
    mesh.set_program_cache_misses_allowed(False)
    try:
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        fn(src, dst)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
    finally:
        mesh.set_program_cache_misses_allowed(True)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    check("replay 1", v1, prev)

    prev2 = torch.randint(0, 248320, (B,))
    v2 = torch.randint(0, 248320, (b,))
    ttnn.copy_host_to_device_tensor(host(prev2), dst)
    ttnn.copy_host_to_device_tensor(host(v2), src)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    check("replay 2", v2, prev2)
    ttnn.release_trace(mesh, tid)
