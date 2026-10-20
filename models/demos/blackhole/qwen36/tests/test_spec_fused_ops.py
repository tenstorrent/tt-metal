# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ONE-chip device test of FusedGreedyDevice (tt/spec_fused.py): the fused greedy accept + drafter-input op sequence
equals fused_greedy_reference bit for bit, eagerly (>= 200 random cases) and in a trace (>= 50 new cases that only
re-stage the inputs, execute_trace, read the outputs). Also times 30 trace replays. No model, no weights.
Run: TT_VISIBLE_DEVICES=0 pytest -o addopts="" --timeout=1800 --import-mode=importlib -sv \
  models/demos/blackhole/qwen36/tests/test_spec_fused_ops.py"""

import os
import statistics
import time

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_fused_host import SCRATCH, K, T, _case
from models.demos.blackhole.qwen36.tt.spec_engine import sanitize_verify_block, table_capacity
from models.demos.blackhole.qwen36.tt.spec_fused import FusedGreedyDevice, fused_accept_aux_host, fused_greedy_reference

BS = 64
ROPE = [(0, 4096), (5, 150), (-7, 60)]  # (rope_delta, rope table rows): the last two hit both clamps


@pytest.fixture(scope="module")
def mesh():
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield m
    ttnn.close_mesh_device(m)


def _make(g, B, nb, edge):
    """One staged-input case -> (ids [R], tok [R], aux, pt_rows, reference) or None (a 'mixed' row: host fallback)."""
    pad, skip, nv, start_pos, tokens, ids, pt = _case(g, B, nb, edge)
    delta, S = ROPE[int(torch.randint(0, len(ROPE), (1,), generator=g))]
    clean, start0, active = sanitize_verify_block(tokens, start_pos, nv, K)
    try:
        aux = fused_accept_aux_host(nv, start0, active, skip, table_capacity(nb, BS), K, SCRATCH, delta, S)
    except ValueError:
        return None
    ids_w, tok_w = ids.reshape(-1).to(torch.int32), clean.reshape(-1).to(torch.int32)
    ref = fused_greedy_reference(ids_w, tok_w, aux, pt, B=B, K=K)
    return ids_w, tok_w, aux, pt.repeat_interleave(T, dim=0), ref


def _check(fd, ref, tag):
    B, R, W = fd.B, fd.R, fd.W
    u = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(fd.mesh, dim=0))[: t.shape[0]]
    pack = u(fd.pack_out).to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(pack, ref.pack.to(torch.int64) & 0xFFFFFFFF), f"{tag}: pack"
    assert torch.equal(u(fd.rs_pt_out).to(torch.int32), ref.rs_pt), f"{tag}: rs_pt"
    res = u(fd.result_out).to(torch.int32)
    want = torch.zeros(2, W, dtype=torch.int32)
    want[0, :R] = ref.committed.reshape(-1)
    want[1, :B] = ref.counts
    assert torch.equal(res, want), f"{tag}: result"


def _gen(g, n, B, nb):
    out = []
    while len(out) < n:
        c = _make(g, B, nb, edge=len(out) % 3 == 2)
        if c is not None:
            out.append(c)
    return out


def test_fused_ops_eager_and_trace(mesh):
    """B in {1,2,4,8} x nb in {4,64}: 30 eager + 8 trace cases each (240 eager, 64 trace in all), one device open."""
    for B in (1, 2, 4, 8):
        for nb in (4, 64):
            _run(mesh, B, nb)


def _run(mesh, B, nb):
    g = torch.Generator().manual_seed(100 * B + nb)
    eager, traced = _gen(g, 30, B, nb), _gen(g, 8, B, nb)
    fd = FusedGreedyDevice.allocate(mesh, B, K, nb, 32)
    seen_m, n_dec = set(), 0
    for i, (ids, tok, aux, ptr, ref) in enumerate(eager):  # the first call also compiles every program
        fd.stage(ids, tok, aux, ptr)
        fd.body()
        _check(fd, ref, f"eager {i}")
        seen_m.update((ref.counts - 1).tolist())
        n_dec += int(ref.declined.sum())
    assert seen_m == set(range(K + 1)) or B == 1, seen_m
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    fd.body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    times = []
    for i, (ids, tok, aux, ptr, ref) in enumerate(traced):
        fd.stage(ids, tok, aux, ptr)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        times.append((time.perf_counter() - t0) * 1e6)
        _check(fd, ref, f"trace {i}")
    for _ in range(30):  # timing: the replay alone (inputs already staged)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        times.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    print(
        f"RESULT B={B} nb={nb} eager {len(eager)}/{len(eager)} trace {len(traced)}/{len(traced)} "
        f"declined_rows={n_dec} replay_median_us={statistics.median(times[len(traced):]):.0f}"
    )
