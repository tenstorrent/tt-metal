# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""ONE-chip device test of ChainDevice (tt/spec_chain.py): the chain body equals chain_reference bit for bit, eagerly
(>= 200 random cases) and in a trace (>= 50 new cases that only re-stage inputs). Also times 30 trace replays.
Run: TT_VISIBLE_DEVICES=0 pytest -o addopts="" --timeout=1800 --import-mode=importlib -sv \
  models/demos/blackhole/qwen36/tests/test_spec_chain_ops.py"""

import os
import statistics
import time

os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_chain_host import SCRATCH, K, _case
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_words
from models.demos.blackhole.qwen36.tt.spec_chain import ChainDevice, chain_reference, pack_ovr_host, res_c_reference

N_EAGER, N_TRACE = 30, 8


@pytest.fixture(scope="module")
def mesh():
    m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield m
    ttnn.close_mesh_device(m)


def _make(g, B, Nv, parity):
    c = _case(g, B, Nv, parity)
    prev_aux = torch.randint(0, 1000, (6, 32), generator=g, dtype=torch.int32)  # junk except the p0 row
    prev_aux[4, :B] = c["p0_prev"].to(torch.int32)
    res = res_c_reference(c["committed"], c["counts"], c["p0_prev"], torch.randint(0, 2, (B,), generator=g))
    ovr = pack_ovr_host(
        c["nv"],
        c["act"],
        c["dec"],
        [int(k) for k in c["keep"]],
        c["p0_host"],
        c["mask"],
        c["host_tok"],
        c["mi_host"],
        parity,
        K=K,
        scratch_block=SCRATCH,
        rope_delta=c["delta"],
        rope_rows=c["rope_rows"],
    )
    ref = chain_reference(
        res,
        c["drafts"],
        prev_aux,
        ovr,
        B=B,
        K=K,
        Nv=Nv,
        rope_delta=c["delta"],
        rope_rows=c["rope_rows"],
        ctrl_words=spec_ctrl_words(B, Nv),
    )
    return res, prev_aux, c["drafts"].to(torch.int32), ovr, ref


def _h2d(mesh, dst, t, dt):
    host = ttnn.from_torch(
        torch.as_tensor(t).to(torch.int32),
        dtype=dt,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    ttnn.copy_host_to_device_tensor(host, dst)


def _stage(cd, case):
    res, prev_aux, drafts, ovr, _ = case
    rows = int(cd.result_in.shape[0])  # S1 [2,W] or S2 [3,W] (row 2 = overflow, junk for CH)
    _h2d(cd.mesh, cd.result_in, res[:rows], ttnn.int32)
    _h2d(cd.mesh, cd.aux_prev_in, prev_aux, ttnn.int32)
    for k, d in enumerate(cd.draft_ins):
        _h2d(cd.mesh, d, drafts[:, k].reshape(1, 1, cd.B, 1), ttnn.uint32)
    cd.stage_ovr(ovr)


def _check(cd, ref, tag):
    u = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(cd.mesh, dim=0))[: t.shape[0]]
    tok, vaux, aux = ref
    w = lambda t: t.to(torch.int64) & 0xFFFFFFFF
    assert torch.equal(w(u(cd.tok_out).reshape(-1)), w(tok)), f"{tag}: tok"
    assert torch.equal(w(u(cd.vaux_out)), w(vaux)), f"{tag}: vaux"
    assert torch.equal(u(cd.aux_prev_in).to(torch.int32), aux), f"{tag}: next aux"


def _count_ops(cd):
    names = (
        "slice",
        "reshape",
        "transpose",
        "typecast",
        "to_layout",
        "concat",
        "pad",
        "copy",
        "add",
        "subtract",
        "multiply",
        "rsub",
        "eq",
        "lt",
        "sum",
        "minimum",
        "relu",
    )
    n, orig = [0], {k: getattr(ttnn, k) for k in names}
    for k in names:
        setattr(ttnn, k, (lambda f: (lambda *a, **kw: (n.__setitem__(0, n[0] + 1), f(*a, **kw))[1]))(orig[k]))
    try:
        cd.body()
    finally:
        for k in names:
            setattr(ttnn, k, orig[k])
    return n[0]


def test_chain_ops_eager_and_trace(mesh):
    """B in {1,2,4,8} x Nv in {2,6}: 30 eager + 8 trace cases each (240 / 64 in all); S1 [2,W] and S2 [3,W] result."""
    for i, (B, Nv) in enumerate([(b, nv) for b in (1, 2, 4, 8) for nv in (2, 6)]):
        _run(mesh, B, Nv, result_rows=2 + i % 2)


def _run(mesh, B, Nv, result_rows):
    g = torch.Generator().manual_seed(1000 * B + Nv)
    gen = lambda n, off: [_make(g, B, Nv, (off + i) % 2) for i in range(n)]
    eager, traced = gen(N_EAGER, 0), gen(N_TRACE, 1)
    cd = ChainDevice.allocate(mesh, B, K, Nv, result_rows=result_rows)
    for i, case in enumerate(eager):
        _stage(cd, case)
        cd.body()
        _check(cd, case[4], f"eager {i}")
    nops = _count_ops(cd)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    cd.body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    for i, case in enumerate(traced):
        _stage(cd, case)
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        _check(cd, case[4], f"trace {i}")
    times = []
    for _ in range(30):  # replay alone (inputs staged; outputs are overwritten, results not checked)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        times.append((time.perf_counter() - t0) * 1e6)
    ttnn.release_trace(mesh, tid)
    print(
        f"RESULT B={B} Nv={Nv} res_rows={result_rows} eager {len(eager)}/{len(eager)} trace {len(traced)}/{len(traced)} "
        f"body_ops={nops} replay_median_us={statistics.median(times):.0f}"
    )
