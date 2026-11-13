# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the packed per-step inputs of the MTP verify and drafter traces give the values the host staged before.

* test_rope_lookup_gives_the_host_rows: Qwen36Model.rope_lookup of table rows == _rope_tp_cos_sin_decode_torch of the
  positions plus deltas, byte for byte, at the verify's and the drafter's row counts.
* test_the_drafter_inputs_are_the_staged_values: MTPSpecEngine._drafter_inputs over a drafter pack: tokens, KV
  positions, cos / sin of the real rows and the anchor one-hot equal the per-buffer values the drafter staged before;
  the anchor matmul picks the exact feed rows also when B * T < 32 and the feed's tile padding holds data.

    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_spec_step.py
"""

import random
from types import SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import _resolve_mesh_shape
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_engine import (
    MTPSpecEngine,
    anchor_selector,
    reseed_row_plan,
    reseed_rows_host,
)
from models.demos.blackhole.qwen36.tt.spec_step import drafter_pack, pack_width, rope_rows

MESH = _resolve_mesh_shape(8)
RD = 64
MAX_SEQ_LEN = 262144


def _parametrize(fn):
    params = {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 64 << 20}
    fn = pytest.mark.parametrize("device_params", [params], indirect=True)(fn)
    return pytest.mark.parametrize("mesh_device", [pytest.param(MESH, id=f"{MESH[0]}x{MESH[1]}")], indirect=True)(fn)


def _rope_model(mesh):
    m = Qwen36Model.__new__(Qwen36Model)
    m.args = SimpleNamespace(rope_head_dim=RD, rope_theta=10_000_000.0, max_seq_len=MAX_SEQ_LEN)
    m.rope = SimpleNamespace(rope_delta=0)
    m.device = m.mesh_device = mesh
    return m


def _rep(mesh, t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _host(t):
    """Device-0 replica, after checking that every device holds the same bytes."""
    per_dev = [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]
    for d in per_dev[1:]:
        assert torch.equal(d, per_dev[0]), "devices disagree"
    return per_dev[0]


def _bits(t):
    return t.contiguous().view(torch.int16) if t.dtype == torch.bfloat16 else t


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("n", [4, 8, 32])
def test_rope_lookup_gives_the_host_rows(mesh_device, n):
    model = _rope_model(mesh_device)
    rows = model.rope_table_rows()
    rng = random.Random(n)
    for case in range(4):
        pos = [rng.randrange(0, MAX_SEQ_LEN) for _ in range(n)]
        if case == 0:
            pos[-1] = MAX_SEQ_LEN - 1
        deltas = [rng.choice([0, -min(p, rng.randrange(1, 4000))]) for p in pos]
        idx = rope_rows(pos, deltas, rows)
        dev_idx = _rep(mesh_device, idx.reshape(1, n).to(torch.int32), ttnn.uint32)
        cos, sin = model.rope_lookup(dev_idx, n)
        assert tuple(cos.shape) == (1, 1, n, RD) and cos.layout == ttnn.TILE_LAYOUT
        want = model._rope_tp_cos_sin_decode_torch(torch.tensor(pos, dtype=torch.int32), deltas)
        for got, w in zip((cos, sin), want):
            assert torch.equal(_bits(_host(got).reshape(n, RD)), _bits(w.reshape(n, RD)))
        for t in (dev_idx, cos, sin):
            ttnn.deallocate(t)


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("B", [1, 8])
def test_the_drafter_inputs_are_the_staged_values(mesh_device, B):
    K = 3
    T, R = K + 1, B * (K + 1)
    model = _rope_model(mesh_device)
    rows = model.rope_table_rows()
    rng = random.Random(B)
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e.model = B, K, T, model
    e._dr_iota = _rep(
        mesh_device, torch.arange(1, R + 1, dtype=torch.float32).reshape(1, R), ttnn.float32, ttnn.TILE_LAYOUT
    )
    for _ in range(3):
        m = [rng.randrange(T) for _ in range(B)]
        deltas = [rng.choice([0, -rng.randrange(1, 30)]) for _ in range(B)]
        p0 = [rng.randrange(100, 200000) for _ in range(B)]
        decline = [B > 1 and u == 1 for u in range(B)]
        keep = list(decline)
        ctok = torch.full((B, T), -1, dtype=torch.int64)
        for u in range(B):
            ctok[u, : m[u] + 1] = torch.tensor([rng.randrange(248320) for _ in range(m[u] + 1)])
        cpos = torch.tensor([[p0[u] + 1 + j for j in range(T)] for u in range(B)])
        pack = drafter_pack(ctok, m, p0, decline, keep, rope_rows(p0, deltas, rows), K, rows)
        assert pack.shape[1] == pack_width(R)
        e._dr_pack = _rep(mesh_device, pack, ttnn.uint32)
        x, owned = MTPSpecEngine._drafter_inputs(e)
        # tokens and positions: the pack rows the drafter staged per buffer before
        m_kv = [mu + int(d and k) for mu, d, k in zip(m, decline, keep)]
        tok, pos, _ = reseed_rows_host(reseed_row_plan(ctok, cpos, m_kv, B, T), torch.zeros(B, 1), T, 1, 0)
        assert _host(x["rs_tok"]).reshape(-1).tolist() == tok.reshape(-1).tolist()
        assert _host(x["rs_pos"]).reshape(-1).tolist() == pos.tolist()
        assert _host(x["tok0"]).reshape(-1)[:B].tolist() == pack[0, :B].tolist()
        for k in range(K):
            assert _host(x["pos"][k]).reshape(-1).tolist() == pack[1 + k, :B].tolist()
        # cos / sin of every real reseed row and every drafting row: the host values at position + delta
        rs_cos = _host(x["rs_cos"]).reshape(R, RD)
        for u in range(B):
            for i in range(m_kv[u]):
                want = model._rope_tp_cos_sin_decode_torch(torch.tensor([p0[u] + i]), [deltas[u]])[0].reshape(RD)
                assert torch.equal(_bits(rs_cos[u * T + i]), _bits(want))
            if decline[u]:
                continue
            for k in range(K):
                want = model._rope_tp_cos_sin_decode_torch(torch.tensor([p0[u] + m[u] + k]), [deltas[u]])[1]
                assert torch.equal(_bits(_host(x["sin"][k]).reshape(B, RD)[u]), _bits(want.reshape(RD)))
        # the anchor one-hot, and its matmul against a feed whose tile padding rows hold data
        sel = _host(x["anchor_sel"]).reshape(B, R).float()
        assert torch.equal(sel, anchor_selector(m, B, T)[0, 0].float())
        full = torch.randn(1, 1, 32, 256).to(torch.bfloat16)
        feed_full = _rep(mesh_device, full, ttnn.bfloat16, ttnn.TILE_LAYOUT)
        feed = ttnn.slice(feed_full, (0, 0, 0, 0), (1, 1, R, 256)) if R < 32 else feed_full
        cc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        hp = _host(ttnn.matmul(x["anchor_sel"], feed, compute_kernel_config=cc)).reshape(B, 256)
        want = full[0, 0, [u * T + m[u] for u in range(B)]]
        assert torch.equal(_bits(hp), _bits(want)), "the anchor matmul must pick the exact feed rows"
        for t in owned:
            ttnn.deallocate(t)
        ttnn.deallocate(e._dr_pack)
