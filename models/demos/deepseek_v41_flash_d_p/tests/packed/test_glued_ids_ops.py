# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.9X (OV-69): a glued chunk's embedding ids made on the device (tt/page_copy.GluedIdsReorder, tt/packed_device.glued_dev_ids)
== the host ids it replaces (tt/packed_device.ids_from_input -> model_ids, PAD clamped to vocab - 1 as
tt/model._host_emb_ids), exactly, op level.

Layouts: test_glued_derive_ops' hand cases (two segments meeting inside a chip at every tile-row offset, 1-token and
idle segments, six segments, a full chunk), the lane headers sent as glued chunks (4 x 2048, padded 4 x ~1.6k, an empty
lane: ChunkDescriptor.as_glued) and AgentX-like fills, each with random ids (some past the vocabulary). The engine
input is tt/packed_device.make_input's (ids + header, one upload). Also counted: program-cache entries and prepared
programs over the layouts after the first (== 0: one program for every layout). The held model gives the mesh and its
vocabulary only (no layer runs).

    <the walls env> scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_ids_ops.py
"""

from __future__ import annotations

import random

import pytest
import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"


def _layouts(chunk: int, R: int, rnd: random.Random) -> list:
    from models.demos.deepseek_v41_flash_d_p.tt.packed import ChunkDescriptor, GluedChunk, Segment

    out = []
    for t in range(1, R // 32):
        a = R + 32 * t
        s1 = 4096 + (32 if a % 64 else 0)
        out.append(GluedChunk(chunk, (Segment(1, s1, s1 + a + 5), Segment(2, 0, 2048))))
    out.append(GluedChunk(chunk, (Segment(1, 128, 129), Segment(2, 64, 64 + 700), Segment(3, 0, 1))))
    out.append(GluedChunk(chunk, tuple(Segment(10 + k, 64 * k * 37, 64 * k * 37 + 1300) for k in range(6))))
    out.append(GluedChunk(chunk, (Segment(1, 139264, 139264 + chunk),)))
    L = chunk // 4
    out.append(ChunkDescriptor.packed(chunk, [Segment(30 + k, 139264 + 64 * k, 139264 + 64 * k + L) for k in range(4)]))
    out.append(ChunkDescriptor.packed(chunk, [Segment(40 + k, 4096 * k, 4096 * k + 1536 + 37 * k) for k in range(4)]))
    out.append(ChunkDescriptor.packed(chunk, [Segment(50, 0, 2048), None, Segment(51, 64, 64 + 900), None]))
    for _ in range(12):
        reqs = []
        for k in range(rnd.randint(1, 9)):
            h = rnd.choice([0, 64, 4096, 61440, 139264, 307200, 546816]) // 64 * 64
            n = rnd.choice([1, 31, 33, 64, 640, 1600, 2900, 5900])
            reqs.append(Segment(60 + k, h, h + n))
        c, _ = GluedChunk.fill(chunk, reqs)
        out.append(c)
    return out


@mesh_parametrize(timeout_s=3600)
def test_glued_ids_ops(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.common.bringup.testing.profile import cache_entries
    from models.demos.deepseek_v41_flash_d_p.tests.fresh_common import PrepCount
    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import switch
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import (
        as_run_desc,
        glued_dev_ids,
        ids_from_input,
        make_input,
    )

    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    deploy = int(S.get("deploy.seq") or S.get("target.seq"))
    chunk = int(S.data["target"]["chunk"])
    cap = -(-deploy // chunk) * chunk  # (the walls' one load: test_packed_walls._cap)
    monkeypatch.setitem(S.data["target"], "seq", max(cap, int(S.data["target"]["seq"])))
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    mesh = model.mesh
    sp, tp = int(mesh.shape[0]), int(mesh.shape[1])
    W = chunk // sp
    vmax = int(model.vocab) - 1
    rnd = random.Random(9)
    bad, n_ok, builds, preps = [], 0, 0, 0
    pc = PrepCount()
    with switch("PACKED", "on"), switch("PACKED_GLUED", "on"), switch("PACKED_GLUED_DYNAMIC", "on"), switch(
        "PACKED_ROUTE", "glued"
    ), switch("POS_DYNAMIC", "on"):
        for ci, desc in enumerate(_layouts(chunk, chunk // (sp * tp), rnd)):
            rd = as_run_desc(desc)
            assert getattr(rd, "glued", False), desc
            if getattr(desc, "glued", False):
                ids = [torch.randint(0, vmax + 200, (s.length,)) for _, s in desc.live()]
            else:
                ids = [None if s is None else torch.randint(0, vmax + 200, (s.length,)) for s in desc.segments]
            inp = make_input(mesh, desc, ids)
            e0, p0 = cache_entries(mesh), pc.n
            with pc:
                dev = glued_dev_ids(model, rd, inp)
            assert dev is not None, "no device ids for a header input"
            if ci > 0:
                e1 = cache_entries(mesh)
                builds += 0 if e0 is None or e1 is None else e1 - e0
                preps += pc.n - p0
            want = ids_from_input(mesh, inp, rd).clamp(0, vmax)  # (the host path: model rows, PAD -> vocab - 1)
            got_dev = ttnn.get_device_tensors(dev)
            for r in range(sp):
                for c in range(tp):
                    got = ttnn.to_torch(got_dev[r * tp + c]).reshape(-1).to(torch.int64) & 0xFFFFFFFF
                    exp = want[r * W : (r + 1) * W]
                    if not torch.equal(got, exp):
                        k = int((got != exp).nonzero()[0])
                        bad.append(f"case {ci} chip ({r}, {c}) row {r * W + k}: {int(got[k])} vs {int(exp[k])}")
                        break
            n_ok += 1
            ttnn.deallocate(dev)
            ttnn.deallocate(inp)
    metrics.record("glued_ids_reorder_mismatches", len(bad))
    metrics.record("glued_ids_reorder_layouts", n_ok)
    metrics.record("glued_ids_reorder_builds_after_first", int(builds) + int(preps))
    print(
        f"glued ids reorder: {n_ok} layouts, mismatches {len(bad)}, builds after the first {builds} + {preps} prepared"
    )
    for b in bad[:10]:
        print("  " + b)
    assert not bad, "device ids differ from the host ids"
    assert builds == 0 and preps == 0, "a layout built a program"
