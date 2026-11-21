# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PB.7C attempt 3: tt/glued_derive's device-derived per-chunk tables == the host functions they replace, op level.

Random glued chunks (tt/packed.GluedChunk.fill over AgentX-like requests, plus hand cases: two segments meeting inside
a chip at every tile-row offset, 1-token segments, an idle segment (cap <= start), six segments, a used-rows == chunk
chunk). For each: the header from the engine input (tt/packed_device.make_input with PACKED_GLUED_DYNAMIC=on: ids +
header, one upload) and from the fallback header upload; every output read back and compared exactly with
GluedGeo.keep_rows / region_lists (no mask and a random mask), the MoE end words (tt/packed.real_prefix), the compressed
keep, per pass the valid lengths / candidate valid lengths / candidate caps and the ChipRows piece tables (Call's
phase / keep pieces), and the server-order rows of random segment lists (WindowGeometry.server_rows + off). The held
model gives the mesh and its indexers' parameters only (no layer runs).

    <the walls env> scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tests/packed/test_glued_derive_ops.py
"""

from __future__ import annotations

import random
import types

import pytest
import torch

from models.demos.common.bringup.testing import batch
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()
BATCH_SHARED = True
BATCH_LAYERS = "executed"


def _cases(chunk: int, R: int, rnd: random.Random) -> list:
    from models.demos.deepseek_v41_flash_d_p.tt.packed import GluedChunk, Segment

    out = []
    # two segments meeting inside chip 1 at every tile-row offset, the second fresh / continuing
    for t in range(1, R // 32):
        a = R + 32 * t  # (segment 1's rows: cap(end) - start = a, so segment 2 starts at chip 1's row 32 t)
        s1 = 4096 + (32 if a % 64 else 0)
        out.append(GluedChunk(chunk, (Segment(1, s1, s1 + a + 5), Segment(2, 0, 2048))))
        out.append(GluedChunk(chunk, (Segment(1, s1, s1 + a), Segment(2, 8192 * 3, 8192 * 3 + 3000))))
    # 1-token and idle segments, six segments, a full chunk
    out.append(GluedChunk(chunk, (Segment(1, 128, 129), Segment(2, 64, 64 + 700), Segment(3, 0, 1))))
    out.append(GluedChunk(chunk, (Segment(1, 1024, 1024 + 64 * 7 + 5), Segment(2, 300032, 300032 + 1250))))
    out.append(GluedChunk(chunk, tuple(Segment(10 + k, 64 * k * 37, 64 * k * 37 + 1300) for k in range(6))))
    out.append(GluedChunk(chunk, (Segment(1, 139264, 139264 + chunk),)))
    # AgentX-like fills
    for _ in range(12):
        reqs = []
        for k in range(rnd.randint(1, 9)):
            h = rnd.choice([0, 64, 4096, 61440, 139264, 307200, 546816]) // 64 * 64
            n = rnd.choice([1, 31, 33, 64, 640, 1600, 2900, 5900])
            reqs.append(Segment(20 + k, h, h + n))
        c, _ = GluedChunk.fill(chunk, reqs)
        out.append(c)
    # PB.9G: chip-aligned lanes (align = R): 1-chip, 1-token, idle, padded, split, replay segments, AgentX-like fills
    A = R
    out.append(
        GluedChunk(chunk, (Segment(1, 2048, 2048 + A), Segment(2, 4096, 4097), Segment(3, 64, 64 + 40)), align=A)
    )
    out.append(GluedChunk(chunk, tuple(Segment(30 + k, 139264, 139264 + 1600) for k in range(4)), align=A))
    out.append(GluedChunk(chunk, (Segment(1, 8192, 8392, 8192 + 128), Segment(2, 4096, 6144, 4096 + 128)), align=A))
    out.append(GluedChunk(chunk, tuple(Segment(10 + k, 64 * k * 37, 64 * k * 37 + 300) for k in range(6)), align=A))
    out.append(GluedChunk(chunk, (Segment(1, 139264, 139264 + chunk),), align=A))
    for _ in range(12):
        reqs = []
        for k in range(rnd.randint(1, 9)):
            h = rnd.choice([0, 64, 4096, 61440, 139264, 307200, 546816]) // 64 * 64
            n = rnd.choice([1, 31, 33, 64, 200, 640, 1600, 2900, 5900])
            ws = h + 128 if (h and n > 256 and rnd.random() < 0.4) else None
            reqs.append(Segment(20 + k, h, h + n, ws))
        c, _ = GluedChunk.fill(chunk, reqs, align=A)
        out.append(c)
    return out


def _segruns(desc) -> list:
    out = []
    for k, s in desc.live():
        re_ = s.run_end(desc.prefill_end, desc.block)
        if re_ > s.start:
            out.append(types.SimpleNamespace(lane=k, start=s.start, run_end=re_, seg=s))
    return out


@mesh_parametrize(timeout_s=3600)
def test_glued_derive_ops(mesh_device, monkeypatch):
    if mesh_device is None:
        pytest.skip("device test")
    import ttnn
    from models.demos.deepseek_v41_flash_d_p.tests.packed.packed_common import switch
    from models.demos.deepseek_v41_flash_d_p.tt.attention import WindowGeometry
    from models.demos.deepseek_v41_flash_d_p.tt.glued_derive import GLUED_SEGS, N_TABLES, TABLE_WORDS, Derive, slot
    from models.demos.deepseek_v41_flash_d_p.tt.indexer import TtCandidateIndexer
    from models.demos.deepseek_v41_flash_d_p.tt.packed import real_prefix
    from models.demos.deepseek_v41_flash_d_p.tt.packed_device import make_input
    from models.demos.deepseek_v41_flash_d_p.tt.packed_glued import Call, GluedGeo

    full = list(S.layers())
    layers = batch.run_layers(S, full, last_chunk=False)
    model = batch.device_model(S, mesh_device, layers, lm_head=False)
    mesh = model.mesh
    chunk = int(model.chunk)
    sp, tp = int(mesh.shape[0]), int(mesh.shape[1])
    n = sp * tp
    R, W = chunk // n, chunk // sp
    ixs = [getattr(model.blocks[i], "indexer", None) for i in model.layers]
    cix = next((ix for ix in ixs if isinstance(ix, TtCandidateIndexer)), None)
    pix = next((ix for ix in ixs if ix is not None), None)
    assert pix is not None
    dvs = {0: Derive(mesh, chunk, 64), R: Derive(mesh, chunk, 64, R)}  # (PB.9G: back to back / chip-aligned)
    comp = ttnn.ConcatMesh2dToTensor(mesh, mesh_shape=(sp, tp), dims=(0, 1))

    def host(t):
        x = ttnn.to_torch(t, mesh_composer=comp)
        return x.to(torch.int64) & 0xFFFFFFFF if t.dtype == ttnn.uint32 else x.float()

    wgeo = WindowGeometry(mesh, chunk)
    rnd = random.Random(7)
    bad = []
    cases = _cases(chunk, R, rnd)
    for ci, desc in enumerate(cases):
        segs = _segruns(desc)
        if not segs:
            continue
        g = GluedGeo(mesh, desc, segs)
        dv = dvs[int(desc.align)]
        chips = "on" if desc.align else "off"
        for src in ("input", "upload"):
            if src == "input":
                with switch("PACKED_GLUED_DYNAMIC", "on"), switch("PACKED_CHIPS", chips):
                    ids = [list(range(s.length)) for _, s in desc.live()]
                    inp = make_input(mesh, desc, ids)
                dv.begin(desc, inp)
            else:
                inp = None
                dv.begin(desc, None)
            tag = f"case {ci} ({src}) off {g.off} rows {g.rows}"
            # keep + MoE end words
            keep, moe = dv.run("keep")
            want = g.keep_rows()
            if not torch.equal(host(keep)[..., :1], want):
                bad.append(f"{tag}: keep")
            nr, _ = real_prefix(chunk, sp, g.keep_rows().reshape(-1) > 0)
            ends = torch.tensor([[r * W + nr[r]] * tp for r in range(sp)]).view(sp, tp, 1, 1)
            if not torch.equal(host(moe), ends):
                bad.append(f"{tag}: moe ends {host(moe).view(-1).tolist()} vs {ends.view(-1).tolist()}")
            # region (no mask, a random mask)
            blocks = [int(r.lane) for r in segs]
            for mask in (0, rnd.randrange(1 << len(segs))):
                classes = tuple(0 if (mask >> j) & 1 else min(int(r.start), 128) for j, r in enumerate(segs))
                got = host(dv.run("region", (), (mask,))[0])
                want = g.region_lists(classes, blocks).to(torch.int64) & 0xFFFFFFFF
                if not torch.equal(got, want):
                    d = (got != want).nonzero()[:4].tolist()
                    bad.append(f"{tag}: region mask {mask} differs at {d}")
            # compressed keep
            for nn in (chunk // 2, chunk // 4):
                got = host(dv.run("ckeep", (nn,))[0])[0, 0, :, 0]
                want = torch.zeros(nn)
                for o, kk in zip(g.off, g.rows):
                    want[o * nn // chunk : (o + kk) * nn // chunk] = 1.0
                if not torch.equal(got, want):
                    bad.append(f"{tag}: ckeep {nn}")
            # idx: valid lengths, candidate caps, piece tables
            for ix in [x for x in (pix, cix) if x is not None]:
                cand = isinstance(ix, TtCandidateIndexer)
                cap = 262144
                prm = (int(ix.chunk), int(ix.ratio), cap, int(ix.block) if cand else 1, int(ix.ctopk) if cand else 0)
                outs = dv.run("idx", prm + (cand,))
                lens = [min(cap, -(-((r.start + ix.chunk) // ix.ratio) // 32) * 32) for r in segs]
                nbv = [v // ix.block for v in lens] if cand else [0] * len(segs)
                tabs = host(outs[-1]).view(n, N_TABLES * TABLE_WORDS)
                for p in range(GLUED_SEGS):
                    ent = [g.entry(q, p) for q in range(n)]
                    clens = [lens[e[0]] if e is not None else 32 for e in ent]
                    if host(outs[p]).view(-1).tolist() != clens:
                        bad.append(f"{tag}: vl pass {p}")
                    if cand:
                        cn = [nbv[e[0]] if e is not None else 0 for e in ent]
                        real = [e is not None for e in ent]
                        vlc = [v if r_ and v >= ix.ctopk else ix.ctopk for r_, v in zip(real, cn)]
                        if host(outs[GLUED_SEGS + p]).view(-1).tolist() != vlc:
                            bad.append(f"{tag}: vlc pass {p}")
                        cv = torch.tensor([v if r_ and v < ix.ctopk else 0 for r_, v in zip(real, cn)]).view(-1, 1)
                        wantc = torch.where(torch.arange(ix.ctopk).view(1, -1) < cv, float("inf"), float("-inf"))
                        gotc = host(outs[2 * GLUED_SEGS + p]).view(n, -1)
                        if not torch.equal(gotc, wantc):
                            bad.append(f"{tag}: ccap pass {p}")
                    js = []
                    for e in ent:
                        if e is not None and e[0] not in js:
                            js.append(e[0])

                    def words(pieces):
                        out = torch.zeros(n, TABLE_WORDS, dtype=torch.int64)
                        for q, ps in enumerate(pieces):
                            ps = [x for x in ps if x[2] > 0]
                            w = [len(ps)] + [v for x in ps for v in x]
                            out[q, : len(w)] = torch.tensor(w)
                        return out

                    def at(t):
                        return tabs[:, t * TABLE_WORDS : (t + 1) * TABLE_WORDS]

                    if js:
                        c = Call(g, p, js, roll=False)
                        exp = {slot("phase", p): words(c.phase_pieces(g))}
                        for ph in (False, True):
                            kp = c.keep_pieces(g, ph)
                            exp[slot("keep", p, ph)] = words(kp)
                            if cand:
                                for lg in (False, True):
                                    pcs = [
                                        (ps if (e is not None and (nbv[e[0]] >= ix.ctopk) == lg) else [])
                                        for ps, e in zip(kp, ent)
                                    ]
                                    exp[slot("cand", p, ph, lg)] = words(pcs)
                    else:
                        exp = {slot("phase", p): words([[(0, 0, R)]] * n)}
                    for t, w in exp.items():
                        if not torch.equal(at(t), w):
                            q = int((at(t) != w).any(-1).nonzero()[0])
                            bad.append(f"{tag}: table {t} chip {q}: {at(t)[q][:7].tolist()} vs {w[q][:7].tolist()}")
            # server-order rows of random segment lists
            for _ in range(2):
                k = rnd.randint(1, len(segs))
                js = rnd.sample(range(len(segs)), k)
                got = host(dv.run("srv", (k,), tuple(js))[0]).view(sp, tp, -1)
                parts = []
                for j in js:
                    lo = max(int(segs[j].start), int(segs[j].seg.gws))  # (PB.9G: a replay writes from gws)
                    d = wgeo.server_rows(32 * ((lo % chunk) // 32))
                    parts.append((d + g.off[j] + lo - int(segs[j].start)) % chunk)
                want = torch.cat(parts, dim=-1).view(sp, 1, -1).expand(sp, tp, -1)
                if not torch.equal(got, want):
                    bad.append(f"{tag}: srv {js}")
            dv.end()
            if inp is not None:
                ttnn.deallocate(inp)
    print(
        f"DERIVE {len(cases)} cases x 2 header sources: {len(bad)} mismatches; runs {[d.stats for d in dvs.values()]}"
    )
    for b in bad[:40]:
        print("DERIVE BAD", b)
    assert not bad, f"{len(bad)} derived tables differ from the host's"
