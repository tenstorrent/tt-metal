# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device checks of the serving engine (resident_m3_2/engine.py ResidentEngine). Run through bench/dflash2/guarded.sh.

  equiv: the frozen step-2 engine (resident_s2) and this one on the same synthetic caches with identity tables: one
         step (argmax, raw logits, taps) and a 16-step device-fed greedy run must agree bitwise.
  perm:  the same cache contents behind aliased pools (sliding blocks + full-geometry views, b -> 2b) at random,
         disjoint per-group pages must give the same outputs as identity tables (V6: per-group vLLM tables).
  bounds: V1 / N7: positions at capacity - 1 decode; position = capacity, slot = max_users and a table entry past the
         pool are flagged, give id 0, do not advance, and leave every non-scratch page bitwise unchanged.
  modes: feed modes, the logits cap, taps parity.
"""

import argparse
import json
import os
import sys
import time

import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_s2.run_step2 import CACHE, close_mesh, open_mesh, read_aiclk
from models.demos.gemma4_31b_qb2.resident_m3_2.common import is_full

if os.environ.get("GRES_ENGINE_PKG") == "resident_s3":  # (baselines: the frozen M1 engine under this harness)
    from models.demos.gemma4_31b_qb2.resident_s3.engine import (
        F_PAGE, F_POSITION, F_SLOT, EngineConfig, EngineError, KVPools, ResidentEngine,
    )
else:
    from models.demos.gemma4_31b_qb2.resident_m3_2.engine import (
        F_PAGE, F_POSITION, F_SLOT, EngineConfig, EngineError, KVPools, ResidentEngine,
    )

TAPS = (0, 16, 28, 46, 58)
RAW = "/home/ttuser/experiments/gemma4/resident_cache_raw"  # the step-3 default (raw weights); CACHE: step 2's folded
GROUPS = [0 if is_full(l) else 1 + l % 6 for l in range(60)]
CAP = 2048
P = CAP // 128


def concat3(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).float()


def engine(mesh, kv, rope, scratch, cache=CACHE, max_users=1):
    cfg = EngineConfig(weight_cache=cache, rows=(1,), max_users=max_users, capacity=CAP, tap_layers=TAPS)
    e = ResidentEngine(mesh, cfg, KVPools(kv=kv, layer_group=GROUPS, scratch=scratch, rope=rope))
    e.warmup()
    return e


def identity_tables(pages, scratch, users=1):
    t = torch.zeros(users, 6, P, dtype=torch.int32)
    for g in range(6):
        for u in range(users):
            t[u, g, :] = scratch[g][u]
            n = min(P, pages[g] - users)  # leave the scratch pages
            t[u, g, :n] = torch.arange(n)
    return t


def step3(e, mesh, tok, p, mode="none"):
    e.write_rows([0], [p], [tok])
    e.set_mode(mode, True)
    e.launch(1)
    ttnn.synchronize_device(mesh)
    raw = concat3(mesh, e.logits_raw)[0, 0, 0]
    taps = ttnn.to_torch(ttnn.get_device_tensors(e.taps_buffer(e.launches - 1))[0]).float()[0, 0, 0]
    return int(e.read_tokens(1)[0]), raw, e.read_logits(1)[0], taps


def cmd_equiv(a):
    from models.demos.gemma4_31b_qb2.resident_s2.attention import AttnInputs
    from models.demos.gemma4_31b_qb2.resident_s2.model import ResidentGemma as Gemma2

    mesh = open_mesh()
    res = {}
    try:
        torch.manual_seed(0)
        attn = AttnInputs.synthetic(mesh, CAP, random=True)
        m2 = Gemma2(mesh, CACHE, attn, rows=1, dump_rounds=True, taps=TAPS)
        pages = {g: int(attn.kv[l][0].shape[0]) for l, g in enumerate(GROUPS)}
        scratch = [[pages[g] - 1] for g in range(6)]
        e = engine(mesh, attn.kv, attn.rope, scratch)
        e.write_tables(identity_tables(pages, scratch))
        for p in (100, 700, 1100):
            tok = 818 + p
            m2.reset()
            m2.set_rows([tok], p)
            m2.launch()
            ttnn.synchronize_device(mesh)
            t2, l2 = m2.tokens()[0], m2.logits()[0]
            taps2 = torch.cat([m2.taps()[l][0, 0] for l in TAPS])
            t3, raw3, cap3, taps3 = step3(e, mesh, tok, p)
            res[f"p{p}"] = {
                "token_s2": t2, "token_s3": t3, "logits_bitwise": bool(torch.equal(raw3.bfloat16(), l2.bfloat16())),
                "logits_max_abs": float((raw3 - l2).abs().max()),
                "cap_max_abs": float((cap3 - 30 * torch.tanh(raw3 / 30)).abs().max()),
                "taps_bitwise": bool(torch.equal(taps3.bfloat16(), taps2.bfloat16())),
                "taps_max_abs": float((taps3 - taps2).abs().max()),
            }
            print(json.dumps({f"p{p}": res[f"p{p}"]}), flush=True)
        # 16 device-fed greedy steps from p = 300
        p0, tok = 300, 1234
        m2.reset()
        m2.set_rows([tok], p0)
        seq2 = []
        for _ in range(16):
            m2.launch()
            ttnn.synchronize_device(mesh)
            seq2.append(m2.tokens()[0])
        e.write_rows([0], [p0], [tok])
        e.set_mode("argmax", False)
        seq3 = []
        for _ in range(16):
            e.launch(1)
            ttnn.synchronize_device(mesh)
            seq3.append(int(e.read_tokens(1)[0]))
        pos3 = int(ttnn.to_torch(ttnn.get_device_tensors(e.positions)[0]).reshape(-1)[0])
        res["feed16"] = {"s2": seq2, "s3": seq3, "equal": seq2 == seq3, "position_after": pos3,
                         "status": e.read_status().__dict__}
        print(json.dumps({"feed16": res["feed16"]}), flush=True)
        e.close()
    finally:
        close_mesh(mesh)
    ok = all(v["token_s2"] == v["token_s3"] and v["logits_bitwise"] and v["taps_bitwise"] and v["cap_max_abs"] < 0.5
             for k, v in res.items() if k.startswith("p")) and res["feed16"]["equal"] and \
        res["feed16"]["position_after"] == 316 and res["feed16"]["status"]["flags"] == 0
    res["ok"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    return ok


def tiles(x):
    a, b, r, c = x.shape
    return x.reshape(a, b, r // 32, 32, c // 32, 32).permute(0, 1, 2, 4, 3, 5).reshape(-1, 32, 32)


def untiles(t, shape):
    a, b, r, c = shape
    return t.reshape(a, b, r // 32, c // 32, 32, 32).permute(0, 1, 2, 4, 3, 5).reshape(a, b, r, c)


def host(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()


def up_bf8(mesh, x):
    return ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=mesh,
                           mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), memory_config=ttnn.DRAM_MEMORY_CONFIG)


def aliased_pools(mesh, cache_host, n_blocks, seed):
    """the synthetic per-layer caches' pages behind 10 aliased pools (pool i: layers 6i .. 6i + 5) at random disjoint
    per-group blocks: returns kv, tables [1, 6, P] (full entries 2b), scratch (block n_blocks per pool)"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_blocks, generator=g)
    blocks = {0: perm[:16]}
    for grp in range(1, 6):
        blocks[grp] = perm[16 + 10 * (grp - 1) : 16 + 10 * grp]
    kv = [None] * 60
    for i in range(10):
        pool = [torch.zeros(n_blocks + 1, 4, 128, 256) for _ in range(2)]
        for l in range(6 * i, 6 * i + 6):
            grp = GROUPS[l]
            for kvi in range(2):
                src = cache_host[l][kvi]
                if grp == 0:
                    for j in range(16):
                        b = int(blocks[0][j])
                        t = tiles(pool[kvi][b : b + 1])
                        t[:64] = tiles(src[j : j + 1])
                        pool[kvi][b : b + 1] = untiles(t, (1, 4, 128, 256))
                else:
                    for j in range(10):
                        pool[kvi][int(blocks[grp][j])] = src[j]
        sk, sv = up_bf8(mesh, pool[0]), up_bf8(mesh, pool[1])
        fk = ttnn.experimental.view(sk, [2 * (n_blocks + 1), 1, 128, 512])
        fv = ttnn.experimental.view(sv, [2 * (n_blocks + 1), 1, 128, 512])
        for l in range(6 * i, 6 * i + 6):
            kv[l] = (fk, fv) if GROUPS[l] == 0 else (sk, sv)
    scratch = [[2 * n_blocks]] + [[n_blocks] for _ in range(5)]
    t = torch.zeros(1, 6, P, dtype=torch.int32)
    for grp in range(6):
        t[0, grp, :] = scratch[grp][0]
    t[0, 0, :16] = 2 * blocks[0]
    for grp in range(1, 6):
        t[0, grp, :10] = blocks[grp]
    return kv, t, scratch


CASES = ((918, 100), (1518, 700), (1918, 1100))


def cmd_perm(a):
    from models.demos.gemma4_31b_qb2.resident_s2.attention import AttnInputs

    mesh = open_mesh()
    res = {}
    try:
        torch.manual_seed(0)
        attn = AttnInputs.synthetic(mesh, CAP, random=True)
        cache_host = [(host(k), host(v)) for k, v in attn.kv]
        kv2, tab2, scr2 = aliased_pools(mesh, cache_host, 80, seed=7)
        pages = {g: int(attn.kv[l][0].shape[0]) for l, g in enumerate(GROUPS)}
        scr1 = [[pages[g] - 1] for g in range(6)]
        e1 = engine(mesh, attn.kv, attn.rope, scr1)
        e1.write_tables(identity_tables(pages, scr1))
        out1 = [step3(e1, mesh, tok, p) for tok, p in CASES]
        e1.close()
        e2 = engine(mesh, kv2, attn.rope, scr2)
        e2.write_tables(tab2)
        out2 = [step3(e2, mesh, tok, p) for tok, p in CASES]
        for (tok, p), o1, o2 in zip(CASES, out1, out2):
            res[f"p{p}"] = {
                "token_identity": o1[0], "token_aliased": o2[0],
                "logits_bitwise": bool(torch.equal(o1[1].bfloat16(), o2[1].bfloat16())),
                "taps_bitwise": bool(torch.equal(o1[3].bfloat16(), o2[3].bfloat16())),
            }
        res["status"] = e2.read_status().__dict__
        e2.close()
        print(json.dumps(res), flush=True)
    finally:
        close_mesh(mesh)
    ok = all(v["token_identity"] == v["token_aliased"] and v["logits_bitwise"] and v["taps_bitwise"]
             for k, v in res.items() if k.startswith("p")) and res["status"]["flags"] == 0
    res["ok"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    return ok


def cmd_bounds(a):
    from models.demos.gemma4_31b_qb2.resident_s2.attention import AttnInputs

    mesh = open_mesh()
    res = {}
    try:
        torch.manual_seed(0)
        attn = AttnInputs.synthetic(mesh, CAP, random=True)
        cache_host = [(host(k), host(v)) for k, v in attn.kv]
        kv, tab, scr = aliased_pools(mesh, cache_host, 80, seed=7)
        pools = sorted({id(t): t for pair in kv for t in pair}.values(), key=lambda t: t.buffer_address())
        sliding = [t for t in pools if t.shape[1] == 4]
        e = engine(mesh, kv, attn.rope, scr)
        e.write_tables(tab)

        def snap():
            return [host(t)[:80] for t in sliding]  # every pool block but the scratch (block 80)

        def changed(before):
            return [i for i, (b, c) in enumerate(zip(before, snap())) if not torch.equal(b, c)]

        def raw_write(t, words):
            e._write(torch.tensor(words, dtype=torch.int32).reshape(1, 1, 1, -1), t)

        def run(mode="none"):
            e.set_mode(mode, False)
            e.launch(1)
            ttnn.synchronize_device(mesh)
            pos = int(host(e.positions).reshape(-1)[0])
            return int(e.read_tokens(1)[0]), pos, e.read_status().__dict__

        # host validation (V1 host side)
        for name, args in {"pos_cap": ([0], [CAP], [5]), "slot": ([1], [5], [5])}.items():
            try:
                e.write_rows(*args)
                res[f"host_rejects_{name}"] = False
            except EngineError as err:
                res[f"host_rejects_{name}"] = str(err)
        # (a) the last position decodes, no flag
        before = snap()
        e.write_rows([0], [CAP - 1], [77])
        tok, pos, st = run()
        res["last_position"] = {"flags": st["flags"], "pools_changed": changed(before)}
        # (b) position = capacity behind the host's back: flagged, id 0, nothing written
        e.clear_status()
        before = snap()
        raw_write(e.positions, [CAP] + [-1] * 31)
        raw_write(e.slots, [0] * 32)
        tok, pos, st = run("argmax")
        res["position_capacity"] = {"flags": st["flags"], "token": tok, "position": pos, "status": st,
                                    "pools_changed": changed(before)}
        # (c) slot = max_users
        e.clear_status()
        before = snap()
        raw_write(e.positions, [50] + [-1] * 31)
        raw_write(e.slots, [1] + [0] * 31)
        tok, pos, st = run("argmax")
        res["slot"] = {"flags": st["flags"], "token": tok, "position": pos, "pools_changed": changed(before)}
        # (d) a full-group entry past the pool (column 0, behind write_tables' validation)
        e.clear_status()
        bad = tab.clone()
        bad[0, 0, 0] = 999
        e._tables_host[:] = bad
        e._write(e._tables_host, e.tables)
        before = snap()
        raw_write(e.positions, [50] + [-1] * 31)
        raw_write(e.slots, [0] * 32)
        tok, pos, st = run("none")
        res["page"] = {"flags": st["flags"], "token": tok, "status": st, "pools_changed": changed(before)}
        e.write_tables(tab)
        # (e) the device feed runs into capacity: advances to capacity - 1, then flags and stops
        e.clear_status()
        e.write_rows([0], [CAP - 3], [77])
        trail = [run("argmax")[:2] for _ in range(5)]
        res["feed_into_capacity"] = {"trail": trail, "status": e.read_status().__dict__}
        e.close()
        print(json.dumps(res), flush=True)
    finally:
        close_mesh(mesh)
    ok = (all(isinstance(res[k], str) for k in res if k.startswith("host_rejects"))
          and res["last_position"]["flags"] == 0
          and res["position_capacity"]["flags"] == F_POSITION and res["position_capacity"]["token"] == 0
          and res["position_capacity"]["position"] == CAP and not res["position_capacity"]["pools_changed"]
          and res["slot"]["flags"] == F_SLOT and res["slot"]["token"] == 0 and not res["slot"]["pools_changed"]
          and res["page"]["flags"] == F_PAGE and res["page"]["token"] == 0 and res["page"]["status"]["entry"] == 999
          and res["page"]["status"]["page"] == 0
          and [p for _, p in res["feed_into_capacity"]["trail"]] == [CAP - 2, CAP - 1, CAP, CAP, CAP]
          and res["feed_into_capacity"]["status"]["flags"] == F_POSITION)
    res["ok"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    return ok


def cmd_warm(a):
    """build one engine on synthetic caches and run single eager launches, each followed by a synchronize, in the
    order given by --seq (r = a real row at p = --pos, i = an inert launch); prints after every step (hang hunting)"""
    from models.demos.gemma4_31b_qb2.resident_s2.attention import AttnInputs

    mesh = open_mesh()
    res = {"steps": []}
    try:
        torch.manual_seed(0)
        attn = AttnInputs.synthetic(mesh, CAP, random=True)
        pages = {g: int(attn.kv[l][0].shape[0]) for l, g in enumerate(GROUPS)}
        scratch = [[pages[g] - 1] for g in range(6)]
        cfg = EngineConfig(weight_cache=CACHE, rows=(1,), max_users=1, capacity=CAP,
                           tap_layers=TAPS if a.taps else (), logits=bool(a.logits))
        e = ResidentEngine(mesh, cfg, KVPools(kv=attn.kv, layer_group=GROUPS, scratch=scratch, rope=attn.rope))
        m = e.models[1]
        e.write_tables(identity_tables(pages, scratch))
        m.reset()
        for i, c in enumerate(a.seq):
            if c == "r":
                e.write_rows([0], [a.pos + i], [1234])
            else:
                e.write_rows([], [], [])
            e.set_mode("none", bool(a.logits))
            print(f"step {i} {c}: launch", flush=True)
            m.launch()
            ttnn.synchronize_device(mesh)
            st = {"kind": c, "token": int(e.read_tokens(1)[0]), "status": e.read_status().__dict__}
            print(f"step {i} {c}: done {st}", flush=True)
            res["steps"].append(st)
    finally:
        close_mesh(mesh)
    res["ok"] = True
    json.dump(res, open(a.out, "w"), indent=1)
    return True

def cmd_perf(a):
    """traced replays, device-fed (argmax) at each context, synthetic zero caches at capacity 32768, per-group
    identity tables; with and without the logits output and taps"""
    from models.demos.gemma4_31b_qb2.resident_s2.attention import AttnInputs

    cap = a.cap
    mesh = open_mesh()
    res = {"runs": []}
    try:
        attn = AttnInputs.synthetic(mesh, cap, random=False)
        pages = {g: int(attn.kv[l][0].shape[0]) for l, g in enumerate(GROUPS)}
        scratch = [[pages[g] - 1] for g in range(6)]
        for taps, logits in ((False, False),) if a.lean else ((False, False), (True, True)):
            cfg = EngineConfig(weight_cache=RAW if a.raw else CACHE, rows=tuple(sorted({1, a.rows})), max_users=1,
                               capacity=cap, tap_layers=TAPS if taps else (), logits=logits)
            e = ResidentEngine(mesh, cfg, KVPools(kv=attn.kv, layer_group=GROUPS, scratch=scratch, rope=attn.rope))
            e.warmup()
            t = torch.zeros(1, 6, cap // 128, dtype=torch.int32)
            for g in range(6):
                t[0, g, :] = scratch[g][0]
                n = min(cap // 128, pages[g] - 1)
                t[0, g, :n] = torch.arange(n)
            # sliding pools are 11 pages here: windows past them read the scratch page (timing only)
            e.write_tables(t)
            for R, ctx in [(R, int(c)) for R in sorted({1, a.rows}) if a.both or R == a.rows for c in a.ctxs.split(",")]:
                e.write_rows([0] * R, [ctx + r for r in range(R)], [1000 + r for r in range(R)])
                e.set_mode("argmax" if R == 1 else "none", logits)
                for _ in range(5):
                    e.launch(R)
                ttnn.synchronize_device(mesh)
                n = a.replays
                t0 = time.perf_counter()
                for _ in range(n):
                    e.launch(R)
                ttnn.synchronize_device(mesh)
                ms = 1e3 * (time.perf_counter() - t0) / n
                row = {"ctx": ctx, "rows": R, "taps": taps, "logits": logits, "ms": round(ms, 3),
                       "status": e.read_status().__dict__, "aiclk": read_aiclk()}
                print(json.dumps(row), flush=True)
                res["runs"].append(row)
            e.close()
            del e
    finally:
        close_mesh(mesh)
    res["ok"] = True
    json.dump(res, open(a.out, "w"), indent=1)
    return True


class Served:
    """the ttnn generator (prefill, the per-layer reference) on vLLM-style aliased pools with per-group tables, and a
    resident engine on the same pools (the serving layout of STEP3_PLAN_v3 / M1-0). Sliding groups own SLIDE_PAGES
    blocks each, mapped (set_ctx) to the last logical pages before a context, the older pages null (0), as vLLM's
    sliding-window groups keep only the window."""

    SLIDE_PAGES = 20

    def __init__(self, mesh, capacity, seed=11, tap_layers=tuple(range(60)), rows=(1,)):
        from models.demos.gemma4_31b_qb2.tt.generator import CacheState, build_generator

        t0 = time.time()
        self.mesh, self.capacity = mesh, capacity
        self.gen = build_generator(mesh, max_seq_len=capacity)
        print(f"ttnn generator built: {time.time() - t0:.0f} s", flush=True)
        layers = self.gen.model.layers
        self.pages = pages = capacity // 128
        n_blocks = pages + 5 * self.SLIDE_PAGES + 64
        g = torch.Generator().manual_seed(seed)
        perm = 1 + torch.randperm(n_blocks - 1, generator=g)  # block 0 stays the null block
        self.blocks = {0: perm[:pages]}
        for grp in range(1, 6):
            self.blocks[grp] = perm[pages + (grp - 1) * self.SLIDE_PAGES : pages + grp * self.SLIDE_PAGES]
        kv, scratch_pages, pool_pairs = [], {}, []
        for i in range(10):
            sl = layers[6 * i].allocate_cache(physical_pages=n_blocks + 1)
            full = tuple(ttnn.experimental.view(t, [2 * (n_blocks + 1), 1, 128, 512]) for t in sl)
            pool_pairs.append((sl, full))
        for l in range(60):
            grp = GROUPS[l]
            sl, full = pool_pairs[l // 6]
            kv.append(full if grp == 0 else sl)
            scratch_pages[l] = [2 * n_blocks] if grp == 0 else [n_blocks]
        self.n_blocks = n_blocks
        self.tables = {l: torch.zeros(1, pages, dtype=torch.int32) for l in range(60)}
        self.state = CacheState(kv, self.tables, 1, capacity, scratch_pages=scratch_pages, vllm_owned=True)
        shared = self.gen.model.shared_setup
        rope = {k: (shared[("rope", k)]["decode_cos"], shared[("rope", k)]["decode_sin"])
                for k in ("sliding_attention", "full_attention")}
        scratch = [[2 * n_blocks]] + [[n_blocks] for _ in range(5)]
        t0 = time.time()
        cfg = EngineConfig(weight_cache=RAW, rows=rows, max_users=1, capacity=capacity, tap_layers=tap_layers)
        self.e = ResidentEngine(mesh, cfg, KVPools(kv=kv, layer_group=GROUPS, scratch=scratch, rope=rope))
        self.e.warmup()
        self.set_ctx(0)
        self.rope = rope
        self.emb = self.e._owner.emb_host
        self.tok = self.gen.tokenizer
        print(f"resident engine built and warm: {time.time() - t0:.0f} s", flush=True)

    def set_ctx(self, ctx, rows=8):
        """tables for a prompt of ctx tokens plus `rows` decode rows: full all pages, sliding the last SLIDE_PAGES
        logical pages up to the last decode row (older ones null)"""
        last = (ctx + rows - 1) // 128
        first = max(0, last - self.SLIDE_PAGES + 1)
        for l in range(60):
            grp = GROUPS[l]
            t = torch.zeros(1, self.pages, dtype=torch.int32)
            if grp == 0:
                t[0] = 2 * self.blocks[0]
            else:
                t[0, first : last + 1] = self.blocks[grp][: last + 1 - first]
            self.tables[l].copy_(t)
        self.gen._tables(self.state, self.tables)
        et = torch.zeros(1, 6, self.pages, dtype=torch.int32)
        for grp in range(6):
            et[0, grp] = self.tables[next(l for l in range(60) if GROUPS[l] == grp)][0]
        self.e.write_tables(et)

    def prefill(self, ids):
        logits = self.gen.prefill_forward(
            torch.tensor([ids]), page_table=self.tables, kv_cache=self.state, prompt_lens=[len(ids)]
        )
        return logits[0, -1].float()

    def kv_row(self, l, pos):
        page = int(self.tables[l][0, pos // 128])
        out = []
        for t in self.state.kv[l]:
            sl = ttnn.slice(t, [page, 0, 0, 0], [page + 1, t.shape[1], 128, t.shape[3]])
            out.append(torch.stack([x[0, :, pos % 128].float() for x in
                                    [ttnn.to_torch(c) for c in ttnn.get_device_tensors(sl)]]))
            ttnn.deallocate(sl)
        return out

    def ttnn_layer(self, l, x, pos):
        layer = self.gen.model.layers[l]
        rep = ttnn.ReplicateTensorToMesh(self.mesh)
        xt = ttnn.from_torch(x.reshape(1, 1, 1, -1).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                             device=self.mesh, mesh_mapper=rep, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        xt = ttnn.to_memory_config(xt, layer._mem(5376, 28))
        positions = self.gen._upload(torch.tensor([pos], dtype=torch.int32))
        y = layer.decode_forward(xt, positions=positions, page_table=self.state.device_tables[l],
                                 kv_cache=self.state.kv[l], rope_positions=positions, cyclic_cache=False)
        return ttnn.to_torch(ttnn.get_device_tensors(y)[0]).float().reshape(-1, 5376)[0]

    def full_history(self, l, pos):
        """a full layer's (K, V) [4 chips' heads, pos, 512] bf16 at positions [0, pos) (the pools to the host)"""
        ps = torch.arange(pos)
        pages = self.tables[l][0, ps // 128].long()
        out = []
        for t in self.state.kv[l]:
            chips = [ttnn.to_torch(c) for c in ttnn.get_device_tensors(t)]
            out.append(torch.cat([c[pages, :, ps % 128].transpose(0, 1) for c in chips]).bfloat16())
            del chips
        return out


def cmd_tf(a):
    """teacher-forced, per layer, resident vs the ttnn decoder layer under vLLM-style aliased per-group tables, at
    each context (ttnn prefill of a filler prompt, then one resident step at position ctx)"""
    from models.demos.gemma4_31b_qb2.resident_s2.reference_s2 import pcc

    ctxs = [int(c) for c in a.ctxs.split(",")]
    cap = -(-(max(ctxs) + 512) // 2048) * 2048
    hf_layers = [int(v) for v in a.hf_layers.split(",")] if a.hf_layers else []
    assert all(is_full(l) for l in hf_layers), "the HF check takes full layers"
    hf_in = {}
    mesh = open_mesh()
    res = {"capacity": cap, "tf": {}, "env": {k: v for k, v in os.environ.items() if k.startswith("GRES_")}}
    try:
        h = Served(mesh, cap)
        e = h.e
        for ctx in ctxs:
            ids, nxt = doc_rows(a, h.tok, ctx)
            ctx = len(ids)
            t0 = time.time()
            h.set_ctx(ctx)
            logits = h.prefill(ids)
            tok = int(nxt[0]) if a.doc == "fillcode" else int(logits.argmax())  # (natural tail: the true token)
            e.write_rows([0], [ctx], [tok])
            e.set_mode("none", True)
            e.launch(1)
            ttnn.synchronize_device(mesh)
            nxt, st = int(e.read_tokens(1)[0]), e.read_status().__dict__
            buf = ttnn.to_torch(ttnn.get_device_tensors(e.taps_buffer(e.launches - 1))[0]).float()[0, 0, 0]
            taps = buf.reshape(60, 5376)
            x = h.emb[tok].float()
            rows = []
            for l in range(60):
                if l in hf_layers:  # (before this layer's writes at ctx: the history is [0, ctx))
                    kh, vh = h.full_history(l, ctx)
                kv_res = h.kv_row(l, ctx)
                y = h.ttnn_layer(l, x, ctx)
                kv_tt = h.kv_row(l, ctx)
                rows.append({"layer": l, "x_pcc": pcc(taps[l], y), "k_pcc": pcc(kv_res[0], kv_tt[0]),
                             "v_pcc": pcc(kv_res[1], kv_tt[1])})
                if l in hf_layers:
                    hf_in[(ctx, l)] = (x.clone(), taps[l].clone(), y.clone(), kh, vh)
                x = taps[l]
            worst = min(rows, key=lambda r: r["x_pcc"])
            xs = sorted(r["x_pcc"] for r in rows)
            res["tf"][ctx] = {"prefill_token": tok, "resident_next": nxt, "status": st, "layers": rows,
                              "min_x_pcc": worst["x_pcc"], "worst_layer": worst["layer"], "median_x_pcc": xs[30],
                              "min_k_pcc": min(r["k_pcc"] for r in rows), "min_v_pcc": min(r["v_pcc"] for r in rows),
                              "seconds": round(time.time() - t0)}
            print(f"tf ctx {ctx}: min x pcc {worst['x_pcc']:.5f} (layer {worst['layer']}), median {xs[30]:.5f}, "
                  f"min k {res['tf'][ctx]['min_k_pcc']:.5f}, v {res['tf'][ctx]['min_v_pcc']:.5f}, status {st}",
                  flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        e.close()
    finally:
        close_mesh(mesh)
    if hf_in:
        # resident and ttnn against the unquantized layer (bf16 checkpoint weights, fp32 math), same input x_l
        from models.demos.gemma4_31b_qb2.resident_s2.reference_s2 import HFLayer
        from models.demos.gemma4_31b_qb2.resident_s2.weights import Checkpoint
        from transformers import AutoConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

        ck = Checkpoint()
        cfg = AutoConfig.from_pretrained(ck.path, local_files_only=True).text_config
        for l in hf_layers:
            layer = HFLayer(ck, l)
            for (ctx, l_), (x, xr, y, kh, vh) in hf_in.items():
                if l_ != l:
                    continue
                cos, sin = (t[0, 0].bfloat16().float() for t in Gemma4TextRotaryEmbedding(cfg)(
                    torch.empty(1, dtype=torch.bfloat16), torch.tensor([[ctx]]), layer_type="full_attention"))
                parts = {}
                ref = layer(x, kh, vh, cos, sin, parts=parts)
                row = res["tf"][ctx]["layers"][l]
                row["hf_pcc_resident"], row["hf_pcc_ttnn"] = pcc(xr, ref), pcc(y, ref)
                print(f"hf ctx {ctx} layer {l}: resident {row['hf_pcc_resident']:.5f}, ttnn {row['hf_pcc_ttnn']:.5f}",
                      flush=True)
    ok = all(v["min_x_pcc"] >= 0.99 and v["status"]["flags"] == 0 for v in res["tf"].values())
    res["ok"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    return ok


def doc_rows(a, tok, ctx):
    """(context ids, the next 16 ids) for a test at ctx: the filler document, or (--doc fillcode) the filler-prefix +
    natural-tail sequence ids_fillcode_<ctx>.pt, scored 32 tokens into its code continuation"""
    if a.doc == "fillcode":
        d = torch.load(f"/home/ttuser/experiments/gemma4/bench/resident/m1l/ids_fillcode_{ctx}.pt")
        pos = d["prompt_len"] + 32
        return d["ids"][:pos], d["ids"][pos : pos + 16]
    return cached_context_ids(tok, ctx)


def cached_context_ids(tok, n):
    """context_ids, kept on disk (the long ones take minutes to build)"""
    from models.demos.gemma4_31b_qb2.resident_s2.run_step2 import context_ids

    path = f"/home/ttuser/experiments/gemma4/bench/resident/m1l/ids_{n}.pt"
    if os.path.exists(path):
        return torch.load(path)
    ids = context_ids(tok, n)
    torch.save(ids, path)
    return ids


def cmd_r8(a):
    """R = 8 at long context against R = 1 on the same aliased pools: after the ttnn prefill of ctx tokens, one R = 8
    launch at ctx .. ctx + 7 (the filler's next 8 tokens), then R = 1 launches at ctx + r with token r (the earlier
    rows' KV from those launches); per row and layer the PCC of the R = 8 taps against the R = 1 ones"""
    from models.demos.gemma4_31b_qb2.resident_s2.reference_s2 import pcc

    ctxs = [int(c) for c in a.ctxs.split(",")]
    cap = -(-(max(ctxs) + 512) // 2048) * 2048
    mesh = open_mesh()
    res = {"capacity": cap, "r8": {}}
    try:
        h = Served(mesh, cap, rows=(1, 8))
        e = h.e
        natural = None
        if a.doc == "code":  # natural text: the M1 code prompt + ttnn's own 256-token greedy continuation
            from models.demos.gemma4_31b_qb2.resident_s2.run_step2 import PROMPTS, chat_ids

            e2e = json.load(open("/home/ttuser/experiments/gemma4/bench/resident/m1_0/e2e.json"))["prompts"]["code"]
            natural = chat_ids(h.tok, PROMPTS["code"]) + e2e["ttnn"]
        for ctx in ctxs:
            ids, nxt = (natural[:ctx], natural[ctx : ctx + 16]) if natural else doc_rows(a, h.tok, ctx)
            ctx = len(ids)
            blk = [int(t) for t in nxt[:8]]
            t0 = time.time()
            h.set_ctx(ctx)
            h.prefill(ids)
            e.write_rows([0] * 8, [ctx + r for r in range(8)], blk)
            e.set_mode("none", True)
            e.launch(8)
            ttnn.synchronize_device(mesh)
            st8 = e.read_status().__dict__
            lg8 = e.read_logits(8)
            buf = ttnn.to_torch(ttnn.get_device_tensors(e.taps_buffer(e.launches - 1))[0]).float()[0, 0, :8]
            taps8 = buf.reshape(8, 60, 5376)
            rows, lrows, lg1s = [], [], []
            for r in range(8):
                e.write_rows([0], [ctx + r], [blk[r]])
                e.launch(1)
                ttnn.synchronize_device(mesh)
                t1 = ttnn.to_torch(ttnn.get_device_tensors(e.taps_buffer(e.launches - 1))[0]).float()[0, 0, 0]
                t1 = t1.reshape(60, 5376)
                rows.append([pcc(taps8[r, l], t1[l]) for l in range(60)])
                lg1 = e.read_logits(1)[0]
                lg1s.append(lg1.clone())
                v1, i1 = lg1.topk(2)
                gap = float(v1[0] - v1[1])
                lrows.append({"logits_pcc": round(pcc(lg8[r], lg1), 6), "argmax8": int(lg8[r].argmax()),
                              "argmax1": int(i1[0]), "gap1": round(gap, 4),
                              "ok": pcc(lg8[r], lg1) >= 0.988 and (gap < 0.25 or int(lg8[r].argmax()) == int(i1[0]))})
            st1 = e.read_status().__dict__
            torch.save({"ctx": ctx, "block": blk, "lg8": lg8[:8].clone(), "lg1": torch.stack(lg1s)},
                       a.out.replace(".json", f"_{ctx}.pt"))
            mins = [min(x) for x in rows]
            res["r8"][ctx] = {"status8": st8, "status1": st1, "min_pcc_per_row": mins, "min_pcc": min(mins),
                              "median_pcc": sorted(v for x in rows for v in x)[240], "pcc": rows, "logits": lrows,
                              "min_logits_pcc": min(x["logits_pcc"] for x in lrows),
                              "v8_ok": all(x["ok"] for x in lrows), "seconds": round(time.time() - t0)}
            print(f"r8 ctx {ctx}: min taps pcc {min(mins):.5f} per row {[round(m, 4) for m in mins]}, logits pcc "
                  f"{[x['logits_pcc'] for x in lrows]}, argmax equal {[x['argmax8'] == x['argmax1'] for x in lrows]}, "
                  f"gaps {[x['gap1'] for x in lrows]}, V8 {res['r8'][ctx]['v8_ok']}, status {st8} / {st1}", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        e.close()
    finally:
        close_mesh(mesh)
    ok = all(v["v8_ok"] and v["status8"]["flags"] == 0 and v["status1"]["flags"] == 0 for v in res["r8"].values())
    res["ok"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    return ok


def cmd_e2e(a):
    """greedy agreement: the ttnn B = 1 decode vs the resident engine (device-fed argmax) on the step-2 prompts, both
    on vLLM-style aliased per-group tables; capped top-2 gaps of both at the first divergence"""
    from models.demos.gemma4_31b_qb2.resident_s2.run_step2 import PROMPTS, chat_ids

    mesh = open_mesh()
    res = {"tokens": a.tokens, "prompts": {}}
    try:
        h = Served(mesh, 4096, tap_layers=())
        e = h.e
        for name, text in PROMPTS.items():
            ids = chat_ids(h.tok, text)
            n = len(ids)
            h.set_ctx(0, rows=n + a.tokens)  # short prompts: every sliding page real
            first = int(h.prefill(ids).argmax())
            # ttnn greedy (host logits, capped by the model)
            tt_seq, tt_top = [first], []
            for i in range(a.tokens - 1):
                lg = h.gen.decode_forward(torch.tensor([[tt_seq[-1]]]), torch.tensor([n + i]), page_table=h.tables,
                                          kv_cache=h.state, host_sampling=True)[0].float()
                v, idx = lg.topk(2)
                tt_top.append((int(idx[0]), int(idx[1]), float(v[0] - v[1])))
                tt_seq.append(int(idx[0]))
            # resident greedy from the same prefill (re-prefilled: the ttnn run wrote its own rows)
            assert int(h.prefill(ids).argmax()) == first
            e.write_rows([0], [n], [first])
            e.set_mode("argmax", True)
            rs_seq, rs_top = [first], []
            for i in range(a.tokens - 1):
                e.launch(1)
                ttnn.synchronize_device(mesh)
                lg = e.read_logits(1)[0]
                v, idx = lg.topk(2)
                rs_top.append((int(idx[0]), int(idx[1]), float(v[0] - v[1])))
                rs_seq.append(int(e.read_tokens(1)[0]))
            div = next((i for i in range(a.tokens) if tt_seq[i] != rs_seq[i]), None)
            row = {"prompt_len": n, "first_divergence": div, "equal_tokens": div is None,
                   "status": e.read_status().__dict__, "ttnn": tt_seq, "resident": rs_seq}
            if div is not None:
                row["ttnn_gap_at_div"] = tt_top[div - 1][2]
                row["resident_gap_at_div"] = rs_top[div - 1][2]
                row["ttnn_top2_at_div"] = tt_top[div - 1][:2]
                row["resident_top2_at_div"] = rs_top[div - 1][:2]
            res["prompts"][name] = row
            print(name, {k: v for k, v in row.items() if k not in ("ttnn", "resident")}, flush=True)
            json.dump(res, open(a.out, "w"), indent=1)
        e.close()
    finally:
        close_mesh(mesh)
    ok = all(r["equal_tokens"] or min(r["ttnn_gap_at_div"], r["resident_gap_at_div"]) <= 0.5
             for r in res["prompts"].values())
    res["ok"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    return ok


def cmd_l1(a):
    """cb_top() against the allocator's view, and rebuild(l1_top) keeping outputs (the integrator's L1 guard)"""
    from models.demos.gemma4_31b_qb2.resident_s2.attention import AttnInputs

    mesh = open_mesh()
    res = {}
    try:
        torch.manual_seed(0)
        attn = AttnInputs.synthetic(mesh, CAP, random=True)
        pages = {g: int(attn.kv[l][0].shape[0]) for l, g in enumerate(GROUPS)}
        scratch = [[pages[g] - 1] for g in range(6)]
        e = engine(mesh, attn.kv, attn.rope, scratch, cache=RAW)
        e.write_tables(identity_tables(pages, scratch))
        view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
        res["cb_top"] = e.cb_top()
        res["l1_base"] = ttnn.get_allocator_base_address(mesh, ttnn.BufferType.L1)
        res["l1_view"] = {"banks": view.num_banks, "per_bank": view.total_bytes_per_bank,
                          "allocated_per_bank": view.total_bytes_allocated_per_bank,
                          "blocks": str(view.block_table)[:400]}
        t1 = step3(e, mesh, 1234, 500)
        e.rebuild(1_250_000)
        res["cb_top_after"] = e.cb_top()
        e.write_tables(identity_tables(pages, scratch))
        t2 = step3(e, mesh, 1234, 500)
        res["same_after_rebuild"] = t1[0] == t2[0] and bool(torch.equal(t1[1].bfloat16(), t2[1].bfloat16()))
        res["tokens"] = (t1[0], t2[0])
        e.close()
        print(json.dumps(res), flush=True)
    finally:
        close_mesh(mesh)
    ok = res["same_after_rebuild"] and res["cb_top_after"] <= 1_250_000 + 64 * 1024 and res["cb_top"] > res["l1_base"]
    res["ok"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    return ok


def cmd_equiv8(a):
    """R = 8 (one slot, 8 consecutive rows): the frozen step-2 engine vs this one, ids / raw logits / taps rows"""
    from models.demos.gemma4_31b_qb2.resident_s2.attention import AttnInputs
    from models.demos.gemma4_31b_qb2.resident_s2.model import ResidentGemma as Gemma2

    mesh = open_mesh()
    res = {}
    try:
        torch.manual_seed(0)
        attn = AttnInputs.synthetic(mesh, CAP, random=True)
        m2 = Gemma2(mesh, CACHE, attn, rows=8, dump_rounds=True, taps=TAPS)
        pages = {g: int(attn.kv[l][0].shape[0]) for l, g in enumerate(GROUPS)}
        scratch = [[pages[g] - 1] for g in range(6)]
        cfg = EngineConfig(weight_cache=CACHE, rows=(1, 8), max_users=1, capacity=CAP, tap_layers=TAPS)
        e = ResidentEngine(mesh, cfg, KVPools(kv=attn.kv, layer_group=GROUPS, scratch=scratch, rope=attn.rope))
        e.warmup()
        e.write_tables(identity_tables(pages, scratch))
        for p in (300, 1000):
            toks = [900 + 37 * r for r in range(8)]
            m2.reset()
            m2.set_rows(toks, p)
            m2.launch()
            ttnn.synchronize_device(mesh)
            t2, l2 = m2.tokens(), m2.logits()
            taps2 = torch.stack([torch.cat([m2.taps()[l][0, r] for l in TAPS]) for r in range(8)])
            e.write_rows([0] * 8, [p + r for r in range(8)], toks)
            e.set_mode("none", True)
            e.launch(8)
            ttnn.synchronize_device(mesh)
            t3 = e.read_tokens(8).tolist()
            raw3 = concat3(mesh, e.logits_raw)[0, 0, :8]
            taps3 = ttnn.to_torch(ttnn.get_device_tensors(e.taps_buffer(e.launches - 1))[0]).float()[0, 0, :8]
            res[f"p{p}"] = {"tokens_s2": t2, "tokens_s3": t3,
                            "logits_bitwise": bool(torch.equal(raw3.bfloat16(), l2.bfloat16())),
                            "logits_max_abs": float((raw3 - l2).abs().max()),
                            "taps_bitwise": bool(torch.equal(taps3.bfloat16(), taps2.bfloat16())),
                            "taps_max_abs": float((taps3 - taps2).abs().max()), "status": e.read_status().__dict__}
            print(json.dumps({f"p{p}": res[f"p{p}"]}), flush=True)
        e.close()
    finally:
        close_mesh(mesh)
    ok = all(v["tokens_s2"] == v["tokens_s3"] and v["logits_bitwise"] and v["taps_bitwise"] for v in res.values())
    res["ok"] = ok
    json.dump(res, open(a.out, "w"), indent=1)
    return ok


def cmd_layers(a):
    """per-layer taps of the frozen step-2 engine vs this one at --pos (first diverging layer)"""
    from models.demos.gemma4_31b_qb2.resident_s2.attention import AttnInputs
    from models.demos.gemma4_31b_qb2.resident_s2.model import ResidentGemma as Gemma2

    mesh = open_mesh()
    res = {}
    try:
        torch.manual_seed(0)
        attn = AttnInputs.synthetic(mesh, CAP, random=True)
        allt = tuple(range(60))
        m2 = Gemma2(mesh, CACHE, attn, rows=1, taps=allt)
        pages = {g: int(attn.kv[l][0].shape[0]) for l, g in enumerate(GROUPS)}
        scratch = [[pages[g] - 1] for g in range(6)]
        cfg = EngineConfig(weight_cache=CACHE, rows=(1,), max_users=1, capacity=CAP, tap_layers=allt)
        e = ResidentEngine(mesh, cfg, KVPools(kv=attn.kv, layer_group=GROUPS, scratch=scratch, rope=attn.rope))
        e.warmup()
        e.write_tables(identity_tables(pages, scratch))
        m2.reset()
        m2.set_rows([818 + a.pos], a.pos)
        m2.launch()
        ttnn.synchronize_device(mesh)
        t2 = m2.taps()
        _, _, _, t3 = step3(e, mesh, 818 + a.pos, a.pos)
        t3 = t3.reshape(60, -1)
        res["diff"] = [float((t3[l] - t2[l][0, 0]).abs().max()) for l in range(60)]
        res["first"] = next((l for l in range(60) if res["diff"][l] > 0), None)
        print(json.dumps(res), flush=True)
        e.close()
    finally:
        close_mesh(mesh)
    res["ok"] = res["first"] is None
    json.dump(res, open(a.out, "w"), indent=1)
    return res["ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["equiv", "perm", "bounds", "warm", "perf", "tf", "e2e", "l1", "equiv8", "layers", "r8"])
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--ctxs", default="128,1100,1500,8000")
    ap.add_argument("--replays", type=int, default=100)
    ap.add_argument("--cap", type=int, default=32768)
    ap.add_argument("--lean", type=int, default=0)
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--both", type=int, default=0)  # perf: also time R = 1 when --rows > 1
    ap.add_argument("--raw", type=int, default=0)  # perf: raw weights (the step-3 default) instead of step 2's folded
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", default="ri")
    ap.add_argument("--pos", type=int, default=300)
    ap.add_argument("--logits", type=int, default=0)
    ap.add_argument("--taps", type=int, default=0)
    ap.add_argument("--doc", default="filler")  # r8: filler (context_ids) or code (natural text, ctx <= 300)
    ap.add_argument("--hf_layers", default="")  # tf: full layers also checked against the unquantized HF layer
    a = ap.parse_args()
    t0 = time.time()
    ok = {"equiv": cmd_equiv, "perm": cmd_perm, "bounds": cmd_bounds, "warm": cmd_warm, "perf": cmd_perf,
          "tf": cmd_tf, "e2e": cmd_e2e, "l1": cmd_l1, "equiv8": cmd_equiv8, "layers": cmd_layers,
          "r8": cmd_r8}[a.cmd](a)
    print("done", a.cmd, "ok" if ok else "FAIL", round(time.time() - t0), "s", flush=True)
    sys.exit(0)  # (a failed check is in the JSON; non-zero exits make guarded.sh reset the chips)


if __name__ == "__main__":
    main()
