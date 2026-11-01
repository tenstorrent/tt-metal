# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gate of PPPrefill.run_batch on the 1 x 4 mesh: prompts prefilled in one pipelined pass into rows-mode slots must
equal the same prompts prefilled one at a time (reset + run + handoff, the rows engine's path), bit for bit.

Built as the served rows profile (PP prefill, rows model of 8 slots on a paged pool, warm-ups, traces captured last).
Per case (a list of prompt lengths at one PP chunk): every prompt is prefilled into its own slot one at a time on
block tables A, then `steps` teacher-forced steps run all slots together (logits per slot and step); the slots are
cleared (model.clear_slot) and the same prompts go through run_batch on fresh block tables B, then the same steps.
The logits must be identical. Also the wall time of both prefills.

--save F: also save the sequential logits of every case (tensor per case); --ref F: compare the sequential logits
with F's (a run of this script on another tree, e.g. --sequential-only on the tree before a handoff change).

usage: python -m qwen36_resident.tests.pp_batch_gate [--cases "1024:1000,2048,1500,3000,700,2500,4100,1024;1024:2049*8"]
           [--steps 4] [--sequential-only] [--save F] [--ref F]
"""
import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State, is_attn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

B = 8
BLOCK = 64
CORPUS = "/home/ttuser/experiments/qwen36_27b/profiles/resident/256k/corpus_ids.pt"
CASES = "1024:1000,2048,1500,3000,700,2500,4100,1024;512:600,1500,513,2000,64,1100,900,2;1024:900,1000,1024,300,64,1000,800,500"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=CASES, help="';'-separated chunk:len,len,...")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=16384)
    ap.add_argument("--chunks", default="128,256,512,1024")
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--sequential-only", action="store_true")
    ap.add_argument("--save", default="")
    ap.add_argument("--ref", default="")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(1800, repeat=True, file=sys.stderr)
    corpus = torch.load(a.corpus)["ids"]
    cases = []
    for spec in a.cases.split(";"):
        C, lens = spec.split(":")
        # (len*k: k prompts of len tokens)
        lens = [int(x.split("*")[0]) for x in lens.split(",") for _ in range(int(x.split("*")[1]) if "*" in x else 1)]
        cases.append((int(C), lens))
    assert all(len(lens) <= B for _, lens in cases)
    mesh = open_mesh()
    ok, saved = True, []
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=vocab)
        emb = Q.embedding(ck)
        chunks = [int(c) for c in a.chunks.split(",")]
        max_len = -(-a.max_len // max(chunks)) * max(chunks)
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        empty = lambda: State(d, layers - n_attn, n_attn, 0, max_pos=max_len + 64, zero=True, kv=False)
        pp = PPPrefill(mesh, ck, layers, interval, chunk=chunks, max_len=max_len)
        log("prefill model")
        need = max(sum(-(-(T + a.steps + BLOCK) // BLOCK) for T in lens) for _, lens in cases)
        nb = 2 * need + 64
        m = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            [empty() for _ in range(B)],
            layers,
            interval,
            lm_head=True,
            embed=pp.embed,
            kv_pool_blocks=nb,
            block_tables=[[0]] * B,
            rows=True,
        )
        pp.handoff(m, 1, block_table=[0], slot=0)
        m.clear_slot(0)
        m.capture_trace()
        pp.capture()
        log(f"rows model (pool {nb} blocks), traces")
        order = m.row_order()

        if not a.sequential_only:
            # _zero_chips on a chip subset (twice: the second call hits the program cache with other chips' ranges)
            for chips in ([1, 3], [0, 2]):
                probe = [
                    ttnn.from_torch(
                        torch.ones(pp.n, 1, 32, 1024),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                    )
                    for _ in range(2)
                ]
                pp._zero_chips(chips, probe)
                got = [ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)) for t in probe]
                want = torch.ones(pp.n, 1, 32, 1024)
                want[chips] = 0
                good = all(bool((g == want).all()) for g in got)
                ok &= good
                log(f"_zero_chips {chips}: {'ok' if good else 'WRONG'}")
                for t in probe:
                    ttnn.deallocate(t)

        def slot_states(users, key):
            """per user: its slot's state as stored (GDN states, conv histories, K / V of the prompt's positions)"""
            m.set_page_table(tables(key))
            out = []
            pools, pool_rows = {}, m._pool_rows
            m._pool_rows = lambda t: pools[id(t)] if id(t) in pools else pools.setdefault(id(t), pool_rows(t))
            try:  # (read_slot reads the whole KV pool: once per call here, not per user)
                for u in users:
                    st = m.read_slot(u["slot"])
                    n = u["T"] - 1
                    for kv in ("K", "V"):  # [n, copies, pages, TILE, HD] in position order: the prefilled rows only
                        x = st[kv]
                        st[kv] = x.reshape(*x.shape[:2], -1, x.shape[-1])[:, :, :n]
                    out.append(st)
            finally:
                del m._pool_rows
            return out

        def rnd(x, bits, rne):
            """fp32 with the low `bits` mantissa bits dropped (13: TF32, 16: bf16) by truncation or round-to-nearest-even"""
            b = x.contiguous().view(torch.int32)
            if rne:
                b = b + ((1 << (bits - 1)) - 1) + ((b >> bits) & 1)
            return (b & ~((1 << bits) - 1)).view(torch.float32)

        def state_diff(got, ref):
            """GDN states batch vs sequential: the copies that differ (per prefill stage of 12) and whether the
            sequential ones are TF32(batch) by truncation / round-to-nearest"""
            g, r = got["state"], ref["state"]
            bad = (g != r).flatten(2).any(-1).any(0)  # per GDN copy
            idx = bad.nonzero().flatten().tolist()
            if not idx:
                return "states identical"
            gi, ri = g[:, idx], r[:, idx]
            frac = float((gi != ri).double().mean())
            # per candidate f: the share of these values where seq == f(batch); and the largest seq - batch in ulps of
            # TF32 at batch's magnitude
            match = {
                f"{name}_{'rne' if rne else 'trunc'}": float((rnd(gi, bits, rne) == ri).double().mean())
                for name, bits in (("tf32", 13), ("bf16", 16))
                for rne in (False, True)
            }
            ulp = torch.pow(2.0, torch.floor(torch.log2(gi.abs().clamp(min=1e-30))) - 10)
            d_ulps = float(((ri - gi).abs() / ulp).max())
            return (
                f"{len(idx)} copies differ (by stage of 12: {[sum(1 for c in idx if c // 12 == s) for s in range(4)]}, "
                f"{100 * frac:.2f}% of their values); seq == f(batch) share {({k: round(v, 5) for k, v in match.items()})}; "
                f"max |seq - batch| {d_ulps:.2f} TF32 ulps; seq tf32-exact {bool(torch.equal(rnd(ri, 13, False), ri))}"
            )

        def teacher(users):
            """logits [users, steps, vocab]: every slot fed its prompt's tokens from its last prompt position"""
            out = [[] for _ in users]
            for k in range(a.steps):
                x0, pos, active = torch.zeros(B, 5120), [0] * B, [False] * B
                for u in users:
                    p = u["T"] - 1 + k
                    x0[u["slot"]], pos[u["slot"]], active[u["slot"]] = emb[int(u["ids"][p])].float(), p, True
                m.step(m.token_state(x0, pos, active=active))
                ttnn.synchronize_device(mesh)
                lg = m.logits()[:, :, : d.vocab_chip].reshape(B, -1)[:, :vocab].float()
                for i, u in enumerate(users):
                    out[i].append(lg[u["slot"]].clone())
            return torch.stack([torch.stack(o) for o in out])

        users = []
        tables = lambda key: [next((u[key] for u in users if u["slot"] == s), [0]) for s in range(B)]
        for ci, (C, lens) in enumerate(cases):
            users = []
            off, blk = 0, 1
            for i, T in enumerate(lens):
                u = dict(T=T, slot=order[i], ids=corpus[off : off + T + a.steps])
                off += T + a.steps + 777
                k = -(-(T + a.steps + BLOCK) // BLOCK)
                u["A"], u["B"] = list(range(blk, blk + k)), list(range(blk + need, blk + need + k))
                blk += k
                users.append(u)
            for u in users:
                m.clear_slot(u["slot"])
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for u in users:
                if u["T"] > 1:
                    pp.reset()
                    pp.run(u["ids"][: u["T"] - 1], chunk=C)
                    pp.handoff(m, u["T"] - 1, block_table=u["A"], slot=u["slot"])
            ttnn.synchronize_device(mesh)
            t_seq = time.perf_counter() - t0
            ref_states = None if a.sequential_only else slot_states(users, "A")
            m.set_page_table(tables("A"))
            ref = teacher(users)
            saved.append(ref)
            line = f"case {ci} C={C} lens {lens}: sequential {1e3 * t_seq:.1f} ms"
            if not a.sequential_only:
                for u in users:
                    m.clear_slot(u["slot"])
                ttnn.synchronize_device(mesh)
                t0 = time.perf_counter()
                parts = {}
                pp.run_batch(
                    [(u["ids"][: u["T"] - 1], u["B"], u["slot"]) for u in users if u["T"] > 1],
                    m,
                    chunk=C,
                    timings=parts,
                )
                t_batch = time.perf_counter() - t0
                got_states = slot_states(users, "B")
                parts_d = [
                    {k: float((x[k] - y[k]).abs().max()) if x[k].numel() else 0.0 for k in x}
                    for x, y in zip(got_states, ref_states)
                ]
                diag = [state_diff(x, y) for x, y in zip(got_states, ref_states)]
                m.set_page_table(tables("B"))
                got = teacher(users)
                diff = (got - ref).abs().amax(dim=(1, 2))
                same = bool((diff == 0).all())
                ok &= same
                toks = sum(lens) - len(lens)
                line += (
                    f", run_batch {1e3 * t_batch:.1f} ms (ticks {1e3 * parts['ticks']:.1f}, histories "
                    f"{1e3 * parts['histories']:.1f}) = {t_seq / t_batch:.2f}x, {toks / t_batch:.0f} vs "
                    f"{toks / t_seq:.0f} prefill tok/s; logits {'IDENTICAL' if same else 'DIFFER'} "
                    f"(max |d| per prompt {[round(float(x), 4) for x in diff]})"
                )
                if not same or any(v for d in parts_d for v in d.values()):
                    line += f"; state max |d| per prompt {[{k: round(v, 4) for k, v in d.items()} for d in parts_d]}"
                    line += "; GDN: " + " | ".join(f"{i}: {x}" for i, x in enumerate(diag))
            log(line)
        if a.ref:
            refs = torch.load(a.ref)
            for ci, (r0, r1) in enumerate(zip(refs, saved)):
                dd = float((r0 - r1).abs().max())
                ok &= dd == 0
                log(f"case {ci}: sequential logits vs {a.ref}: max |d| {dd:.4g} {'IDENTICAL' if dd == 0 else 'DIFFER'}")
        if a.save:
            torch.save(saved, a.save)
        m.release_trace()
    finally:
        close_mesh(mesh)
    log(f"pp_batch_gate: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
