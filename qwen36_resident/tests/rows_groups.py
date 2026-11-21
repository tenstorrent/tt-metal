# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Row groups (ResidentModel groups=, kernels/active_rows.hpp): 16 users as one launch of 2 groups x 8 rows give the
same results as the same 16 users as two launches of 8 rows (random weights, lm_head with a small vocab, a random
embedding table on device, paged KV with shuffled block tables, 1 x 4 mesh).

Run once per build, each in its own process (one model on the chips at a time), then compare:
  python -m qwen36_resident.tests.rows_groups --groups 1 --out g1.pt
  python -m qwen36_resident.tests.rows_groups --groups 2 --out g2.pt --ref g1.pt [--json out.json]
Recorded per run: the host-driven greedy steps' logits (every active user, every step) and tokens, the fed greedy run's
tokens and positions, a fed run with per-row sampling, the step time, and slots 0 / 9's state after the host-driven
steps (GDN states, conv histories, their KV pages). The gate: everything bit-identical (the row groups run each
group's arithmetic exactly as its own launch would: the same weights, the same all-reduce order).
"""

import argparse
import faulthandler
import json
import sys
import time

import torch

import ttnn
from qwen36_resident.model import HIDDEN, Dims, ResidentModel, State, is_attn, random_weights
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh, shuffled_tables

VOCAB = 4096
# per user count (16: G=2 vs two launches, 32: G=4 vs four): rows with nobody (16: one in group 0, three in group 1;
# 32: and group 3 down to its last row), slots whose state is read back, who runs step by step ("gIJ": groups I, J)
SETUP = {
    16: dict(
        empty=(3, 11, 13, 14), probe=(0, 9), patterns=["all", "g0", "g1", "all", "g1", "g1", "g0", "all", "g0", "g0"]
    ),
    32: dict(
        empty=(3, 11, 13, 14, 19, 22, 24, 25, 26, 27, 28, 29, 30),
        probe=(0, 9, 31),
        patterns=["all", "g0", "g3", "g12", "all", "g2", "g13", "g0", "all", "g01"],
    ),
}
USERS = 16
EMPTY, PROBE, PATTERNS = SETUP[16]["empty"], SETUP[16]["probe"], SETUP[16]["patterns"]


def run(a):
    mesh = open_mesh()
    rec = {}
    try:
        n = mesh.get_num_devices()
        d = Dims(n, mesh.dram_grid_size().x, vocab=VOCAB)
        interval = 4
        n_attn = sum(is_attn(l, interval) for l in range(a.layers))
        n_gdn = a.layers - n_attn
        w = random_weights(d, n_gdn, n_attn, a.layers)
        g = torch.Generator().manual_seed(3)
        emb_host = (torch.randn(VOCAB, HIDDEN, generator=g) * 0.5).bfloat16()
        embed = ttnn.from_torch(
            emb_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        # positions spread over several tiles and banks, some on a tile's last column
        sts = [
            State(d, n_gdn, n_attn, 50 + 37 * u + (31 if u % 5 == 0 else 0), a.max_pos, seed=30 + u)
            for u in range(USERS)
        ]
        blocks, tables = shuffled_tables(USERS, a.max_pos, seed=4)
        m = ResidentModel(
            mesh,
            d,
            w,
            sts,
            a.layers,
            interval,
            lm_head=True,
            embed=embed,
            kv_pool_blocks=blocks,
            block_tables=tables,
            rows=True,
            groups=a.groups,
        )
        m.capture_trace()
        active = [u not in EMPTY for u in range(USERS)]
        first = [int(t) for t in torch.randint(0, VOCAB, (USERS,), generator=g)]
        print(f"rows groups model: groups={a.groups}, {m.n_sub} launch(es), {sum(active)}/{USERS} active", flush=True)

        def restart():
            for s in range(USERS):
                m.write_slot(s, sts[s])
            m.reset(state=False)

        def tok_host(tokens, pos):
            x0 = torch.stack([emb_host[t].float() for t in tokens])
            return m.token_state(x0, pos, active=active)

        def logits_rows():
            lg = m.logits()
            return lg[:, :, : d.vocab_chip].reshape(USERS, -1)[:, :VOCAB].float()

        start = [st.pos for st in sts]
        restart()
        m.set_sampling(0.0, 1, 1.0, 0)
        toks, pos, lgs, ref = list(first), list(start), [], []
        for _ in range(a.steps):
            m.step(tok_host(toks, pos))
            ttnn.synchronize_device(mesh)
            lg = logits_rows()
            lgs.append(lg[torch.tensor(active)].bfloat16())
            toks = [int(lg[u].argmax()) if active[u] else 0 for u in range(USERS)]
            pos = [p + 1 if active[u] else p for u, p in enumerate(pos)]
            ref.append(list(toks))
        rec["host_logits"] = torch.stack(lgs)
        rec["host_tokens"] = ref
        rec["x"] = m.x().bfloat16()
        rec["slots"] = {s: {k: v.clone() for k, v in m.read_slot(s).items()} for s in PROBE}

        def fed_run():
            restart()
            m.step(tok_host(first, start))
            got = []
            for k in range(a.steps):
                if k:
                    m.step(None)
                ttnn.synchronize_device(mesh)
                got.append(m.fed_token())
            return got

        rec["fed_greedy"] = fed_run()
        fed = [[t if act else 0 for t, _, act in step] for step in rec["fed_greedy"]]
        rec["fed_matches_host"] = sum(f == r for f, r in zip(fed, ref))
        m.set_sampling(rows=[(0.8, 20, 0.9, 1000 + u) for u in range(USERS)])
        rec["fed_sampled"] = fed_run()

        # the active groups change from step to step (row groups run only the groups with an active row): per step
        # who runs (all, group 0's users, group 1's users), preserve set (the others hold live state); host-driven
        # greedy. A G=1 build runs only the launches with an active row.
        m.set_sampling(0.0, 1, 1.0, 0)
        restart()
        toks, pos, plg = list(first), list(start), []

        def runs(pat):
            return [active[u] and (pat == "all" or str(u // 8) in pat[1:]) for u in range(USERS)]

        def launches_of(run):
            return None if m.n_sub == 1 else [k for k in range(m.n_sub) if any(run[k * 8 : (k + 1) * 8])]

        for pat in PATTERNS:
            run = runs(pat)
            x0 = torch.stack([emb_host[t].float() for t in toks])
            m.step(m.token_state(x0, pos, active=run, preserve=True), launches=launches_of(run))
            ttnn.synchronize_device(mesh)
            lg = logits_rows()
            plg.append(lg[torch.tensor(run)].bfloat16())
            toks = [int(lg[u].argmax()) if run[u] else toks[u] for u in range(USERS)]
            pos = [p + 1 if run[u] else p for u, p in enumerate(pos)]
        rec["pattern_logits"] = plg
        # step time with group 0's users only (group 1 empty), fed back to back
        restart()
        run = runs("g0")
        x0 = torch.stack([emb_host[t].float() for t in first])
        m.step(m.token_state(x0, start, active=run, preserve=True), launches=launches_of(run))
        for _ in range(4):
            m.step(None, launches=launches_of(run))
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(a.time_steps):
            m.step(None, launches=launches_of(run))
        ttnn.synchronize_device(mesh)
        rec["step_ms_g0"] = (time.time() - t0) / a.time_steps * 1e3
        # step time, fed back to back (all users active but the empty rows)
        restart()
        m.step(tok_host(first, start))
        for _ in range(4):
            m.step(None)
        ttnn.synchronize_device(mesh)
        t0 = time.time()
        for _ in range(a.time_steps):
            m.step(None)
        ttnn.synchronize_device(mesh)
        rec["step_ms"] = (time.time() - t0) / a.time_steps * 1e3
        print(
            f"groups={a.groups}: fed greedy matches host-driven {rec['fed_matches_host']}/{a.steps}; "
            f"step {rec['step_ms']:.3f} ms, group 0 only {rec['step_ms_g0']:.3f} ms",
            flush=True,
        )
    finally:
        close_mesh(mesh)
    return rec


def compare(got, ref):
    res = {}
    res["host_logits_identical"] = bool(torch.equal(got["host_logits"], ref["host_logits"]))
    res["host_logits_max_diff"] = float((got["host_logits"].float() - ref["host_logits"].float()).abs().max())
    res["host_tokens_identical"] = got["host_tokens"] == ref["host_tokens"]
    res["x_identical"] = bool(torch.equal(got["x"], ref["x"]))
    res["fed_greedy_identical"] = got["fed_greedy"] == ref["fed_greedy"]
    res["fed_sampled_identical"] = got["fed_sampled"] == ref["fed_sampled"]
    res["fed_matches_host"] = [got["fed_matches_host"], ref["fed_matches_host"]]
    slots = {}
    for s in ref["slots"]:
        slots[s] = {k: bool(torch.equal(got["slots"][s][k], ref["slots"][s][k])) for k in ref["slots"][s]}
    res["slots_identical"] = slots
    res["pattern_logits_identical"] = all(
        torch.equal(x, y) for x, y in zip(got["pattern_logits"], ref["pattern_logits"])
    ) and len(got["pattern_logits"]) == len(ref["pattern_logits"])
    res["step_ms"] = dict(got=got["step_ms"], ref=ref["step_ms"])
    res["step_ms_group0_only"] = dict(got=got["step_ms_g0"], ref=ref["step_ms_g0"])
    res["pass"] = (
        res["pattern_logits_identical"]
        and res["host_logits_identical"]
        and res["host_tokens_identical"]
        and res["x_identical"]
        and res["fed_greedy_identical"]
        and res["fed_sampled_identical"]
        and all(all(v.values()) for v in slots.values())
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", type=int, required=True)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--time-steps", type=int, default=32)
    ap.add_argument("--max-pos", type=int, default=1024)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--json")
    ap.add_argument("--users", type=int, default=16, choices=sorted(SETUP))
    a = ap.parse_args()
    global USERS, EMPTY, PROBE, PATTERNS
    USERS = a.users
    EMPTY, PROBE, PATTERNS = SETUP[USERS]["empty"], SETUP[USERS]["probe"], SETUP[USERS]["patterns"]
    if USERS > 16:
        a.max_pos = max(a.max_pos, 2048)  # (the users' start positions reach past 1024)
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    rec = run(a)
    torch.save(rec, a.out)
    if a.ref:
        res = compare(rec, torch.load(a.ref))
        print(json.dumps(res, indent=1), flush=True)
        print(f"rows groups gate: {'PASS' if res['pass'] else 'FAIL'}", flush=True)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
