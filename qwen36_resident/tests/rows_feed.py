# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user rows mode with the hub feeding every row (WP7), random weights on a 1 x 4 mesh.

A rows-mode model with the lm_head (random, small vocab) and a random embedding table on device, 2 launches of 8 rows,
13 active users (3 rows of launch 1 empty):
  greedy:   the host-driven run (host argmax of each row's logits, host-written token states) and the fed run (first
            token state from the host, then step(None): every hub writes its launch's next token states) give the
            same tokens for every user at every step; inactive rows report inactive and keep their position;
  back to back: the fed steps also run without any host sync in between, ending at the same tokens;
  sampling: per-row temperature / top-k / top-p / seed; two fed runs with the same seeds draw the same tokens, every
            drawn token is in its row's top-k of the logits of its step (host check), and other seeds draw others.

usage: python -m qwen36_resident.tests.rows_feed [--layers 4] [--steps 32] [--out rows_feed.json]
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

USERS, VOCAB = 16, 4096
EMPTY = (11, 13, 14)  # rows of launch 1 with nobody


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-pos", type=int, default=1024)
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    mesh = open_mesh()
    out = {}
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
        sts = [State(d, n_gdn, n_attn, 50 + 37 * u, a.max_pos, seed=30 + u) for u in range(USERS)]
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
        )
        m.capture_trace()
        active = [u not in EMPTY for u in range(USERS)]
        first = [int(t) for t in torch.randint(0, VOCAB, (USERS,), generator=g)]
        print(f"rows feed model: {USERS} slots, {sum(active)} active, vocab {VOCAB}", flush=True)

        def restart():
            for s in range(USERS):
                m.write_slot(s, sts[s])
            m.reset(state=False)  # the feeding hubs' counters

        def tok_host(tokens, pos):
            x0 = torch.stack([emb_host[t].float() for t in tokens])
            return m.token_state(x0, pos, active=active)

        def logits_rows():
            lg = m.logits()  # [users, n, vocab_chip]
            return lg[:, :, : d.vocab_chip].reshape(USERS, -1)[:, :VOCAB].float()

        start = [st.pos for st in sts]
        # host-driven greedy reference
        restart()
        m.set_sampling(0.0, 1, 1.0, 0)
        toks, pos, ref = list(first), list(start), []
        for _ in range(a.steps):
            m.step(tok_host(toks, pos))
            ttnn.synchronize_device(mesh)
            lg = logits_rows()
            toks = [int(lg[u].argmax()) if active[u] else 0 for u in range(USERS)]
            pos = [p + 1 if active[u] else p for u, p in enumerate(pos)]
            ref.append(list(toks))

        def fed_run(sync=True):
            restart()
            m.step(tok_host(first, start))
            got = []
            for k in range(a.steps):
                if k:
                    m.step(None)
                if sync:
                    ttnn.synchronize_device(mesh)
                    got.append(m.fed_token())
            ttnn.synchronize_device(mesh)
            return got, m.fed_token()

        got, last = fed_run()
        fed = [[t if act else 0 for t, _, act in step] for step in got]
        same = sum(f == r for f, r in zip(fed, ref))
        inactive_ok = all(
            not step[u][2] and step[u][1] == start[u] for step in got for u in range(USERS) if not active[u]
        )
        pos_ok = all(got[-1][u][1] == start[u] + a.steps for u in range(USERS) if active[u])
        print(
            f"greedy fed vs host-driven: {same}/{a.steps} steps identical for all users; inactive rows untouched "
            f"{inactive_ok}; positions advanced {pos_ok}",
            flush=True,
        )
        _, last_b2b = fed_run(sync=False)
        b2b = [t for t, _, _ in last_b2b] == [t for t, _, _ in last]
        print(f"back to back (no host sync): final tokens equal {b2b}", flush=True)
        out["greedy"] = dict(identical=same, of=a.steps, inactive_ok=inactive_ok, pos_ok=pos_ok, back_to_back=b2b)

        # sampling: per-row parameters
        params = [(0.8, 20, 0.9, 1000 + u) for u in range(USERS)]
        m.set_sampling(rows=params)
        draws, in_topk, total = [], 0, 0
        for rep in range(2):
            restart()
            m.step(tok_host(first, start))
            seq = []
            for k in range(a.steps):
                if k:
                    m.step(None)
                ttnn.synchronize_device(mesh)
                fed_now = m.fed_token()
                if rep == 0:
                    lg = logits_rows()
                    for u in range(USERS):
                        if active[u]:
                            total += 1
                            in_topk += fed_now[u][0] in lg[u].topk(20).indices.tolist()
                seq.append([t for t, _, _ in fed_now])
            draws.append(seq)
        m.set_sampling(rows=[(0.8, 20, 0.9, 5000 + u) for u in range(USERS)])
        restart()
        m.step(tok_host(first, start))
        for _ in range(a.steps - 1):
            m.step(None)
        ttnn.synchronize_device(mesh)
        other = [t for t, _, _ in m.fed_token()]
        repro = draws[0] == draws[1]
        differs = other != draws[0][-1]
        print(
            f"sampling: reproducible {repro}; drawn tokens in their row's top-20 {in_topk}/{total}; other seeds "
            f"draw other tokens {differs}",
            flush=True,
        )
        out["sampling"] = dict(reproducible=repro, in_topk=in_topk, total=total, other_seeds_differ=differs)
        ok = same == a.steps and inactive_ok and pos_ok and b2b and repro and in_topk == total and differs
        print(f"rows feed gate: {'PASS' if ok else 'FAIL'}", flush=True)
        out["pass"] = ok
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
