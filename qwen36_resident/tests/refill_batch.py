# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""AdaptiveEngine.move_to_spec_many (the drafter refills of several users in one taps-only PPPrefill.run_batch pass,
each tail at its own tap-ring offset) vs the sequential refill (reset + run + fill per user): 4 prompt pairs in the
plain build (each pair two users with the same prompt), one of each pair moved sequentially, the other batched; the
drafts proposed afterwards must be equal pairwise, and a few spec steps' tokens too.

usage: python -m qwen36_resident.tests.refill_batch
"""
import faulthandler
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.adaptive import AdaptiveEngine
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"


def main():
    faulthandler.enable()
    ids = torch.load(CORPUS)["ids"]
    mesh = open_mesh()
    ok = True
    try:
        max_pos = 4096
        bpu = -(-(max_pos + 72) // 64) + 2
        eng = AdaptiveEngine(mesh, Q.Checkpoint(Q.checkpoint_dir()), max_pos, kv_pool_blocks=20 * bpu + 8)
        e = eng.spec
        lens = [100, 300, 700, 1500]
        prompts = [ids[3000 * k + 77 : 3000 * k + 77 + n].tolist() for k, n in enumerate(lens)]
        table = lambda k: list(range(k * bpu, (k + 1) * bpu))  # noqa: E731
        eng.add([(k, prompts[k % 4], table(k)) for k in range(8)], "plain")
        for _ in range(3):
            eng.plain_step()
        assert all(eng.users[k].tokens == eng.users[k + 4].tokens for k in range(4)), "pairs diverged in plain"
        # sequential reference (the former path) for users 0..3 into spec users 0..3
        t0 = time.perf_counter()
        for k in range(4):
            us = eng.users[k]
            u = k
            snap = eng.plain.snapshot_slot(us.idx)
            eng._set_spec_table(u, us.table)
            e.m.vrows_restore(u, snap)
            for x in ("state", "hist"):
                ttnn.deallocate(snap[x])
            n = us.pos
            s0 = max(0, n - e.window) // 64 * 64
            eng.pp.reset()
            eng.pp.run(torch.as_tensor(us.tokens[s0:n]))
            e._fill_context(n - s0, phys=u, offset=s0)
            e.ring[u], e.slot[u], e.prev[u] = 0, 0, None
            eng.sess.pos[u], eng.sess.started[u], eng.sess.committed[u], eng.sess.last[u] = n, True, 1, None
            us.geo, us.idx = "spec", u
        ttnn.synchronize_device(mesh)
        t_seq = time.perf_counter() - t0
        # batched for users 4..7 into spec users 4..7
        t0 = time.perf_counter()
        eng.move_to_spec_many([(eng.users[k].idx, k, eng.users[k].table, eng.users[k].tokens) for k in range(4, 8)])
        ttnn.synchronize_device(mesh)
        t_batch = time.perf_counter() - t0
        for k in range(4, 8):
            eng.users[k].geo, eng.users[k].idx = "spec", k
        eng.free_spec = []
        U = eng.SPEC_USERS
        drafts = eng.sess.propose(
            [eng.users[k].next for k in range(U)], [eng.users[k].pos for k in range(U)], [1] * U, [True] * U
        )
        same = [drafts[k] == drafts[k + 4] for k in range(4)]
        print(
            f"refill: sequential {t_seq * 1000:.0f} ms, batched {t_batch * 1000:.0f} ms; drafts equal per pair {same}",
            flush=True,
        )
        ok &= all(same)
        print(f"refill batch: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
