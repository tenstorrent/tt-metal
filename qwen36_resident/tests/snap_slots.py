# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strata S1 / S3 primitive: a rows-mode slot's recurrent state snapshot (ResidentModel.snapshot_slot / restore_slot:
GDN states by page-copy jobs, the conv history row by hist_scatter's gather / scatter) moves a user to another slot.

Random weights, a small vocab, paged KV with shuffled tables, 8 users in one launch (rows mode, one group): a few
steps for every user; slot A's snapshot (with its KV pages below its position); A alone for N steps (the reference
logits); then the snapshot into slot B — its GDN states, conv history and KV pages into B's own blocks (--share-table:
the GDN part only, B's block table := A's) — and B alone over the same N tokens and positions:
B's logits must equal A's bit for bit (a row's arithmetic does not depend on its row). The other slots' states are
checked untouched by the restore.

usage: python -m qwen36_resident.tests.snap_slots [--layers 4] [--steps 6]
"""
import argparse
import faulthandler
import sys

import torch

import ttnn
from qwen36_resident.model import HIDDEN, Dims, ResidentModel, State, is_attn, random_weights
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh, shuffled_tables

VOCAB = 4096
A, B = 1, 5


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--max-pos", type=int, default=512)
    ap.add_argument("--share-table", action="store_true")
    ap.add_argument("--via-host", action="store_true", help="the snapshot through host memory (device copies freed)")
    a = ap.parse_args()
    mesh = open_mesh()
    ok = False
    try:
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=VOCAB)
        interval = 4
        n_attn = sum(is_attn(l, interval) for l in range(a.layers))
        n_gdn = a.layers - n_attn
        w = random_weights(d, n_gdn, n_attn, a.layers)
        sts = [State(d, n_gdn, n_attn, 40 + 23 * u, a.max_pos, seed=50 + u) for u in range(8)]
        blocks, tables = shuffled_tables(8, a.max_pos, seed=9)
        m = ResidentModel(
            mesh, d, w, sts, a.layers, interval, lm_head=True, kv_pool_blocks=blocks, block_tables=tables, rows=True
        )
        g = torch.Generator().manual_seed(1)
        pos = [st.pos for st in sts]

        def step(active, xs):
            x0 = torch.zeros(8, HIDDEN)
            for u, x in xs.items():
                x0[u] = x
            m.step(m.token_state(x0, list(pos), active=[u in active for u in range(8)]))
            ttnn.synchronize_device(mesh)
            return m.logits()[:, :, : d.vocab_chip].reshape(8, -1)[:, :VOCAB].float()

        for _ in range(3):  # every user a few steps
            step(set(range(8)), {u: torch.randn(HIDDEN, generator=g) for u in range(8)})
            pos = [p + 1 for p in pos]
        p_snap = pos[A]
        snap = m.snapshot_slot(A, n_tokens=0 if a.share_table else p_snap)
        xs = [torch.randn(HIDDEN, generator=g) for _ in range(a.steps)]
        ref = []
        for x in xs:  # A alone
            ref.append(step({A}, {A: x})[A])
            pos[A] += 1
        others = {u: m.read_slot(u) for u in range(8) if u not in (A, B)}
        if a.via_host:
            import time

            t0 = time.perf_counter()
            host = m.snapshot_to_host(snap)
            t1 = time.perf_counter()
            snap = m.snapshot_from_host(host)
            ttnn.synchronize_device(mesh)
            t2 = time.perf_counter()
            mb = sum(int(host[k].volume() * host[k].element_size()) for k in m.SNAP_TENSORS if k in host) / 1e6
            print(f"via host: {mb:.1f} MB (all chips) out {1e3 * (t1 - t0):.1f} ms, in {1e3 * (t2 - t1):.1f} ms")
        m.restore_slot(B, snap)
        if a.share_table:
            tbl = [list(t) for t in m.block_tables]
            tbl[B] = list(tbl[A])
            m.set_page_table(tbl)
        pos[B] = p_snap
        got = []
        for x in xs:  # B alone over A's tokens and positions
            got.append(step({B}, {B: x})[B])
            pos[B] += 1
        diffs = [float((g_ - r_).abs().max()) for g_, r_ in zip(got, ref)]
        untouched = all(
            all(torch.equal(v, m.read_slot(u)[k]) for k, v in st.items() if k != "kv") for u, st in others.items()
        )
        ok = max(diffs) == 0.0 and untouched
        print(f"restored slot {B} vs slot {A}: max |logit diff| per step {diffs}; other slots untouched {untouched}")
        print(f"snap slots: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
