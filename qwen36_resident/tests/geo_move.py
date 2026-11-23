# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB3 phase 2, P2-0b: moving a user between the 8 x 4 verify-rows build and a plain 32-row build that shares its
weights / KV pool / L1 (ResidentModel(share=)). The interchange is snapshot_slot's format (GDN states + the conv
history's 3 newest inputs): vrows_snapshot(u, ring, slot) / vrows_restore(u, snap) on the vrows side, snapshot_slot /
restore_slot on the plain side; the KV stays where it is (the plain row takes the user's block table).

Gates: (1) bit-exact round trips: vrows user -> plain row -> snapshot again == the first snapshot; -> vrows user v at
ring 0 -> snapshot == the first. (2) continuation from the moved state: the user's greedy tokens unmoved (vrows),
moved to another vrows user (must be identical) and moved to the plain build (equal up to near-ties: another build's
summation order), N tokens each.

usage: python -m qwen36_resident.tests.geo_move [--prompt-len 256] [--steps 6] [--new 48]
"""
import argparse
import faulthandler

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.dflash import ResidentDFlashRows, SpecRowsSession
from qwen36_resident.model import ATTN_TAILS, TOP_K, Dims, ResidentModel, State, is_attn
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"


def host(t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=0))


def same(a, b):
    return all(torch.equal(host(a[k]), host(b[k])) for k in ("state", "hist"))


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6, help="spec steps before the move (ring / slot past 0)")
    ap.add_argument("--new", type=int, default=48)
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"]
    mesh = open_mesh()
    ok = True
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=cfg["vocab_size"])
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        users, R, P = 8, 4, a.prompt_len
        max_pos = P + 4 * a.steps + 2 * a.new + 256
        bpu = -(-(max_pos + 72) // 64) + 2
        blocks = 40 * bpu + 8
        eng = ResidentDFlashRows(
            mesh,
            ck,
            users=users,
            max_pos=max_pos,
            R=R,
            prefill=True,
            kv_pool_blocks=blocks,
            l1_union=dict(cand_words=d.n * 2 * TOP_K * 32, attn_tails=ATTN_TAILS),
        )
        sts = [State(d, layers - n_attn, n_attn, 0, eng.max_pos, zero=True, kv=False) for _ in range(32)]
        plain = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            sts,
            layers,
            interval,
            lm_head=True,
            embed=eng.pp.embed,
            kv_pool_blocks=blocks,
            block_tables=[[0]] * 32,
            rows=True,
            groups=4,
            share=eng.m,
        )
        plain.reset()
        plain.capture_trace()
        eng.pp.capture()  # (after the second build: its L1 layout)
        tbl = {u: list(range(u * bpu, (u + 1) * bpu)) for u in range(users)}
        emb = Q.embedding(ck).float()

        # ---- users 0..5 started and advanced by spec steps (6, 7 kept free as move targets)
        live = list(range(6))
        sess = SpecRowsSession(eng)
        prompts = {u: ids[1000 * u + 101 : 1000 * u + 101 + P].tolist() for u in live}
        t, pos, acc = [0] * users, [0] * users, [1] * users
        for u, lg in zip(live, sess.start_many([(u, prompts[u], tbl[u]) for u in live])):
            t[u], pos[u] = int(lg.argmax()), P
        act = [u in live for u in range(users)]
        for _ in range(a.steps):
            drafts = sess.propose(t, pos, acc, act)
            g = sess.verify([[t[u]] + list(drafts[u]) if act[u] else [0] * R for u in range(users)], pos, act)
            for u in live:
                k = 0
                while k < R - 1 and drafts[u][k] == g[u][k]:
                    k += 1
                t[u], pos[u], acc[u] = g[u][k], pos[u] + k + 1, k + 1
        u = 2
        a_u = acc[u] - 1  # (the user's last commit, applied as the next verify would)
        ring_u, slot_u = eng.ring[u] + 1 + a_u, (eng.slot[u] + a_u) % R
        print(f"user {u}: position {pos[u]}, ring {ring_u}, slot {slot_u} (last commit {acc[u]})", flush=True)

        # ---- (1) bit-exact round trips
        snap_a = eng.m.vrows_snapshot(u, ring_u, slot_u)
        j = plain.row_order()[5]
        plain.restore_slot(j, snap_a)
        snap_b = plain.snapshot_slot(j)
        rt1 = same(snap_a, snap_b)
        v = 6
        eng.m.vrows_restore(v, snap_b)
        snap_c = eng.m.vrows_snapshot(v, 0, 0)
        rt2 = same(snap_a, snap_c)
        print(
            f"round trip vrows -> plain: {'bit-exact' if rt1 else 'DIFFERS'}; -> vrows user {v} at ring 0: "
            f"{'bit-exact' if rt2 else 'DIFFERS'}",
            flush=True,
        )
        ok &= rt1 and rt2

        # ---- (2) continuation, N greedy tokens from (t, pos) of user u: unmoved / moved to vrows v / moved to plain j
        vlogits = {}

        def vrows_greedy(w, ring, slot, prev, t0, p0):
            ring_, slot_, prev_ = list(eng.ring), list(eng.slot), list(eng.prev)
            ring_[w], slot_[w], prev_[w] = ring, slot, prev
            out, tok, p = [], t0, p0
            lgs = []
            for _ in range(a.new):
                toks = [[0] * R for _ in range(users)]
                toks[w] = [tok] * R
                x0 = torch.stack([eng.emb[list(x)] for x in toks])
                ps = list(pos)
                ps[w] = p
                eng.m.step(eng.m.token_state_vrows(x0, ps, ring_, slot_, prev_, active=[k == w for k in range(users)]))
                tok = eng.greedy_rows()[w][0]
                lgs.append(eng.m.logits()[w * R, :, : d.vocab_chip].reshape(-1)[: cfg["vocab_size"]].float().clone())
                out.append(tok)
                ring_[w], slot_[w], prev_[w] = ring_[w] + 1, slot_[w], 0
                p += 1
            vlogits[w] = lgs
            return out

        plain.set_page_table([tbl[u] if r == j else [0] for r in range(32)])
        eng.m.set_page_table([tbl[k] if k < 6 else (tbl[u] if k == v else [0]) for k in range(users) for _ in range(R)])
        moved_v = vrows_greedy(v, 0, 0, None, t[u], pos[u])  # (with v on u's blocks: it rewrites u's KV identically)
        unmoved = vrows_greedy(u, ring_u, slot_u, a_u, t[u], pos[u])  # (prev: the drafts its last commit accepted)
        out_p, tok, p, plg = [], t[u], pos[u], []
        for _ in range(a.new):
            plain.step(plain.token_state_rows({j: (emb[tok], p)}))
            lg = plain.logits()[j, :, : d.vocab_chip].reshape(-1)[: cfg["vocab_size"]].float()
            plg.append(lg.clone())
            tok = int(lg.argmax())
            out_p.append(tok)
            p += 1
        fd = lambda x, y: next((i for i, (p_, q_) in enumerate(zip(x, y)) if p_ != q_), None)  # noqa: E731
        print(
            f"continuation: moved to vrows {v} vs unmoved first difference {fd(moved_v, unmoved)}; plain vs "
            f"unmoved {fd(out_p, unmoved)} (of {a.new})",
            flush=True,
        )
        ok &= fd(moved_v, unmoved) is None
        # (at each token before the first difference and at it: the logits' distance, and the top-2 gaps there)
        f = fd(out_p, unmoved)
        n_cmp = a.new if f is None else f + 1
        dl = [float((plg[i] - vlogits[u][i]).abs().max()) for i in range(n_cmp)]
        print(f"max |dlogit| plain vs unmoved per token (to the first difference): {[round(x, 3) for x in dl]}")
        host_v = [int(x.argmax()) for x in vlogits[u]]
        print(
            f"tokens unmoved (device greedy) {unmoved[: n_cmp + 2]}\n       unmoved (host argmax)   "
            f"{host_v[: n_cmp + 2]}\n       plain                   {out_p[: n_cmp + 2]}",
            flush=True,
        )
        if f is not None:
            vt, pt_ = vlogits[u][f].topk(2), plg[f].topk(2)
            print(
                f"at token {f}: vrows top-2 {vt.indices.tolist()} gap {float(vt.values[0] - vt.values[1]):.4f}; "
                f"plain top-2 {pt_.indices.tolist()} gap {float(pt_.values[0] - pt_.values[1]):.4f}",
                flush=True,
            )
        print(f"geo move: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
