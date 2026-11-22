# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB1: verify rows (ResidentModel(verify=8, rows=True, groups=G)): row group g is user g's verify of 8 rows (a token
and 7 drafts at p_g .. p_g + 7), the G users in one launch.

random: a few layers with random weights (a small vocab, paged KV with shuffled block tables, a random embedding table
        for the hub's feed). Each step runs every user's R rows from its committed state; every row is compared with
        torch_step from the user's reference state (rows chained as plain decode). Then user g accepts a_g drafts (a_g
        differs per user and cycles over 0 .. R - 1) and its next step continues from row a_g (p + a + 1, ring + 1 + a,
        slot + a): the reference continues from its state after row a_g, so every later step also checks the commit.
        The GDN state the step left in the accepted row's slot and the taps (x after the tap layers, group g in rows
        8 g ..) are checked too. Some steps leave a user inactive (its rows do not run; its state must stay).

real:   the Qwen3.8 checkpoint. A plain batch-1 model teacher-forces each user's corpus sequence (positions 0 .. L - 1)
        for reference logits; then the verify-rows model runs the same sequences R rows per step (the drafts are the
        true next tokens), user g accepting a_g of them (cycling per user, so the users' positions drift apart and every
        commit is used); every row's logits are compared with the plain model's at its position (top-1 agreement, max
        |diff|, the plain top-1 - top-2 gap where the top-1 differs) and the device's greedy token of every row (the
        hub's argmax, fed_token) with the host argmax. Then (--contexts) the traced fed step time with every user's
        rows at position T (KV pages zero; the time does not depend on them).

usage: python -m qwen36_resident.tests.db_verify random [--groups 4] [--layers 4] [--steps 12] [--json out.json]
       python -m qwen36_resident.tests.db_verify real --corpus corpus_ids.pt [--groups 4] [--len 256]
           [--contexts 128,8192,32768] [--json out.json]
"""
import argparse
import copy
import faulthandler
import gc
import json
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import HIDDEN, Dims, ResidentModel, State, is_attn, random_weights, torch_step
from qwen36_resident.tests.dfl_verify import pcc, read_tap, ref_states, rel, tap_tensors
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh, shuffled_tables

VOCAB = 4096
R = 8


def dev_states(model):
    """the GDN state tensor as [chips, copies, heads, launch rows (slots), DK, DV]"""
    d, LR = model.d, model.launch_rows
    t = ttnn.to_torch(model.subs[0]["state"], mesh_composer=ttnn.ConcatMeshToTensor(model.mesh, dim=0)).float()
    t = t.reshape(d.n, model._gdn_copies, d.nv, LR, 4, 4, 32, 32).permute(0, 1, 2, 3, 4, 6, 5, 7)
    return t.reshape(d.n, model._gdn_copies, d.nv, LR, 128, 128)


def run_random(mesh, G, layers, interval, steps, max_pos=512, seed=0, use_taps=True, only=None, starts=None, R=8):
    U = G * 8 // R  # (users: one per group at R = 8, two at R = 4)
    n = mesh.get_num_devices()
    d = Dims(n, mesh.dram_grid_size().x, vocab=VOCAB)
    n_attn = sum(is_attn(l, interval) for l in range(layers))
    n_gdn = layers - n_attn
    w = random_weights(d, n_gdn, n_attn, layers, seed=seed)
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
    # start positions over several KV tiles, some near a tile's end (the rows cross into the next tile)
    starts = starts or [40 + 53 * u + (27 if u % 2 else 0) for u in range(U)]
    sts = [State(d, n_gdn, n_attn, starts[u], max_pos, seed=30 + u) for u in range(U)]
    refs = [copy.deepcopy(s) for s in sts]
    blocks, tables = shuffled_tables(U, max_pos, seed=4)
    tap_layers = sorted({0, layers // 2, layers - 1}) if use_taps else []
    taps = tap_tensors(mesh, len(tap_layers), 32)
    t0 = time.time()
    m = ResidentModel(
        mesh,
        d,
        w,
        sts,
        layers,
        interval,
        lm_head=True,
        embed=embed,
        kv_pool_blocks=blocks,
        block_tables=tables,
        rows=True,
        groups=G,
        verify=R,
        taps=list(zip(tap_layers, taps)),
    )
    print(f"verify rows: {U} users x {R} rows, layers={layers} ({n_attn} attention), built in {time.time() - t0:.1f} s")
    gen = torch.Generator().manual_seed(100)
    pos, ring, slot, prev = list(starts), [0] * U, [0] * U, [None] * U
    rows, worst = [], dict(rel=0.0, pcc=1.0, tap_pcc=1.0, state=0.0, idle_state=0.0)
    for s in range(steps):
        active = [not (s % 4 == 2 and u == s % U) for u in range(U)]  # every 4th step one user sits out
        if only is not None:
            active = [u in only for u in range(U)]
        acc = [(3 * s + 1 + 2 * u) % R for u in range(U)]
        x0 = torch.randn(U, R, HIDDEN, generator=gen)
        before = dev_states(m) if not all(active) else None
        m.step(m.token_state_vrows(x0, pos, ring, slot, prev, active=active))
        ttnn.synchronize_device(mesh)
        got = m.x().reshape(U * R, HIDDEN)
        got_taps = [read_tap(mesh, t, 32) for t in taps]
        dev = dev_states(m)
        line = []
        for u in range(U):
            if not active[u]:
                base = u * R
                err = float((dev[:, :, :, base : base + R] - before[:, :, :, base : base + R]).abs().max())
                worst["idle_state"] = max(worst["idle_state"], err)
                line.append(f"u{u}: idle (state change {err:.1e})")
                continue
            snap, nxt = copy.deepcopy(refs[u]), None
            for r in range(R):
                xs = []
                want = torch_step(x0[u, r], d, w, snap, layers, interval, xs=xs)
                row = u * R + r
                e, c = rel(got[row], want), pcc(got[row], want)
                tc = min([pcc(got_taps[k][row], xs[l]) for k, l in enumerate(tap_layers)] or [1.0])
                each = [round(pcc(got_taps[k][row], xs[l]), 5) for k, l in enumerate(tap_layers)]
                rows.append(
                    dict(
                        step=s, user=u, p=pos[u], row=r, ring=ring[u], slot=slot[u], rel=e, pcc=c, tap_pcc=tc, taps=each
                    )
                )
                worst.update(rel=max(worst["rel"], e), pcc=min(worst["pcc"], c), tap_pcc=min(worst["tap_pcc"], tc))
                if r == acc[u]:
                    nxt = copy.deepcopy(snap)
            # the accepted row's state: slot u R + (slot + a) % R
            got_st = dev[:, :, :, u * R + (slot[u] + acc[u]) % R]
            want_st = ref_states(nxt)
            st_err = float((got_st - want_st).abs().max() / want_st.abs().max())
            worst["state"] = max(worst["state"], st_err)
            mine = rows[-R:]
            line.append(
                f"u{u}: p={pos[u]} a={acc[u]} pcc min {min(x['pcc'] for x in mine):.5f} taps "
                f"{min(x['tap_pcc'] for x in mine):.5f} state {st_err:.1e}"
            )
            refs[u] = nxt
            pos[u], ring[u], slot[u], prev[u] = (
                pos[u] + acc[u] + 1,
                ring[u] + 1 + acc[u],
                (slot[u] + acc[u]) % R,
                acc[u],
            )
        print(f"step {s}: " + "; ".join(line), flush=True)
    # (bars by depth: single-user verify, dfl_verify random R=8, reaches pcc 0.9974 / state 0.065 at 4 layers and
    # 0.9905 / 0.16 at 16 — bf16 rounding over random layers)
    bar, st_bar = (0.995, 0.15) if layers <= 4 else (0.985, 0.25)
    ok = worst["pcc"] >= bar and worst["tap_pcc"] >= bar and worst["state"] < st_bar and worst["idle_state"] == 0.0
    print(f"verify rows: worst {json.dumps(worst)}  {'PASS' if ok else 'FAIL'}", flush=True)
    return dict(U=U, layers=layers, interval=interval, worst=worst, ok=ok, rows=rows)


def run_leak(mesh, G, R, layers=4, interval=4, max_pos=512):
    """bit-exactness of a verify row against its later rows: the same step twice from the same state (reset in
    between), the second with rows 1 .. R - 1 of every user replaced — row 0's x after every layer (taps) and at the end
    must be bit-identical (it sees none of the later rows)"""
    U = G * 8 // R
    n = mesh.get_num_devices()
    d = Dims(n, mesh.dram_grid_size().x, vocab=VOCAB)
    n_attn = sum(is_attn(l, interval) for l in range(layers))
    n_gdn = layers - n_attn
    w = random_weights(d, n_gdn, n_attn, layers, seed=0)
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
    starts = [40 + 53 * u + (27 if u % 2 else 0) for u in range(U)]
    sts = [State(d, n_gdn, n_attn, starts[u], max_pos, seed=30 + u) for u in range(U)]
    blocks, tables = shuffled_tables(U, max_pos, seed=4)
    tap_layers = list(range(min(layers, 5)))
    taps = tap_tensors(mesh, len(tap_layers), 32)
    m = ResidentModel(
        mesh,
        d,
        w,
        sts,
        layers,
        interval,
        lm_head=True,
        embed=embed,
        kv_pool_blocks=blocks,
        block_tables=tables,
        rows=True,
        groups=G,
        verify=R,
        taps=list(zip(tap_layers, taps)),
    )
    gen = torch.Generator().manual_seed(7)
    x0 = torch.randn(U, R, HIDDEN, generator=gen)
    outs = []
    for variant in range(2):
        xv = x0.clone()
        if variant:
            xv[:, 1:] = torch.randn(U, R - 1, HIDDEN, generator=gen)
        m.reset()
        m.step(m.token_state_vrows(xv, starts, [0] * U, [0] * U))
        ttnn.synchronize_device(mesh)
        outs.append(([read_tap(mesh, t, 32) for t in taps], m.x().reshape(U * R, HIDDEN)))
    res = {}
    for u in range(U):
        row = u * R
        per = [float((outs[0][0][k][row] - outs[1][0][k][row]).abs().max()) for k in range(len(tap_layers))]
        res[u] = dict(taps=per, x=float((outs[0][1][row] - outs[1][1][row]).abs().max()))
        print(f"user {u}: row 0 |diff| after layers {tap_layers} = {per}, x {res[u]['x']}", flush=True)
    ok = all(v["x"] == 0.0 for v in res.values())
    print(f"leak check ({U} x {R}, {layers} layers): {'bit-identical' if ok else 'DIFFERS'}", flush=True)
    return dict(ok=ok, users=res)


def run_real(mesh, U, L, corpus, contexts, time_steps=32, warm_steps=4, block=64):
    ck = Q.Checkpoint(Q.checkpoint_dir())
    cfg = ck.config
    layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
    n = mesh.get_num_devices()
    d = Dims(n, mesh.dram_grid_size().x, vocab=vocab)
    n_attn = sum(is_attn(l, interval) for l in range(layers))
    emb = Q.embedding(ck).float()
    ids = torch.load(corpus)["ids"]
    seqs = [ids[5000 * u + 17 : 5000 * u + 17 + L + R].tolist() for u in range(U)]
    max_pos = max([L + 2 * R + 64] + [T + R + warm_steps + time_steps + 64 for T in contexts])
    w = Q.load(ck, d, layers, interval)
    empty = lambda: State(d, layers - n_attn, n_attn, 0, max_pos, zero=True, kv=False)  # noqa: E731
    out = dict(U=U, L=L, R=R)
    # ---- plain teacher-forced reference (alone on the mesh, see dfl_verify)
    t0 = time.time()
    plain = ResidentModel(mesh, d, w, empty(), layers, interval, True)
    want = []
    for u in range(U):
        plain.reset()
        lg = []
        for p in range(L):
            plain.step(plain.token_state(emb[seqs[u][p]], p))
            lg.append(plain.logits()[:, : d.vocab_chip].reshape(-1)[:vocab].bfloat16())
        want.append(torch.stack(lg))
    print(f"plain reference: {U} x {L} positions in {time.time() - t0:.0f} s", flush=True)
    del plain
    gc.collect()
    ttnn.synchronize_device(mesh)
    # ---- verify rows
    embed = ttnn.from_torch(
        emb.bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    blocks, tables = shuffled_tables(U, max_pos, block=block, seed=7)
    t0 = time.time()
    m = ResidentModel(
        mesh,
        d,
        w,
        [empty() for _ in range(U)],
        layers,
        interval,
        lm_head=True,
        embed=embed,
        kv_pool_blocks=blocks,
        block_tables=tables,
        kv_block_size=block,
        rows=True,
        groups=U,
        verify=R,
    )
    del w
    gc.collect()
    print(f"verify rows model: {U} users x {R} rows built in {time.time() - t0:.0f} s", flush=True)
    LR = U * R
    pos, ring, slot, prev = [0] * U, [0] * U, [0] * U, [None] * U
    agree = total = dev_agree = 0
    worst, gaps, s = 0.0, [], 0
    while min(pos) < L:
        active = [p < L for p in pos]
        x0 = torch.zeros(U, R, HIDDEN)
        for u in range(U):
            for r in range(R):
                x0[u, r] = emb[seqs[u][min(pos[u] + r, len(seqs[u]) - 1)]]
        print(f"step {s}: pos {pos}", flush=True)
        m.step(m.token_state_vrows(x0, pos, ring, slot, prev, active=active))
        ttnn.synchronize_device(mesh)
        lg = m.logits()[:, :, : d.vocab_chip].reshape(LR, -1)[:, :vocab].float()
        fed = m.fed_token()
        for u in range(U):
            if not active[u]:
                continue
            for r in range(R):
                p = pos[u] + r
                if p >= L:
                    break
                got, ref = lg[u * R + r], want[u][p].float()
                total += 1
                worst = max(worst, float((got - ref).abs().max()))
                g1, w1 = int(got.argmax()), int(ref.argmax())
                if g1 == w1:
                    agree += 1
                else:
                    gaps.append(float(ref.max() - ref[g1]))
                dev_agree += int(fed[u * R + r][0] == g1)
            # accept a_u drafts: cycles per user (all R - 1 every other step, so the sequences advance)
            a = R - 1 if s % 2 == 0 else (s // 2 + 3 * u) % R
            pos[u], ring[u], slot[u], prev[u] = pos[u] + a + 1, ring[u] + 1 + a, (slot[u] + a) % R, a
        s += 1
    out.update(
        steps=s,
        rows=total,
        top1_agree=agree / total,
        max_abs_diff=worst,
        mismatch_gaps=sorted(gaps),
        device_argmax_agree=dev_agree / total,
    )
    print(
        f"teacher-forced {total} rows in {s} steps: top-1 agreement {agree}/{total} ({agree / total:.4f}), max |diff| "
        f"{worst:.3f}, mismatches' plain gap max {max(gaps) if gaps else 0:.4f}; device argmax = host argmax "
        f"{dev_agree}/{total}",
        flush=True,
    )
    # ---- traced fed step time at context T (every user's rows at T .. T + R - 1)
    if contexts:
        m.capture_trace()
        out["step_ms"] = {}
        for T in contexts:
            m.reset(state=False)
            tok = m.token_state_vrows(torch.zeros(U, R, HIDDEN), [T] * U, [0] * U, [0] * U)
            m.step(tok)
            for _ in range(warm_steps):
                m.step(None)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(time_steps):
                m.step(None)
            ttnn.synchronize_device(mesh)
            ms = (time.time() - t0) / time_steps * 1e3
            out["step_ms"][T] = ms
            print(f"T={T}: {ms:.2f} ms per {U} x {R}-row verify step (traced, fed)", flush=True)
    ok = out["top1_agree"] >= 0.97 and (not gaps or max(gaps) < 0.5) and out["device_argmax_agree"] == 1.0
    out["ok"] = ok
    print(f"verify rows real: {'PASS' if ok else 'FAIL'}", flush=True)
    return out


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["random", "real", "leak"])
    ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt")
    ap.add_argument("--len", type=int, default=256)
    ap.add_argument("--contexts", default="128,8192,32768")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--rows", type=int, default=8, help="verify rows per user (8: a user per group, 4: two)")
    ap.add_argument("--layers", type=int, default=4)  # (row groups need an attention layer: its leader runs the rows)
    ap.add_argument("--interval", type=int, default=4)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--json", default=None)
    ap.add_argument("--no-taps", action="store_true")
    ap.add_argument("--only", default=None, help="comma list: only these users run (diagnostics)")
    ap.add_argument("--starts", default=None, help="comma list: the users' start positions")
    a = ap.parse_args()
    mesh = open_mesh()
    try:
        if a.mode == "leak":
            out = run_leak(mesh, a.groups, a.rows, layers=a.layers)
        elif a.mode == "real":
            ctx = [int(t) for t in a.contexts.split(",")] if a.contexts else []
            out = run_real(mesh, a.groups, a.len, a.corpus, ctx)
        else:
            only = None if a.only is None else [int(x) for x in a.only.split(",")]
            starts = None if a.starts is None else [int(x) for x in a.starts.split(",")]
            out = run_random(
                mesh,
                a.groups,
                a.layers,
                a.interval,
                a.steps,
                use_taps=not a.no_taps,
                only=only,
                starts=starts,
                R=a.rows,
            )
    finally:
        close_mesh(mesh)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
