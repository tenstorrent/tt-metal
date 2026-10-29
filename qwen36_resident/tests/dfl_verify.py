# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checks of the R-row verify step (ResidentModel(verify=R), DFlash2: a token and R - 1 drafts of one user).

random: a few layers with random weights against torch_step (model.py). Each step runs R rows at p .. p + R - 1
        and is compared row by row with R reference decode steps from the committed state; then a of the drafts
        are accepted (a cycles over 0 .. R - 1, so every commit is exercised) and the next step continues from
        row a (p + a + 1, ring + 1 + a, slot + a): the reference rolls back to its state after row a, so every
        later step also checks that the commit left exactly the state of plain decode of the accepted rows. The
        taps (x after the tap layers, written by the hubs) are compared with the reference's x after those layers.
        Start positions near a tile end make the rows cross into the next KV tile.
real:   the Qwen3.8 checkpoint, an R-row verify model and a plain batch-1 model on the same mesh.
        rows:   a corpus sequence teacher-forced through both (verify: R rows per step, all accepted); per position
                the logits of the verify row against the plain step (max |diff|, top-1 agreement, the plain model's
                top-1 - top-2 gap where the top-1 differs);
        commit: for every a in 0 .. R - 1, from the state after a prefix: a verify step with a drafts right, commit a,
                then the next verify step, against the same model reaching that position by one-token steps (the
                accepted tokens one at a time); all R rows' logits compared (expected bit-equal), and row 0 with the
                plain model;
        taps:   the verify model's taps for the teacher-forced positions, saved (--taps-out) for the comparison with
                our ttnn target's taps;
        speed:  the traced verify step (token state write, replay, per-row argmax readback) at a few positions.

usage: python -m qwen36_resident.tests.dfl_verify random [--rows 8] [--layers 4] [--interval 4] [--steps 16]
       python -m qwen36_resident.tests.dfl_verify real [--rows 8] --out real.json [--taps-out taps.pt]
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
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh


def pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-30))


def tap_tensors(mesh, n, rows):
    return [
        ttnn.from_torch(
            torch.zeros(1, 1, rows, HIDDEN),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        for _ in range(n)
    ]


def read_tap(mesh, t, rows):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float().reshape(-1, HIDDEN)[:rows]


def run_random(mesh, R, layers, interval, steps, start_pos, max_pos=256, seed=0):
    d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x)
    n_attn = sum(is_attn(l, interval) for l in range(layers))
    n_gdn = layers - n_attn
    w = random_weights(d, n_gdn, n_attn, layers, seed=seed)
    st = State(d, n_gdn, n_attn, start_pos, max_pos, seed=11)
    ref = copy.deepcopy(st)
    tap_layers = sorted({0, layers // 2, layers - 1})
    taps = tap_tensors(mesh, len(tap_layers), R)
    t0 = time.time()
    model = ResidentModel(
        mesh, d, w, st, layers, interval, verify=True if R == 2 else R, taps=list(zip(tap_layers, taps))
    )
    log(f"R={R} layers={layers} interval={interval}: built in {time.time() - t0:.1f} s (ring {model.ring_bytes})")
    g = torch.Generator().manual_seed(100 + R)
    p, ring, slot = start_pos, 0, 0
    rows, worst = [], dict(rel=0.0, pcc=1.0, tap_rel=0.0, tap_pcc=1.0)
    for s in range(steps):
        a = (3 * s + 1) % R  # every accept count over R steps
        x0 = torch.randn(R, HIDDEN, generator=g)
        model.step(model.token_state(x0, list(range(p, p + R)), ring=ring, slot=slot))
        ttnn.synchronize_device(mesh)
        got = model.x().reshape(R, HIDDEN)
        got_taps = [read_tap(mesh, t, R) for t in taps]
        snap, nxt = copy.deepcopy(ref), None
        for r in range(R):
            xs = []
            want = torch_step(x0[r], d, w, snap, layers, interval, xs=xs)
            e, c = rel(got[r], want), pcc(got[r], want)
            te = max(rel(got_taps[k][r], xs[l]) for k, l in enumerate(tap_layers))
            tc = min(pcc(got_taps[k][r], xs[l]) for k, l in enumerate(tap_layers))
            rows.append(dict(step=s, p=p, row=r, ring=ring, slot=slot, accept=a, rel=e, pcc=c, tap_rel=te, tap_pcc=tc))
            worst = dict(
                rel=max(worst["rel"], e),
                pcc=min(worst["pcc"], c),
                tap_rel=max(worst["tap_rel"], te),
                tap_pcc=min(worst["tap_pcc"], tc),
            )
            if r == a:
                nxt = copy.deepcopy(snap)
        bad = [x for x in rows[-R:] if x["pcc"] < 0.995 or x["tap_pcc"] < 0.995]
        log(
            f"step {s}: p={p} ring={ring} slot={slot} accept {a}: rows rel max "
            f"{max(x['rel'] for x in rows[-R:]):.4f} pcc min {min(x['pcc'] for x in rows[-R:]):.5f}, taps pcc min "
            f"{min(x['tap_pcc'] for x in rows[-R:]):.5f}" + (f"  BAD rows {[x['row'] for x in bad]}" if bad else "")
        )
        ref = nxt
        p, ring, slot = p + a + 1, ring + 1 + a, (slot + a) % R
    log(f"R={R} layers={layers}: worst {json.dumps(worst)}")
    del model
    return dict(R=R, layers=layers, interval=interval, worst=worst, rows=rows)


REF = "/home/ttuser/experiments/qwen36_27b/profiles/resident/hf_ref_qwen38.pt"
TAP_LAYERS = (5, 19, 33, 47, 61)  # the DFlash2 drafter's (incoai/Qwen3.8-27B-DFlash2 target_layer_ids)


class Verifier:
    """an R-row verify model with taps, driven from the host: step(tokens, p) runs rows tokens at p .. p + R - 1
    and commit(a) continues from row a"""

    def __init__(self, mesh, d, w, layers, interval, R, max_pos, emb, taps=True):
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        self.R, self.emb, self.d, self.mesh = R, emb, d, mesh
        self.taps = tap_tensors(mesh, len(TAP_LAYERS), R) if taps else []
        st = State(d, layers - n_attn, n_attn, 0, max_pos, zero=True, kv=False)
        self.m = ResidentModel(mesh, d, w, st, layers, interval, True, verify=R, taps=list(zip(TAP_LAYERS, self.taps)))
        self.ring = self.slot = 0

    def reset(self):
        self.m.reset()
        self.ring = self.slot = 0

    def step(self, toks, p):
        assert len(toks) == self.R
        self.m.step(
            self.m.token_state(self.emb[list(toks)], list(range(p, p + self.R)), ring=self.ring, slot=self.slot)
        )

    def commit(self, a):
        self.ring, self.slot = self.ring + 1 + a, (self.slot + a) % self.R

    def logits(self, vocab):
        lg = self.m.logits()  # [R, n, vocab per chip padded]
        return lg[:, :, : self.d.vocab_chip].reshape(self.R, -1)[:, :vocab].clone()

    def tap_rows(self):
        return torch.stack([read_tap(self.mesh, t, self.R) for t in self.taps])  # [taps, R, H]


def gap_info(want_row, got_top):
    """the plain row's top-1 - (its value at the verify's top-1)"""
    return float(want_row.max() - want_row[got_top])


def run_real(mesh, R, prompt_len, gen, prefix, taps_out, positions):
    ck = Q.Checkpoint(Q.checkpoint_dir())
    cfg = ck.config
    layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
    d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=vocab)
    n_attn = sum(is_attn(l, interval) for l in range(layers))
    emb = Q.embedding(ck).float()
    max_pos = max(positions + [prompt_len + gen]) + 64
    w = Q.load(ck, d, layers, interval)
    seq = torch.load(REF)["seqs"][0]["ids"].tolist()
    L = -(-(prompt_len + gen) // R) * R
    seq = (seq * (L // len(seq) + 1))[:L]
    out = dict(R=R, L=L)
    # the plain model first, alone (two models' weight copies do not keep the fixed per-copy DRAM stride the
    # streamers need once the first has left holes): its teacher-forced logits are the reference
    t0 = time.time()
    plain = ResidentModel(
        mesh, d, w, State(d, layers - n_attn, n_attn, 0, max_pos, zero=True, kv=False), layers, interval, True
    )
    log(f"plain model built ({time.time() - t0:.0f} s)")

    def plain_logits(t, p):
        plain.step(plain.token_state(emb[t], p))
        return plain.logits()[:, : d.vocab_chip].reshape(-1)[:vocab].clone()

    want = torch.stack([plain_logits(seq[p], p) for p in range(L)])
    del plain
    gc.collect()
    ttnn.synchronize_device(mesh)
    t0 = time.time()
    ver = Verifier(mesh, d, w, layers, interval, R, max_pos, emb)
    del w
    log(f"verify model R={R} built in {time.time() - t0:.0f} s (ring {ver.m.ring_bytes})")

    # ---- rows: teacher-forced, every row accepted
    ver.reset()
    got, taps = [], []
    for p0 in range(0, L, R):
        ver.step(seq[p0 : p0 + R], p0)
        got.append(ver.logits(vocab))
        taps.append(ver.tap_rows())
        ver.commit(R - 1)
    got, taps = torch.cat(got), torch.cat(taps, dim=1)  # [L, vocab], [taps, L, H]
    diff = (got - want).abs().max(dim=1).values
    top_g, top_w = got.argmax(1), want.argmax(1)
    miss = [
        dict(pos=p, verify=int(top_g[p]), plain=int(top_w[p]), plain_gap=gap_info(want[p], int(top_g[p])))
        for p in range(L)
        if top_g[p] != top_w[p]
    ]
    out["rows"] = dict(
        positions=L,
        max_abs=float(diff.max()),
        median_max_abs=float(diff.median()),
        top1_agree=1 - len(miss) / L,
        misses=miss,
        logit_scale=float(want.abs().max()),
    )
    log(
        f"rows: {L} positions, max |verify - plain| {diff.max():.4f} (median per row {diff.median():.4f}), "
        f"top-1 agree {1 - len(miss) / L:.4f}, misses {miss}"
    )
    if taps_out:
        torch.save(dict(ids=seq, taps=taps.bfloat16(), tap_layers=TAP_LAYERS), taps_out)

    # ---- commit: every accept count from the state after `prefix` positions
    assert prefix % R == 0 and prefix + 2 * R <= L
    rows_c = []

    def replay():
        ver.reset()
        for p0 in range(0, prefix, R):
            ver.step(seq[p0 : p0 + R], p0)
            ver.commit(R - 1)

    for a in range(R):
        nxt = prefix + a + 1
        drafts = [seq[prefix + i] if i <= a else (seq[prefix + i] + 1) % vocab for i in range(1, R)]
        replay()
        ver.step([seq[prefix]] + drafts, prefix)
        ver.commit(a)
        ver.step(seq[nxt : nxt + R], nxt)
        lg_spec = ver.logits(vocab)
        replay()
        for q in range(prefix, nxt):
            ver.step([seq[q]] * R, q)
            ver.commit(0)
        ver.step(seq[nxt : nxt + R], nxt)
        lg_ref = ver.logits(vocab)
        e = float((lg_spec - lg_ref).abs().max())
        e_plain = float((lg_spec[0] - want[nxt]).abs().max())
        rows_c.append(
            dict(
                a=a,
                max_abs_vs_one_token_steps=e,
                bit_equal=bool(torch.equal(lg_spec, lg_ref)),
                row0_max_abs_vs_plain=e_plain,
                top1_equal=bool((lg_spec.argmax(1) == lg_ref.argmax(1)).all()),
            )
        )
        log(
            f"commit a={a}: vs one-token steps max |d| {e:.5f} bit-equal {rows_c[-1]['bit_equal']}, "
            f"row 0 vs plain {e_plain:.4f}"
        )
    out["commit"] = rows_c

    # ---- speed: traced verify step at a few positions (state contents are irrelevant to the time)
    ver.reset()
    ver.step(seq[:R], 0)
    ttnn.synchronize_device(mesh)
    ver.m.capture_trace()
    sp = {}
    for p in positions:
        ver.ring = ver.slot = 0
        toks = seq[:R]
        time.sleep(1.5)
        ts = []
        for _ in range(16):
            t1 = time.perf_counter()
            ver.step(toks, p)
            ver.m.argmax()
            ts.append(1e3 * (time.perf_counter() - t1))
        ts.sort()
        tb = time.perf_counter()
        for _ in range(16):
            ver.m.step(None)
        ttnn.synchronize_device(mesh)
        sp[p] = dict(host_loop_ms=ts[len(ts) // 2], device_ms=1e3 * (time.perf_counter() - tb) / 16)
        log(f"speed pos {p}: verify step {sp[p]['host_loop_ms']:.2f} ms with I/O, {sp[p]['device_ms']:.2f} ms device")
    out["speed"] = sp
    ver.m.release_trace()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["random", "real"])
    ap.add_argument("--prompt", type=int, default=96)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--prefix", type=int, default=48)
    ap.add_argument("--taps-out")
    ap.add_argument("--positions", default="128,2048,8192")
    ap.add_argument("--rows", default="8")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--interval", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--start", type=int, default=29)
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    mesh = open_mesh()
    res = []
    try:
        for R in [int(r) for r in a.rows.split(",")]:
            if a.mode == "random":
                res.append(run_random(mesh, R, a.layers, a.interval, a.steps, a.start))
            else:
                pos = [int(p) for p in a.positions.split(",")]
                res.append(run_real(mesh, R, a.prompt, a.gen, a.prefix, a.taps_out, pos))
    finally:
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1)
        close_mesh(mesh)


if __name__ == "__main__":
    main()
