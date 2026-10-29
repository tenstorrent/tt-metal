# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rows mode against the B = 1 build on the Qwen3.8 checkpoint, all teacher-forced on the same text.

One PP prefill model; in this order:
  B=1        a B = 1 paged build (fresh): user A's prompt through PP prefill + handoff, then `steps` + 1 decode steps
             fed the text's next tokens (positions P - 1 ...), logits kept;
  rows       a B = 8 rows build: A in slot `slot` (PP prefill, handoff into the slot, first-token launch with only A
             active), the same teacher-forced steps, alone and then beside 7 users prefilled into the other slots
             (their own greedy tokens);
  B=1 again  a second B = 1 build after the rows build, the same run.
Reports: rows alone vs beside neighbours (must be bit-identical), B=1 vs B=1 again (must be bit-identical: the builds
must not leak into each other), rows vs B=1 top-1 agreement with the margins of every disagreement, and each run's
top-1 agreement with the text.

usage: python -m qwen36_resident.tests.rows_check --corpus corpus_ids.pt [--prompt-len 2000] [--steps 32] [--slot 3]
"""

import argparse
import faulthandler
import gc
import weakref
import json
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State, is_attn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.tests.e0_bringup import PREFILL_CHUNKS, close_mesh, open_mesh, shuffled_tables

B = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--prompt-len", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--slot", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    ids = torch.load(a.corpus)["ids"]
    t_start = time.time()

    def phase(name):
        print(f"[{time.time() - t_start:7.1f} s] {name}", flush=True)

    mesh = open_mesh()
    out = {}
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
        n = mesh.get_num_devices()
        d = Dims(n, mesh.dram_grid_size().x, vocab=vocab)
        emb = Q.embedding(ck)
        C = max(PREFILL_CHUNKS)
        max_len = -(-a.max_len // C) * C
        max_pos = max_len + 64
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
        phase("prefill model")
        empty = lambda: State(d, layers - n_attn, n_attn, 0, max_pos=max_pos, zero=True, kv=False)
        P, S = a.prompt_len, a.steps
        prompt = ids[:P]
        feed = [int(ids[P - 1 + k]) for k in range(S + 1)]  # step k's input, at position P - 1 + k
        text = [int(ids[P + k]) for k in range(S + 1)]  # what step k should predict
        captured = [False]

        def l1_addrs(m):
            """the model's persistent L1 tensors' addresses (a later build placed lower means L1 still held)"""
            names = ("o_in_t", "logits_t", "argmax_t")
            return {k: getattr(m, k).buffer_address() for k in names if getattr(m, k, None) is not None}

        def gone(ref, name):
            """after the caller dropped its names: the model must be collected; else name what holds it"""
            gc.collect()
            ttnn.synchronize_device(mesh)
            if ref() is None:
                print(f"{name}: freed", flush=True)
                return
            holders = [f"{type(r).__name__}:{str(r)[:120]}" for r in gc.get_referrers(ref()) if r is not ref]
            print(f"{name}: NOT freed, held by {holders}", flush=True)

        def b1_run():
            blocks, tables = shuffled_tables(1, max_pos, seed=9)
            m = ResidentModel(
                mesh,
                d,
                Q.load(ck, d, layers, interval),
                empty(),
                layers,
                interval,
                lm_head=True,
                kv_pool_blocks=blocks,
                block_tables=tables,
            )
            print(f"B=1 build L1 addresses {l1_addrs(m)}", flush=True)
            m.step(m.token_state(emb[0].float(), 0))  # compile before the trace
            ttnn.synchronize_device(mesh)
            m.reset()
            m.capture_trace()
            if not captured[0]:
                pp.capture()
                captured[0] = True
            pp.reset()
            pp.run(prompt[: P - 1])
            pp.handoff(m, P - 1)
            lg = []
            for k, tok in enumerate(feed):
                m.step(m.token_state(emb[tok].float(), P - 1 + k))
                ttnn.synchronize_device(mesh)
                lg.append(m.logits()[:, : d.vocab_chip].reshape(-1)[:vocab].float().clone())
            ref = weakref.ref(m)
            m.release_trace()
            pp._ho.pop("model", None)  # the handoff's job cache holds the model
            del m
            gone(ref, "B=1 model")
            return torch.stack(lg)

        l1 = b1_run()
        phase("B=1 (fresh): teacher-forced")

        # ---------------- rows build
        blocks, tables = shuffled_tables(B, max_pos, seed=11)
        m = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            [empty() for _ in range(B)],
            layers,
            interval,
            lm_head=True,
            kv_pool_blocks=blocks,
            block_tables=tables,
            rows=True,
        )
        print(f"rows build L1 addresses {l1_addrs(m)}", flush=True)
        m.capture_trace()
        phase("rows model")

        def rows_logits(slot):
            lg = m.logits()  # [8, n, vocab_chip]
            return lg[slot, :, : d.vocab_chip].reshape(-1)[:vocab].float().clone()

        def launch(rows, preserve):
            x0 = torch.zeros(B, 5120)
            pos, active = [0] * B, [False] * B
            for s, (tok, p) in rows.items():
                x0[s], pos[s], active[s] = emb[int(tok)].float(), int(p), True
            m.step(m.token_state(x0, pos, active=active, preserve=preserve))
            ttnn.synchronize_device(mesh)

        def join(slot, pr, live):
            T = len(pr)
            pp.reset()
            pp.run(pr[: T - 1])
            pp.handoff(m, T - 1, slot=slot)
            launch({slot: (int(pr[-1]), T - 1)}, preserve=bool(live))
            return rows_logits(slot)

        others = {s: ids[(s + 1) * 4096 : (s + 1) * 4096 + 1500 - 97 * s] for s in range(B) if s != a.slot}

        def rows_run(with_neighbours):
            m.reset()
            cur = {}
            if with_neighbours:
                for s, pr in others.items():
                    cur[s] = (int(join(s, pr, cur).argmax()), len(pr))
            lg = [join(a.slot, prompt, cur)]
            for k in range(1, S + 1):
                cur[a.slot] = (feed[k], P - 1 + k)
                launch(cur, preserve=False)
                full = m.logits()
                lg.append(full[a.slot, :, : d.vocab_chip].reshape(-1)[:vocab].float().clone())
                for s in cur:
                    if s != a.slot:
                        cur[s] = (int(full[s, :, : d.vocab_chip].reshape(-1)[:vocab].argmax()), cur[s][1] + 1)
            return torch.stack(lg)

        lr = rows_run(False)
        phase(f"rows, A alone in slot {a.slot}: teacher-forced")
        lr_nb = rows_run(True)
        phase(f"rows, A in slot {a.slot} beside 7 users: teacher-forced")
        ref = weakref.ref(m)
        m.release_trace()
        pp._ho.pop("model", None)
        del m
        gone(ref, "rows model")

        l1b = b1_run()
        phase("B=1 again, after the rows build: teacher-forced")

        def top1_text(lg):
            return sum(int(lg[k].argmax()) == text[k] for k in range(len(text)))

        same_nb = sum(torch.equal(x, y) for x, y in zip(lr, lr_nb))
        same_b1 = sum(torch.equal(x, y) for x, y in zip(l1, l1b))
        agree, gaps = 0, []
        for k in range(S + 1):
            r, b = int(lr[k].argmax()), int(l1[k].argmax())
            agree += r == b
            if r != b:
                gaps.append(
                    dict(step=k, rows=r, b1=b, b1_gap=float(l1[k][b] - l1[k][r]), rows_gap=float(lr[k][r] - lr[k][b]))
                )
        print(f"rows alone vs beside 7 users: bit-identical at {same_nb}/{S + 1} steps", flush=True)
        print(f"B=1 fresh vs B=1 after the rows build: bit-identical at {same_b1}/{S + 1} steps", flush=True)
        print(f"rows vs B=1: top-1 equal at {agree}/{S + 1} steps; disagreements {gaps}", flush=True)
        print(
            f"top-1 vs text: rows {top1_text(lr)}/{S + 1}, B=1 {top1_text(l1)}/{S + 1}, B=1 again {top1_text(l1b)}/{S + 1}",
            flush=True,
        )
        out = dict(
            rows_alone_vs_neighbours=same_nb,
            b1_vs_b1_again=same_b1,
            rows_vs_b1_top1=agree,
            of=S + 1,
            disagreements=gaps,
            top1_vs_text=dict(rows=top1_text(lr), b1=top1_text(l1), b1_again=top1_text(l1b)),
        )
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
