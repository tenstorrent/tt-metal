# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-user rows mode on the Qwen3.8 checkpoint (1 x 4 mesh): correctness and step time.

Correctness. A rows-mode model (B = 8, paged KV, shuffled block tables) serves user A, whose prompt goes through the
PP prefill, the handoff into its slot and a first-token launch with only A active.
- A decodes `steps` greedy tokens (host argmax, host-written token states): alone, then beside 7 users prefilled into
  the other slots. A's logits must be bit-identical in both runs.
- The same prompt then runs on a B = 1 paged build. Its greedy tokens are compared with the rows run; they are not
  expected to be bit-identical, because B = 1 runs another rmsnorm path.

Step time. Traced replays of fixed token states (no feed, no host work between replays):
- the B = 8 rows build with 1 and with 8 active rows;
- the B = 1 build;
at each context (positions T - 1; the KV pages hold the prefill's or zeros, the time does not depend on them).

usage: python -m qwen36_resident.tests.rows_real --corpus corpus_ids.pt [--prompt-len 2000] [--steps 32]
           [--contexts 128,4096,32000] [--out rows_real.json]
"""
import argparse
import faulthandler
import gc
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
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--prompt-len", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--contexts", default="128,4096,32000")
    ap.add_argument("--time-steps", type=int, default=64)
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    ids = torch.load(a.corpus)["ids"]
    contexts = [int(t) for t in a.contexts.split(",")]
    t_start = time.time()

    def phase(name):
        print(f"[{time.time() - t_start:7.1f} s] {name}", flush=True)

    mesh = open_mesh()
    out = dict(step_ms={})
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
        blocks, tables = shuffled_tables(B, max_pos, seed=9)
        empty = lambda: State(d, layers - n_attn, n_attn, 0, max_pos=max_pos, zero=True, kv=False)

        def logits_rows(model):
            lg = model.logits()  # [8, n, vocab_chip]
            return lg[:, :, : d.vocab_chip].reshape(B, -1)[:, :vocab].float().clone()

        # ---------------- rows build
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
        phase(f"rows decode model (pool {blocks} blocks)")
        m.capture_trace()
        pp.capture()
        phase("traces")

        def step_rows(feed):
            """feed: slot -> (token, pos); the other slots inactive. Returns [8, vocab] logits"""
            x0 = torch.zeros(B, 5120)
            pos, active = [0] * B, [False] * B
            for s, (tok, p) in feed.items():
                x0[s], pos[s], active[s] = emb[int(tok)].float(), int(p), True
            m.step(m.token_state(x0, pos, active=active, preserve=feed.get("_preserve", False)))
            ttnn.synchronize_device(mesh)
            return logits_rows(m)

        def join(slot, prompt, live):
            """PP prefill + handoff into slot, then its first-token launch (preserve when others are live)"""
            T = len(prompt)
            pp.reset()
            pp.run(prompt[: T - 1])
            pp.handoff(m, T - 1, slot=slot)
            x0 = torch.zeros(B, 5120)
            x0[slot] = emb[int(prompt[-1])].float()
            pos = [0] * B
            pos[slot] = T - 1
            active = [s == slot for s in range(B)]
            m.step(m.token_state(x0, pos, active=active, preserve=bool(live)))
            ttnn.synchronize_device(mesh)
            return logits_rows(m)[slot]

        A_slot, P = 3, a.prompt_len
        prompt_A = ids[:P]
        prompts = {s: ids[(s + 1) * 4096 : (s + 1) * 4096 + P - 97 * s] for s in range(B) if s != A_slot}

        def decode_A(with_neighbours):
            m.reset()
            live, cur = {}, {}
            if with_neighbours:
                for s, pr in prompts.items():
                    lg = join(s, pr, live)
                    live[s] = True
                    cur[s] = (int(lg.argmax()), len(pr))
            lg = join(A_slot, prompt_A, live)
            live[A_slot] = True
            cur[A_slot] = (int(lg.argmax()), P)
            seq, toks = [lg], [cur[A_slot][0]]
            for _ in range(a.steps):
                L = step_rows(cur)
                for s in cur:
                    cur[s] = (int(L[s].argmax()), cur[s][1] + 1)
                seq.append(L[A_slot])
                toks.append(cur[A_slot][0])
            return torch.stack(seq), toks

        lg_alone, tok_alone = decode_A(False)
        phase(f"A alone: {a.steps} greedy steps")
        lg_nb, tok_nb = decode_A(True)
        phase(f"A with 7 neighbours: {a.steps} greedy steps")
        same = sum(torch.equal(x, y) for x, y in zip(lg_alone, lg_nb))
        print(
            f"rows: A's logits alone vs with 7 neighbours bit-identical at {same}/{len(lg_alone)} steps "
            f"(max |diff| {float((lg_alone - lg_nb).abs().max())})",
            flush=True,
        )
        out["alone_vs_neighbours"] = dict(identical=same, of=len(lg_alone))

        # step time: traced replays of fixed token states
        def time_replays(model, tok_host, launches=None):
            model.step(tok_host, launches=launches) if launches is not None else model.step(tok_host)
            for _ in range(8):
                model.step(None, launches=launches) if launches is not None else model.step(None)
            ttnn.synchronize_device(mesh)
            t0 = time.time()
            for _ in range(a.time_steps):
                model.step(None, launches=launches) if launches is not None else model.step(None)
            ttnn.synchronize_device(mesh)
            return (time.time() - t0) / a.time_steps * 1e3

        for T in contexts:
            for act in (1, 8):
                x0 = torch.zeros(B, 5120)
                for s in range(act):
                    x0[s] = emb[int(ids[T - 1 + s])].float()
                tok = m.token_state(x0, [T - 1] * B, active=[s < act for s in range(B)])
                out["step_ms"][f"B8 rows, {act} active, T={T}"] = ms = time_replays(m, tok, launches=[0])
                phase(f"B=8 rows build, {act} active, T={T}: step {ms:.3f} ms")
        m.release_trace()
        pp._ho.pop("model", None)
        del m
        gc.collect()
        ttnn.synchronize_device(mesh)

        # ---------------- B = 1 paged build (same prompt, greedy; then step time)
        b1_blocks, b1_tables = shuffled_tables(1, max_pos, seed=9)
        m1 = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            empty(),
            layers,
            interval,
            lm_head=True,
            kv_pool_blocks=b1_blocks,
            block_tables=b1_tables,
        )
        m1.step(m1.token_state(emb[0].float(), 0))  # compile before the trace
        ttnn.synchronize_device(mesh)
        m1.reset()
        m1.capture_trace()
        phase("B=1 decode model")
        m1.reset()
        pp.reset()
        pp.run(prompt_A[: P - 1])
        pp.handoff(m1, P - 1)
        toks1, tok, p = [], int(prompt_A[-1]), P - 1
        for _ in range(a.steps + 1):
            m1.step(m1.token_state(emb[tok].float(), p))
            ttnn.synchronize_device(mesh)
            tok, p = int(m1.logits()[:, : d.vocab_chip].reshape(-1)[:vocab].argmax()), p + 1
            toks1.append(tok)
        agree = sum(x == y for x, y in zip(toks1, tok_alone))
        first_diff = next((i for i, (x, y) in enumerate(zip(toks1, tok_alone)) if x != y), None)
        print(
            f"rows (B=8) vs B=1 greedy tokens: {agree}/{len(toks1)} equal, first difference at step {first_diff}",
            flush=True,
        )
        out["rows_vs_b1_tokens"] = dict(equal=agree, of=len(toks1), first_diff=first_diff)
        for T in contexts:
            out["step_ms"][f"B1, T={T}"] = ms = time_replays(m1, m1.token_state(emb[int(ids[T - 1])].float(), T - 1))
            phase(f"B=1 build, T={T}: step {ms:.3f} ms")
        m1.release_trace()
        pp._ho.pop("model", None)
        del m1
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
