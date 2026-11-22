# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strata S1: a prompt's prefill resumed from a cached prefix (PPPrefill.load_prefix / run_from) hands the slot the
same state as a whole-prompt prefill (the Qwen3.8 checkpoint, the PP prefill and a rows decode build of 8 slots).

A corpus prompt of P tokens, prefix h (a multiple of 1,024):
  slot 0  reference: the whole prompt's prefill (P - 1 tokens) and handoff;
  slot 1  resume: the prefix's prefill and handoff into slot 1, then load_prefix from slot 1, run_from h and handoff;
  slot 2  through the host tier: slot 1's prefix state (after its prefix handoff) snapshotted to host memory and restored
          into slot 2 (its own blocks), then load_prefix from slot 2, run_from h and handoff.
Each slot then takes the last prompt token and decodes greedily alone; reported: the first logits' top-1 / max |diff|
vs slot 0 and how many greedy tokens agree. Not bit-identical by design (the prefix's pipeline drain rounds the GDN
state differently from an uninterrupted run); the gate: the first token equal and >= 90% of the greedy tokens.

usage: python -m qwen36_resident.tests.pp_resume --corpus corpus_ids.pt [--prompt-len 2600] [--prefix 2048] [--steps 32]
"""
import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.generator_vllm import PREFILL_CHUNKS
from qwen36_resident.model import Dims, ResidentModel, State, is_attn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.tests.e0_bringup import shuffled_tables
from qwen36_resident.tests.long_ctx import close_mesh, open_mesh

B = 8


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt")
    ap.add_argument("--prompt-len", type=int, default=2600)
    ap.add_argument("--prefix", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    P, h = a.prompt_len, a.prefix
    assert h % 1024 == 0 and h < P - 1
    ids = torch.load(a.corpus)["ids"]
    prompt = ids[3000 : 3000 + P]
    t_start = time.time()

    def phase(name):
        print(f"[{time.time() - t_start:7.1f} s] {name}", flush=True)

    mesh = open_mesh()
    ok = False
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval, vocab = cfg["num_hidden_layers"], cfg["full_attention_interval"], cfg["vocab_size"]
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=vocab)
        emb = Q.embedding(ck)
        C = max(PREFILL_CHUNKS)
        max_len = -(-a.max_len // C) * C
        max_pos = max_len + 64
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
        phase("prefill model")
        blocks, tables = shuffled_tables(B, max_pos, seed=21)
        m = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            [State(d, layers - n_attn, n_attn, 0, max_pos=max_pos, zero=True, kv=False) for _ in range(B)],
            layers,
            interval,
            lm_head=True,
            kv_pool_blocks=blocks,
            block_tables=tables,
            rows=True,
        )
        m.capture_trace()
        pp.capture()
        m.reset()
        phase("rows model")

        def step(slot, tok, pos):
            x0 = torch.zeros(B, 5120)
            x0[slot] = emb[int(tok)].float()
            p = [0] * B
            p[slot] = pos
            m.step(m.token_state(x0, p, active=[s == slot for s in range(B)], preserve=True))
            ttnn.synchronize_device(mesh)
            return m.logits()[slot, :, : d.vocab_chip].reshape(-1)[:vocab].float().clone()

        def decode(slot):
            first = step(slot, prompt[-1], P - 1)
            toks, tok = [], int(first.argmax())
            for k in range(a.steps):
                toks.append(tok)
                tok = int(step(slot, tok, P + k).argmax())
            return first, toks

        # slot 0: the whole prompt
        pp.reset()
        pp.run(prompt[: P - 1])
        pp.handoff(m, P - 1, slot=0)
        ref_first, ref_toks = decode(0)
        phase("slot 0: whole-prompt prefill + decode")
        # slot 1: the prefix, then resumed from slot 1
        pp.reset()
        pp.run(prompt[:h])
        pp.handoff(m, h, slot=1)
        snap = m.snapshot_to_host(m.snapshot_slot(1, n_tokens=h))  # (slot 2's prefix, via the host tier)
        t0 = time.perf_counter()
        pp.load_prefix(m, h, slot=1)
        t1 = time.perf_counter()
        pp.run_from(prompt[: P - 1], h)
        pp.handoff(m, P - 1, slot=1)
        r1_first, r1_toks = decode(1)
        phase(f"slot 1: prefix + load_prefix ({1e3 * (t1 - t0):.0f} ms) + run_from + decode")
        # slot 2: the prefix snapshot from the host into slot 2's own blocks, then resumed
        m.restore_slot(2, m.snapshot_from_host(snap))
        pp.load_prefix(m, h, slot=2)
        pp.run_from(prompt[: P - 1], h)
        pp.handoff(m, P - 1, slot=2)
        r2_first, r2_toks = decode(2)
        phase("slot 2: host snapshot -> restore -> load_prefix + run_from + decode")
        results = []
        for name, first, toks in (("resume", r1_first, r1_toks), ("host tier", r2_first, r2_toks)):
            agree = sum(int(x == y) for x, y in zip(toks, ref_toks))
            same_top = int(first.argmax()) == int(ref_first.argmax())
            diff = float((first - ref_first).abs().max())
            results.append(same_top and agree >= 0.9 * a.steps)
            print(
                f"{name}: first top-1 equal {same_top} (max |diff| {diff:.3f}), greedy tokens equal {agree}/{a.steps}",
                flush=True,
            )
        ok = all(results)
        print(f"pp resume: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
