# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strata S1 option C for split prompts (the batch profile splits a prompt over its gang cap into 1,024-token spans):
the adapter's _prefill_row with the host-tier prefix cache, spans as vLLM's chunk core sends them. Turn 1 (P1 tokens)
stores its final span's start; turn 2 (the prompt extended to P2, another vLLM slot) defers the spans the cached entry
covers, resumes from it (non-draining) and continues with run_span; its decode must match a cache-free split prefill.
usage: python -m qwen36_resident.tests.prefix_cache_split [--p1 12000] [--p2 14000] [--steps 32]
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
from qwen36_resident.prefix_cache import PrefixCache
from qwen36_resident.vllm_rows import ResidentRowsAdapter, RowsEngine

B = 8


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt")
    ap.add_argument("--p1", type=int, default=12000)
    ap.add_argument("--p2", type=int, default=14000)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=16384)
    a = ap.parse_args()
    ids = torch.load(a.corpus)["ids"]
    p1, p2 = ids[5000 : 5000 + a.p1].tolist(), ids[5000 : 5000 + a.p2].tolist()
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
        blocks, tables = shuffled_tables(B, max_pos, seed=31)
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
        eng = RowsEngine(m, pp, emb, vocab, d.vocab_chip, B, 1024, mesh=mesh)
        phase("models")

        def step(slot, tok, pos):
            x0 = torch.zeros(B, 5120)
            x0[slot] = emb[int(tok)].float()
            p = [0] * B
            p[slot] = pos
            m.step(m.token_state(x0, p, active=[s == slot for s in range(B)], preserve=True))
            ttnn.synchronize_device(mesh)
            return m.logits()[slot, :, : d.vocab_chip].reshape(-1)[:vocab].float().clone()

        def decode(slot, prompt):
            P = len(prompt)
            first = step(slot, prompt[-1], P - 1)
            toks, tok = [], int(first.argmax())
            for k in range(a.steps):
                toks.append(tok)
                tok = int(step(slot, tok, P + k).argmax())
            return first, toks

        ad = ResidentRowsAdapter(eng, vocab, max_pos, B, 1024)
        model_tables = [list(t) for t in tables]

        def prefill(vslot, prompt, cached):
            """the prompt's spans through the adapter (vLLM slot vslot, its block table tables[vslot]); returns the
            resident slot it landed in and the prefill time"""
            eng.prefix_cache = cache if cached else None
            ids, T, row = torch.tensor(prompt), len(prompt), torch.tensor(tables[vslot])
            t0 = time.perf_counter()
            for start in range(0, T, 1024):
                end = min(start + 1024, T)
                out = ad._prefill_row(ids[:end], row, vslot, start, end, end == T, T)
            ttnn.synchronize_device(mesh)
            r = out[0]
            model_tables[r] = list(tables[vslot])  # (the decode reads the resident slot's table)
            m.set_page_table(model_tables)
            return r, time.perf_counter() - t0

        cache = PrefixCache(20e9)
        r0, t_ref = prefill(0, p2, cached=False)
        ref_first, ref_toks = decode(r0, p2)
        phase(f"turn 2 without the cache (slot {r0}, split prefill {1e3 * t_ref:.0f} ms)")
        r1, t1 = prefill(1, p1, cached=True)
        phase(f"turn 1, cache store (slot {r1}, {1e3 * t1:.0f} ms); cache {dict(cache.stats)}")
        r2, t2 = prefill(2, p2, cached=True)
        got_first, got_toks = decode(r2, p2)
        phase(f"turn 2 via the cache (slot {r2}, {1e3 * t2:.0f} ms); cache {dict(cache.stats)}, {len(cache)} entries")
        agree = sum(int(x == y) for x, y in zip(got_toks, ref_toks))
        same_top = int(got_first.argmax()) == int(ref_first.argmax())
        diff = float((got_first - ref_first).abs().max())
        hit = cache.stats["hits"] == 1 and cache.stats["hit_tokens"] == (a.p1 - 1) // 1024 * 1024
        ok = same_top and agree >= 0.9 * a.steps and hit
        print(
            f"turn 2 via the cache: hit {hit}, first top-1 equal {same_top} (max |diff| {diff:.3f}), greedy "
            f"{agree}/{a.steps}",
            flush=True,
        )
        print(f"prefix cache split: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
