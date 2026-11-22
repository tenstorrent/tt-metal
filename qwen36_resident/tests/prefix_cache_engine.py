# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strata S1 option C end to end through the rows adapter's engine (vllm_rows.RowsEngine.prefill_whole with the
host-tier prefix cache): turn 1 (a P1-token prompt) misses and stores its last 1,024 boundary; turn 2 (the same prompt
extended to P2 tokens, another slot) hits it, prefills only the rest and stores its own boundary; turn 2's decode must
match a cache-free prefill of the same prompt (slot 0). The Qwen3.8 checkpoint, the PP prefill, a rows build of 8.
usage: python -m qwen36_resident.tests.prefix_cache_engine [--p1 2600] [--p2 3500] [--steps 32]
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
from qwen36_resident.vllm_rows import RowsEngine

B = 8


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt")
    ap.add_argument("--p1", type=int, default=2600)
    ap.add_argument("--p2", type=int, default=3500)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=4096)
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

        def prefill(slot, prompt, cached):
            eng.prefix_cache = cache if cached else None
            t0 = time.perf_counter()
            eng.prefill_whole(prompt[:-1], slot, tables[slot], eng.pick_chunk(len(prompt) - 1))
            ttnn.synchronize_device(mesh)
            return time.perf_counter() - t0

        cache = PrefixCache(20e9)
        eng.prefix_laps = []
        t_ref = prefill(0, p2, cached=False)
        ref_first, ref_toks = decode(0, p2)
        phase(f"slot 0: turn 2 without the cache (prefill {1e3 * t_ref:.0f} ms)")
        eng.prefix_laps.clear()
        t1 = prefill(1, p1, cached=True)
        print("  turn 1 parts (ms):", [(k, round(1e3 * v)) for k, v in eng.prefix_laps], flush=True)
        eng.prefix_laps.clear()
        phase(f"slot 1: turn 1, cache miss + store (prefill {1e3 * t1:.0f} ms); cache {dict(cache.stats)}")
        t2 = prefill(2, p2, cached=True)
        print("  turn 2 parts (ms):", [(k, round(1e3 * v)) for k, v in eng.prefix_laps], flush=True)
        got_first, got_toks = decode(2, p2)
        phase(f"slot 2: turn 2, cache hit (prefill {1e3 * t2:.0f} ms); cache {dict(cache.stats)}, {len(cache)} entries")
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
        print(f"prefix cache engine: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
