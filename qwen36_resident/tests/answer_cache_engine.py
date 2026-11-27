# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strata S4-1 on the real engine (vllm_rows.RowsEngine cross_snapshot / answer_store, ResidentModel.snapshot_state /
entry_from): slot 0 prefills a prompt through the prefix cache (its boundary stored), decodes greedy past the next
1,024 boundary (its state snapshotted to host there), and finishes: an entry through the answer at that boundary, its
KV the prompt entry's pieces + the pool's past them.

  mechanics   the entry restored into slot 2 (other blocks) and fed the same tokens from the boundary: logits
              bit-identical to slot 0's (the same state, the same KV bytes, the same kernels)
  next turn   prompt = slot 0's sequence + a question (corpus tokens): prefilled through the cache (a hit at the
              answer's boundary, the PP prefill from there) vs without the cache (the whole prompt by the PP prefill):
              first-token logits close and greedy tokens compared — not bit-identical (the answer's positions were
              computed by decode steps instead of the PP prefill), the level of a near-tie
usage: python -m qwen36_resident.tests.answer_cache_engine [--prompt 3000] [--answer 120] [--question 300]
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
from qwen36_resident.prefix_cache import PrefixCache
from qwen36_resident.tests.e0_bringup import shuffled_tables
from qwen36_resident.tests.long_ctx import close_mesh, open_mesh
from qwen36_resident.vllm_rows import ANSWER_BLOCK, RowsEngine

B = 8


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt")
    ap.add_argument("--prompt", type=int, default=3000)
    ap.add_argument("--answer", type=int, default=120)
    ap.add_argument("--question", type=int, default=300)
    ap.add_argument("--greedy", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=4096)
    a = ap.parse_args()
    ids = torch.load(a.corpus)["ids"]
    prompt = ids[5000 : 5000 + a.prompt].tolist()
    question = ids[30000 : 30000 + a.question].tolist()
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
        blocks, tables = shuffled_tables(B, max_pos, seed=41)
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
        pc = eng.prefix_cache = PrefixCache(int(40e9))
        eng.answer = True
        phase("models")

        def step(slot, tok, pos):
            x0 = torch.zeros(B, 5120)
            x0[slot] = emb[int(tok)].float()
            p = [0] * B
            p[slot] = pos
            m.step(m.token_state(x0, p, active=[s == slot for s in range(B)], preserve=True))
            ttnn.synchronize_device(mesh)
            return m.logits()[slot, :, : d.vocab_chip].reshape(-1)[:vocab].float().clone()

        def prefill(slot, toks):
            t0 = time.perf_counter()
            eng.prefill_whole(toks[:-1], slot, tables[slot], eng.pick_chunk(len(toks) - 1))
            ttnn.synchronize_device(mesh)
            return time.perf_counter() - t0

        # turn 1: slot 0 prefills (the prompt's boundary stored), decodes past the next boundary, finishes
        P = len(prompt)
        t_p = prefill(0, prompt)
        H0 = eng.entry_of[0][0]
        phase(f"slot 0: prompt {P} prefilled ({1e3 * t_p:.0f} ms), its entry at {H0}")
        seq, lgs0 = list(prompt), {}
        lg = step(0, seq[-1], P - 1)
        if P % ANSWER_BLOCK == 0:
            eng.cross_snapshot(0, P)
        for k in range(a.answer):
            seq.append(int(lg.argmax()))
            p = len(seq) - 1
            lg = step(0, seq[-1], p)
            lgs0[p] = lg
            if (p + 1) % ANSWER_BLOCK == 0:
                eng.cross_snapshot(0, p + 1)
        seq.append(int(lg.argmax()))  # (the last token, not run: the request's finish)
        Bnd = eng.cross[0][0]
        st = eng.answer_stats
        phase(f"slot 0: {a.answer} tokens decoded, boundary state at {Bnd} ({st['cross_ms']:.0f} ms)")
        t0 = time.perf_counter()
        kept = eng.answer_store(0, seq, tables[0])
        t_store = time.perf_counter() - t0
        eng.forget(0)
        h, snap = pc.lookup(seq[: Bnd + 1])
        n_pieces = len(snap["kv_host"]) if snap else 0
        phase(f"entry at {Bnd}: kept {kept} ({1e3 * t_store:.0f} ms), lookup -> {h}, {n_pieces} KV pieces")
        assert kept and h == Bnd, (kept, h, Bnd)

        # mechanics: the entry into slot 2, the same tokens from the boundary -> bit-identical logits
        m.restore_slot(2, m.snapshot_from_host(snap), table=tables[2])
        same, diff = True, 0.0
        for p in range(Bnd, len(seq) - 1):
            lg2 = step(2, seq[p], p)
            same &= torch.equal(lg2, lgs0[p])
            diff = max(diff, float((lg2 - lgs0[p]).abs().max()))
        phase(
            f"mechanics: {len(seq) - 1 - Bnd} steps from the restored entry, bit-identical {same} (max |diff| {diff})"
        )

        # next turn: through the cache (hit at the answer's boundary) vs without
        turn2 = seq + question
        t_hit = prefill(3, turn2)
        hits = pc.stats["hit_tokens"]
        lg_hit = step(3, turn2[-1], len(turn2) - 1)
        cache, eng.prefix_cache = eng.prefix_cache, None
        t_ref = prefill(4, turn2)
        eng.prefix_cache = cache
        lg_ref = step(4, turn2[-1], len(turn2) - 1)
        g_hit, g_ref = [int(lg_hit.argmax())], [int(lg_ref.argmax())]
        for k in range(a.greedy - 1):
            n = len(turn2) + k
            g_hit.append(int(step(3, g_hit[-1], n).argmax()))
            g_ref.append(int(step(4, g_ref[-1], n).argmax()))
        agree = next((k for k, (x, y) in enumerate(zip(g_hit, g_ref)) if x != y), len(g_hit))
        dlg = float((lg_hit - lg_ref).abs().max())
        top = (lg_ref.max() - lg_ref.topk(2).values[1]).item()
        phase(
            f"next turn ({len(turn2)} tokens): through the answer entry {1e3 * t_hit:.0f} ms vs uncached "
            f"{1e3 * t_ref:.0f} ms; first logits max |diff| {dlg:.3f} (ref top-1 margin {top:.3f}), argmax equal "
            f"{g_hit[0] == g_ref[0]}, greedy prefix equal {agree}/{len(g_hit)}"
        )
        ok = same and kept and g_hit[0] == g_ref[0]
        print(f"answer cache engine: {'PASS' if ok else 'FAIL'}", flush=True)
    finally:
        close_mesh(mesh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
