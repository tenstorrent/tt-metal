# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paged KV against contiguous KV on the Qwen3.8 checkpoint (1 x 4 mesh).

For each KV mode (contiguous, then paged with a shuffled block table drawn from a larger pool) the decode model is
built next to one PP prefill model; per context T: PP prefill of T - 1 tokens, handoff (paged: into the pool pages
of the block table), then `steps` teacher-forced decode steps (tokens of a natural-text corpus) whose full logits
are kept. The two modes' logits must be bit-identical at every step. Then the traced greedy decode TPOT (the hub
feeding itself, B = 1) of each mode at a few contexts.

usage: python -m qwen36_resident.tests.paged_real --corpus corpus_ids.pt [--contexts 4096,32000] [--steps 64]
           [--tpot 128,4096,32000] [--out paged_real.json]
"""

import argparse
import faulthandler
import gc
import hashlib
import json
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import Dims, ResidentModel, State, is_attn
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.tests.e0_bringup import PREFILL_CHUNKS, close_mesh, open_mesh, shuffled_tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="{'ids': int64 [N]} natural-text token ids")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--contexts", default="4096,32000")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--tpot", default="128,4096,32000")
    ap.add_argument("--tpot-steps", type=int, default=128)
    ap.add_argument("--tpot-reps", type=int, default=3)
    ap.add_argument("--modes", default="contiguous,paged")
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(300, repeat=True, file=sys.stderr)
    ids = torch.load(a.corpus)["ids"]
    contexts = [int(t) for t in a.contexts.split(",")]
    tpot_ctx = [int(t) for t in a.tpot.split(",")]
    t_start = time.time()

    def phase(name):
        print(f"[{time.time() - t_start:7.1f} s] {name}", flush=True)

    mesh = open_mesh()
    out = dict(modes={}, identical={})
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
        assert max(contexts) + a.steps <= max_pos and max(tpot_ctx) + 8 + a.tpot_steps * a.tpot_reps <= max_pos
        pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
        phase("prefill model")
        blocks, tables = shuffled_tables(1, max_pos, seed=7)
        logits = {}
        captured = False
        runs = []  # (run name, KV mode): a mode named again runs again as "mode#2", ...
        for mode in a.modes.split(","):
            k = sum(m == mode for _, m in runs)
            runs.append((mode if k == 0 else f"{mode}#{k + 1}", mode))
        for run, mode in runs:
            st = State(d, layers - n_attn, n_attn, 0, max_pos=max_pos, zero=True, kv=False)
            w = Q.load(ck, d, layers, interval)
            kw = dict(kv_pool_blocks=blocks, block_tables=tables) if mode == "paged" else {}
            model = ResidentModel(mesh, d, w, st, layers, interval, lm_head=True, embed=pp.embed, **kw)
            del w, st
            phase(f"{run} decode model" + (f" (pool {blocks} blocks, table {tables[0][:6]}...)" if kw else ""))
            model.step(model.token_state(emb[0].float(), 0))
            ttnn.synchronize_device(mesh)
            model.reset()
            model.capture_trace()
            if not captured:
                pp.capture()
                captured = True
            phase(f"{run} traces")

            def prefill(prompt):
                model.reset()
                pp.reset()
                T = len(prompt)
                pp.run(prompt[: T - 1])
                pp.handoff(model, T - 1)

            def logits_at(token, pos):
                model.step(model.token_state(emb[int(token)].float(), int(pos)))
                ttnn.synchronize_device(mesh)
                return model.logits()[:, : d.vocab_chip].reshape(-1)[:vocab].float().clone()

            res = dict(steps={}, tpot={})
            for T in contexts:
                prefill(ids[:T])
                lg = [logits_at(ids[T - 1 + k], T - 1 + k) for k in range(a.steps)]
                logits[(run, T)] = torch.stack(lg)
                res["steps"][T] = [
                    dict(
                        pos=T - 1 + k,
                        sha=hashlib.sha256(x.numpy().tobytes()).hexdigest()[:16],
                        top=x.topk(5).indices.tolist(),
                        hit=int(x.argmax()) == int(ids[T + k]),
                    )
                    for k, x in enumerate(lg)
                ]
                hits = sum(s["hit"] for s in res["steps"][T])
                phase(f"{run} T={T}: {a.steps} teacher-forced steps, top-1 vs text {hits}/{a.steps}")
            for T in tpot_ctx:
                prefill(ids[:T])
                model.set_sampling(0.0, 1, 1.0, 0)
                model.step(model.token_state(emb[int(ids[T - 1])].float(), T - 1))
                for _ in range(8):
                    model.step(None)
                ttnn.synchronize_device(mesh)
                reps = []
                for _ in range(a.tpot_reps):
                    t0 = time.time()
                    for _ in range(a.tpot_steps):
                        model.step(None)
                    ttnn.synchronize_device(mesh)
                    reps.append((time.time() - t0) / a.tpot_steps * 1e3)
                res["tpot"][T] = reps
                phase(f"{run} T={T}: TPOT {' / '.join(f'{r:.3f}' for r in reps)} ms")
            out["modes"][run] = res
            model.release_trace()
            pp._ho.pop("model", None)  # the handoff's job cache holds the model
            del model
            gc.collect()
            ttnn.synchronize_device(mesh)
        if "contiguous" in out["modes"]:
            # every other run's logits against the first contiguous run's
            for run, _ in runs[1:] if runs[0][0] == "contiguous" else runs:
                if run == "contiguous":
                    continue
                for T in contexts:
                    c, p = logits[("contiguous", T)], logits[(run, T)]
                    same = [bool(torch.equal(c[k], p[k])) for k in range(a.steps)]
                    diff = float((c - p).abs().max())
                    out["identical"][f"{run} T={T}"] = dict(steps=sum(same), of=a.steps, max_abs_diff=diff)
                    print(
                        f"T={T}: {run} vs contiguous logits bit-identical at {sum(same)}/{a.steps} steps "
                        f"(max |diff| {diff})",
                        flush=True,
                    )
            med = lambda r, T: sorted(out["modes"][r]["tpot"][T])[a.tpot_reps // 2]
            for T in tpot_ctx:
                by = {m: [med(r, T) for r, rm in runs if rm == m] for m in ("contiguous", "paged")}
                if by["contiguous"] and by["paged"]:
                    tc, tp = sum(by["contiguous"]) / len(by["contiguous"]), sum(by["paged"]) / len(by["paged"])
                    print(
                        f"T={T}: TPOT (median per run) contiguous {by['contiguous']} ms, paged {by['paged']} ms: "
                        f"paged {100 * (tp / tc - 1):+.2f}% vs the contiguous mean",
                        flush=True,
                    )
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
