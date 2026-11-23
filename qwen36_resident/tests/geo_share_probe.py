# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DB3 phase 2, P2-0a: the 8 x 4 verify-rows build (ResidentDFlashRows: + PP with taps + drafter) and a plain 32-row
build in one process, the plain one adopting the first's weights, KV pool and persistent L1 scratch
(ResidentModel(share=)). Modes: vrows / plain (each alone, the references) / both. Per mode: the weight rings (logged
by the builds), the DRAM left (128 MB probes), traced step times (median of --steps, synchronized), and greedy tokens
(vrows: ResidentDFlashRows.greedy of 8 users through the PP; plain: 4 users, prompts fed token by token) printed as a
digest to compare across modes (both must equal the standalone runs).

usage: python -m qwen36_resident.tests.geo_share_probe --mode both [--max-pos 4096] [--new 32]
"""
import argparse
import faulthandler
import hashlib
import statistics
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.model import ATTN_TAILS, TOP_K, Dims, ResidentModel, State, is_attn
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh

CORPUS = "/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt"
KV_BYTES_PER_BLOCK = 2 * 278528


def free_dram(mesh):
    held, n = [], 0
    while True:
        try:
            held.append(ttnn.empty([1, 1, 8192, 8192], ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG))
            n += 1
        except RuntimeError:
            break
    for t in held:
        ttnn.deallocate(t)
    return n * (128 << 20)


def digest(tokens):
    return hashlib.sha1(str(tokens).encode()).hexdigest()[:12]


def timed(fn, sync, steps):
    ts = []
    for _ in range(steps):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts)


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["vrows", "plain", "both"], default="both")
    ap.add_argument("--max-pos", type=int, default=4096)
    ap.add_argument("--prompt-len", type=int, default=64)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    ids = torch.load(CORPUS)["ids"]
    mesh = open_mesh()
    sync = lambda: ttnn.synchronize_device(mesh)  # noqa: E731
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=cfg["vocab_size"])
        n_attn = sum(is_attn(l, interval) for l in range(layers))
        bpu = -(-(a.max_pos + 2 * 4 + 64) // 64) + 2
        blocks = 40 * bpu + 8  # (8 vrows users + 32 plain rows, each its own blocks)
        users, R, P = 8, 4, a.prompt_len
        out = {}
        eng = plain = None
        if a.mode in ("vrows", "both"):
            from qwen36_resident.dflash import ResidentDFlashRows

            union = dict(cand_words=d.n * 2 * TOP_K * 32, attn_tails=ATTN_TAILS) if a.mode == "both" else None
            eng = ResidentDFlashRows(
                mesh, ck, users=users, max_pos=a.max_pos, R=R, prefill=True, kv_pool_blocks=blocks, l1_union=union
            )
            eng.m.set_page_table([list(range(u * bpu, (u + 1) * bpu)) for u in range(users) for _ in range(R)])
            out["vrows_ring"] = eng.m.ring_bytes
        if a.mode in ("plain", "both"):
            if eng is not None:
                embed, share = eng.pp.embed, eng.m
            else:
                embed = ttnn.from_torch(
                    Q.embedding(ck).bfloat16(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                )
                share = None
            sts = [
                State(d, layers - n_attn, n_attn, 0, eng.max_pos if eng else a.max_pos + 72, zero=True, kv=False)
                for _ in range(32)
            ]
            t0 = time.time()
            plain = ResidentModel(
                mesh,
                d,
                Q.load(ck, d, layers, interval),
                sts,
                layers,
                interval,
                lm_head=True,
                embed=embed,
                kv_pool_blocks=blocks,
                block_tables=[[0]] * 32,
                rows=True,
                groups=4,
                share=share,
            )
            plain.set_page_table([list(range((8 + j) * bpu, (9 + j) * bpu)) for j in range(32)])
            plain.reset()
            plain.capture_trace()
            if eng is not None:  # (the PP traces replay L1 placed around the decode models live at capture: again)
                eng.pp.capture()
            print(f"plain build{' (shared)' if share else ''} in {time.time() - t0:.0f} s", flush=True)
            out["plain_ring"] = plain.ring_bytes
        sync()
        out["free_gb"] = round(free_dram(mesh) / 2**30, 2)
        emb = Q.embedding(ck).float()
        # ---- vrows: greedy of 8 users through the PP + its traced step time
        if eng is not None:
            prompts = [ids[1000 * u + 101 : 1000 * u + 101 + P].tolist() for u in range(users)]
            g = eng.greedy(prompts, a.new)
            out["vrows_digest"] = digest(g)
            pos = [P + a.new] * users
            out["vrows_step_ms"] = round(timed(lambda: eng._step([[0] * R] * users, pos), sync, a.steps), 2)
        # ---- plain: 4 users fed token by token, greedy; then 32 active rows' step time
        if plain is not None:
            order = plain.row_order()[:4]
            prompts = {r: ids[5000 + 1000 * k : 5000 + 1000 * k + P].tolist() for k, r in enumerate(order)}
            toks = {r: [] for r in order}
            plain.reset()
            for p in range(P + a.new - 1):
                feed = {r: (emb[prompts[r][p] if p < P else toks[r][-1]], p) for r in order}
                plain.step(plain.token_state_rows(feed))
                lg = plain.logits()
                for r in order:
                    if p >= P - 1:
                        toks[r].append(int(lg[r, :, : d.vocab_chip].reshape(-1)[: cfg["vocab_size"]].argmax()))
            out["plain_digest"] = digest([toks[r] for r in order])
            feed = {r: (emb[0], P + a.new) for r in range(32)}
            out["plain_step_ms"] = round(timed(lambda: plain.step(plain.token_state_rows(feed)), sync, a.steps), 2)
        if eng is not None and plain is not None:  # (both: the vrows greedy again after the plain runs)
            g2 = eng.greedy(
                prompts_v := [ids[1000 * u + 101 : 1000 * u + 101 + P].tolist() for u in range(users)], a.new
            )
            out["vrows_digest_after_plain"] = digest(g2)
        print(f"geo share probe ({a.mode}): {out}", flush=True)
    finally:
        close_mesh(mesh)


if __name__ == "__main__":
    main()
