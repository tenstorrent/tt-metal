# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""S0 hang probe: the served batch build (generator_vllm.allocate_kv_cache: 8 x RESIDENT_ROW_GROUPS rows, the hub feed,
a full KV pool, PP prefill at max_len) without vLLM. Prompts of --prompt tokens are prefilled one by one into slots
0, 1, ... (RowsEngine.prefill_whole: the PP chunk their length picks), each followed by its first-token launch and
--steps device-sampled decode steps of every live slot. With RESIDENT_KV_V_BFP4=1 RESIDENT_KV_V_RTN=1 the served
batch profile hung at 2K prompts; this prints a line per phase so a hang shows where.
usage: python -m qwen36_resident.tests.s0_g4_probe [--prompt 2048] [--users 4] [--steps 8] [--max-len 262144]
"""

import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.generator_vllm import PREFILL_CHUNKS, ROW_GROUPS
from qwen36_resident.model import Dims, ResidentModel, State
from qwen36_resident.prefill.pp_prefill import PPPrefill
from qwen36_resident.tests.long_ctx import close_mesh, open_mesh
from qwen36_resident.vllm_rows import RowsEngine


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/ttuser/experiments/qwen38/results/native/corpus_ids_40k.pt")
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--users", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=262144)
    ap.add_argument("--blocks", type=int, default=22000)
    a = ap.parse_args()
    ids = torch.load(a.corpus)["ids"]
    t_start = time.time()

    def phase(name):
        print(f"[{time.time() - t_start:7.1f} s] {name}", flush=True)

    mesh = open_mesh()
    ok = False
    try:
        ck = Q.Checkpoint(Q.checkpoint_dir())
        cfg = ck.config
        layers, interval = cfg["num_hidden_layers"], cfg["full_attention_interval"]
        d = Dims(mesh.get_num_devices(), mesh.dram_grid_size().x, vocab=cfg["vocab_size"])
        emb = Q.embedding(ck)
        C = max(PREFILL_CHUNKS)
        max_len = -(-a.max_len // C) * C
        pp = PPPrefill(mesh, ck, layers, interval, chunk=list(PREFILL_CHUNKS), max_len=max_len)
        n_slots = 8 * ROW_GROUPS
        n_attn = sum((l + 1) % interval == 0 for l in range(layers))
        sts = [State(d, layers - n_attn, n_attn, 0, max_pos=max_len + 64, zero=True, kv=False) for _ in range(n_slots)]
        m = ResidentModel(
            mesh,
            d,
            Q.load(ck, d, layers, interval),
            sts,
            layers,
            interval,
            lm_head=True,
            embed=pp.embed,
            kv_pool_blocks=a.blocks,
            block_tables=[[0]] * n_slots,
            rows=True,
            groups=ROW_GROUPS,
        )
        pp.handoff(m, 1, block_table=[0], slot=0)
        m.clear_slot(0)
        m.capture_trace()
        pp.capture()
        if m.sample:
            m.reset(state=False)
        eng = RowsEngine(m, pp, emb, cfg["vocab_size"], d.vocab_chip, n_slots, 1024, mesh=mesh)
        phase(f"built: {n_slots} slots (groups {ROW_GROUPS}), pool {a.blocks} blocks, chunks {list(PREFILL_CHUNKS)}")
        order = eng.row_order()
        bpu = -(-(a.prompt + a.steps * a.users + 64) // 64)
        tables = [[1 + u * bpu + b for b in range(bpu)] for u in range(n_slots)]
        live, pos, nxt = [], {}, {}
        eng.set_sampling({r: (0.0, 1, 1.0, 0) for r in range(n_slots)})
        for u in range(a.users):
            r = order[u]
            prompt = ids[3000 + u * 5000 : 3000 + u * 5000 + a.prompt].tolist()
            m.set_page_table([tables[s] if s in live or s == r else [0] for s in range(n_slots)])
            c = eng.pick_chunk(len(prompt) - 1)
            t0 = time.perf_counter()
            eng.prefill_whole(prompt[:-1], r, tables[r][: -(-len(prompt) // 64)], c)
            eng.sync()
            phase(
                f"user {u} (slot {r}): prefill {len(prompt)} tokens at chunk {c} ({1e3 * (time.perf_counter() - t0):.0f} ms)"
            )
            launches = sorted({s // 8 // ROW_GROUPS for s in live + [r]})
            lg = eng.first_tokens(
                {r: (prompt[-1], len(prompt) - 1)},
                [r // (8 * ROW_GROUPS)],
                preserve_launches=set(launches) if live else (),
            )
            nxt[r], pos[r] = int(lg[0].argmax()), len(prompt)
            live.append(r)
            phase(f"user {u}: first token {nxt[r]}")
            for k in range(a.steps):
                feed = {s: (nxt[s], pos[s]) for s in live}
                eng.decode(feed, [0])
                got = eng.tokens_from(eng.read_tokens(live, False))
                for s, tok in zip(live, got):
                    nxt[s], pos[s] = tok, pos[s] + 1
            phase(f"user {u}: {a.steps} decode steps of {len(live)} users done")
        ok = True
        print("s0 g4 probe: PASS (no hang)", flush=True)
    finally:
        close_mesh(mesh)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
