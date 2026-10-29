# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device gate (WP8): a prompt prefilled in chunks across calls, with live users decoding between the chunks, hands
the same state to its slot as one PP run of the same prompt (review item 4: x_last, S, carry, the KV scratch and the
shift semaphores of the PP pipeline must survive the rows-mode decode launches run in between).

The serving build (Qwen36ResidentRowsForCausalLM: rows-mode model, paged pool, traces, PP prefill) with 9 live users
(resident slots 0-8, launches 0 and 1) decoding greedily through the adapter, and a target prompt of --prompt-len
tokens for resident slot 9 (launch 1, beside live slot 8), all with distinct physical KV blocks:
A  pp.run(prompt[:T-1], chunk=C) -> handoff into slot 9 -> read_slot(9) -> first-token launch (preserve) logits
B  pp.run_span over the same prompt in chunks of C, --decode-steps decode steps of the live users after every span,
   the draining span -> handoff -> read_slot -> first-token logits
C  as B without the decode steps (separates span mechanics from decode interference)
Pass: B and C equal A bit for bit: GDN state, conv history, KV pages of positions 0..T-2, first-token logits.
Also printed: A against run() at the chunk pick_chunk(T) would choose (served whole prompts), information only.

usage: python -m qwen36_resident.tests.split_prefill_gate --doc METALIUM_GUIDE.md [--prompt-len 5000] [--out split.json]
(the serving venv: it builds the vLLM class; needs the rows-mode model of qwen38-res-mu-wp7 or later)
"""
import argparse
import faulthandler
import json
import random
import sys

import torch
from transformers import AutoTokenizer

import ttnn
from qwen36_resident import weights as Q
from qwen36_resident.generator_vllm import Qwen36ResidentRowsForCausalLM
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh
from qwen36_resident.vllm_rows import BLOCK

TARGET = 9  # resident slot of the split prompt: launch 1, beside live slot 8
LIVE = 9  # live users in vLLM slots 0..8 -> resident slots 0..8


def kv_rows(slot_state, n_pos):
    """K and V of positions 0..n_pos-1: [n, copies, n_pos, HD] each (pages are position tiles of 32 rows)"""
    out = []
    for which in ("K", "V"):
        t = slot_state[which]  # [n, copies, pages, 32, HD]
        out.append(t.reshape(t.shape[0], t.shape[1], -1, t.shape[-1])[:, :, :n_pos])
    return out


def compare(a, b, n_pos):
    res = dict(state=torch.equal(a["state"], b["state"]), hist=torch.equal(a["hist"], b["hist"]))
    ka, va = kv_rows(a, n_pos)
    kb, vb = kv_rows(b, n_pos)
    res["kv"] = torch.equal(ka, kb) and torch.equal(va, vb)
    res["max_abs"] = {
        k: float((x.float() - y.float()).abs().max())
        for k, x, y in (("state", a["state"], b["state"]), ("hist", a["hist"], b["hist"]), ("K", ka, kb), ("V", va, vb))
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--prompt-len", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--decode-steps", type=int, default=3)
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
    path = Q.checkpoint_dir()
    ck = Q.Checkpoint(path)
    tok = AutoTokenizer.from_pretrained(path)
    text = tok(open(a.doc).read(), add_special_tokens=False)["input_ids"]
    T, C = a.prompt_len, a.chunk
    ids = torch.tensor(text[:T])
    rng = random.Random(5)
    nb = LIVE * 4 + -(-T // BLOCK) + 64  # 4 blocks per live user (prompt + decode steps), the target's, spare
    free = list(range(1, nb))
    rng.shuffle(free)
    take = lambda n: [free.pop() for _ in range(n)]
    mesh = open_mesh()
    res = {}
    try:
        m = Qwen36ResidentRowsForCausalLM(mesh, ck, a.max_len, 16, False)
        m.allocate_kv_cache((nb, 1, BLOCK, 256), None, 16)
        eng, pp, model = m.engine, m.pp, m.engine.model
        width = -(-a.max_len // BLOCK)

        # live users: short prompts, whole, through the adapter
        live = []
        for v in range(LIVE):
            p = torch.tensor(text[1000 + 300 * v : 1000 + 300 * v + 40 + 7 * v])
            blocks = take(4)
            pt = torch.zeros(1, width, dtype=torch.int32)
            pt[0, :4] = torch.tensor(blocks, dtype=torch.int32)
            lg = m.prefill_forward(
                p[None], pt, [], torch.tensor([len(p)]), start_pos=torch.tensor([0]), empty_slots=[v]
            )
            live.append(dict(tokens=p.tolist() + [int(lg[0, 0].argmax())], table=blocks))

        def decode_live(steps):
            for _ in range(steps):
                toks = torch.zeros(16, 1, dtype=torch.int64)
                pos = torch.full((16,), -1, dtype=torch.int64)
                tables = torch.zeros(16, width, dtype=torch.int32)
                for v, u in enumerate(live):
                    toks[v, 0], pos[v] = u["tokens"][-1], len(u["tokens"]) - 1
                    tables[v, : len(u["table"])] = torch.tensor(u["table"], dtype=torch.int32)
                out = m.decode_forward(toks, pos, tables, reload_inputs=True, read_from_device=True)
                for v, u in enumerate(live):
                    u["tokens"].append(int(out[v, 0].argmax()))

        decode_live(2)
        target_blocks = take(-(-T // BLOCK))

        def finish(name):
            """handoff into the target slot, its state as stored, then its first-token launch's logits"""
            pp.handoff(model, T - 1, block_table=target_blocks, slot=TARGET)
            m.tables[TARGET] = list(target_blocks)  # read_slot and the first-token launch read the slot's table
            m._write_tables()
            ttnn.synchronize_device(mesh)
            st = model.read_slot(TARGET)
            return st, eng.first_token(TARGET, int(ids[T - 1]), T - 1, preserve=True)

        # A: one PP run at chunk C
        pp.reset()
        pp.run(ids[: T - 1], chunk=C)
        state_a, logits_a = finish("A")
        decode_live(2)

        # B / C: spans of C, with and without decode launches between them
        def split(steps):
            pp.reset()
            full = (T - 1) // C
            for c in range(full):
                pp.run_span(ids[c * C : (c + 1) * C], c, (c + 1) * C, drain=False, chunk=C)
                decode_live(steps)
            pp.run_span(ids[full * C : T - 1], full, T - 1, drain=True, chunk=C)
            return finish("split")

        state_b, logits_b = split(a.decode_steps)
        state_c, logits_c = split(0)
        res["B_interleaved"] = dict(compare(state_a, state_b, T - 1), logits=torch.equal(logits_a, logits_b))
        res["C_spans_only"] = dict(compare(state_a, state_c, T - 1), logits=torch.equal(logits_a, logits_c))
        # information: run() at the chunk it would pick for a whole prompt
        pp.reset()
        pp.run(ids[: T - 1])
        state_p, logits_p = finish("pick")
        res["pick_chunk_info"] = dict(
            chunk=pp.pick_chunk(T - 1), **compare(state_a, state_p, T - 1), logits=torch.equal(logits_a, logits_p)
        )
        ok = all(r[k] for r in (res["B_interleaved"], res["C_spans_only"]) for k in ("state", "hist", "kv", "logits"))
        res["ok"] = ok
        for k, v in res.items():
            print(f"{k}: {v}", flush=True)
        print("SPLIT PREFILL GATE " + ("PASS" if ok else "FAIL"), flush=True)
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
