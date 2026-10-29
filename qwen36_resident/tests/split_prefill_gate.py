# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device gate (WP8): a prompt prefilled in chunks across calls, with live users decoding between the chunks, hands
the same state to its slot as one PP run of the same prompt (review item 4: x_last, S, carry, the KV scratch and the
shift semaphores of the PP pipeline must survive the rows-mode decode launches run in between).

The serving build (Qwen36ResidentRowsForCausalLM: rows-mode model, paged KV pool, traces, PP prefill) with 9 live users
(resident slots 0-8, launches 0 and 1) decoding greedily through the adapter, and a target prompt of --prompt-len
tokens for resident slot 9 (launch 1, beside live slot 8), all with distinct physical KV blocks. The PP chunk c is the
serving one: the chunk the prompt's whole length picks (the adapter's _pp_chunk), for whole and split runs alike.
A  pp.run(prompt[:T-1], chunk=c) -> handoff into slot 9 -> read_slot(9) -> first-token launch (preserve), then
   --extra greedy steps of slot 9 (preserve): their logits
B  pp.run_span over the same prompt in CHUNK_UNIT-aligned spans at chunk c, --decode-steps decode steps of the live
   users after every span, the draining span -> handoff -> read_slot -> the same launches
C  as B without the decode steps (separates span mechanics from decode interference)
Pass: B and C equal A bit for bit: GDN state, the conv history elements the decode reads (the q|k|v tiles the
handoff writes; hist_index), KV pages of positions 0..T-2, the logits of every launch.
Information: the history pad (the other elements of the slot's row: a non-preserve launch rewrites them, the handoff
does not, no step reads them), and A against a whole run at the other chunk (rounding: PCC, argmax agreement).

usage: python -m qwen36_resident.tests.split_prefill_gate --doc METALIUM_GUIDE.md [--prompt-len 5000] [--out split.json]
(the serving venv: it builds the vLLM class; needs the rows-mode model of qwen38-res-mu-wp6 or later)
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
from qwen36_resident.model import CONV_K, TILE
from qwen36_resident.tests.e0_bringup import close_mesh, open_mesh, pcc
from qwen36_resident.vllm_rows import BLOCK, CHUNK_UNIT

TARGET = 9  # resident slot of the split prompt: launch 1, beside live slot 8
LIVE = 9  # live users in vLLM slots 0..8 -> resident slots 0..8


def kv_rows(slot_state, n_pos):
    """K and V of positions 0..n_pos-1: [n, copies, n_pos, HD] each (pages are position tiles of 32 rows)"""
    out = []
    for which in ("K", "V"):
        t = slot_state[which]  # [n, copies, pages, 32, HD]
        out.append(t.reshape(t.shape[0], t.shape[1], -1, t.shape[-1])[:, :, :n_pos])
    return out


def conv_mask(model, shape):
    """the elements of read_slot's history row [n * S * copies, slots * blk] the decode reads: per (chip, streamer,
    copy) row the q|k|v columns of hist_index in each of the CONV_K - 1 slots"""
    rows, cols, _ = model.hist_index()
    n, S, G, blk = model.d.n, model._S, model._gdn_copies, model._blk * TILE
    assert tuple(shape) == (n * S * G, (CONV_K - 1) * blk), (tuple(shape), n, S, G, blk)
    mask = torch.zeros(shape, dtype=torch.bool)
    r = rows.repeat_interleave(TILE)
    for chip in range(n):
        for g in range(G):
            for j in range(CONV_K - 1):
                mask[chip * S * G + r * G + g, j * blk + cols] = True
    assert int(mask.sum()) == n * G * (CONV_K - 1) * len(cols) > 0, "conv elements overlap or none"
    return mask


def compare(a, b, n_pos, mask):
    ka, va = kv_rows(a, n_pos)
    kb, vb = kv_rows(b, n_pos)
    ha, hb = a["hist"], b["hist"]
    res = dict(
        state=torch.equal(a["state"], b["state"]),
        hist=torch.equal(ha[mask], hb[mask]),
        kv=torch.equal(ka, kb) and torch.equal(va, vb),
    )
    absmax = lambda x, y: float((x.float() - y.float()).abs().max()) if x.numel() else 0.0
    res["max_abs"] = dict(
        state=absmax(a["state"], b["state"]), hist=absmax(ha[mask], hb[mask]), K=absmax(ka, kb), V=absmax(va, vb)
    )
    res["hist_pad_max_abs (info)"] = absmax(ha[~mask], hb[~mask])
    return res


def compare_logits(la, lb):
    return dict(logits=torch.equal(la, lb), per_step=[torch.equal(x, y) for x, y in zip(la, lb)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--prompt-len", type=int, default=5000)
    ap.add_argument("--chunk", type=int, default=None, help="PP chunk (default: the serving pick for the prompt)")
    ap.add_argument("--decode-steps", type=int, default=3)
    ap.add_argument("--extra", type=int, default=3, help="greedy steps of the target after its first token")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--out")
    a = ap.parse_args()
    faulthandler.dump_traceback_later(900, repeat=True, file=sys.stderr)
    path = Q.checkpoint_dir()
    ck = Q.Checkpoint(path)
    tok = AutoTokenizer.from_pretrained(path)
    text = tok(open(a.doc).read(), add_special_tokens=False)["input_ids"]
    T, U = a.prompt_len, CHUNK_UNIT
    ids = torch.tensor(text[:T])
    rng = random.Random(5)
    n_target = -(-(T + a.extra) // BLOCK)
    nb = LIVE * 4 + n_target + 64  # 4 blocks per live user (prompt + decode steps), the target's, spare
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
        c = a.chunk or m._pp_chunk(T)
        assert U % c == 0
        res["chunk"] = c

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
        target_blocks = take(n_target)

        def finish():
            """handoff into the target slot, its state as stored, then the logits of its first-token launch and of
            --extra greedy steps (preserve: live slot 8 shares launch 1)"""
            pp.handoff(model, T - 1, block_table=target_blocks, slot=TARGET)
            m.tables[TARGET] = list(target_blocks)  # read_slot and the target's launches read the slot's table
            m._write_tables()
            ttnn.synchronize_device(mesh)
            st = model.read_slot(TARGET)
            lg = [eng.first_token(TARGET, int(ids[T - 1]), T - 1, preserve=True)]
            for i in range(a.extra):
                lg.append(eng.first_token(TARGET, int(lg[-1].argmax()), T + i, preserve=True))
            return st, torch.stack(lg)

        def whole(chunk):
            pp.reset()
            pp.run(ids[: T - 1], chunk=chunk)
            return finish()

        def split(steps):
            """CHUNK_UNIT-aligned spans (the chunk core's) at the sequence's chunk c"""
            pp.reset()
            full = (T - 1) // U
            for s in range(full):
                pp.run_span(ids[s * U : (s + 1) * U], s * U // c, (s + 1) * U, drain=False, chunk=c)
                decode_live(steps)
            pp.run_span(ids[full * U : T - 1], full * U // c, T - 1, drain=True, chunk=c)
            return finish()

        state_a, logits_a = whole(c)
        mask = conv_mask(model, state_a["hist"].shape)
        res["hist_conv_fraction"] = float(mask.float().mean())
        decode_live(2)
        state_b, logits_b = split(a.decode_steps)
        state_c, logits_c = split(0)
        res["B_interleaved"] = dict(compare(state_a, state_b, T - 1, mask), **compare_logits(logits_a, logits_b))
        res["C_spans_only"] = dict(compare(state_a, state_c, T - 1, mask), **compare_logits(logits_a, logits_c))
        # information: a whole run at the other chunk (rounding, not a gate)
        other = 512 if c == 1024 else 1024
        state_o, logits_o = whole(other)
        ko, vo = kv_rows(state_o, T - 1)
        ka, va = kv_rows(state_a, T - 1)
        res["other_chunk_info"] = dict(
            chunk=other,
            **compare(state_a, state_o, T - 1, mask),
            pcc=dict(
                state=pcc(state_a["state"], state_o["state"]),
                K=pcc(ka, ko),
                V=pcc(va, vo),
                logits=[pcc(x, y) for x, y in zip(logits_a, logits_o)],
            ),
            argmax_equal=[int(x.argmax()) == int(y.argmax()) for x, y in zip(logits_a, logits_o)],
        )
        gates = ("state", "hist", "kv", "logits")
        ok = all(r[k] for r in (res["B_interleaved"], res["C_spans_only"]) for k in gates)
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
