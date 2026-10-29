# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device gate (WP8): a prompt prefilled in chunks across calls, with live users decoding between the chunks, hands
the same state to its slot as one PP run of the same prompt (review item 4: x_last, S, carry, the KV scratch and the
shift semaphores of the PP pipeline must survive the rows-mode decode launches run in between).

The serving build (Qwen36ResidentRowsForCausalLM: rows-mode model, paged KV pool, traces, PP prefill) with 9 live users
(resident slots 0-8, launches 0 and 1) decoding greedily through the adapter, and, for each of --prompt-lens, a target
prompt of T tokens for resident slot 9 (launch 1, beside live slot 8), all with distinct physical KV blocks. The
lengths straddle the edges where whole and split runs used to take different PP chunks (at most one unit, just over
one, ~2k, 5000) and include ones no unit divides. The PP chunk c is the serving one: the chunk the prompt's whole
length picks (the adapter's _pp_chunk), for whole and split runs alike.
A  pp.run(prompt[:T-1], chunk=c) -> handoff into slot 9 -> read_slot(9) -> first-token launch (preserve), then
   --extra greedy steps of slot 9 (preserve): their logits
B  pp.run_span over the same prompt in CHUNK_UNIT-aligned spans at chunk c (asserted: every span but the draining
   one ends on a unit), --decode-steps decode steps of the live users after every span, the draining span ->
   handoff -> read_slot -> the same launches
C  as B without the decode steps (separates span mechanics from decode interference)
Pass, at every length: B and C equal A bit for bit: GDN state, the conv history elements the decode reads (the q|k|v
tiles the handoff writes; hist_index), KV pages of positions 0..T-2, the logits of every launch.
Information: the history pad (the other elements of the slot's row: a non-preserve launch rewrites them, the handoff
does not, no step reads them), and A against a whole run at the other chunk (rounding expected: PCC, argmax
agreement; a low PCC would be a bug of one chunk's path, see ROUNDING).

usage: python -m qwen36_resident.tests.split_prefill_gate --doc METALIUM_GUIDE.md [--prompt-lens 1000,1025,...]
    [--out split.json]
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
LENGTHS = "1000,1025,1100,2100,3333,5000"
# the other-chunk comparison counts as rounding when every launch's logits PCC and the KV PCC reach these
ROUNDING = dict(logits=0.99, kv=0.999)


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
    ap.add_argument("--prompt-lens", default=LENGTHS)
    ap.add_argument("--chunk", type=int, default=None, help="PP chunk (default: the serving pick for each prompt)")
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
    lens = [int(x) for x in a.prompt_lens.split(",")]
    U = CHUNK_UNIT
    rng = random.Random(5)
    n_target = -(-(max(lens) + a.extra) // BLOCK)
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
        all_target_blocks = take(n_target)
        mask = None

        def gate_length(T):
            nonlocal mask
            ids = torch.tensor(text[:T])
            target_blocks = all_target_blocks[: -(-(T + a.extra) // BLOCK)]
            c = a.chunk or m._pp_chunk(T)
            assert U % c == 0
            r = dict(chunk=c)

            def finish():
                """handoff into the target slot, its state as stored, then the logits of its first-token launch and
                of --extra greedy steps (preserve: live slot 8 shares launch 1)"""
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
                spans = [(s * U, (s + 1) * U) for s in range((T - 1) // U)] + [((T - 1) // U * U, T - 1)]
                for i, (lo, hi) in enumerate(spans):
                    last = i == len(spans) - 1
                    assert lo % U == 0 and (last or hi % U == 0), f"span {lo}..{hi} off the unit {U}"
                    pp.run_span(ids[lo:hi], lo // c, hi, drain=last, chunk=c)
                    if not last:
                        decode_live(steps)
                r["spans"] = [hi - lo for lo, hi in spans]
                return finish()

            state_a, logits_a = whole(c)
            if mask is None:
                mask = conv_mask(model, state_a["hist"].shape)
                res["hist_conv_fraction"] = float(mask.float().mean())
            decode_live(2)
            state_b, logits_b = split(a.decode_steps)
            state_c, logits_c = split(0)
            r["B_interleaved"] = dict(compare(state_a, state_b, T - 1, mask), **compare_logits(logits_a, logits_b))
            r["C_spans_only"] = dict(compare(state_a, state_c, T - 1, mask), **compare_logits(logits_a, logits_c))
            # information: a whole run at the other chunk (rounding, not a gate)
            other = 512 if c == 1024 else 1024
            state_o, logits_o = whole(other)
            ko, vo = kv_rows(state_o, T - 1)
            ka, va = kv_rows(state_a, T - 1)
            p_lg = [pcc(x, y) for x, y in zip(logits_a, logits_o)]
            p_kv = min(pcc(ka, ko), pcc(va, vo))
            r["other_chunk_info"] = dict(
                chunk=other,
                **compare(state_a, state_o, T - 1, mask),
                pcc=dict(state=pcc(state_a["state"], state_o["state"]), K=pcc(ka, ko), V=pcc(va, vo), logits=p_lg),
                argmax_equal=[int(x.argmax()) == int(y.argmax()) for x, y in zip(logits_a, logits_o)],
                rounding=min(p_lg) >= ROUNDING["logits"] and p_kv >= ROUNDING["kv"],
            )
            gates = ("state", "hist", "kv", "logits")
            r["ok"] = all(r[v][k] for v in ("B_interleaved", "C_spans_only") for k in gates)
            return r

        for T in lens:
            res[T] = r = gate_length(T)
            for k, v in r.items():
                print(f"T={T} {k}: {v}", flush=True)
        ok = all(res[T]["ok"] for T in lens)
        rounding = all(res[T]["other_chunk_info"]["rounding"] for T in lens)
        res["ok"], res["other_chunk_rounding"] = ok, rounding
        print(f"other chunk vs serving chunk: {'rounding' if rounding else 'NOT ROUNDING: check the chunk paths'}")
        print("SPLIT PREFILL GATE " + ("PASS" if ok else "FAIL"), flush=True)
    finally:
        close_mesh(mesh)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
