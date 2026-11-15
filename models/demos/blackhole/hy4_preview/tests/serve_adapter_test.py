# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapter mechanics test (no vLLM): HYV4ForCausalLM driven the way the plugin drives it. Env: HY4_SERVE_LAYERS (default 4 here), HY4_SERVE_TCAP, OUT.
 A: prompt alone in slot 0 -> N greedy decode steps.
 B: A in slot 2 + other prompts in slots 0/1 (one prefill call), remap, decode: A's tokens must match run A (same numerics up to batch composition).
 C: live decode of A (slot 0), a new prefill of C into slot 3 in the middle: A's logits continue as in run A (decode caches of running rows untouched)."""
import json
import os
import time

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tests.dev_common import close_mesh, open_mesh, pcc
from models.demos.blackhole.hy4_preview.tt.generator_vllm import HYV4ForCausalLM

OUT = os.environ.get("OUT", "/out")
NSTEP = int(os.environ.get("NSTEPS", "12"))


def res(**kw):
    print("RES " + json.dumps(kw), flush=True)


def greedy_run(m, prompts_slots, steps, remap_after=None, mid_prefill=None):
    """prompts_slots {slot: tokens}; returns {slot: [tokens]}, {slot: [logit rows]} keyed by the ORIGINAL slot."""
    B = m.max_num_seqs
    slots = sorted(prompts_slots)
    toks = torch.zeros(len(slots), max(len(prompts_slots[s]) for s in slots), dtype=torch.long)
    for i, s in enumerate(slots):
        toks[i, : len(prompts_slots[s])] = torch.tensor(prompts_slots[s])
    lens = [len(prompts_slots[s]) for s in slots]
    lg = m.prefill_forward(toks, start_pos=[0] * len(slots), prompt_lens=lens, empty_slots=slots)
    cur = {s: int(lg[i, 0].argmax()) for i, s in enumerate(slots)}
    pos = {s: lens[i] for i, s in enumerate(slots)}
    gen = {s: [cur[s]] for s in slots}
    logs = {s: [lg[i, 0].clone()] for i, s in enumerate(slots)}
    row_of = {s: s for s in slots}  # decode row of the original slot
    for st in range(steps):
        if mid_prefill and st == mid_prefill[0]:
            s2, t2 = mid_prefill[1], mid_prefill[2]
            l2 = m.prefill_forward(torch.tensor([t2]), start_pos=[0], prompt_lens=[len(t2)], empty_slots=[s2])
            cur[s2] = int(l2[0, 0].argmax())
            pos[s2] = len(t2)
            gen[s2] = [cur[s2]]
            logs[s2] = [l2[0, 0].clone()]
            row_of[s2] = s2
        remap = None
        if remap_after is not None and st == remap_after[0]:
            remap = torch.tensor(remap_after[1], dtype=torch.int32)
            # new slot i reads old slot remap[i]: original slot s now sits at row i with remap[i] == row_of[s]
            inv = {int(remap[i]): i for i in range(B)}
            row_of = {s: inv[r] for s, r in row_of.items()}
        act = sorted(row_of[s] for s in cur)
        tk = torch.zeros(B, 1, dtype=torch.long)
        sp = torch.full((B,), -1, dtype=torch.long)
        for s in cur:
            tk[row_of[s], 0] = cur[s]
            sp[row_of[s]] = pos[s]
        out = m.decode_forward(tk, sp, slot_remap=remap)
        for s in list(cur):
            row = out[row_of[s], 0]
            nxt = int(row.argmax())
            logs[s].append(row.clone())
            gen[s].append(nxt)
            cur[s] = nxt
            pos[s] += 1
    return gen, logs


def main():
    md = open_mesh(trace=int(float(os.environ.get("TRACE_BYTES", "600e6"))))
    try:
        m = HYV4ForCausalLM.initialize_vllm_model(None, md, 4, int(os.environ.get("HY4_SERVE_TCAP", "2048")))
        t0 = time.time()
        m.warm_serving()
        res(event="warm", sec=round(time.time() - t0, 1))
        g = torch.Generator().manual_seed(1)
        mk = lambda n: torch.randint(1000, 50000, (n,), generator=g).tolist()
        A, Bp, Cp, Dp = mk(37), mk(600), mk(1300), mk(90)
        t0 = time.time()
        gA, lA = greedy_run(m, {0: A}, NSTEP)
        res(event="runA", sec=round(time.time() - t0, 1), toks=gA[0])
        # B: A at slot 2, others at 0, 1 (different lengths: 1 and 3 chunks), remap rotates the slots after 3 steps
        gB, lB = greedy_run(m, {0: Bp, 1: Dp, 2: A}, NSTEP, remap_after=(3, [1, 2, 0, 3]))
        res(event="runB", toks=gB[2], match=sum(a == b for a, b in zip(gA[0], gB[2])), n=len(gA[0]))
        res(event="runB_logit_pcc", pcc=[round(pcc(lA[0][j], lB[2][j]), 4) for j in range(len(lA[0]))])
        # C: A live, a new prompt prefilled mid-decode into slot 3
        gC, lC = greedy_run(m, {0: A}, NSTEP, mid_prefill=(4, 3, Cp))
        res(event="runC", toks=gC[0], match=sum(a == b for a, b in zip(gA[0], gC[0])), n=len(gA[0]))
        res(event="runC_logit_pcc", pcc=[round(pcc(lA[0][j], lC[0][j]), 4) for j in range(len(lA[0]))])
        # D: the mid-prefilled prompt alone gives the same continuation as inside run C
        gD, lD = greedy_run(m, {0: Cp}, NSTEP - 4)
        res(event="runD", match=sum(a == b for a, b in zip(gD[0], gC[3])), n=len(gD[0]))
    finally:
        close_mesh(md)


if __name__ == "__main__":
    main()
