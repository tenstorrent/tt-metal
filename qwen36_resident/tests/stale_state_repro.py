# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A served single-user prompt after long-decoded ones (lane ttft2, 2026-10-30): the ttft2 session's OSL 1024 sweep
(ttft2_session_20261030-1248) gave wrong first tokens for some ISL >= 2048 DFlash2 prompts, in async and synchronous
serving alike, where the same prompts after 128-token requests were right; the prefill paths skip the initial-state
upload (reset(state=False)) on the premise that the handoff overwrites everything the first step reads.

The serving class itself (--model dflash2 | mtp | plain: generator_vllm's Qwen36ResidentDFlashForCausalLM /
Qwen36ResidentMTPForCausalLM / Qwen36ResidentForCausalLM, built as served at --max-seq-len), driven as the plugin
drives it: prefill_forward, then decode_forward (spec classes: blocks; plain: one token per step, greedy on the host
logits), release_request. Per case (isl:prev:cur, the vllm bench prompts of ttft2_prompts.pt), the predecessors
prev - 1 and prev are served, then cur, as variants:
  ctl:   predecessors of --short tokens (the OSL 128 sweep's history)
  bad:   predecessors of --long tokens (the OSL 1024 sweep's history)
  bad1:  prev alone, --long tokens
  full:  as bad, then the decode model's whole initial state restored before cur (what reset(state=True) uploads)
  state: as bad, then only its GDN states restored
  hist:  as bad, then only its conv histories restored
Logs cur's first token, its prefill logits (NaNs, max |d| against ctl's) and how many of cur's --tokens tokens equal
the full variant's (the fresh-reset reference); dflash2 also gets the verify model's own greedy decode of cur (its
prompt through verify steps after a full reset: no PP prefill, no handoff).

usage (vLLM venv, plugin on PYTHONPATH): python -m qwen36_resident.tests.stale_state_repro --model dflash2
    [--cases 8192:4:5,8192:6:7,4096:1:2] [--variants ctl,bad,full,state,hist,bad1]
"""
import argparse
import faulthandler
import sys
import time

import torch

import ttnn
from qwen36_resident import generator_vllm as G
from qwen36_resident import weights as Q
from qwen36_resident.tests.long_ctx import close_mesh, log, open_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=("dflash2", "mtp", "plain"), required=True)
ap.add_argument("--cases", default="8192:4:5,8192:6:7,4096:1:2")
ap.add_argument("--prompts", default="/home/ttuser/experiments/qwen36_27b/profiles/resident/dfl/ttft/ttft2_prompts.pt")
ap.add_argument("--max-seq-len", type=int, default=65536)
ap.add_argument("--short", type=int, default=128)
ap.add_argument("--long", type=int, default=1024)
ap.add_argument("--tokens", type=int, default=64)
ap.add_argument("--variants", default="ctl,bad,full,state,hist,bad1")
a = ap.parse_args()
faulthandler.dump_traceback_later(1800, repeat=True, file=sys.stderr)
prompts = torch.load(a.prompts)
CLS = dict(
    dflash2=G.Qwen36ResidentDFlashForCausalLM, mtp=G.Qwen36ResidentMTPForCausalLM, plain=G.Qwen36ResidentForCausalLM
)[a.model]
mesh = open_mesh()
try:
    t0 = time.time()
    obj = CLS(mesh, Q.Checkpoint(Q.checkpoint_dir()), a.max_seq_len, False)
    target = obj.model if a.model == "plain" else (obj.spec.m if a.model == "dflash2" else obj.spec.main)
    log(f"{CLS.__name__} built in {time.time() - t0:.0f} s; decode model initial-state entries {len(target._initial)}")
    raw = {}
    if a.model != "plain":  # the spec classes return start()'s logits masked to the argmax: keep the raw ones
        start = obj.spec.start

        def start_raw(ids):
            raw["lg"] = start(ids)
            return raw["lg"]

        obj.spec.start = start_raw

    def serve(ids, n):
        """one request as the plugin serves it: (prefill logits [vocab], n tokens)"""
        T = len(ids)
        lg = obj.prefill_forward(torch.tensor([ids]), None, None, torch.tensor([T]))
        lg = (raw.pop("lg") if a.model != "plain" else lg).reshape(-1).float()
        out = [int(lg.argmax())]
        if a.model == "plain":
            while len(out) < n:
                out.append(
                    int(obj.decode_forward(torch.tensor([[out[-1]]]), torch.tensor([T + len(out) - 1])).argmax())
                )
        else:
            while len(out) < n:
                out += obj.decode_forward(torch.tensor([[out[0]]]), torch.tensor([T])).reshape(-1).tolist()
            obj.release_request(0)
        return lg, out[:n]

    def restore(which):
        """the decode model's initial contents of entries `which` (0: conv histories, 1: GDN states; reset uploads all)"""
        for i in which:
            ttnn.copy_host_to_device_tensor(*target._initial[i])
        ttnn.synchronize_device(mesh)

    VARIANTS = dict(
        ctl=(2, a.short, ()),
        bad=(2, a.long, ()),
        bad1=(1, a.long, ()),
        full=(2, a.long, tuple(range(len(target._initial)))),
        state=(2, a.long, (1,)),
        hist=(2, a.long, (0,)),
    )
    for case in a.cases.split(","):
        isl, prev, cur = (int(x) for x in case.split(":"))
        P = prompts[isl, 1024]
        ref = None
        if a.model == "dflash2":
            ref = obj.spec.greedy(P[cur], a.tokens)
            log(f"[{a.model}] ISL {isl} cur {cur}: verify greedy (no PP prefill / handoff) first 16 {ref[:16]}")
        res = {}
        for name in a.variants.split(","):
            n_prev, n_tok, which = VARIANTS[name]
            t1 = time.time()
            for p in range(prev - n_prev + 1, prev + 1):
                serve(P[p], n_tok)
            restore(which)
            lg, out = serve(P[cur], a.tokens)
            res[name] = (lg, out)
            base = res.get("ctl")
            d = float((lg - base[0]).abs().max()) if base else float("nan")
            refs = {"full": res["full"][1]} if "full" in res else {}
            if ref is not None:
                refs["verify-greedy"] = ref
            agree = ", ".join(
                f"== {k} for {next((i for i, (x, y) in enumerate(zip(out, r)) if x != y), len(out))}"
                for k, r in refs.items()
            )
            log(
                f"[{a.model}] ISL {isl} prev {prev} -> cur {cur} [{name}]: first token {out[0]}, prefill logits nan "
                f"{int(torch.isnan(lg).sum())} max {float(lg.nan_to_num().max()):.3g}, max|d| vs ctl {d:.4g}; "
                f"{agree}; first 12 {out[:12]} ({time.time() - t1:.0f} s)"
            )
        for k, r in (("full", res.get("full")), ("ctl", res.get("ctl"))):
            if r is None:
                continue
            log(
                f"[{a.model}] ISL {isl} cur {cur} summary vs {k}: "
                + ", ".join(
                    f"{name} {next((i for i, (x, y) in enumerate(zip(o, r[1])) if x != y), len(o))}/{len(o)}"
                    for name, (_, o) in res.items()
                )
            )
finally:
    close_mesh(mesh)
