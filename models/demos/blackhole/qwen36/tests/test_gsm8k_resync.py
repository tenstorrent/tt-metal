# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Rigorous GSM8K eval of Option A (periodic decode re-sync) on the demo path.

0-shot chat (short prompt → re-sync stays cheap; the few-shot prompt would dominate re-prefill cost),
EOS-terminated generation, robust answer extraction (number after '####', else the last number).
Compares pure decode vs re-sync (full / sliding-window) over N GSM8K test problems.

Run: GSM8K_DATA=/data/gsm8k.json GSM8K_N=12 MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B \
     pytest .../test_gsm8k_resync.py -s
"""
import json
import os
import re
import time

import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

CONFIGS = [("pure", 0, None), ("N4_full", 4, None)]


def _gold(item):
    if "gold" in item:
        return str(item["gold"]).replace(",", "").strip()
    m = re.search(r"####\s*(-?[\d,]+)", item.get("a", ""))
    return m.group(1).replace(",", "").strip() if m else None


def _extract(txt):
    m = re.findall(r"####\s*(-?[\d,]+)", txt)
    if m:
        return m[-1].replace(",", "").strip()
    nums = re.findall(r"-?[\d,]+", txt.replace(",", ""))
    return nums[-1] if nums else None


@parametrize_mesh_tp()
def test_gsm8k_resync(mesh_device):
    from loguru import logger
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    eos = tok.eos_token_id

    data = json.load(open(os.environ["GSM8K_DATA"]))
    tests = data["test"][: int(os.environ.get("GSM8K_N", "12"))]
    MAXNEW = int(os.environ.get("GSM8K_MAXNEW", "480"))

    think = os.environ.get("THINK", "0") == "1"

    def ids_of(q):
        msg = [{"role": "user", "content":
                q + "\n\nSolve step by step. End with the final answer on its own line as: #### <number>"}]
        kw = {} if think else {"enable_thinking": False}
        try:
            return list(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=False, **kw))
        except TypeError:
            return list(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=False))

    summary = {}
    for name, N, win in CONFIGS:
        ok = 0
        t0 = time.perf_counter()
        ntok = 0
        for i, item in enumerate(tests):
            gold = _gold(item)
            out = model.generate_tp_resync(ids_of(item["q"]), max_new_tokens=MAXNEW,
                                           resync_every=N, eos_id=eos, window=win)
            ntok += len(out)
            pred = _extract(tok.decode(out, skip_special_tokens=True))
            hit = pred is not None and gold is not None and pred == gold
            ok += int(hit)
            logger.info(f"GSM8KRS cfg={name} #{i} gold={gold} pred={pred} hit={hit} ntok={len(out)}")
        dt = time.perf_counter() - t0
        acc = ok / len(tests)
        summary[name] = (ok, len(tests), acc, ntok / dt if dt else 0)
        logger.info(f"GSM8KRS_SUMMARY cfg={name} acc={ok}/{len(tests)}={acc:.3f} tok/s={ntok/dt:.2f} elapsed={dt:.0f}s")

    for name, (ok, n, acc, tps) in summary.items():
        logger.info(f"GSM8KRS_FINAL {name}: {ok}/{n} = {acc:.1%}  {tps:.2f} tok/s")
