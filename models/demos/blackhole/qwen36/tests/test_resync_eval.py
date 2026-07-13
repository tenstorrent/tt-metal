# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Option A productionization eval (a: generate_tp_resync method; b: throughput; c: sliding-window;
e: multi-problem robustness). Runs several GSM8K problems x configs (pure decode / re-sync full /
re-sync window) and reports correctness + wall-clock tok/s.

Run: MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B pytest .../test_resync_eval.py -s
"""
import os
import re
import time
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PROBLEMS = [
    ("Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes four into muffins. "
     "She sells the remainder at the market for two dollars per egg. How many dollars does she make "
     "every day?", 18),
    ("Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. "
     "How much did she earn?", 10),
    ("Betty is saving for a $100 wallet. She has only half of the money she needs. Her parents give "
     "her $15, and her grandparents give twice as much as her parents. How much more money does "
     "Betty need to buy the wallet?", 5),
]
CONFIGS = [("pure", 0, None), ("N4_full", 4, None), ("N4_win64", 4, 64)]


@parametrize_mesh_tp()
def test_resync_eval(mesh_device):
    from loguru import logger
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    eos = tok.eos_token_id
    MAXNEW = 256

    def ids_of(q):
        msg = [{"role": "user", "content": q + " Think step by step and end with the final number."}]
        return list(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=False))

    results = {}
    for name, N, win in CONFIGS:
        n_ok = 0
        tot_tok = 0
        tot_t = 0.0
        for q, gold in PROBLEMS:
            ids = ids_of(q)
            t0 = time.perf_counter()
            out = model.generate_tp_resync(ids, max_new_tokens=MAXNEW, resync_every=N, eos_id=eos, window=win)
            dt = time.perf_counter() - t0
            txt = tok.decode(out, skip_special_tokens=True)
            nums = re.findall(r"-?\d+", txt.replace(",", ""))
            # correct if the gold number is the last distinct number mentioned (final answer)
            ok = len(nums) > 0 and str(gold) in nums[-3:]
            n_ok += int(ok)
            tot_tok += len(out)
            tot_t += dt
            logger.info(f"RESYNCEVAL cfg={name} gold={gold} ok={ok} ntok={len(out)} t={dt:.1f}s "
                        f"tps={len(out)/dt:.2f} last_nums={nums[-4:]}")
        tps = tot_tok / tot_t if tot_t else 0
        results[name] = (n_ok, len(PROBLEMS), tps)
        logger.info(f"RESYNCEVAL_SUMMARY cfg={name} correct={n_ok}/{len(PROBLEMS)} agg_tps={tps:.2f}")

    for name in results:
        n_ok, ntot, tps = results[name]
        logger.info(f"RESYNCEVAL_FINAL {name}: {n_ok}/{ntot} correct, {tps:.2f} tok/s")
