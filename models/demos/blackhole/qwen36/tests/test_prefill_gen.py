# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decisive: autoregressive generation via the PREFILL/score path (no decode path at all).

Each step re-prefills the whole sequence (score_tp) and greedily takes the last-position argmax.
Slow (O(L) prefills), but it uses the HEALTHY prefill path for generation, bypassing the decode
path entirely. If GSM8K is solved here -> the decode path IS the culprit (prefill-gen works). If it
also fails -> the model/prompt behavior is the cause and the TT decode path is exonerated.

Runs a few GSM8K problems (chat-style CoT). Env: PFGEN_N (default 3), PFGEN_MAXNEW (default 160).
"""
import os
import re
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


def _pad128(n):
    return ((n + 127) // 128) * 128


@parametrize_mesh_tp()
def test_prefill_gen(mesh_device):
    from loguru import logger
    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    te = load_dataset("openai/gsm8k", "main", split="test")
    N = int(os.environ.get("PFGEN_N", "3"))
    MAXNEW = int(os.environ.get("PFGEN_MAXNEW", "160"))

    def gen(prompt_ids, n):
        ids = list(prompt_ids)
        for _ in range(n):
            L = len(ids)
            Tp = _pad128(L)
            padded = torch.zeros(1, Tp, dtype=torch.long)
            padded[0, :L] = torch.tensor(ids)
            logits = model.score_tp(padded, valid_len=L)  # [Tp, vocab]
            nxt = int(logits[L - 1].argmax().item())
            ids.append(nxt)
            if nxt == tok.eos_token_id:
                break
        return ids[len(prompt_ids):]

    correct = 0
    for i in range(N):
        q = te[i]["question"]
        gold = te[i]["answer"].split("####")[-1].strip().replace(",", "")
        prompt = f"Question: {q}\nAnswer: Let's think step by step."
        pid = tok(prompt, return_tensors="pt").input_ids[0].tolist()
        out_ids = gen(pid, MAXNEW)
        txt = tok.decode(out_ids)
        nums = re.findall(r"[\-0-9][0-9,]*\.?[0-9]*", txt.replace(",", ""))
        pred = nums[-1].rstrip(".") if nums else None
        ok = pred == gold
        correct += int(ok)
        logger.info(f"PFGEN#{i} gold={gold} pred={pred} ok={ok} :: {txt[:300]!r}")
    logger.info(f"PFGEN_SUMMARY prefill-path-gen N={N} correct={correct} (decode path bypassed)")
