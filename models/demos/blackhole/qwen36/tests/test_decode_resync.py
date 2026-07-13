# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Option A validation: periodic re-sync of the decode path to prefill's chunk-space.

Root cause: decode (recurrent, 0.99998 vs torch) continues a context prefill built in the chunk
kernel's "chunk-space" (0.988 vs torch); the space mismatch compounds → long-generation drift. N=1
(re-prefill every step = test_prefill_gen) already reasons correctly. This sweeps resync_every=N:
every N generated tokens, re-prefill the whole sequence-so-far (rebuilds attention KV cache + GDN
state in chunk-space via prefill_seed_tp + finalize_seed); between re-syncs, fast recurrent decode.
Finds the largest N that keeps the Janet answer ($18) correct.

Run: MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B pytest .../test_decode_resync.py -s
"""
import math
import os
import re
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@parametrize_mesh_tp()
def test_decode_resync(mesh_device):
    from loguru import logger
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    eos = tok.eos_token_id

    prompt = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes four into "
              "muffins. She sells the remainder at the market for two dollars per egg. How many dollars "
              "does she make every day? Think step by step and give the final number.")
    ids = list(tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       add_generation_prompt=True, tokenize=True, return_dict=False))
    MAXNEW = 240

    def prefill_full(seq):
        T = len(seq)
        Tp = max(128, math.ceil(T / 128) * 128)
        padded = seq + [0] * (Tp - T)
        lg = model.prefill_seed_tp(torch.tensor([padded], dtype=torch.long), valid_len=T, batch_slot=0)
        for layer in model.layers:
            if not layer.is_full_attention:
                layer.attention.finalize_seed(1)
        return int(torch.argmax(lg).item())

    def gen(N):
        model.reset_tp()
        seq = list(ids)
        nxt = prefill_full(seq)
        seq.append(nxt)
        out = [nxt]
        since = 0
        for _ in range(MAXNEW - 1):
            if nxt == eos:
                break
            if since >= N:  # re-sync: rebuild caches+state in chunk-space
                model.reset_tp()
                nxt = prefill_full(seq)
                since = 0
            else:
                lg = model.decode_tp_batched([seq[-1]], [len(seq) - 1])
                nxt = int(torch.as_tensor(lg).float().reshape(-1)[: model.vocab_size].argmax().item())
                since += 1
            seq.append(nxt)
            out.append(nxt)
        return tok.decode(out, skip_special_tokens=True)

    for N in (10 ** 9, 8, 4):
        txt = gen(N)
        nums = re.findall(r"\$?\s*(\d+)", txt)
        has18 = "18" in nums
        tag = "PURE_DECODE" if N > 10 ** 6 else f"RESYNC_N={N}"
        logger.info(f"RESYNC[{tag}] reaches_18={has18} last_nums={nums[-6:]}\n---\n{txt[-400:]}\n---END---")
