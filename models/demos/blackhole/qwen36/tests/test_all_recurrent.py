# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""(a) drift-reduction probe: 'all-recurrent' generation vs chunk-prefill+decode.

Decode drift = the accurate recurrent decode (0.99998 vs torch) continuing a context prefill built in
the chunk kernel's inaccurate 'chunk-space' (0.988). Hypothesis: feeding the PROMPT through the
recurrent decode path too (no chunk kernel) makes prefill-space = decode-space = true-space (self-
consistent, like GPU), removing the mismatch WITHOUT per-step re-sync cost. If all-recurrent reasons
correctly where chunk-prefill+decode drifts, "recurrent prefill" is the fix direction.

Run: MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B pytest .../test_all_recurrent.py -s
"""
import os
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@parametrize_mesh_tp()
def test_all_recurrent(mesh_device):
    from loguru import logger
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    eos = tok.eos_token_id
    vocab = model.vocab_size
    MAXNEW = 160

    q = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes four into muffins. "
         "She sells the remainder at two dollars per egg. How many dollars does she make every day?")
    msg = [{"role": "user", "content": q + "\n\nSolve step by step. End with: #### <number>"}]
    ids = list(tok.apply_chat_template(msg, add_generation_prompt=True, tokenize=True, return_dict=False))
    P = len(ids)

    def gen_all_recurrent():
        # PROMPT through the recurrent decode path (no chunk kernel) → true-space state, then continue.
        model.reset_tp()
        nxt = None
        for pos in range(P):
            lg = model.decode_tp(ids[pos], pos).reshape(-1)[:vocab]
            nxt = int(torch.argmax(lg))  # prediction after consuming ids[:pos+1]
        out = [nxt]
        for pos in range(P, P + MAXNEW - 1):
            if nxt == eos:
                break
            lg = model.decode_tp(nxt, pos).reshape(-1)[:vocab]
            nxt = int(torch.argmax(lg))
            out.append(nxt)
        return out

    def gen_chunk_prefill():
        # Standard: chunk-kernel prefill, then recurrent decode (the drifting path).
        return model.generate_tp(ids, max_new_tokens=MAXNEW)

    for name, fn in (("ALLREC", gen_all_recurrent), ("CHUNKPF", gen_chunk_prefill)):
        out = fn()
        txt = tok.decode(out, skip_special_tokens=True)
        import re
        nums = re.findall(r"-?\d+", txt)
        logger.info(f"ALLREC_PROBE[{name}] reaches_18={'18' in nums} last_nums={nums[-6:]}\n---\n{txt[:500]}\n---END---")
