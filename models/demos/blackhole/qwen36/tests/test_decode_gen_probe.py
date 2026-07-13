# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real decode-path generation quality probe: does the model reason correctly through the
scaled_dot_product_attention_decode path? Compares the decode-vs-prefill PCC proxy (which favors the
HF-INCORRECT flat q/k-norm) against actual generated text. Run twice via QWEN_ATTN_SHARP_DECODE:
  - unset  -> decode uses FLAT q/k-norm (main default, HF-incorrect, missing the +1 offset)
  - =1     -> decode uses SHARP q/k-norm (HF-correct: output*(1+weight); matches prefill + branch)
The Janet problem's answer is 18 dollars (16-3-4=9 eggs * $2). Coherent, correct-number reasoning
vs confabulation/looping is the real signal.
Run: MESH_DEVICE=P150x4 HF_MODEL=Qwen/Qwen3.6-27B pytest .../test_decode_gen_probe.py -s
"""
import os
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import model_path, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.model import Qwen36Model


@parametrize_mesh_tp()
def test_decode_gen_probe(mesh_device):
    from loguru import logger
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_MODEL", model_path())
    mesh_device.enable_program_cache()
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=2048)
    tok = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    sharp = os.environ.get("QWEN_ATTN_SHARP_DECODE") == "1"
    logger.info(f"GENPROBE decode_norm={'SHARP(+1, HF-correct)' if sharp else 'FLAT(no +1, main default)'}")

    prompt = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes four into "
              "muffins. She sells the remainder at the market for two dollars per egg. How many dollars "
              "does she make every day? Think step by step.")
    msgs = [{"role": "user", "content": prompt}]
    ids = list(tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=False))
    logger.info(f"GENPROBE prompt_tokens={len(ids)}")

    out = model.generate_tp_batched([ids], max_new_tokens=200, eos_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    logger.info(f"GENPROBE_OUTPUT[{'SHARP' if sharp else 'FLAT'}]:\n{text}\n---END---")
