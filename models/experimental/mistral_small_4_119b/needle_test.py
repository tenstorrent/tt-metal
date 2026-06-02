# SPDX-License-Identifier: Apache-2.0
"""Long-context QUALITY check (text needle-in-a-haystack).

Buries a unique passcode near the START of a long filler prompt, asks for it at the
END, runs the real text model (chunked prefill + paged decode), and checks the model
retrieves it. Validates that long-context attention + chunked prefill preserve
information across chunk boundaries (i.e. quality, not just that it runs).

  MESH_DEVICE=P150x8 NEEDLE_TOKENS=32000 MISTRAL4_WEIGHT_CACHE_DIR=... \
    python models/experimental/mistral_small_4_119b/needle_test.py
"""
import os
import time

import torch
import ttnn
from loguru import logger

from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID
from models.experimental.mistral_small_4_119b.langauge_demo import _open_mesh_device, _state_dict_prefixes
from models.experimental.mistral_small_4_119b.tt.mistral4_text_model import TtMistral4TextModel
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

TARGET_TOKENS = int(os.environ.get("NEEDLE_TOKENS", "32000"))
PASSCODE = os.environ.get("NEEDLE_CODE", "7391")
N_LAYERS = int(os.environ.get("AB_LAYERS", "36"))


def _build_prompt(tokenizer) -> str:
    needle = f"IMPORTANT — remember this exactly: the secret passcode is {PASSCODE}.\n\n"
    # Distinct, numbered filler (big enough to exceed 100k tokens; trimmed below).
    filler = "".join(
        f"Log entry {i}: routine system check {i} completed, status nominal, value {i * 7 % 1000}.\n"
        for i in range(1, 40000)
    )
    question = "\n\nBased on the text above, what is the secret passcode? Answer with only the number."
    body = needle + filler
    ids = tokenizer(body, return_tensors="pt").input_ids[0]
    if ids.shape[0] > TARGET_TOKENS:
        body = tokenizer.decode(ids[:TARGET_TOKENS], skip_special_tokens=True)
    messages = [{"role": "user", "content": body + question}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding
    from models.experimental.mistral_small_4_119b.tt_demo_agent import _precompute_rope_table

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    text_cfg = cfg.text_config
    for a in ("attn_implementation", "_attn_implementation"):
        if hasattr(text_cfg, a):
            setattr(text_cfg, a, "eager")

    prompt = _build_prompt(tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    seq_len = input_ids.shape[1]
    max_new = 16
    logger.info(f"Needle prompt: {seq_len} tokens (passcode {PASSCODE} at the start)")

    sd = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(N_LAYERS))
    max_seq = seq_len + max_new + 64
    cos_full, sin_full = _precompute_rope_table(Mistral4RotaryEmbedding, text_cfg, max_seq)

    mesh = _open_mesh_device()
    try:
        model = TtMistral4TextModel(
            mesh_device=mesh,
            state_dict=sd,
            text_config=text_cfg,
            num_decoder_layers=N_LAYERS,
            max_seq_len=max_seq,
        )
        model.cache_rope_tables(cos_full, sin_full)
        del sd

        chunk = int(os.environ.get("MISTRAL4_PREFILL_CHUNK", "512"))
        logger.info(f"Chunked prefill: {seq_len} tokens, chunk={chunk} → {(seq_len + chunk - 1)//chunk} chunks…")
        t0 = time.time()
        next_id = model.prefill_next_token(input_ids)
        logger.info(f"Prefill done in {time.time() - t0:.1f}s")

        gen = [next_id]
        cur = torch.tensor([[next_id]], dtype=torch.long)
        dts = []
        for step in range(1, max_new):
            td = time.time()
            tok = model.decode_next_token(cur, seq_len + step - 1)
            dts.append((time.time() - td) * 1000)
            gen.append(tok)
            cur = torch.tensor([[tok]], dtype=torch.long)
            if tokenizer.eos_token_id is not None and tok == tokenizer.eos_token_id:
                break
        if dts:
            avg = sum(dts) / len(dts)
            logger.info(f"Decode: {avg:.0f} ms/tok ({1000/avg:.1f} tok/s) @ {seq_len} tokens")
        answer = tokenizer.decode(gen, skip_special_tokens=True)
        logger.info("=" * 60)
        logger.info(f"Q: secret passcode?  (needle={PASSCODE}, context={seq_len} tok)")
        logger.info(f"A: {answer!r}")
        logger.info(f"{'✅ PASS — needle retrieved' if PASSCODE in answer else '❌ FAIL — needle not found'}")
        logger.info("=" * 60)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
