# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the fp32 host reference predicts along the served answer to OpenRouter's audio transcription request.

The prompt is the request as the server renders it (system prompt, a user turn holding only the audio clip, thinking
on, so the assistant prefix has no ``<think></think>``), followed by a continuation that is teacher-forced. The
reference's top next tokens are printed at every continuation position, so one run shows what the checkpoint itself
wants after ``assistant\\n``, after ``<think>``, and at the end of the transcription. All 48 layers on the host CPU
(~28 s per layer with 24 threads). The device answered ``<think><think>the words<|im_end|>``, and so does the
reference: doc/multimodal_evals.md, "OpenRouter's provider checks".

    python -m models.experimental.mimo_v2_6_flash.reference.answer_audio_fp32 clip.wav --words "This nation ..."
"""

import argparse
import time

import torch
from loguru import logger

from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig
from models.experimental.mimo_v2_6_flash.reference.make_golden_mm import (
    AUDIO_TOKEN,
    PLACEHOLDER,
    ReferenceEncoders,
    load_wav,
)

SYSTEM = "You are a transcription engine. Reply with only the exact words spoken in the audio, with no commentary."
# Teacher-forced continuations after the generation prompt: the served answer, the same with one <think>, and the
# non-thinking form (an empty closed block before the words).
CONTINUATIONS = {
    "served_answer": "<think><think>{words}<|im_end|>",
    "one_think": "<think>{words}<|im_end|>",
    "closed_think": "<think></think>{words}<|im_end|>",
}


def main():
    from transformers import AutoConfig, AutoTokenizer
    from vllm.transformers_utils.processors.mimo_v2_omni import MiMoOmniProcessor

    from models.experimental.mimo_v2_6_flash.tt import multimodal as mm

    parser = argparse.ArgumentParser()
    parser.add_argument("wav")
    parser.add_argument("--words", required=True, help="the transcription the server answered")
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--out", default=None)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--no-system", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    proc = MiMoOmniProcessor.from_hf_config(tok, AutoConfig.from_pretrained(args.model_path, trust_remote_code=True))
    messages = [] if args.no_system else [{"role": "system", "content": SYSTEM}]
    messages.append({"role": "user", "content": PLACEHOLDER["audio"]})
    prefix = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    logger.info(f"rendered prompt: {prefix!r}")

    cfg = MiMoConfig.from_model_path(args.model_path)
    ckpt = MiMoCheckpoint(args.model_path, cfg)
    backend = ReferenceEncoders(args.model_path)
    embedding = ckpt.embedding()
    records = []
    for name, cont in CONTINUATIONS.items():
        cont = cont.format(words=args.words)
        p = proc(text=prefix + cont, audio=[load_wav(args.wav)], return_tensors="pt")
        ids = p["input_ids"][0]
        n_cont = len(tok(cont, add_special_tokens=False)["input_ids"])
        (offset, length), *rest = mm.placeholder_rows(ids, AUDIO_TOKEN)
        assert not rest
        item = mm.MMItem("audio", name, offset, length, None, {"audio_features": p["audio_features"][0]})
        rows = mm.encode_item(backend, item)
        embeds = embedding[ids].to(torch.bfloat16)
        embeds[offset : offset + length] = rows
        logger.info(
            f"{name}: {ids.numel()} tokens ({ids.numel() - n_cont} prompt), audio rows [{offset}, {offset + length})"
        )
        records.append(dict(name=name, ids=ids, n_cont=n_cont, hidden=embeds.float()))
    del embedding

    for layer_idx in range(cfg.num_hidden_layers):
        t0 = time.time()
        layer = ref.ReferenceDecoderLayer(cfg, layer_idx, ckpt=ckpt)
        with torch.no_grad():
            for r in records:
                r["hidden"] = layer(r["hidden"], torch.arange(r["ids"].numel()))
        del layer
        logger.info(f"layer {layer_idx}: {time.time() - t0:.1f}s")

    norm = ckpt.final_norm()
    lm_head = ckpt.lm_head().float()
    with torch.no_grad():
        for r in records:
            n = r["n_cont"] + 1  # the last prompt position predicts the first continuation token
            logp = torch.log_softmax(ref.rms_norm(r["hidden"][-n:], norm, cfg.layernorm_epsilon) @ lm_head.T, dim=-1)
            r["logp_tail"] = logp
            print(f"=== {r['name']}")
            for k in range(n):
                given = tok.decode(r["ids"][-n + k : -n + k + 1])
                forced = tok.decode(r["ids"][-n + k + 1 : -n + k + 2]) if k + 1 < n else "(end)"
                top = logp[k].topk(5)
                alts = ", ".join(f"{tok.decode([int(t)])!r}:{float(v):.2f}" for v, t in zip(top.values, top.indices))
                forced_lp = float(logp[k][r["ids"][-n + k + 1]]) if k + 1 < n else float("nan")
                print(f"  after {given!r:16s} forced {forced!r:14s} {forced_lp:8.2f}   top [{alts}]")
    if args.out:
        torch.save([{k: v for k, v in r.items() if k != "hidden"} for r in records], args.out)


if __name__ == "__main__":
    main()
