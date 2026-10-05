# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Writes the teacher-forcing reference `test_e2e.py` scores the device against.

One CustomVoice utterance sampled on CPU by the fp32 reference, with the checkpoint's own
settings from a fixed seed, and the reference's argmax at every codebook of every frame.
The device is then fed these codes and scored on whether its argmax agrees: top-1 and
top-5 token accuracy, the way the LLM demos score themselves.

CPU only, a few minutes at 1.7B. Rerun it when the reference or the checkpoint changes:
    HF_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-Base python -m models.demos.audio.qwen3_tts.tests.e2e.generate_reference
    HF_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-Base python -m models.demos.audio.qwen3_tts.tests.e2e.generate_reference
"""

import os

import torch

from models.demos.audio.qwen3_tts import sampling, weights
from models.demos.audio.qwen3_tts.reference.qwen3_code_predictor_ref import (
    CodePredictorReference,
    build_input_embeddings,
)
from models.demos.audio.qwen3_tts.reference.qwen3_talker_ref import TalkerReference
from models.demos.audio.qwen3_tts.tests.checkpoints import use_release
from models.demos.audio.qwen3_tts.tests.e2e.references import LANGUAGE, MAX_FRAMES, SEED, SPEAKER, TEXT, path
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_pipeline import (
    CONTROL_ID_COUNT,
    MIN_FRAMES,
    HostEmbeddings,
    build_custom_voice_prefill,
)


@torch.inference_mode()
def generate():
    tables = HostEmbeddings()
    talker = TalkerReference()
    predictor = CodePredictorReference()
    generation = weights.generation_config()
    config = weights.talker_config()
    eos, vocab = config["codec_eos_token_id"], config["vocab_size"]
    control = [index for index in range(vocab - CONTROL_ID_COUNT, vocab) if index != eos]
    generator = torch.Generator().manual_seed(SEED)

    embeddings, _ = build_custom_voice_prefill(TEXT, SPEAKER, LANGUAGE, tables)
    codes, top1, seen = [], [], []
    for step in range(MAX_FRAMES):
        hidden = talker(embeddings)[:, -1:, :]
        logits = (hidden.reshape(-1) @ tables.codec_head.T).reshape(-1)
        first = sampling.sample(
            logits,
            seen=seen,
            temperature=generation.get("temperature", 1.0),
            top_k=generation.get("top_k", 0),
            top_p=generation.get("top_p", 1.0),
            penalty=generation.get("repetition_penalty", 1.0),
            generator=generator,
            suppress=control if step >= MIN_FRAMES else control + [eos],
        )
        if first == eos:
            break
        seen.append(first)

        frame, picks = [first], [int(logits.argmax())]
        for group in range(predictor.groups - 1):
            projected = predictor.model.small_to_mtp_projection(build_input_embeddings(hidden, frame))
            state = predictor.model.model(inputs_embeds=projected).last_hidden_state
            row = predictor.model.lm_head[group](state[:, -1]).reshape(-1)
            picks.append(int(row.argmax()))
            frame.append(
                sampling.sample(
                    row,
                    temperature=generation.get("subtalker_temperature", 1.0),
                    top_k=generation.get("subtalker_top_k", 0),
                    top_p=generation.get("subtalker_top_p", 1.0),
                    generator=generator,
                )
            )
        codes.append(frame)
        top1.append(picks)
        embeddings = torch.cat([embeddings, tables.frames(frame) + tables.tts_pad], dim=1)
        print(f"  frame {step}: {frame[0]}", flush=True)
    else:
        raise RuntimeError(f"no end of speech in {MAX_FRAMES} frames")

    return {"codes": torch.tensor(codes, dtype=torch.int16), "top1": torch.tensor(top1, dtype=torch.int16)}


def main():
    release = use_release("custom_voice")
    next(release)
    try:
        reference = generate()
        os.makedirs(os.path.dirname(path()), exist_ok=True)
        torch.save(reference, path())
        print(f"{path()}: {reference['codes'].shape[0]} frames")
    finally:
        next(release, None)


if __name__ == "__main__":
    main()
