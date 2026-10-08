# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Greedy text generation on the device model by repeated prefill (no decode path): each step prefills the prompt plus
the tokens so far as one padded chunk from position 0 and appends the argmax at the last real row. Same loop as the
bring-up smoke (models/demos/common/bringup/testing/smoke.py), with a free prompt, more tokens and streamed output.

    TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 PYTHONPATH=$PWD \\
    BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml \\
    GLM_GEN_PROMPT="Write a haiku about the sea." GLM_GEN_TOKENS=64 \\
    scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_generate.py -s

GLM_GEN_RAW=1 feeds the prompt as is (no chat template). The chunk is the smallest of the spec's chunk sizes (ladder
chunks and target.chunk, the sizes the model builds its per-chunk tables for) that holds the sequence.
"""

import os
import time

import torch

from models.demos.common.bringup.testing.harness import device_timeout, mesh_parametrize, spec
from models.demos.common.bringup.testing.smoke import tokenizer

S = spec()
pytestmark = device_timeout(S)


def _ids(tok, text: str) -> list[int]:
    from models.demos.common.bringup.reference.prompt import template_kwargs

    if os.environ.get("GLM_GEN_RAW") != "1" and getattr(tok, "chat_template", None):
        text = tok.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False, **template_kwargs(S)
        )
    return tok(text, add_special_tokens=False)["input_ids"]


@mesh_parametrize
def test_generate(mesh_device):
    from models.demos.glm53_flash_d_p.bringup.hooks import _chunks

    prompt = os.environ.get("GLM_GEN_PROMPT", "Explain in a few sentences why the sky is blue.")
    max_new = int(os.environ.get("GLM_GEN_TOKENS", "64"))
    tok = tokenizer(S)
    ids = _ids(tok, prompt)
    chunks = sorted(set(_chunks(S)))
    assert len(ids) + max_new <= chunks[-1], f"{len(ids)} + {max_new} tokens exceed the largest chunk {chunks[-1]}"
    layers = S.layers()
    assert layers[0] == 0 and layers[-1] == S.num_layers - 1, "generation needs the whole model"

    t_load = time.time()
    model = S.hooks().device_model(mesh_device, S, layers, lm_head=True)
    print(f"model loaded in {time.time() - t_load:.0f}s; prompt {len(ids)} tokens, chunks {chunks}", flush=True)

    stop = {t for t in (tok.eos_token_id,) if t is not None}
    stop |= {tok.convert_tokens_to_ids(t) for t in ("<|user|>", "<|endoftext|>") if t in tok.get_vocab()}
    out, times = [], []
    for step in range(max_new):
        seq = ids + out
        chunk = next(c for c in chunks if c >= len(seq))
        tokens = torch.tensor(seq + [0] * (chunk - len(seq)), dtype=torch.long)  # pad after: causal, never read
        state = model.new_state(chunk)
        t0 = time.time()
        h = model.embed(tokens)
        for i in layers:
            h2 = model.layer(i, h, 0, state)
            model.free(h)
            h = h2
        hidden = model.final_norm(h)
        model.free(h)
        nxt = int(model.logits(hidden, [len(seq) - 1]).float()[0].argmax())
        model.free(hidden)
        times.append(time.time() - t0)
        print(f"[{step:3d} {times[-1]:5.2f}s chunk {chunk}] {tok.decode([nxt])!r}", flush=True)
        if nxt in stop:
            break
        out.append(nxt)

    text = tok.decode(out)
    warm = times[1:] or times
    print(
        f"\nPROMPT: {prompt}\nOUTPUT ({len(out)} tokens, {sum(warm) / len(warm):.2f} s/token warm):\n{text}\n",
        flush=True,
    )
    assert out, "no tokens generated"
