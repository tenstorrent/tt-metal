# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Hy4-preview on one Blackhole Galaxy: a chat prompt in, generated text out.

    prompt --chat template, tokenizer--> ids --prefill (the prefill stack, all 78 layers)-->
    KV / index-key caches --bridge--> decode cache --greedy decode on the device--> text

One command (through the lock every device job on this machine takes; ``-s`` shows the text):

    HY4_DECODE_WEIGHT_CACHE=/mnt/data/zni/hy4/decode-cache/weights \
        /mnt/data/zni/hy4/tools/run_device_tests.sh demo models/experimental/hy4_preview/demo/demo.py -s

or, where pytest's fixtures are not wanted, ``python -m models.experimental.hy4_preview.demo.demo``
(opens the 8x4 mesh itself). Environment:

    HY4_HF_MODEL              the checkpoint directory (weights, tokenizer, chat template)
    HY4_DECODE_WEIGHT_CACHE   converted decode weights; written on the first run (about 30 s per
                              layer from the checkpoint), loaded after that
    HY4_DEMO_PROMPT           the user turn (default: DEFAULT_PROMPT)
    HY4_DEMO_TOKENS           tokens to generate (default 32)
    HY4_DEMO_LAYERS           cut the model to its first layers (default 78: the model)

What it does not do: sample (greedy only), stop at the end-of-sequence token on the device (the
steps are issued without reading anything back; the text is cut at the first EOS afterwards), a
second prompt (the prefill and the decode weights do not fit a chip together, tt/model.py), more
than one user. Nothing is traced: a step of the 78 layers takes about two seconds.
"""

import os
import time

import pytest
import torch
from loguru import logger

from models.experimental.hy4_preview.demo.record_reference import DEFAULT_PROMPT, DEFAULT_STEPS
from models.experimental.hy4_preview.tt import prefill_bridge as pb
from models.experimental.hy4_preview.tt.model import WEIGHT_CACHE_ENV, Hy4Engine

MESH = pb.MESH


def chat_ids(tokenizer, prompt: str) -> torch.Tensor:
    """One user turn through the checkpoint's chat template, with the generation prompt: ids [N]."""
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True
    )
    return torch.tensor(encoded if isinstance(encoded, list) else encoded["input_ids"], dtype=torch.long).flatten()


def run_demo(mesh_device, prompt: str = DEFAULT_PROMPT, max_new_tokens: int = DEFAULT_STEPS, num_layers: int = 78):
    """Prefill ``prompt``, decode ``max_new_tokens`` greedy tokens, return {"text", "token_ids",
    "prompt_tokens", "seconds"}. ``mesh_device`` is the 8x4 mesh opened with
    prefill_bridge.device_params()."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(os.environ["HY4_HF_MODEL"])
    ids = chat_ids(tokenizer, prompt)
    logger.info(f"prompt, {ids.numel()} tokens: {tokenizer.decode(ids.tolist())!r}")
    engine = Hy4Engine(
        mesh_device,
        num_layers=num_layers,
        max_context=ids.numel() + max_new_tokens,
        weight_cache=os.environ.get(WEIGHT_CACHE_ENV) or None,
    )
    try:
        t0 = time.perf_counter()
        n = engine.prefill({0: ids})[0]
        engine.load()
        first = engine.generate(1)
        first_token_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        rest = engine.generate(max_new_tokens - 1) if max_new_tokens > 1 else first[:0]
        decode_s = time.perf_counter() - t1
    finally:
        engine.close()
    tokens = torch.cat([first, rest])[:, 0].tolist()
    eos = tokenizer.eos_token_id
    if eos in tokens:
        tokens = tokens[: tokens.index(eos)]
    text = tokenizer.decode(tokens)
    seconds = dict(engine.seconds, first_token=first_token_s, decode=decode_s)
    same_as_hf = _against_hf(num_layers, ids, max_new_tokens, torch.cat([first, rest])[:, 0])
    prefill_s = engine.seconds["prefill / prefill"][0]
    logger.info(
        f"{num_layers} layers, untraced: first token after {first_token_s:.1f} s (prefill of {n} tokens "
        f"{prefill_s:.2f} s, decode weights {engine.seconds['load_decode_weights']:.1f} s); then "
        + (f"{(max_new_tokens - 1) / decode_s:.2f} tokens per second" if max_new_tokens > 1 else "nothing more")
    )
    logger.info(f"generated: {text!r}")
    print(f"\n=== prompt ===\n{prompt}\n=== Hy4-preview, {len(tokens)} tokens ===\n{text}\n")
    return {"text": text, "token_ids": tokens, "prompt_tokens": n, "seconds": seconds, "same_as_hf": same_as_hf}


def _against_hf(num_layers: int, ids: torch.Tensor, steps: int, generated: torch.Tensor):
    """The generated ids against HF's greedy generation of the same prompt, where its golden has been
    recorded (demo/record_reference.py): (tokens equal, tokens equal in a row from the first, HF's
    top-1 margin at the first token that differs or None), or None when there is no golden."""
    from models.experimental.hy4_preview.reference import hf_harness

    path = hf_harness.decode_golden_path(num_layers, ids.numel(), steps, ids)
    if not path.is_file():
        logger.warning(f"no HF generation recorded for this prompt at {num_layers} layers ({path}): not compared")
        return None
    golden = torch.load(path, weights_only=False)
    assert torch.equal(golden["input_ids"][: ids.numel()], ids), f"{path} was recorded for another prompt"
    hf_tokens = golden["input_ids"][ids.numel() : ids.numel() + steps]
    same = generated == hf_tokens
    prefix = int(same.long().cumprod(0).sum())
    margin = None if prefix == steps else golden["logits"][prefix].topk(2).values.diff().abs().item()
    logger.info(
        f"against HF's greedy generation ({num_layers} layers, BF16): {int(same.sum())} of {steps} tokens equal, the "
        f"first {prefix} in a row" + ("" if margin is None else f"; HF's top-1 margin where they part: {margin:.3f}")
    )
    return int(same.sum()), prefix, margin


@pytest.mark.parametrize("device_params, mesh_device", [(pb.device_params(), MESH)], indirect=True, ids=["8x4-torus"])
@pytest.mark.timeout(0)
def test_demo(mesh_device):
    result = run_demo(
        mesh_device,
        prompt=os.environ.get("HY4_DEMO_PROMPT", DEFAULT_PROMPT),
        max_new_tokens=int(os.environ.get("HY4_DEMO_TOKENS", DEFAULT_STEPS)),
        num_layers=int(os.environ.get("HY4_DEMO_LAYERS", 78)),
    )
    assert result["token_ids"], "nothing was generated"
    if result["same_as_hf"] is not None:
        equal, prefix, margin = result["same_as_hf"]
        # the device may leave HF's greedy path only where HF's own top-1 leads by less than a logit
        assert margin is None or margin < 1.0, f"left HF's generation after {prefix} tokens at a margin of {margin:.3f}"


def main():
    import ttnn

    mesh_device = pb.open_mesh()
    try:
        run_demo(
            mesh_device,
            prompt=os.environ.get("HY4_DEMO_PROMPT", DEFAULT_PROMPT),
            max_new_tokens=int(os.environ.get("HY4_DEMO_TOKENS", DEFAULT_STEPS)),
            num_layers=int(os.environ.get("HY4_DEMO_LAYERS", 78)),
        )
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()
