# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end ttml -> tt-transformers weight transfer test.

Scenario:

    1. Build a tt-transformers (TTT) completer with the **base**
       ``meta-llama/Llama-3.2-1B`` weights.
    2. Build a ttml completer with the **instruct**
       ``meta-llama/Llama-3.2-1B-Instruct`` weights (sharing the same
       open mesh device).
    3. Greedy-generate from TTT (base) -> ``tokens_base``.  Captures
       the prefill+decode traces against the base-weight buffers.
    4. Call ``ttml_model.export_to_hf_dict()`` and feed the resulting
       HF-keyed dict of on-device tensors into
       ``ttt.model.update_weights(..., hf_rope=False)``.
    5. Greedy-generate from TTT again with ``enable_trace=True`` ->
       ``tokens_instr``.  Replays the cached traces; the recorded
       buffer addresses are still valid because every leaf
       ``.update()`` uses ``_inplace_copy``.
    6. Greedy-generate from TTT once more with ``enable_trace=False``
       -> ``tokens_instr_no_trace``.  Bypasses the trace cache and
       executes the graph eagerly against the same in-place-updated
       weights.
    7. Assert ``tokens_base != tokens_instr`` and
       ``tokens_instr == tokens_instr_no_trace``.

This is a coarse smoke test that verifies:

* ``Transformer.update_weights`` actually wired every kwarg to every
  leaf ``.update()`` (otherwise the dispatch would leave some weights
  pointing at base values and the assertion could pass for the wrong
  reason -- the per-key strict consumption check in
  ``Transformer.update_weights`` catches dispatch typos earlier).
* ``LlamaCompositeKV.export_to_hf_dict`` produces a dict that matches
  the consumer's expectations: HF dot-keys, 4D shapes, Meta-permuted
  Q/K rows, K/V split rows, tied embed/lm_head fan-out.
* The trace replay in step 5 is reading the *updated* weight bytes,
  not stale memory.  If any leaf ``.update()`` ever reallocated a
  buffer (regression) the trace would read freed memory and step 5
  would crash or diverge from step 6.  Greedy sampling
  (``temperature=0``) makes the comparison bit-exact.

Token IDs and decoded strings are both printed so a failure investigation
can see whether the output drifted in tokenisation or in vocabulary.

HF auth: requires ``HF_TOKEN`` set in the environment; both repos are
gated.
"""

from __future__ import annotations

import gc
from typing import Any

import pytest

from _completer_utils import (
    build_completer,
    close_device,
    load_device_config,
    open_device,
)

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
INSTRUCT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # greedy -> deterministic, exact-equality comparable


def _build_ttml_completer(mesh_device: Any, *, enable_ddp: bool, raw: dict, model_source: str) -> Any:
    """Build a :class:`LlamaCompleterTtml` on ``mesh_device``.

    Uses the same ``TransformerConfig`` the standard ttml completer would
    build for this training yaml so RoPE scaling, vocab, and block count
    match what HF Llama-3.2-1B-Instruct expects.
    """
    from ttml.common.config import get_model_config

    from utils.llama_completer_ttml import LlamaCompleterTtml, LlamaCompletionCtx

    tf_config = get_model_config(raw["training_config"]["model_config"])

    ctx = LlamaCompletionCtx(
        max_tokens_to_complete=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    return LlamaCompleterTtml(
        ctx=ctx,
        transformer_config=tf_config,
        mesh_device=mesh_device,
        model_source=model_source,
        enable_ddp=enable_ddp,
    )


def _decode(completer: Any, ids: list[int]) -> str:
    return completer.tokenizer.decode(ids, skip_special_tokens=False)


@pytest.fixture(scope="module")
def completers():
    """Open one mesh, host TTT(base) + ttml(instruct) on it, tear down on exit.

    Both completers take an already-open ``mesh_device`` kwarg, so a
    single :func:`open_device` call serves both. The finally clause
    drops ttml first, then TTT, then closes the AutoContext mesh, so
    every on-device tensor is freed before the mesh goes away.
    """
    device_config, raw = load_device_config()
    mesh_device = open_device(device_config)
    ttt = ttml_completer = None
    try:
        ttt = build_completer(
            mesh_device,
            dummy_weights=False,
            max_batch_size=1,
            model_source=BASE_MODEL_ID,
            instruct=False,
        )
        ttml_completer = _build_ttml_completer(
            mesh_device,
            enable_ddp=device_config.enable_ddp,
            raw=raw,
            model_source=INSTRUCT_MODEL_ID,
        )
        yield ttt, ttml_completer
    finally:
        ttml_completer = None
        ttt = None
        gc.collect()
        close_device()


def _generate_ttt(completer: Any, prompt_ids: list[int], *, enable_trace: bool = True) -> list[int]:
    return completer.generate(
        [prompt_ids],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        enable_trace=enable_trace,
    )[0]


def _generate_ttml(completer: Any, prompt_ids: list[int]) -> list[int]:
    """``LlamaCompleterTtml.generate`` takes ``List[List[int]]`` and uses
    ``ctx.max_tokens_to_complete`` / ``ctx.temperature`` (set in the fixture)
    instead of per-call kwargs. Returns the completion tokens for one prompt.
    """
    completions = completer.generate([prompt_ids])
    return [int(t) for t in completions[0]]


def test_ttml_to_ttt_weight_transfer_changes_output(completers):
    ttt, ttml_completer = completers

    prompt_ids = ttt.tokenizer.encode(PROMPT, add_special_tokens=True)

    tokens_base = _generate_ttt(ttt, prompt_ids)
    text_base = _decode(ttt, tokens_base)

    tokens_ttml_instruct = _generate_ttml(ttml_completer, prompt_ids)
    text_ttml_instruct = ttml_completer.tokenizer.decode(tokens_ttml_instruct, skip_special_tokens=False)

    hf_dict = ttml_completer.model.export_to_hf_dict()
    try:
        ttt.model.update_weights(hf_dict, hf_rope=False)
    finally:
        # Free the per-layer k_proj/v_proj slices that export_to_hf_dict
        # newly allocated (the rest of the dict aliases ttml's params and
        # is owned by ttml).
        del hf_dict
        gc.collect()

    tokens_instr = _generate_ttt(ttt, prompt_ids, enable_trace=True)
    text_instr = _decode(ttt, tokens_instr)

    tokens_instr_no_trace = _generate_ttt(ttt, prompt_ids, enable_trace=False)
    text_instr_no_trace = _decode(ttt, tokens_instr_no_trace)

    print("\n========= ttml -> TTT weight transfer test =========")
    print(f"prompt                          : {PROMPT!r}")
    print(f"tokens_ttt_base           (#{len(tokens_base):2d}): {tokens_base}")
    print(f"completion_ttt_base             : {text_base!r}")
    print(f"tokens_ttml_instruct      (#{len(tokens_ttml_instruct):2d}): {tokens_ttml_instruct}")
    print(f"completion_ttml_instruct        : {text_ttml_instruct!r}")
    print(f"tokens_ttt_instruct       (#{len(tokens_instr):2d}): {tokens_instr}")
    print(f"completion_ttt_instruct         : {text_instr!r}")
    print(f"tokens_ttt_instr_no_trace (#{len(tokens_instr_no_trace):2d}): {tokens_instr_no_trace}")
    print(f"completion_ttt_instr_no_trace   : {text_instr_no_trace!r}")
    print("====================================================\n")

    assert tokens_base != tokens_instr, (
        "TTT generated the same tokens before and after loading ttml's instruct "
        "weights. Either the export, the update_weights dispatch, or one of the "
        "leaf .update() methods is a no-op for this configuration."
    )

    assert tokens_instr == tokens_instr_no_trace, (
        "TTT produced different tokens with enable_trace=True vs "
        "enable_trace=False after the weight transfer. The traces captured "
        "before update_weights() are reading stale or wrong-address memory, "
        "which means at least one leaf .update() did not preserve its buffer "
        "address (regression in the _inplace_copy invariant).\n"
        f"  with trace   : {tokens_instr}\n"
        f"  without trace: {tokens_instr_no_trace}"
    )
