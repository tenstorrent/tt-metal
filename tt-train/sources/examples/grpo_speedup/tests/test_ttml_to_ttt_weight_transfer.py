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
    3. Greedy-generate from TTT (base) -> ``tokens_base``.
    4. Call ``ttml_model.export_to_hf_dict()`` and feed the resulting
       HF-keyed dict of on-device tensors into
       ``ttt.model.update_weights(..., hf_rope=False)``.
    5. Greedy-generate from TTT again -> ``tokens_instr``.
    6. Assert ``tokens_base != tokens_instr``.

This is a coarse smoke test that verifies:

* ``Transformer.update_weights`` actually wired every kwarg to every
  leaf ``.update()`` (otherwise the dispatch would leave some weights
  pointing at base values and the assertion could pass for the wrong
  reason -- the per-key strict consumption check in
  ``Transformer.update_weights`` catches dispatch typos earlier).
* ``LlamaCompositeKV.export_to_hf_dict`` produces a dict that matches
  the consumer's expectations: HF dot-keys, 4D shapes, Meta-permuted
  Q/K rows, K/V split rows, tied embed/lm_head fan-out.

Token IDs and decoded strings are both printed so a failure investigation
can see whether the output drifted in tokenisation or in vocabulary.

HF auth: requires ``HF_TOKEN`` set in the environment; both repos are
gated.
"""

from __future__ import annotations

import gc
import os
from typing import Any

import pytest

from _completer_utils import (
    TTML_DEVICE_CONFIG_REL,
    REPO_ROOT,
    build_completer,
    teardown_completer,
)

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
INSTRUCT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # greedy -> deterministic, exact-equality comparable


def _build_ttml_completer_reusing_device(model_source: str) -> Any:
    """Build a ttml ``LlamaCompleterTtml`` that reuses the already-open device.

    The TTT completer opens the device first; calling ``open_device`` a
    second time fails. We subclass and override ``setup_device`` to
    return the existing ``AutoContext`` mesh, which is the workflow the
    base class documents for tests that co-host multiple completers on
    one device.
    """
    import ttml
    from ttml.common.config import DeviceConfig, get_model_config, load_config

    from utils.llama_completer_ttml import LlamaCompleterTtml, LlamaCompletionCtx
    from utils.llama_completer_ttt import LlamaGRPOCompleter  # noqa: F401  (force import order)

    class _LlamaCompleterTtmlReusingDevice(LlamaCompleterTtml):
        def setup_device(self, device_config):
            return ttml.autograd.AutoContext.get_instance().get_device()

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)

    # Use the same TransformerConfig the standard ttml completer would
    # build for this training yaml so RoPE scaling, vocab, and block
    # count match what HF Llama-3.2-1B-Instruct expects.
    tf_config = get_model_config(raw["training_config"]["model_config"])

    ctx = LlamaCompletionCtx(
        max_tokens_to_complete=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    return _LlamaCompleterTtmlReusingDevice(
        ctx=ctx,
        transformer_config=tf_config,
        device_config=device_config,
        model_source=model_source,
    )


def _decode(completer: Any, ids: list[int]) -> str:
    return completer.tokenizer.decode(ids, skip_special_tokens=False)


@pytest.fixture(scope="module")
def completers():
    """Open the device, build TTT(base) and ttml(instruct), tear down on exit.

    ``LlamaGRPOCompleter`` opens the mesh device in its constructor;
    ``_LlamaCompleterTtmlReusingDevice`` reuses it. We tear down by
    dropping ttml first (no device close), then ``teardown_completer``
    on TTT which releases the device.
    """
    ttt = build_completer(
        dummy_weights=False,
        max_batch_size=1,
        model_source=BASE_MODEL_ID,
        instruct=False,
    )
    try:
        ttml_completer = _build_ttml_completer_reusing_device(INSTRUCT_MODEL_ID)
        try:
            yield ttt, ttml_completer
        finally:
            del ttml_completer
            gc.collect()
    finally:
        teardown_completer(ttt)


def _generate_ttt(completer: Any, prompt_ids: list[int]) -> list[int]:
    return completer.generate(
        [prompt_ids],
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
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

    tokens_instr = _generate_ttt(ttt, prompt_ids)
    text_instr = _decode(ttt, tokens_instr)

    print("\n========= ttml -> TTT weight transfer test =========")
    print(f"prompt                    : {PROMPT!r}")
    print(f"tokens_ttt_base     (#{len(tokens_base):2d}): {tokens_base}")
    print(f"completion_ttt_base       : {text_base!r}")
    print(f"tokens_ttml_instruct (#{len(tokens_ttml_instruct):2d}): {tokens_ttml_instruct}")
    print(f"completion_ttml_instruct  : {text_ttml_instruct!r}")
    print(f"tokens_ttt_instruct (#{len(tokens_instr):2d}): {tokens_instr}")
    print(f"completion_ttt_instruct   : {text_instr!r}")
    print("====================================================\n")

    assert tokens_base != tokens_instr, (
        "TTT generated the same tokens before and after loading ttml's instruct "
        "weights. Either the export, the update_weights dispatch, or one of the "
        "leaf .update() methods is a no-op for this configuration."
    )
