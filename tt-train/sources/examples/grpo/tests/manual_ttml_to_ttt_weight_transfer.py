# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Manual exploration: ttml -> TTT weight transfer with batched sampling.

Usage:
    python manual_ttml_to_ttt_weight_transfer.py

What it does:
  1. Build a tt-transformers (TTT) completer with the **base**
     ``meta-llama/Llama-3.2-1B`` weights, sized for ``BATCH_SIZE``.
  2. Build a ttml completer with the **instruct**
     ``meta-llama/Llama-3.2-1B-Instruct`` weights, reusing the same
     already-open mesh device.
  3. Generate ``BATCH_SIZE`` completions from TTT with the **base**
     weights and print them. (The base model has no instruction-following
     training, so this row tends to look like raw web text -- a useful
     baseline.)
  4. Transfer ttml's weights into TTT in place via
     ``LlamaCompositeKV.export_to_hf_dict`` +
     ``Transformer.update_weights``.
  5. Generate ``BATCH_SIZE`` more completions from TTT with the
     **transferred** instruct weights and print them.

  Both rows are sampled with ``temperature=0.7`` so they show diverse
  draws from the same prompt rather than a single greedy reply.

The prompt is built with the Llama-3.2 instruct chat template
(``system`` + ``user``) so the transferred instruct weights see input
in the format they were trained on. Default content is the canonical
GSM8K "Natalia clips" word problem -- a useful smoke test for whether
the transferred weights can do small step-by-step arithmetic. (Expected
answer: 48 + 24 = 72.)

This is intentionally **not** a pytest test: there is no assertion. The
point is to eyeball the diversity / coherence of the transferred-weight
output, look for obvious regressions (degenerate repetitions, gibberish,
wrong language, wrong arithmetic), and sanity-check that batched
sampling produces distinct completions from identical prompts.

HF auth: requires ``HF_TOKEN`` in the environment (both repos are gated).
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

HERE = Path(__file__).resolve().parent
GRPO_SPEEDUP = HERE.parent
REPO_ROOT = HERE.parents[4]  # .../tt-metal

for _p in (str(HERE), str(GRPO_SPEEDUP), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ttnn  # noqa: E402

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

from _completer_utils import (  # noqa: E402
    build_completer,
    close_device,
    load_device_config,
    open_device,
)

BASE_MODEL_ID = "meta-llama/Llama-3.2-1B"
INSTRUCT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

SYSTEM_PROMPT = (
    "You are a competent math professor. Answer this question step by step, "
    "show your reasoning. Put #### before final answer"
)
USER_PROMPT = (
    "Natalia sold clips to 48 of her friends in April, and then she sold "
    "half as many clips in May. How many clips did Natalia sell altogether "
    "in April and May?"
)

BATCH_SIZE = 32
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.7
TOP_P = 1.0


def _build_ttml_completer(mesh_device: Any, *, enable_ddp: bool, raw: dict, model_source: str) -> Any:
    """Build a :class:`LlamaCompleterTtml` on the already-open ``mesh_device``."""
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


def _build_chat_prompt_ids(tokenizer: Any) -> list[int]:
    """Apply the Llama-3.2 instruct chat template and tokenize.

    ``add_generation_prompt=True`` appends the
    ``<|start_header_id|>assistant<|end_header_id|>`` opener so the model
    continues *as the assistant* from the very first sampled token.
    Returns a flat ``list[int]`` of token IDs (no batch dim).
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    return [int(t) for t in ids]


def _print_completion(idx: int, completion_text: str, n_tokens: int) -> None:
    print(f"--- [{idx:02d}] ({n_tokens} tokens) ---")
    print(completion_text.rstrip())
    print("")


def _print_completions_block(
    label: str,
    completions: list[list[int]],
    tokenizer: Any,
) -> None:
    """Print one batch of completions under a labelled header.

    Decoded with ``skip_special_tokens=True`` for readable prose; token
    counts are still printed per completion so a cap-hit is visible.
    """
    banner = f"===== {label} ====="
    print("\n" + banner)
    print(f"system: {SYSTEM_PROMPT!r}")
    print(f"user  : {USER_PROMPT!r}")
    print(f"temperature={TEMPERATURE}  top_p={TOP_P}  batch_size={BATCH_SIZE}")
    print("(expected answer: 72)")
    print("")
    for i, ids in enumerate(completions):
        text = tokenizer.decode(ids, skip_special_tokens=True)
        _print_completion(i, text, len(ids))
    print("=" * len(banner) + "\n")


def main() -> None:
    print(f"[manual] base TTT       : {BASE_MODEL_ID}")
    print(f"[manual] instruct ttml  : {INSTRUCT_MODEL_ID}")
    print(f"[manual] system prompt  : {SYSTEM_PROMPT!r}")
    print(f"[manual] user prompt    : {USER_PROMPT!r}")
    print(f"[manual] batch_size     : {BATCH_SIZE}")
    print(f"[manual] temperature    : {TEMPERATURE}")
    print(f"[manual] top_p          : {TOP_P}")
    print(f"[manual] max_new_tokens : {MAX_NEW_TOKENS}")
    print("")

    device_config, raw = load_device_config()
    mesh_device = open_device(device_config)
    ttt = ttml_completer = None
    try:
        print("[manual] building TTT completer (base) ...")
        ttt = build_completer(
            mesh_device,
            dummy_weights=False,
            max_batch_size=BATCH_SIZE,
            model_source=BASE_MODEL_ID,
            instruct=False,
        )

        print("[manual] building ttml completer (instruct, shared device) ...")
        ttml_completer = _build_ttml_completer(
            mesh_device,
            enable_ddp=device_config.enable_ddp,
            raw=raw,
            model_source=INSTRUCT_MODEL_ID,
        )

        # Use ttml_completer's tokenizer: it was loaded from the
        # -Instruct repo, which is the only one of the two that ships
        # a chat_template. Vocab is byte-identical to the base
        # tokenizer, so the IDs are valid for TTT too.
        prompt_ids = _build_chat_prompt_ids(ttml_completer.tokenizer)
        print(f"[manual] chat-templated prompt tokens: {len(prompt_ids)}")

        print(f"[manual] generating {BATCH_SIZE} completions from BASE TTT weights (pre-transfer) ...")
        base_completions = ttt.generate(
            [list(prompt_ids) for _ in range(BATCH_SIZE)],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        _print_completions_block(
            "TTT (Llama-3.2-1B BASE weights, pre-transfer)",
            base_completions,
            ttt.tokenizer,
        )

        print("[manual] exporting ttml weights -> HF-keyed on-device dict ...")
        hf_dict = ttml_completer.model.export_to_hf_dict()
        print("[manual] running ttt.model.update_weights(...) in place ...")
        ttt.model.update_weights(hf_dict, hf_rope=False)
        del hf_dict
        gc.collect()
        print("[manual] weight transfer done")

        print(f"[manual] generating {BATCH_SIZE} completions from INSTRUCT TTT weights (post-transfer) ...")
        instr_completions = ttt.generate(
            [list(prompt_ids) for _ in range(BATCH_SIZE)],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        _print_completions_block(
            "TTT (Llama-3.2-1B-Instruct weights via ttml, post-transfer)",
            instr_completions,
            ttt.tokenizer,
        )
    finally:
        ttml_completer = None
        ttt = None
        gc.collect()
        close_device()


if __name__ == "__main__":
    main()
