#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Minimal profiling driver for the old GRPO ``LlamaGRPOCompleter``.

Runs a single batch of completions through ``LlamaGRPOCompleter.generate_str``
so it can be wrapped under Tracy. The ``py_zone`` markers added in
``utils/llama_completer.py`` will fire from inside ``_completion_batched_impl``
during this run.

Run with::

    cd <repo root>
    TT_METAL_DEVICE_PROFILER=1 \\
    TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT=100000 \\
    python -m tracy -r -v -p \\
        tt-train/sources/examples/grpo/profile_llama_completer.py

The resulting ``tracy_profile_log_host.tracy`` lands under
``${TT_METAL_HOME}/generated/profiler/reports/<timestamp>/`` and can be opened
with the Tracy GUI.

Re-uses ``boolq_accuracy_example.yaml`` (single P150, 1x1 mesh, Llama-3.2-1B).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Silence noisy tt-metal warnings so the prompt + completion print is legible.
# Must be set before any tt-metal / ttnn import.
os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

# When invoked via ``python -m tracy <this_script>`` the script's directory is
# inserted to ``sys.path`` automatically. When invoked directly (no tracy) we
# still want ``utils.llama_completer`` to resolve, so do the same insert.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


from transformers import AutoTokenizer
from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config

from utils.llama_completer import LlamaCompletionCtx, LlamaGRPOCompleter


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# Verbose system prompt to push the model toward producing the full
# ``MAX_COMPLETION_LENGTH`` worth of decode tokens (otherwise Llama-Instruct
# tends to stop after a handful of tokens, giving a very short trace).
SYSTEM_PROMPT = (
    "You are a thorough, long-winded tutor. For every question, write a "
    "multi-paragraph answer that explains every concept in detail, defines "
    "all relevant terms, walks through the reasoning step by step, considers "
    "edge cases, and finishes with an extended summary. Never give a short "
    "answer."
)
USER_PROMPT = (
    "Explain in exhaustive detail how the transformer architecture works, "
    "covering tokenisation, embeddings, positional encodings, multi-head "
    "self-attention, feed-forward networks, residual connections, layer "
    "normalisation, the training objective, and how inference proceeds "
    "autoregressively. Spend several paragraphs on each topic."
)

BATCH_SIZE = 1
MAX_COMPLETION_LENGTH = 8
TEMPERATURE = 0.0
NUM_GENERATIONS = 1


_CONFIG_PATH = Path(__file__).with_name("boolq_accuracy_example.yaml")
_RAW_CONFIG = load_config(str(_CONFIG_PATH))
_TRAINING_CONFIG = TrainingConfig(_RAW_CONFIG)
DEVICE_CONFIG = DeviceConfig(_RAW_CONFIG)
assert _TRAINING_CONFIG.model_config, "training_config.model_config must be set"
TRANSFORMER_CONFIG = get_model_config(_TRAINING_CONFIG.model_config)


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompts = [prompt] * BATCH_SIZE

    print(f"[profile] building LlamaGRPOCompleter (model={MODEL_ID})")
    print(f"[profile] batch_size={BATCH_SIZE} max_new={MAX_COMPLETION_LENGTH} temperature={TEMPERATURE}")

    llama = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=MAX_COMPLETION_LENGTH,
            temperature=TEMPERATURE,
            completions_per_prompt=NUM_GENERATIONS,
        ),
        transformer_config=TRANSFORMER_CONFIG,
        device_config=DEVICE_CONFIG,
        model_source=MODEL_ID,
    )

    print(f"[profile] generating completions for {BATCH_SIZE} prompt(s)")
    t0 = time.perf_counter()
    completions = llama.generate_str(prompts)
    elapsed = time.perf_counter() - t0

    total_chars = sum(len(c) for c in completions)
    print(
        f"[profile] done: {len(completions)} completion(s), "
        f"avg_chars={total_chars / max(1, len(completions)):.1f}, "
        f"elapsed={elapsed:.2f}s"
    )

    print()
    print("[profile] first completion (truncated to 500 chars):")
    print(completions[0][:500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
