#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run LlamaGRPOCompleter twice (real weights vs dummy weights) and compare."""

from __future__ import annotations

import os

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import gc
import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"
MAX_NEW_TOKENS = 50
MAX_SEQ_LEN = 2048
TEMPERATURE = 0.7
SEED = 0

PROMPTS: List[str] = [
    "The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization,",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return ",
    "Theorem: For all integers n >= 0, the sum 1 + 2 + ... + n equals n(n+1)/2. Proof:",
    "Once upon a time, in a small village by the sea, there lived an old fisherman named",
]


def run_ttt(prompts: List[str], *, dummy_weights: bool) -> List[str]:
    import ttml
    from ttml.common.config import DeviceConfig, load_config

    from utils.llama_completer_ttt import LlamaGRPOCompleter

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)

    tag = "dummy" if dummy_weights else "real"
    print(f"[ttt-{tag}] building LlamaGRPOCompleter ({MODEL_ID}, max_seq_len={MAX_SEQ_LEN})")
    completer = LlamaGRPOCompleter(
        device_config=device_config,
        model_source=MODEL_ID,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        dummy_weights=dummy_weights,
    )

    out: List[str] = []
    for prompt in prompts:
        prompt_ids = completer.tokenizer.encode(prompt, add_special_tokens=True)
        completions = completer.generate(
            [prompt_ids],
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            seed=SEED,
            stop_at_eos=False,
            enable_trace=False,
        )
        out.append(completer.tokenizer.decode(completions[0], skip_special_tokens=False))

    del completer
    gc.collect()
    ttml.autograd.AutoContext.get_instance().close_device()
    return out


def main() -> None:
    import ttnn

    print("\n>>> set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    print("\n>>> stage ttt-real")
    real_outputs = run_ttt(PROMPTS, dummy_weights=False)

    print("\n>>> stage ttt-dummy")
    dummy_outputs = run_ttt(PROMPTS, dummy_weights=True)

    print("\n>>> side-by-side")
    for i, prompt in enumerate(PROMPTS):
        print()
        print(f"=== Prompt {i + 1}: {prompt!r} ===")
        print(f"  real:  {real_outputs[i]!r}")
        print(f"  dummy: {dummy_outputs[i]!r}")
        print(f"  match: {real_outputs[i] == dummy_outputs[i]}")


if __name__ == "__main__":
    main()
