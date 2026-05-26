#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Side-by-side sampled generation across three backends.

Runs the same completion-style prompts through:

* ``hf``  – HuggingFace ``transformers.AutoModelForCausalLM`` (CPU, fp32)
* ``ttml`` – the ttml ``Llama`` model via :class:`LlamaCompleterTtml`
* ``ttt``  – the new tt-transformers-backed :class:`LlamaGRPOCompleter`

and prints the decoded text outputs side by side. The two TT backends share
a single mesh device: ``ttml`` opens it first, releases its model, the
device is closed, then ``ttt`` reopens it for its own run.

EOS-stopping is disabled and decode keeps special tokens so you can see
exactly what each backend emits (including ``</s>`` markers for base
models that greedy-prefer to terminate early).
"""

from __future__ import annotations

import os

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import sys
from pathlib import Path
from typing import List

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"
TTML_MODEL_CONFIG_REL = "tt-train/configs/model_configs/llama3_2_1B.yaml"
MAX_NEW_TOKENS = 50
MAX_SEQ_LEN = 2048

# Sampling settings shared by all three backends. ``top_p`` and
# ``repetition_penalty`` are intentionally left at HF/library defaults
# (1.0) so all three pipelines see the same softmax-and-multinomial
# decision rule (modulo numerical precision and per-step RNG source).
# Greedy (temperature=0) degenerates to EOS on this base model.
TEMPERATURE = 0.7
SEED = 0

# Completion-style prompts — base models continue text rather than answer Q/A.
# Longer/anchored prompts push EOS probability down enough that sampling
# actually picks non-EOS tokens.
PROMPTS: List[str] = [
    "The TinyLlama project aims to pretrain a 1.1B Llama model on 3 trillion tokens. With some proper optimization,",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return ",
    "Theorem: For all integers n >= 0, the sum 1 + 2 + ... + n equals n(n+1)/2. Proof:",
    "Once upon a time, in a small village by the sea, there lived an old fisherman named",
]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


def run_hf(prompts: List[str], pad_token_id: int) -> List[str]:
    """HuggingFace sampled generation (fp32, CPU). Matches model card defaults."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[hf] loading {MODEL_ID} (fp32, CPU)")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()

    torch.manual_seed(SEED)
    out: List[str] = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
            attention_mask = torch.ones_like(input_ids)  # works on batch_size=1
            generated = model.generate(
                input_ids,
                attention_mask=attention_mask,
                min_new_tokens=MAX_NEW_TOKENS,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=pad_token_id,
            )
            new_tokens = generated[0, input_ids.shape[1] :]
            out.append(tokenizer.decode(new_tokens, skip_special_tokens=False))
    return out


def run_ttml(prompts: List[str]) -> List[str]:
    """ttml ``Llama`` generation via ``LlamaCompleterTtml``.

    Opens the autograd-context mesh device, builds the ttml model, runs
    ``generate_str`` for the batch of prompts, then drops the model and
    closes the device so :func:`run_ttt` can reopen it cleanly.
    """
    import gc

    import numpy as np

    import ttml
    from ttml.common.config import DeviceConfig, get_model_config, load_config

    from utils.llama_completer_ttml import LlamaCompleterTtml, LlamaCompletionCtx

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)
    transformer_config = get_model_config(os.path.join(REPO_ROOT, TTML_MODEL_CONFIG_REL))

    np.random.seed(SEED)

    ctx = LlamaCompletionCtx(
        max_tokens_to_complete=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    print(f"[ttml] building LlamaCompleterTtml ({MODEL_ID})")
    completer = LlamaCompleterTtml(
        ctx=ctx,
        transformer_config=transformer_config,
        device_config=device_config,
        model_source=MODEL_ID,
    )

    out = completer.generate_str(prompts)

    del completer
    gc.collect()
    ttml.autograd.AutoContext.get_instance().close_device()
    return out


def run_ttt(prompts: List[str]) -> List[str]:
    """tt-transformers generation via ``LlamaGRPOCompleter``.

    Reopens the mesh device (closed by :func:`run_ttml`) via the new
    completer's :meth:`setup_device`, runs prefill+decode, then drops the
    model and closes the device for symmetry.
    """
    import gc

    import ttml
    from ttml.common.config import DeviceConfig, load_config

    from utils.llama_completer_ttt import LlamaGRPOCompleter

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)

    # ``LlamaGRPOCompleter`` is hardwired to bf16 weights / activations /
    # collectives, matching ``run_ttml``'s precision so any pre-EOS quality
    # gap comes from the implementation rather than from quantization.
    print(f"[ttt] building LlamaGRPOCompleter ({MODEL_ID}, max_seq_len={MAX_SEQ_LEN})")
    completer = LlamaGRPOCompleter(
        device_config=device_config,
        model_source=MODEL_ID,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import ttnn
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("\n>>> stage hf")
    hf_outputs = run_hf(PROMPTS, pad_token_id=tokenizer.eos_token_id)

    # set_fabric_config must run before any device is opened. Doing it once
    # here keeps the ``ttml``/``ttt`` device-open/close cycles symmetric and
    # avoids re-setting fabric between them (which the runtime rejects).
    print("\n>>> set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    print("\n>>> stage ttml")
    ttml_outputs = run_ttml(PROMPTS)

    print("\n>>> stage ttt")
    ttt_outputs = run_ttt(PROMPTS)

    print("\n>>> side-by-side")
    for i, prompt in enumerate(PROMPTS):
        print()
        print(f"=== Prompt {i + 1}: {prompt!r} ===")
        print(f"  hf:   {hf_outputs[i]!r}")
        print(f"  ttml: {ttml_outputs[i]!r}")
        print(f"  ttt:  {ttt_outputs[i]!r}")


if __name__ == "__main__":
    main()
