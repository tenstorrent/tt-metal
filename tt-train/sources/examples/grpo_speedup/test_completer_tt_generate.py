#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for ``LlamaGRPOCompleter.generate``.

Verifies the simplified completer's full lifecycle on a single device:
   1. construct a ``LlamaGRPOCompleter`` (opens mesh, loads tokenizer, builds ``ModelArgs``),
   2. build a Meta-style state dict via ``completer.model_args.load_state_dict()``,
   3. ``completer.load_weights(state_dict)`` to build the ``Transformer`` + ``Generator``,
   4. ``completer.generate([prompt_ids], max_new_tokens=...)`` and decode the result.

There's no PCC check here; this is purely the integration smoke test that
exercises the new public surface. Edit the constants below to change the
checkpoint or the prompt.

Run with:

    python tt-train/sources/examples/grpo_speedup/test_completer_tt_generate.py
"""

from __future__ import annotations

import os

# Silence noisy tt-metal warnings so the prompt + completion print is
# legible. Must be set before any tt-metal / ttnn import.
os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_ID = "TinyLlama/TinyLlama_v1.1_math_code"
TTML_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"

PROMPT = "Q: A train travels at 60 mph for 2.5 hours. How far does it go?\nA:"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7  # 0 == greedy
MAX_SEQ_LEN = 1024


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import ttnn

    from ttml.common.config import DeviceConfig, load_config

    from utils.llama_completer import LlamaGRPOCompleter

    # tt-transformers expects fabric_config to be set BEFORE any mesh device
    # is opened. ``LlamaGRPOCompleter.setup_device`` opens the mesh inside
    # the constructor for single device without enabling fabric, so we do it
    # here. FABRIC_2D matches ``gen_hf_ttt.py``.
    print("[test] set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    config_path = os.path.join(REPO_ROOT, TTML_CONFIG_REL)
    print(f"[test] loading device config: {config_path}")
    raw = load_config(config_path)
    device_config = DeviceConfig(raw)

    print(f"[test] building LlamaGRPOCompleter ({MODEL_ID}, max_seq_len={MAX_SEQ_LEN})")
    completer = LlamaGRPOCompleter(
        device_config=device_config,
        model_source=MODEL_ID,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
    )

    print("[test] loading weights via completer.model_args.load_state_dict()")
    state_dict = completer.model_args.load_state_dict()
    print(f"[test]   state_dict has {len(state_dict)} tensors")
    completer.load_weights(state_dict)

    tokenizer = completer.tokenizer
    prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)

    print(f"[test] prompt: {PROMPT!r}")
    print(f"[test] tokenized prompt: {len(prompt_ids)} tokens")
    print(f"[test] first 16 token ids: {prompt_ids[:16]}")
    print(f"[test] first 16 tokens:    {tokenizer.convert_ids_to_tokens(prompt_ids[:16])}")

    print(f"[test] tokenizer.decode {tokenizer.decode(prompt_ids)}")

    print(f"[test] generating up to {MAX_NEW_TOKENS} tokens (temperature={TEMPERATURE})")
    completions = completer.generate(
        [prompt_ids], max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, seed=0, enable_trace=False
    )
    assert len(completions) == 1
    completion_ids = completions[0]

    completion_str = tokenizer.decode(completion_ids, skip_special_tokens=False)
    print()
    print(f"[test] completion ({len(completion_ids)} tokens):")
    print(completion_str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
