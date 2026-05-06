#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test for ``LlamaGRPOCompleter.generate_tt_transformers``.

Step 1 of the ttml -> tt-transformers bridge: just verify that we can
   1. construct a ``LlamaGRPOCompleter`` (loads ttml on the mesh device),
   2. build a Meta-style state dict via ``ModelArgs.load_state_dict()``,
   3. call ``completer.init_model(state_dict)`` to build the tt-transformers
      ``Transformer`` mirror on the same mesh device,
   4. call ``completer.generate_tt_transformers([prompt_ids])`` and decode
      the resulting tokens.

There's no PCC check here; this is purely the integration smoke test that
exercises the new public surface. Pairwise PCC against ttml / hf is the
job of ``pcc_hf_ttml_ttt.py``. Edit the constants below to change the
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

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 32
TEMPERATURE = 0.0  # 0 == greedy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    import ttnn
    from transformers import AutoTokenizer

    from ttml.common.config import DeviceConfig, TrainingConfig, get_model_config, load_config

    from utils.llama_completer import LlamaCompletionCtx, LlamaGRPOCompleter

    # tt-transformers expects fabric_config to be set BEFORE any mesh device
    # is opened. ``LlamaGRPOCompleter.setup_device`` opens the mesh inside
    # the constructor for single device without enabling fabric, so we do it
    # here. FABRIC_2D matches ``pcc_hf_ttml_ttt.py``.
    print("[test] set_fabric_config(FABRIC_2D)")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    config_path = os.path.join(REPO_ROOT, TTML_CONFIG_REL)
    print(f"[test] loading ttml config: {config_path}")
    raw = load_config(config_path)
    training_config = TrainingConfig(raw)
    device_config = DeviceConfig(raw)
    transformer_config = get_model_config(training_config.model_config)

    print(f"[test] building LlamaGRPOCompleter (loads ttml weights from {MODEL_ID})")
    completer = LlamaGRPOCompleter(
        ctx=LlamaCompletionCtx(
            max_tokens_to_complete=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        ),
        transformer_config=transformer_config,
        device_config=device_config,
        model_source=MODEL_ID,
    )

    try:
        # Build the Meta-style state dict that tt-transformers wants. For
        # step 1 we source the dict from HF (via tt-transformers' own loader).
        # In a later step this will be replaced by a ttml dump so the mirror
        # tracks the ttml model's current weights.
        os.environ["HF_MODEL"] = MODEL_ID

        from models.tt_transformers.tt.model_config import ModelArgs

        print("[test] building Meta-style state_dict via ModelArgs.load_state_dict()")
        scratch_args = ModelArgs(
            completer._mesh_device,
            instruct=True,
            max_batch_size=1,
            max_seq_len=2048,
            cache_hf=True,
        )
        state_dict = scratch_args.load_state_dict()
        print(f"[test]   state_dict has {len(state_dict)} tensors")

        print("[test] calling completer.init_model(state_dict)")
        completer.init_model(state_dict, max_seq_len=2048)
        assert completer.tt_model is not None

        tokenizer: AutoTokenizer = completer.tokenizer
        prompt_ids = tokenizer.encode(PROMPT, add_special_tokens=True)
        print(f"[test] prompt: {PROMPT!r}  ({len(prompt_ids)} tokens)")

        print(f"[test] generating up to {MAX_NEW_TOKENS} tokens " f"(temperature={TEMPERATURE})")
        # ``generate`` dispatches to the tt-transformers path automatically
        # because ``init_model`` populated ``completer.tt_model``.
        completions = completer.generate([prompt_ids])
        assert len(completions) == 1
        completion_ids = completions[0]

        completion_str = tokenizer.decode(completion_ids, skip_special_tokens=False)
        print()
        print(f"[test] completion ({len(completion_ids)} tokens):")
        print(f"  {completion_str!r}")
        print()
        print(f"[test] full text: {(PROMPT + completion_str)!r}")
        return 0
    finally:
        # Free the temp cache directory the mirror uses. The completer goes
        # out of scope right after, but explicit teardown keeps /tmp clean
        # if the script is re-run in a loop.
        completer._teardown_tt_model()


if __name__ == "__main__":
    sys.exit(main())
