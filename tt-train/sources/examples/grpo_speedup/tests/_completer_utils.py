# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the grpo_speedup update-method tests.

These are intentionally regular functions (not pytest fixtures) so each
test module can wrap them in a module-scoped fixture with whatever
extras it needs (e.g. yielding a specific layer or sub-module).
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
TTML_DEVICE_CONFIG_REL = "tt-train/configs/training_configs/grpo_boolq_llama_1dev.yaml"
MAX_SEQ_LEN = 2048

REPO_ROOT = Path(__file__).resolve().parents[5]  # .../tt-metal


def build_completer(*, dummy_weights: bool, max_batch_size: int = 1, max_seq_len: int = MAX_SEQ_LEN):
    """Construct a fresh ``LlamaGRPOCompleter``.

    Heavy: opens a device and (when ``dummy_weights=False``) loads real
    Llama-3.2-1B-Instruct weights via HF auth. Tests should call this
    from a module-scoped fixture so the cost is paid once per file.
    """
    from ttml.common.config import DeviceConfig, load_config

    from utils.llama_completer_ttt import LlamaGRPOCompleter

    raw = load_config(os.path.join(REPO_ROOT, TTML_DEVICE_CONFIG_REL))
    device_config = DeviceConfig(raw)
    return LlamaGRPOCompleter(
        device_config=device_config,
        model_source=MODEL_ID,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        dummy_weights=dummy_weights,
    )


def teardown_completer(completer) -> None:
    """Drop ``completer`` and close the device. Call from a fixture's
    teardown so the device is released even if a test fails."""
    import ttml

    del completer
    gc.collect()
    ttml.autograd.AutoContext.get_instance().close_device()
