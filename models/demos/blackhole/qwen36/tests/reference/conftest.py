# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the host DFlash reference tests.

Two scales live side by side:

* **tiny** — randomly initialised models that keep Qwen3.6-27B's *structure* (interleaved
  DeltaNet / full attention in the target, sliding-then-full in the drafter) but shrink every
  dimension. These carry the correctness tests: they run in seconds, need no checkpoint and no
  network, and the GDN rollback bug they guard against is structural, not scale-dependent.
* **real** — the actual ``z-lab/Qwen3.6-27B-DFlash`` and ``Qwen/Qwen3.6-27B`` checkpoints. These
  skip unless the checkpoints resolve, and the 27B target additionally needs ``DFLASH_RUN_TARGET=1``
  because a host forward of 27B params takes minutes on CPU.
"""

from __future__ import annotations

import os

import pytest
import torch
from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from models.demos.blackhole.qwen36.reference.dflash.dflash import DFlashDraftModel
from models.demos.blackhole.qwen36.reference.dflash.loader import (
    DFlashDrafterConfig,
    load_drafter,
    load_target,
    resolve_drafter_path,
)

# Tiny-model dimensions. TINY_TAPS must stay inside TINY_TARGET_LAYERS.
TINY_HIDDEN = 128
TINY_TARGET_LAYERS = 8
TINY_DRAFT_LAYERS = 2
TINY_TAPS = (1, 5)
TINY_VOCAB = 512
TINY_BLOCK = 4


@pytest.fixture(scope="module")
def tiny_target():
    """A shrunk Qwen3.6: 6 Gated DeltaNet layers + 2 full-attention layers, real ``layer_types``."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    config = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B").get_text_config()
    config.num_hidden_layers = TINY_TARGET_LAYERS
    config.layer_types = config.layer_types[:TINY_TARGET_LAYERS]
    config.hidden_size, config.intermediate_size = TINY_HIDDEN, 2 * TINY_HIDDEN
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 32
    config.linear_num_key_heads, config.linear_num_value_heads = 4, 8
    config.vocab_size = TINY_VOCAB
    config._attn_implementation = "sdpa"
    assert config.layer_types.count("linear_attention") > 0, "tiny target must keep GDN layers"
    return Qwen3_5ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def tiny_drafter():
    """A shrunk DFlash drafter: 1 causal sliding layer + 1 bidirectional full-attention layer."""
    torch.manual_seed(1)
    config = Qwen3Config(
        hidden_size=TINY_HIDDEN,
        intermediate_size=2 * TINY_HIDDEN,
        num_hidden_layers=TINY_DRAFT_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=TINY_VOCAB,
        # Qwen3Config nulls `sliding_window` unless `use_sliding_window` is set; the real drafter's
        # config.json sets both.
        use_sliding_window=True,
        # 64, not 16: the real drafter's window (2048) is far longer than anything these tests
        # generate, so a window the test sequences actually overrun would not be faithful — and
        # truncate_kv refuses to truncate past it (see test_truncate_kv_refuses_past_the_window).
        sliding_window=64,
        layer_types=["sliding_attention", "full_attention"],
    )
    assert config.sliding_window == 64, "tiny drafter lost its sliding window"
    config.num_target_layers = TINY_TARGET_LAYERS
    config.dflash_config = {"target_layer_ids": list(TINY_TAPS), "mask_token_id": TINY_VOCAB - 1}
    config.block_size = TINY_BLOCK
    config._attn_implementation = "sdpa"
    return DFlashDraftModel(config).eval()


@pytest.fixture(scope="module")
def drafter_path() -> str:
    """Local dir for ``$DFLASH_HF_MODEL``; skips when it cannot be resolved (no cache, no network)."""
    try:
        return resolve_drafter_path()
    except Exception as e:  # noqa: BLE001 - any resolution failure is a skip, not a test failure
        pytest.skip(f"drafter checkpoint unavailable ({type(e).__name__}: {e})")


@pytest.fixture(scope="module")
def real_drafter(drafter_path):
    """The real 1.73B drafter with its real weights, fp32 on CPU."""
    return load_drafter(drafter_path)


@pytest.fixture(scope="module")
def real_target():
    """The real 27B target. Opt-in: loading 54 GB and running it on CPU costs minutes."""
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run against the real 27B target on host")
    return load_target()


@pytest.fixture(scope="module")
def real_drafter_config(drafter_path) -> DFlashDrafterConfig:
    return DFlashDrafterConfig.from_pretrained(drafter_path)
