# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the DFlash drafter tests.

These live in ``tests/dflash/`` rather than ``tests/unit/`` on purpose: ``tests/unit/
conftest.py`` does ``os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")``, so a drafter
test placed there would silently be graded against the 9B.

:data:`CHECKPOINT_CONFIG` is a literal transcription of
``z-lab/Qwen3.6-27B-DFlash/config.json``. Keeping a literal lets the mask and config tests
run with no network, no checkpoint and no device -- which is what makes them the cheapest
gate in the suite and usable on a laptop. ``test_config.py`` asserts the literal still
matches the real checkpoint whenever one is reachable, so it cannot drift silently.
"""

from __future__ import annotations

import pytest
import torch

from models.demos.blackhole.qwen36.tests.dflash.capture_fixtures import fixture_path
from models.demos.blackhole.qwen36.tt.dflash.config import DFlashDrafterConfig

#: Context lengths ``capture_fixtures.py`` writes by default. 4096 is the only one above the
#: 2048 sliding window, i.e. the only one where the window's lower bound is exercised.
FIXTURE_CTX_LENS = (512, 4096)

#: Verbatim from ``z-lab/Qwen3.6-27B-DFlash/config.json`` (only the keys this port reads).
CHECKPOINT_CONFIG = {
    "block_size": 16,
    "dflash_config": {
        "mask_token_id": 248070,
        "target_layer_ids": [1, 16, 31, 46, 61],
    },
    "head_dim": 128,
    "hidden_size": 5120,
    "initializer_range": 0.02,
    "intermediate_size": 17408,
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "num_attention_heads": 32,
    "num_hidden_layers": 5,
    "num_key_value_heads": 8,
    "num_target_layers": 64,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000000,
    "sliding_window": 2048,
    "use_sliding_window": True,
    "vocab_size": 248320,
}

#: The target this drafter attaches to. 64 layers, so ``target_layer_ids`` max (61) fits.
TARGET_NUM_HIDDEN_LAYERS = 64


@pytest.fixture(scope="session")
def drafter_cfg() -> DFlashDrafterConfig:
    """The real checkpoint's config, built hermetically from :data:`CHECKPOINT_CONFIG`."""
    return DFlashDrafterConfig.from_dict(CHECKPOINT_CONFIG)


def load_fixture(ctx_len: int) -> dict:
    """Load a captured reference fixture, or skip the test if it has not been captured.

    Skipping rather than failing is deliberate: the fixtures are multi-hundred-MB
    reproducible artifacts under ``generated/`` and are never committed, so a fresh checkout
    legitimately has none until ``capture_fixtures.py`` has been run once.
    """
    path = fixture_path(ctx_len)
    if not path.is_file():
        pytest.skip(
            f"no fixture at {path}. Capture it first:\n"
            f"  python models/demos/blackhole/qwen36/tests/dflash/capture_fixtures.py --ctx-len {ctx_len}"
        )
    return torch.load(path, weights_only=False)
