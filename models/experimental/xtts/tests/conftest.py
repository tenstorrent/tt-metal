# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the XTTS-v2 test suite.

Loads the checkpoint once per session and skips cleanly if weights are unavailable
(network down, or no HuggingFace cache).
"""

import pytest

from models.experimental.xtts.reference.xtts_gpt_block import HF_REPO_ID, load_xtts_state_dict


@pytest.fixture(scope="session")
def xtts_state_dict():
    """The real coqui/XTTS-v2 checkpoint tensors, loaded once per session."""
    try:
        return load_xtts_state_dict()
    except Exception as exc:  # network down, hub error, revision gone
        pytest.skip(
            f"XTTS-v2 weights unavailable ({type(exc).__name__}: {exc}). "
            f"Needs network access to HuggingFace for {HF_REPO_ID}, or a warm HF hub cache."
        )
