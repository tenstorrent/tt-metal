# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.experimental.xtts.reference.xtts_gpt_block import HF_REPO_ID, load_xtts_state_dict


@pytest.fixture(scope="session")
def xtts_state_dict():
    """Session fixture that loads the XTTS-v2 checkpoint state dict."""
    try:
        return load_xtts_state_dict()
    except Exception as exc:  # network down, hub error, revision gone
        pytest.skip(
            f"XTTS-v2 weights unavailable ({type(exc).__name__}: {exc}). "
            f"Needs network access to HuggingFace for {HF_REPO_ID}, or a warm HF hub cache."
        )
