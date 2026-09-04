# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from models.experimental.xtts.reference.xtts_gpt_block import HF_REPO_ID, load_xtts_state_dict

SKIP_ENV = "XTTS_SKIP_WITHOUT_WEIGHTS"


@pytest.fixture(scope="session")
def xtts_state_dict():
    """Session fixture that loads the XTTS-v2 checkpoint state dict."""
    try:
        return load_xtts_state_dict()
    except Exception as exc:  # network down, hub error, revision gone, corrupt cache
        msg = (
            f"XTTS-v2 weights unavailable ({type(exc).__name__}: {exc}). "
            f"Needs network access to HuggingFace for {HF_REPO_ID}, or a warm HF hub cache."
        )
        # Fail by default: every test here takes this fixture, so skipping on any error lets the
        # gate go green having verified nothing. Developers without network opt in to skipping.
        if os.environ.get(SKIP_ENV) == "1":
            pytest.skip(msg)
        raise RuntimeError(f"{msg} Set {SKIP_ENV}=1 to skip these tests instead of failing.") from exc
