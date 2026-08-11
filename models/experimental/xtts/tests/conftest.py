# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the XTTS-v2 test suite.

Provides the checkpoint once per **session**. ``load_xtts_state_dict()`` is uncached and re-runs a
full ``torch.load`` of the ~1.9 GB ``model.pth`` on every call, and each of the 12 test modules
used to declare its own module-scoped copy of this fixture — so a full ``tests/pcc/`` run paid that
load a dozen times over. Hoisting it here pays it once. Be honest about the size of that win: the
load is ~1.2 s warm (the file stays in the page cache), so this saves ~13 s on a ~363 s suite —
within run-to-run noise. The reason to do it is the single definition and the skip below, not speed.

It also turns a missing checkpoint into a clean skip rather than a stack trace: the weights are
downloaded from HuggingFace on first use, so a box with no network (or a revoked/renamed upstream
repo) would otherwise fail every test with a confusing error far from the cause.
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
