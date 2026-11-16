# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Artifact-availability policy shared by the Qwen3-ASR fixtures and tests.

The PCC / e2e suites need artifacts that are too large to keep in the repo (CPU-reference
golden tensors, the extracted text-decoder checkpoint) plus the HF snapshot. On a dev box
their absence should skip; on a runner that is supposed to have them staged, a skip would
hide the very breakage the suite exists to catch — so an unattended run sets
``QWEN3ASR_REQUIRE_ARTIFACTS=1`` and the same condition becomes a hard failure.

Kept out of ``conftest.py`` so tests can import it directly without depending on how
pytest happens to have imported the conftest module.
"""
import os

import pytest


def require_artifacts():
    """True when missing artifacts must fail instead of skip (set for unattended runs)."""
    return os.environ.get("QWEN3ASR_REQUIRE_ARTIFACTS", "").strip().lower() in ("1", "true", "yes")


def missing_artifact(msg):
    """Skip on a dev box; fail under ``QWEN3ASR_REQUIRE_ARTIFACTS=1``.

    Regenerate the artifacts with ``reference/extract_text_decoder.py`` and
    ``reference/dump_reference.py`` in the CPU-reference virtualenv
    (``requirements-reference.txt``).
    """
    if require_artifacts():
        pytest.fail(f"{msg} [QWEN3ASR_REQUIRE_ARTIFACTS=1]")
    pytest.skip(msg)
