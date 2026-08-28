# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the accuracy gate.

pytest scopes `conftest.py` by directory, so this module re-exports the e2e fixtures
rather than defining a second set. That matters: `qwen_mesh` is the package's ONLY
device opener, and two openers in one session would collide over the command-queue
count the trace lever depends on. Importing the same objects keeps one opener.
"""
from __future__ import annotations

from models.demos.qwen3_coder_next.tests.e2e.conftest import (  # noqa: F401
    CAPACITY,
    LAYERS,
    pipeline,
    qwen_mesh,
    reference,
)
