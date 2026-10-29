# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Select candidate kernel sources while keeping the evaluator/runtime shared."""

import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def nvfp4_candidate_kernel_directory(monkeypatch):
    candidate = os.environ.get("NVFP4_KERNEL_ROOT")
    if candidate:
        # Kernel lookup checks CWD before TT_METAL_KERNEL_PATH. The safe runner
        # initially enters the runtime tree; enter the candidate before fixtures
        # create programs so that the runtime cannot silently use baseline code.
        monkeypatch.chdir(Path(candidate).resolve(strict=True))
