# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``examples/train/train.py`` as a module: it is not part of the installed ``ttml``."""

from __future__ import annotations

import importlib
import os
import sys

TRAIN_EXAMPLE_DIR = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "sources", "examples", "train")
)


def import_train_example():
    """Put ``examples/train`` first on ``sys.path`` and import its ``train`` module.

    examples/qwen3 also ships a top-level ``train`` module and other tests put that dir on sys.path, so a cached
    ``train`` or ``callbacks`` from anywhere else is evicted first.
    """
    if TRAIN_EXAMPLE_DIR in sys.path:
        sys.path.remove(TRAIN_EXAMPLE_DIR)
    sys.path.insert(0, TRAIN_EXAMPLE_DIR)
    for name in ("train", "callbacks"):
        cached = sys.modules.get(name)
        if cached is not None and not os.path.realpath(getattr(cached, "__file__", "") or "").startswith(
            TRAIN_EXAMPLE_DIR
        ):
            del sys.modules[name]
    return importlib.import_module("train")
