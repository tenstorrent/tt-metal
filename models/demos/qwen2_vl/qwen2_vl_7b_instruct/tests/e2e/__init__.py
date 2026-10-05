# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""E2E test package. Exposes a loader for the captured HF golden."""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN = os.path.normpath(os.path.join(_HERE, "..", "..", "_captured", "e2e_golden.pt"))


def _golden():
    import torch

    return torch.load(_GOLDEN, weights_only=False)
