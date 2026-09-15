# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-identity text-generation demo runner for meta-llama/Llama-3.1-8B-Instruct.

This directory is a SELF-CONTAINED copy of the tt_transformers Llama stack:
the real forward pass, pipeline, and demo live under this package's own ``tt/``
and ``demo/simple_text_demo.py``. This runner simply invokes the COPIED demo
node (not the upstream one) so the real model executes on device.

The model identity is pinned to Llama-3.1-8B-Instruct at module scope.
"""
from __future__ import annotations

import os
import sys

os.environ["HF_MODEL"] = "meta-llama/Llama-3.1-8B-Instruct"
os.environ.setdefault("MESH_DEVICE", "P150")

HF_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# The COPIED (self-contained) demo node + default performance selector.
DEMO_NODE = "models/demos/llama3_1_8b_p150/demo/simple_text_demo.py::test_demo_text"
DEMO_SELECTOR = "performance-batch-1"


def run_demo(extra_pytest_args=None):
    """Run the copied Llama-3.1-8B-Instruct simple_text_demo performance node."""
    import pytest

    args = [DEMO_NODE, "-k", DEMO_SELECTOR, "-s"]
    if extra_pytest_args:
        args.extend(extra_pytest_args)
    return pytest.main(args)


if __name__ == "__main__":
    sys.exit(run_demo(sys.argv[1:]))
