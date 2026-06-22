#!/usr/bin/env python3
# Run the NOP test over a list of pytest nodeids (read from $LLK_NODEIDS, one per
# line). Passing nodeids as a Python list avoids shell word-splitting.
# Must run with CWD = python_tests.
import os
import sys

PT = os.environ.get("LLK_PT", os.getcwd())
sys.path.insert(0, PT)  # so `-p nop_plugin` is importable

import pytest

ids = [l.rstrip("\n") for l in open(os.environ["LLK_NODEIDS"]) if l.strip()]
sys.exit(pytest.main([*ids, "-p", "nop_plugin", "-p", "no:randomly", "-q", "-s"]))
