#!/usr/bin/env python3
"""Scratch, not for merge. Runs the pytest node ids of $AB_NODES_FILE with r3_prof_plugin (R3_REPS calls per test)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with open(os.environ["AB_NODES_FILE"]) as f:
    nodes = [line.strip() for line in f if line.strip()]
raise SystemExit(pytest.main(["-q", "-p", "r3_prof_plugin", "-p", "no:cacheprovider", "-o", "timeout_method=thread",
                              "-o", "addopts=", "-rfE", *nodes]))
