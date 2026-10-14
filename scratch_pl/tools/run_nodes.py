#!/usr/bin/env python3
"""Scratch, not for merge. Runs the pytest node ids of $HR_NODES with mm_prof_plugin (tracy re-joins argv through /bin/sh)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
with open(os.environ["HR_NODES"]) as f:
    nodes = [line.strip() for line in f if line.strip()]
raise SystemExit(pytest.main(["-q", "-p", "mm_prof_plugin", "-p", "no:cacheprovider", "-o", "timeout_method=thread",
                              "-rfE", *nodes]))
