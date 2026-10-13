#!/usr/bin/env python3
"""Scratch, not for merge: runs the pytest node ids of ci_bits/ids.txt (LLK asserts on)."""
import os
import sys

import pytest

here = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(here, "ids.txt")) as f:
    nodes = [line.strip() for line in f if line.strip()]
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "-o", "timeout_method=thread", "-rfE", *nodes]))
