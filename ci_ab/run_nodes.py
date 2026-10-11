#!/usr/bin/env python3
"""Scratch, not for merge. Runs the pytest node ids listed in $AB_NODES_FILE, one per line, under tracy."""
import os

import pytest

with open(os.environ["AB_NODES_FILE"]) as f:
    nodes = [line.strip() for line in f if line.strip()]
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "-o", "timeout_method=thread", *nodes]))
