#!/usr/bin/env python3
"""Scratch, not for merge. Runs the pytest node ids listed in $AB_NODES_FILE (tracy re-joins argv through /bin/sh)."""
import os
import sys

import pytest

with open(os.environ["AB_NODES_FILE"]) as f:
    nodes = [line.strip() for line in f if line.strip()]
sys.exit(pytest.main(["-q", "-p", "no:cacheprovider", "-o", "timeout_method=thread", "-o", "addopts=", "-rfE", *nodes]))
