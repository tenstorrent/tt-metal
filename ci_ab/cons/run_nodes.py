#!/usr/bin/env python3
"""Scratch, not for merge. Runs the pytest node ids of $AB_NODES_FILE with r3_prof_plugin (R3_REPS calls per test); every
ttnn.to_torch result is hashed into $OUT_HASH (raw bytes), so two arms' outputs can be compared value for value."""
import hashlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import ttnn

_orig = ttnn.to_torch


def _to_torch(*a, **k):
    t = _orig(*a, **k)
    if os.environ.get("OUT_HASH") and isinstance(t, torch.Tensor):
        b = t.detach().cpu().contiguous()
        b = b.view(torch.int16) if b.dtype == torch.bfloat16 else b
        node = os.environ.get("PYTEST_CURRENT_TEST", "?").rsplit(" (", 1)[0]
        with open(os.environ["OUT_HASH"], "a") as f:
            f.write(f"{node}\t{hashlib.sha1(b.numpy().tobytes()).hexdigest()[:16]}\n")
    return t


ttnn.to_torch = _to_torch
with open(os.environ["AB_NODES_FILE"]) as f:
    nodes = [line.strip() for line in f if line.strip()]
raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", "-o", "timeout_method=thread",
                              "-o", "addopts=", "-rfE", *nodes]))
