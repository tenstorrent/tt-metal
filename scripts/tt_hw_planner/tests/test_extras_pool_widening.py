"""Unit tests for Bundle item #2: extras pool widening.

Before: parallel-extras were drawn from `_ungraduated_now`, which only
returns components whose stubs are STILL ON AUTOFILL (CPU fallback).
This meant components with native ttnn stubs that failed PCC could
NEVER be parallel-extras — they had to wait for their turn as the
primary, one at a time.

The Step 3 dry-run on facebook/sam2-hiera-tiny v14-end-state showed
the symptom directly: vision_config, vision_model, vision_neck were
all "ungraduated" in the candidate_pool, but only vision_config was
on autofill, so _extras_pool would have had {vision_model, vision_neck}
and the dry-run would have written 3 prompt files instead of 1.

After: the pool is `_ungraduated_now ∪ candidate_pool`, which captures
both autofill-on-CPU components AND native-stub-but-PCC-failing ones.

These tests source-grep the wiring."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

cli = importlib.import_module("scripts.tt_hw_planner.cli")


def _auto_iterate_source() -> str:
    return (Path(cli.__file__).parent / "_cli_helpers" / "auto_iterate.py").read_text()
