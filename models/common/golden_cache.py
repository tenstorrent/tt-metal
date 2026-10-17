"""Module-level per-component golden cache (tt_hw_planner optimize --module-level).

The per-component PCC test builds its torch reference by loading the ENTIRE model
(from_pretrained) just to resolve + run one submodule, then extracts that
submodule's weights to build the ttnn port. Under module-level optimize that full
load repeats for the baseline and every candidate, dominating wall-clock.

Because the test seeds torch before building inputs, the resolved submodule, its
inputs, and its golden output are deterministic across runs. So the first run
persists (submodule, inputs, golden) and every subsequent run loads that bundle
and skips the full-model load entirely. The cached submodule is a single layer
(small); the ttnn stub under optimization is re-imported fresh each run, so
candidate edits are still picked up — only the fixed torch reference is cached.

Gated by the caller on TT_PERF_MODULE_LEVEL; whole-model optimize never calls it.
"""

from __future__ import annotations

import os
from pathlib import Path


def golden_cache_path(test_file, component, seed=0) -> str:
    demo = Path(test_file).resolve().parents[2]
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in str(component))
    return str(demo / "_captured" / safe / f"golden_cache_s{seed}.pt")


def load_golden_cache(path):
    """Return (torch_module, sample_kwargs, primary, golden) or None. Never raises."""
    try:
        import torch

        if not os.path.isfile(path):
            return None
        d = torch.load(path, map_location="cpu", weights_only=False)
        return d["module"], d["kwargs"], d["primary"], d["golden"]
    except Exception:
        return None


def save_golden_cache(path, torch_module, sample_kwargs, primary, golden) -> bool:
    """Persist the reference bundle for reuse. Best-effort; never raises."""
    try:
        import torch

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {"module": torch_module, "kwargs": sample_kwargs, "primary": primary, "golden": golden},
            path,
        )
        return True
    except Exception:
        return False
