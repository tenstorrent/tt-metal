# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Teacher-forced accuracy checks of GLM-5.4 device tests, run after the device lock is released.

The safe runner holds the device lock for the whole pytest process, so a CPU reference computed inside a device test
(tens of seconds to minutes) keeps every other user off the device. A device test hands its check to ``run_or_defer``
with the device tensors the check reads (the teacher-forced inputs and the outputs to compare):

* by default the check runs in the test, as before;
* with ``GLM54_DEFER_ACCURACY=<dir>`` set, the test saves those tensors to a bundle in ``<dir>`` and logs
  ``GLM54_ACCURACY_DEFERRED <bundle>``; after the runner exits, ``check_deferred_accuracy`` (host only, no device)
  runs the same check function on the saved tensors:

      GLM54_DEFER_ACCURACY=$D scripts/run_safe_pytest.sh <device items> -vv
      python -m models.demos.deepseek_v3_d_p.tests.glm54.check_deferred_accuracy $D

A deferred test still asserts everything it checks on device (determinism, trace replay, cache isolation); its
accuracy result is the checker's ``GLM54_ACCURACY:`` lines and exit status. A check function takes only what the
bundle holds and raises AssertionError on failure. Each bundle records the checkout HEAD that produced it; the checker
refuses a bundle from another HEAD (its check code or weights may differ). A failed bundle is the diagnosis file.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Callable

import torch
from loguru import logger

DEFER_ENV = "GLM54_DEFER_ACCURACY"
_CHECKOUT = Path(__file__).resolve().parents[5]


def checkout_head() -> str:
    """HEAD of the checkout these modules run from."""
    return subprocess.run(
        ["git", "-C", str(_CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def accuracy_deferred() -> bool:
    return bool(os.environ.get(DEFER_ENV))


def run_or_defer(check: Callable, name: str, **inputs) -> list[str]:
    """Run ``check(**inputs)`` now and return its failure ([] or [message]), or with ``$GLM54_DEFER_ACCURACY`` set
    save ``inputs`` as a bundle for ``check_deferred_accuracy`` and return []."""
    out_dir = os.environ.get(DEFER_ENV)
    if not out_dir:
        try:
            check(**inputs)
        except AssertionError as e:
            return [str(e)]
        return []
    path = Path(out_dir) / f"{name}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{os.getpid()}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save({"check": f"{check.__module__}:{check.__qualname__}", "head": checkout_head(), "inputs": inputs}, tmp)
    os.replace(tmp, path)
    logger.warning(f"GLM54_ACCURACY_DEFERRED {path} (saved in {time.perf_counter() - t0:.1f} s)")
    return []
