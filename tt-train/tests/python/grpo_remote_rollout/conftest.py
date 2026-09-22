# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest config: put this test dir, the examples root, and the repo root
on sys.path, and offer an opt-in fixture that sets ttnn fabric once per session
before any device opens."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("TT_LOGGER_LEVEL", "Error")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
EXAMPLES_DIR = REPO_ROOT / "tt-train" / "sources" / "examples"

for _p in (str(HERE), str(EXAMPLES_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Destroy each module's tt_transformers model graph before the next module runs.

    The graph -- device tensors, ``TT_CCL`` global semaphores, the mesh's program
    cache -- frees device resources from its destructors and must be destroyed while
    the ``MetalEnv`` it was created under is still alive. Left for a later collection,
    it outlives a ``ttml.reset_metal_env`` in the next module and segfaults.

    Two things keep it reachable. ``ModelArgs`` decorates ~30 instance methods with
    ``functools.lru_cache``; the wrapper lives on the class and keys on ``self``, so
    every ``ModelArgs`` these tests build stays alive until the caches are cleared
    (tt_transformers is outside tt-train, so this is the closest place to fix it).
    And pytest keeps the last test's fixture values in ``item.funcargs`` until after
    every teardown has run, so this can't be a module-fixture teardown: the graph is
    only unreachable once the protocol for that item returns.
    """
    yield
    if nextitem is not None and getattr(nextitem, "module", None) is getattr(item, "module", None):
        return
    try:
        from models.tt_transformers.tt.model_config import ModelArgs
    except Exception:  # noqa: BLE001
        pass
    else:
        for attr in vars(ModelArgs).values():
            cache_clear = getattr(attr, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
    gc.collect()


@pytest.fixture(scope="session")
def _set_fabric_2d():
    """Configure ttnn fabric to 2-rank.

    Request this fixture explicitly, e.g. through ``pytest.mark.usefixtures``,
    from a module that already self-skips unless ``OMPI_COMM_WORLD_SIZE == 2``.
    Both ranks must agree on the fabric config before either one opens a device.
    See description in ``_completer_utils.open_device`` for more details.
    """
    import ttnn

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
