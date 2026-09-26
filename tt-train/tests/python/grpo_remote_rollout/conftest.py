# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared pytest config: put this test dir, the examples root, and the repo root
on sys.path, free each test module's tt_transformers objects before the next
module runs, and offer an opt-in fixture that sets ttnn fabric once per session
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
    """Destroy the tt_transformers objects a module built, after its last test.

    Their destructors free device resources, so they must run before the next
    module's ``ttml.reset_metal_env`` destroys the ``MetalEnv``, or they segfault.
    Several of them, including ``Attention`` layers and ``ModelArgs``, sit in
    reference cycles, so only ``gc.collect()`` frees them.

    ``ModelArgs``'s method-level ``lru_cache`` also pins every instance and its
    submesh until ``cache_clear()`` is called. Clearing the caches here is a
    workaround until https://github.com/tenstorrent/tt-metal/issues/57716 is fixed.

    This is a hook wrapper rather than a module-fixture teardown because pytest's
    ``item.funcargs`` holds the last test's fixture values until the protocol
    returns, after every teardown.
    """
    yield
    if nextitem is not None and getattr(nextitem, "module", None) is getattr(item, "module", None):
        return
    model_config = sys.modules.get("models.tt_transformers.tt.model_config")
    if model_config is not None:
        for attr in vars(model_config.ModelArgs).values():
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
