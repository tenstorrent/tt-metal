# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin: clear the device program cache before every test.

Used to decide whether ``test_multichip_decode_batch``'s failure under the
adopted SDPA program config is a *model* fault or a **program-cache keying**
fault. ``tests/test_multichip_decoder.py`` uses a ``module``-scoped mesh device,
and the repo's own ``conftest.py`` warns that "cached programs from earlier
tests may be reused by later tests ... this can cause incorrect results" when
tests need different program configurations. That file runs SDPA-decode both via
``functional_decoder.attention_decode`` (paged, ``program_config=None``) and via
``attention_decode_optimized`` (paged, configured) in one process, which is
exactly that situation. The shipped model never does.

    PYTHONPATH=<this dir> pytest ... -p no_progcache_plugin
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_program_cache(request):
    for name in ("mesh_device", "device"):
        if name in request.fixturenames:
            try:
                dev = request.getfixturevalue(name)
            except Exception:  # noqa: BLE001
                continue
            try:
                dev.clear_program_cache()
            except Exception:  # noqa: BLE001
                pass
            break
    yield
