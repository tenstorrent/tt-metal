# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""CI only (#58723 third review): keep the collected items whose node id matches the regex in EB_SEL (and not EB_SEL_NOT)."""
import os
import re

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    sel = os.environ.get("EB_SEL")
    nsel = os.environ.get("EB_SEL_NOT")
    if not sel and not nsel:
        return
    keep, drop = [], []
    for it in items:
        ok = (not sel or re.search(sel, it.nodeid)) and not (nsel and re.search(nsel, it.nodeid))
        (keep if ok else drop).append(it)
    items[:] = keep
    if drop:
        config.hook.pytest_deselected(items=drop)
