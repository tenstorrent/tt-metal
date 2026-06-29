# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
# Local tt-emule modification: trace capture (enable_trace / trace_mode = True) requires FAST dispatch,
# which the software emulator does not support — SDMeshCommandQueue::record_begin TT_THROWs "Not supported
# for slow dispatch". Under emule (TT_METAL_EMULE_MODE set) we deselect the trace parametrizations so the
# suite runs only the configs emule can execute (the non-trace variants). On real hardware this hook is a
# no-op, so the upstream trace coverage is unchanged. (Mirror of the all_post_commit/conftest.py hook.)
import os

import pytest


def pytest_collection_modifyitems(config, items):
    if not os.environ.get("TT_METAL_EMULE_MODE"):
        return
    kept = []
    deselected = []
    for item in items:
        params = getattr(getattr(item, "callspec", None), "params", {})
        if params.get("enable_trace") is True or params.get("trace_mode") is True:
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept
