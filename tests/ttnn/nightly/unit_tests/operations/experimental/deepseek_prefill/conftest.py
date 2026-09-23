# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import os

from models.common.utility_functions import is_blackhole


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "uncollect_if(pred): deselect parametrized cases for which pred(**params) returns True. "
        "pred receives the test's collection-time param values as keyword args, plus "
        "is_ci_env / is_ci_v2_env / is_bh. Rules live in ci_pruning.py.",
    )


def pytest_collection_modifyitems(config, items):
    is_ci_env = os.getenv("CI") == "true"
    is_ci_v2_env = "TT_GH_CI_INFRA" in os.environ
    is_bh = is_blackhole()
    kept = []
    deselected = []
    for item in items:
        marker = item.get_closest_marker("uncollect_if")
        if marker is None:
            kept.append(item)
            continue
        params = dict(getattr(getattr(item, "callspec", None), "params", {}))
        params.setdefault("is_ci_env", is_ci_env)
        params.setdefault("is_ci_v2_env", is_ci_v2_env)
        params.setdefault("is_bh", is_bh)
        if marker.kwargs["pred"](**params):
            deselected.append(item)
        else:
            kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = kept
