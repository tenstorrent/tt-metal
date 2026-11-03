# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_params(reason_fn=fn, **kwargs): skip the parametrized cases for which fn(**params, **kwargs) "
        "returns a reason, at collection time so their fixtures never run",
    )


def pytest_collection_modifyitems(config, items):
    # A pytest.skip() in the test body only runs after mesh_device has opened the 1x8 mesh. Cases that the
    # parameters alone rule out are marked here instead, so pytest skips them before any fixture is set up.
    for item in items:
        marker = item.get_closest_marker("skip_params")
        callspec = getattr(item, "callspec", None)
        if marker is None or callspec is None:
            continue
        kwargs = dict(marker.kwargs)
        reason_fn = kwargs.pop("reason_fn")
        reason = reason_fn(**callspec.params, **kwargs)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
