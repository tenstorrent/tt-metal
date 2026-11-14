# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open the device once per module for this op's unit tests.

Applies @pytest.mark.use_module_device to every collected test; the root
`device` fixture is function-scoped and the marker switches it to module scope.
Do not define a local `device` fixture here — it shadows the root one and
disables the marker.

Mesh modules (perf_feed_measure.py) instead get `perf_mesh`: the feed mesh with
the golden suite's fabric config and a trace region, opened once per module
through the golden conftest's mesh cache.
"""
import os

import pytest

_MESH_MODULES = ("perf_feed_measure.py",)


def pytest_collection_modifyitems(items):
    for item in items:
        if os.path.basename(str(item.path)) in _MESH_MODULES:
            continue
        item.add_marker(pytest.mark.use_module_device)


@pytest.fixture(scope="module")
def perf_mesh():
    from eval import ccl_perf
    from eval.golden_tests.mhc_post.conftest import _FabricMeshCache
    from eval.golden_tests.mhc_post.feature_spec import FEED_FABRIC_CONFIG
    from eval.golden_tests.mhc_post.topology import feed_mesh_shape

    shape = feed_mesh_shape()
    if shape is None:
        pytest.skip("no feed mesh on this cluster")
    cache = _FabricMeshCache()
    yield cache.get(shape, FEED_FABRIC_CONFIG, trace_region_size=ccl_perf.TRACE_REGION_SIZE)
    cache.close()
