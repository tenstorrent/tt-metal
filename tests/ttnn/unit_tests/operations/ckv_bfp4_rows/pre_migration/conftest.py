# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures of the ckv_bfp4_rows pre-migration suite.

``device_params`` is overridden for this directory only: every ``device`` / ``mesh_device`` opened here gets a trace
region (the output-tensor file's trace replay needs one), unless a test parametrizes its own.
"""

import pytest

TRACE_REGION_BYTES = 1 << 20


@pytest.fixture(scope="function")
def device_params(request):
    params = dict(getattr(request, "param", {}))
    params.setdefault("trace_region_size", TRACE_REGION_BYTES)
    return params
