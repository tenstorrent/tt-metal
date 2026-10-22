# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Pure test for the LTX e2e traced-by-default rule (no device)."""

import pytest

from models.tt_dit.utils.ltx import traced_default


@pytest.mark.parametrize(
    "device_params,env,expected",
    [
        ({"trace_region_size": 500_000_000}, None, True),  # reserves a trace region: traced by default
        ({"l1_small_size": 32768}, None, False),  # no trace region: cannot trace, stays eager
        ({}, None, False),
        ({"trace_region_size": 0}, None, False),  # a zero-sized region is no region
        ({"trace_region_size": 500_000_000}, "0", False),  # explicit override wins either way
        ({"trace_region_size": 500_000_000}, "false", False),
        ({}, "1", True),
        ({}, "true", True),
        ({}, "", False),  # set but empty is an explicit "no"
    ],
)
def test_traced_default_derivation(device_params, env, expected):
    assert traced_default(device_params, env) is expected
