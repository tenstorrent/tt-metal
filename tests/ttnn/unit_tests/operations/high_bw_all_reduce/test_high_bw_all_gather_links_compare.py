# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference for high_bw_all_reduce's link scaling: runs high_bw_all_gather's own QuietBox perf test at
num_links = 2 (as shipped) and 4 (every link between neighbouring QuietBox chips). Run with `-s` to see
the HIGH_BW_ALL_GATHER lines."""

import pytest

import tests.ttnn.unit_tests.operations.experimental.test_high_bw_all_gather as ag


@pytest.fixture(params=[2, 4], ids=["links2", "links4"])
def ag_links(request, monkeypatch):
    monkeypatch.setattr(ag, "_NUM_LINKS", request.param)
    return request.param


test_all_gather_quietbox_links = pytest.mark.usefixtures("ag_links")(ag.test_high_bw_all_gather_quietbox_ci_perf)
