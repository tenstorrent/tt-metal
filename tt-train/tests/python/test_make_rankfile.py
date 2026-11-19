# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Host-only tests for ``examples/python/multihost/pipeline_parallel_training/make_rankfile.py``.

The script is a plain CLI helper with no ttnn dependency, so these tests import it
by path and never open a device.
"""

from __future__ import annotations

import importlib.util
import os

import pytest

_SCRIPT = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "sources",
    "examples",
    "python",
    "multihost",
    "pipeline_parallel_training",
    "make_rankfile.py",
)


def _load_module():
    spec = importlib.util.spec_from_file_location("make_rankfile_under_test", os.path.abspath(_SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def rankfile():
    return _load_module()


def test_canonical_order_is_snake_order(rankfile):
    hosts = ["c10u02", "c09u02", "c10u08", "c09u08"]
    assert rankfile.canonicalize_hosts(hosts) == ["c09u08", "c09u02", "c10u02", "c10u08"]


def test_unrecognised_hostname_is_placed_last_without_raising(rankfile):
    """``scontrol`` can emit hostnames that do not match ``<digits>u<2 digits>``.

    ``_parse_host`` documents that such a host is placed last in canonical order, so
    the call must not raise (it used to hit an undefined ``logger``).
    """
    hosts = ["c09u08", "c09u02", "weird-host"]
    assert rankfile.canonicalize_hosts(hosts) == ["c09u08", "c09u02", "weird-host"]


def test_rankfile_lists_every_slot_of_every_host(rankfile):
    text = rankfile.make_rankfile(["a", "b"], 2)
    assert text.splitlines() == ["rank 0=a slot=0", "rank 1=a slot=1", "rank 2=b slot=0", "rank 3=b slot=1"]
