# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-tree pytest entry. Hooks live in ``helpers.llk_pytest_plugin`` so the
same harness can be driven from outside this repo — though an out-of-tree suite
should load it as ``tt_llk_harness.plugin``, the supported spelling, rather than
naming the implementation module. See ``docs/tests/getting_started.md`` §9.

``from conftest import skip_for_wormhole`` (and the other arch markers) keeps
working for existing tests.
"""

pytest_plugins = ["helpers.llk_pytest_plugin"]

from helpers.llk_pytest_plugin import (  # noqa: E402
    blackhole_only,
    quasar_only,
    skip_for_blackhole,
    skip_for_coverage,
    skip_for_quasar,
    skip_for_wormhole,
    wormhole_only,
)

__all__ = [
    "blackhole_only",
    "quasar_only",
    "skip_for_blackhole",
    "skip_for_coverage",
    "skip_for_quasar",
    "skip_for_wormhole",
    "wormhole_only",
]
