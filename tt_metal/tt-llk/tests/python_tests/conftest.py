# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""In-tree pytest entry. Hooks live in ``helpers.llk_pytest_plugin`` so
out-of-tree suites can load the same harness without depending on this file.

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
