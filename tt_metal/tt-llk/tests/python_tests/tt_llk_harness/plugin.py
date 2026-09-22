# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Stable plugin name for out-of-tree suites.

Put this in your suite's rootdir ``conftest.py``::

    pytest_plugins = ["tt_llk_harness.plugin"]

It forwards to the implementation rather than re-exporting its hooks. That
distinction matters: copying hook functions into a second module would let
pytest register the same hooks twice if a session ever loaded both names, and
every hook would run twice. Forwarding keeps exactly one registration, and lets
the implementation module be renamed without touching a single consumer.
"""

from __future__ import annotations

pytest_plugins = ["helpers.llk_pytest_plugin"]
