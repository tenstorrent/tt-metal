# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Local test config for dots_ocr.

Mirrors ``models/demos/qwen25_vl/conftest.py`` by re-exporting the ``device_params``
fixture from ``models.tt_transformers.conftest`` so tests can parametrize it the same way
(e.g. ``fabric_config``, ``trace_region_size``, ``num_command_queues``).

Tests can still be run in isolation without pulling the repo-wide ``tt-metal/conftest.py``:

  pytest models/demos/dots_ocr/tests --confcutdir=models/demos/dots_ocr/tests
"""

from models.tt_transformers.conftest import device_params  # noqa: F401
