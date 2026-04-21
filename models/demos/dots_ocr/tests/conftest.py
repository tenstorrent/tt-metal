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

try:
    # TT-backed tests need the real fixture; importing it pulls in ``ttnn``.
    from models.tt_transformers.conftest import device_params  # noqa: F401
except ImportError:  # pragma: no cover — CPU-only / no Tenstorrent wheel
    import pytest

    @pytest.fixture
    def device_params(request):
        pytest.skip("ttnn is not installed; run TT tests in an environment with tt-metal + ttnn.")


def pytest_configure(config):
    # Before any HF remote-code import: satisfy ``check_imports`` for ``flash_attn`` while we use
    # ``_attn_implementation="eager"`` (see ``reference/_flash_attn_shim.py``).
    from models.demos.dots_ocr.reference._flash_attn_shim import install

    install()
