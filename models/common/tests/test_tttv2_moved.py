# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTTv2 moved to tenstorrent/tt-transformers: the old import paths must fail loudly and point there.

Host-only: the stubs raise before anything imports ttnn.
"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "models.common.modules",
        "models.common.llm_runtime",
        "models.common.models",
        "models.common.modules.mlp.mlp_1d",
        "models.common.modules.tt_ccl",
    ],
)
def test_tttv2_import_paths_raise_with_a_pointer(module):
    with pytest.raises(ImportError) as excinfo:  # allow-pytest.raises: host-only, runs under --noconftest
        importlib.import_module(module)
    message = str(excinfo.value)
    assert "https://github.com/tenstorrent/tt-transformers" in message
    assert "models.common.{lazy_weight, tt_ccl, moe}" in message
