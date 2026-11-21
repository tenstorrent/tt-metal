# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the global-torch-import hook validate_no_global_torch_imports.py."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import validate_no_global_torch_imports as v  # noqa: E402


def imports(tmp_path, src):
    p = tmp_path / "module.py"
    p.write_text(src)
    return v.check_file(str(p))


@pytest.mark.parametrize(
    "src, expected_stmt",
    [
        ("import torch\n", "import torch"),
        ("import torch.nn as nn\n", "import torch.nn"),
        ("import torch.distributed\n", "import torch.distributed"),
        ("from torch import Tensor\n", "from torch import ..."),
        ("from torch.nn.functional import relu\n", "from torch.nn.functional import ..."),
    ],
)
def test_global_torch_imports_are_reported_including_submodules(tmp_path, src, expected_stmt):
    assert imports(tmp_path, src) == [(1, expected_stmt)]


@pytest.mark.parametrize("src", ["import torchvision\n", "import torchtune\n", "from torchvision import ops\n"])
def test_non_torch_packages_are_not_reported(tmp_path, src):
    assert imports(tmp_path, src) == []


def test_function_local_import_is_allowed(tmp_path):
    assert imports(tmp_path, "def f():\n    import torch.nn as nn\n    return nn\n") == []
