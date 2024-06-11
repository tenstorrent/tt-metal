# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import pathlib
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

from models.utility_functions import skip_for_wormhole_b0

import ttnn

TUTORIALS_PATH = pathlib.Path(ttnn.__path__[0]).parent / "tutorials"


def collect_tutorials():
    for file_name in TUTORIALS_PATH.glob("*.ipynb"):
        if (
            "tutorials/001.ipynb" in str(file_name)
            or "tutorials/002.ipynb" in str(file_name)
            or "tutorials/003.ipynb" in str(file_name)
            or "tutorials/004.ipynb" in str(file_name)
        ):
            yield file_name


@pytest.mark.requires_fast_runtime_mode_off
@pytest.mark.parametrize("notebook_path", collect_tutorials())
def test_tutorials(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
        ep.preprocess(notebook)
