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
        yield file_name


@skip_for_wormhole_b0()
@pytest.mark.parametrize("notebook_path", collect_tutorials())
def test_tutorials(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
        ep.preprocess(notebook)
