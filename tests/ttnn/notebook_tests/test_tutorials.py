# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

from models.common.utility_functions import skip_for_blackhole
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

# Constants
TUTORIALS_NOTEBOOK_PATH = Path("ttnn/tutorials/2025_dx_rework")
TUTORIALS_PYTHON_PATH = Path("ttnn/tutorials/basic_python")
"""
The TUTORIALS_DATA_PATHS section contains paths for data needed by tutorials but stored
on external server. When new tutorial will be added with data in external server,
please update this part.
"""
TUTORIAL_DATA_TTNN_SIMPLECNN_INFERENCE = Path("data")
TUTORIALS_DATA_PATHS = [TUTORIAL_DATA_TTNN_SIMPLECNN_INFERENCE]


# Helper functions
def prepare_test_data(model_location_generator):
    for path in TUTORIALS_DATA_PATHS:
        model_location_generator(path, download_if_ci_v2=True)


def collect_ttnn_tutorials(path: Path, extension: str = "*.py"):
    for file_name in path.glob(extension):
        yield file_name


# Tests
@skip_for_blackhole("Fails on BH. Issue #25579")
@pytest.mark.parametrize("notebook_path", collect_ttnn_tutorials(path=TUTORIALS_NOTEBOOK_PATH, extension="*.ipynb"))
def test_ttnn_notebook_tutorials(notebook_path, model_location_generator):
    prepare_test_data(model_location_generator)
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
        ep.preprocess(notebook)


@skip_for_blackhole("Fails on BH. Issue #25579")
@pytest.mark.parametrize("python_path", collect_ttnn_tutorials(path=TUTORIALS_PYTHON_PATH, extension="*.py"))
def test_ttnn_python_tutorials(python_path, model_location_generator):
    prepare_test_data(model_location_generator)
    result = subprocess.run(
        ["python3", str(python_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to run {python_path}:\n{result.stderr}"
