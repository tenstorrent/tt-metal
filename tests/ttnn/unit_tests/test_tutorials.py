# SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

from models.common.utility_functions import skip_for_blackhole
import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor


TUTORIALS_NOTEBOOK_PATH = Path("ttnn/tutorials/2025_dx_rework")
TUTORIALS_PYTHON_PATH = Path("ttnn/tutorials/basic_python")


def collect_ttnn_tutorials(path: Path, extension: str = "*.py"):
    for file_name in path.glob(extension):
        yield file_name


@skip_for_blackhole("Fails on BH. Issue #25579")
@pytest.mark.parametrize("notebook_path", collect_ttnn_tutorials(path=TUTORIALS_NOTEBOOK_PATH, extension="*.ipynb"))
def test_ttnn_notebook_tutorials(notebook_path):
    with open(notebook_path) as f:
        notebook = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=180, kernel_name="python3")
        ep.preprocess(notebook)


@skip_for_blackhole("Fails on BH. Issue #25579")
@pytest.mark.parametrize("python_path", collect_ttnn_tutorials(path=TUTORIALS_PYTHON_PATH, extension="*.py"))
def test_ttnn_python_tutorials(python_path):
    result = subprocess.run(
        ["python3", str(python_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to run {python_path}:\n{result.stderr}"
