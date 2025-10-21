# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
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
LOCAL_SOURCE_PATH_KEY = "local"
EXTERNAL_SOURCE_PATH_KEY = "external"
EXTERNAL_SERVER_BASE_URL = "http://large-file-cache.large-file-cache.svc.cluster.local//tutorials_data"
LOCAL_BASE_DIRECTORY = "tutorials_data"

TUTORIALS_DATA_PATHS = {
    "ttnn_simplecnn_inference": {LOCAL_SOURCE_PATH_KEY: "./data", EXTERNAL_SOURCE_PATH_KEY: "ttnn_simplecnn_inference"},
    # NOTE: Add entries here for new tutorials that require external data
}


@pytest.fixture(scope="module", autouse=True)
def setup_once(model_location_generator):
    """
    Prepare test data for tutorial tests by setting up symbolic links.

    This function iterates over data needed for tutorial tests. If the data is present
    in CIv2 LFC, it downloads it. To avoid changing paths in the tutorials themselves
    (so customers can download and run them without modifying the paths), we create
    symbolic links between where the data was downloaded and where the tutorial
    expects it to be.

    Args:
        model_location_generator: A callable that takes an external path and returns
            the local path where the data should be placed. Should support
            download_if_ci_v2 parameter for conditional downloading.

    Returns:
        None

    Effect:
        - Downloads data from external servers via model_location_generator
        - Creates symbolic links from local paths to downloaded data locations
        - Removes existing symlinks/files at local paths if they exist
    """
    tt_metal_path = os.environ.get("TT_METAL_HOME", "/work")

    for tutorial_id in TUTORIALS_DATA_PATHS.keys():
        # Download data from external server
        local_path = TUTORIALS_DATA_PATHS[tutorial_id][LOCAL_SOURCE_PATH_KEY]
        external_path = TUTORIALS_DATA_PATHS[tutorial_id][EXTERNAL_SOURCE_PATH_KEY]

        # Skip if local path exists and has content
        local_path_obj = Path(tt_metal_path) / Path(local_path)

        if local_path_obj.exists() and any(local_path_obj.iterdir()):
            continue

        # Download data using model_location_generator
        download_dir_suffix = Path(LOCAL_BASE_DIRECTORY) / Path(external_path)
        data_placement = model_location_generator(
            external_path,
            download_if_ci_v2=True,
            endpoint_prefix=EXTERNAL_SERVER_BASE_URL,
            download_dir_suffix=download_dir_suffix,
        )
        data_placement = Path(data_placement)

        # Create symbolic link from local_path to data_placement
        if local_path_obj.exists():
            local_path_obj.unlink()  # Remove existing symlink/file
        local_path_obj.symlink_to(data_placement.parent)


def collect_ttnn_tutorials(path: Path, extension: str = "*.py"):
    for file_name in path.glob(extension):
        yield file_name


# Tests
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
