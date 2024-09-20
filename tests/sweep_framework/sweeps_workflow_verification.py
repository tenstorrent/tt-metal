# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from yaml import load, Loader
import pathlib

with open(".github/workflows/ttnn-run-sweeps.yaml") as file:
    workflow = load(file, Loader)
    workflow_count = len(workflow[True]["workflow_dispatch"]["inputs"]["sweep_name"]["options"])

    sweeps_path = pathlib.Path(__file__).parent / "sweeps"
    file_count = len(list(sweeps_path.glob("**/*.py")))
    assert (
        file_count + 1 == workflow_count
    ), f"Sweeps workflow options does not match expected number of sweep files ({workflow_count} exist, expected {file_count + 1}). If you added a new sweep file, please add it to the options in the .github/workflows/ttnn-run-sweeps.yaml workflow file."
    print("Sweeps workflow options match expected number of sweep files.")
