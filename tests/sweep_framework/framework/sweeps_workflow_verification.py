# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from yaml import load, Loader
import pathlib

with open(".github/workflows/ttnn-run-sweeps.yaml") as file:
    workflow = load(file, Loader)
    workflow_count = len(workflow[True]["workflow_dispatch"]["inputs"]["sweep_name"]["options"])

    sweeps_path = pathlib.Path(__file__).parent.parent / "sweeps"
    # Exclude model_traced files - they're run via "ALL SWEEPS (Model Traced)" option, not individually listed
    all_files = list(sweeps_path.glob("**/*.py"))
    file_count = len([f for f in all_files if "model_traced" not in str(f)])
    # Account for "ALL SWEEPS (Nightly)", "ALL SWEEPS (Comprehensive)", and "ALL SWEEPS (Model Traced)" options (+3)
    assert (
        file_count + 3 == workflow_count
    ), f"Sweeps workflow options does not match expected number of sweep files ({workflow_count} exist, expected {file_count + 3}). Expected: {file_count} sweep files + 3 'ALL SWEEPS' options. If you added a new sweep file, please add it to the options in the .github/workflows/ttnn-run-sweeps.yaml workflow file."
    print("Sweeps workflow options match expected number of sweep files.")
