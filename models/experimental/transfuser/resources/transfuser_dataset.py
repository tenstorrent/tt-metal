# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def ensure_scenario3_town01_curved_route0(
    *,
    local_root: str | Path = "models/experimental/transfuser/resources/data",
    cleanup_zip: bool = True,
) -> str:
    base_url = "https://s3.eu-central-1.amazonaws.com/avg-projects/transfuser/data/2022_data"
    zip_name = "s3.zip"
    zip_url = f"{base_url}/{zip_name}"

    inner_dir = (
        "coke_dataset_23_11/Routes_Scenario3_Town01_curved_Seed1000/" "Scenario3_Town01_curved_route0_11_23_20_02_59"
    )

    local_root = Path(local_root).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    scenario_path = local_root / inner_dir

    # Skip if already extracted (depth + label_raw exist in this scenario):contentReference[oaicite:0]{index=0}
    if (scenario_path / "depth").is_dir() and (scenario_path / "label_raw").is_dir():
        return str(scenario_path) + "/"

    if os.environ.get("TRANSFUSER_SKIP_DATA_DOWNLOAD", "0") == "1":
        raise FileNotFoundError(f"Scenario not found at {scenario_path}, and TRANSFUSER_SKIP_DATA_DOWNLOAD=1")

    zip_path = local_root / zip_name

    if not zip_path.exists():
        subprocess.run(["wget", "-c", zip_url, "-O", str(zip_path)], check=True)

    subprocess.run(
        ["unzip", "-q", str(zip_path), f"{inner_dir}/*", "-d", str(local_root)],
        check=True,
    )

    if cleanup_zip:
        try:
            zip_path.unlink()
        except Exception:
            pass

    return str(scenario_path) + "/"
