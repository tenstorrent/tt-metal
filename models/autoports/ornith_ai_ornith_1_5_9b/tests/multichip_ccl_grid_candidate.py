# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run an existing diagnostic with mesh-wide CCL semaphore allocation.

Example: --probe isolation --operation rs --order BAB --output result.json
Remaining arguments are passed unchanged to the selected diagnostic. No
production module is edited; bindings are restored after the diagnostic exits.
"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import ttnn
from models.demos.gpt_oss.tt.ccl import CCLManager


class MeshGridCCLManager(CCLManager):
    """Cover every worker eligible for the default CCL core-selection policy."""

    observed_grids = []

    def _init_subdevice(self):
        grid = self.mesh_device.compute_with_storage_grid_size()
        self.ccl_cores = ttnn.CoreRangeSet(
            {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}
        )
        self.ccl_sub_device_id = ttnn.SubDeviceId(0)
        self.observed_grids.append([grid.x, grid.y])
        print("CCL_SEMAPHORE_GRID", grid.x, grid.y, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--probe", choices=("isolation", "workspace", "history"), default="isolation")
    args, remaining = parser.parse_known_args()
    package = "models.autoports.ornith_ai_ornith_1_5_9b"
    targets = {
        "isolation": "multichip_ccl_workspace_isolation",
        "workspace": "multichip_workspace_probe",
        "history": "multichip_history_prefill",
    }
    modules = [importlib.import_module(package + ".tt.multichip_decoder")]
    if args.probe == "history":
        # The TILE history candidate inherits a frozen decoder snapshot whose
        # collective constructor resolves CCLManager in its own module globals.
        modules.append(importlib.import_module(package + ".tests.multichip_fusion_reference"))
    previous = [(module, module.CCLManager) for module in modules]
    original_argv = sys.argv[:]
    MeshGridCCLManager.observed_grids.clear()
    metadata = dict(
        candidate="full mesh compute-grid CCL semaphores",
        probe=args.probe,
        command=[sys.executable, *original_argv],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        patched_modules=[module.__name__ for module in modules],
        grids=MeshGridCCLManager.observed_grids,
        completed=False,
    )
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output")
    output_args, _ = output_parser.parse_known_args(remaining)

    def save():
        if output_args.output:
            sidecar = Path(output_args.output + ".candidate.json")
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            sidecar.write_text(json.dumps(metadata, indent=2) + "\n")

    try:
        for module, _ in previous:
            module.CCLManager = MeshGridCCLManager
        sys.argv = [original_argv[0], *remaining]
        save()
        importlib.import_module(package + ".tests." + targets[args.probe]).main()
        metadata["completed"] = True
    except BaseException as error:
        metadata["error"] = repr(error)
        raise
    finally:
        sys.argv = original_argv
        for module, old_manager in previous:
            module.CCLManager = old_manager
        save()


if __name__ == "__main__":
    main()
