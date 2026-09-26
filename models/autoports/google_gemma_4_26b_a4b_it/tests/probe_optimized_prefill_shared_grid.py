# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Redistribute both shared prefill projections across 110 active cores."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests import probe_optimized_prefill_placement as placement
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import digest


class SharedGridControl(placement.PlacementControl):
    def __init__(self, decoder, mesh, candidate, ttnn):
        super().__init__(decoder, mesh, "shared_l1", ttnn)
        self.baseline_programs = self.programs
        self.grid_programs = {
            (2816, 4224): self._grid_program(8, 14, 2),
            (2112, 2816): self._grid_program(6, 9, 1),
        }
        self.candidate = "shared_grid110"
        self.metadata.update(
            candidate=self.candidate,
            baseline_programs={str(key): str(value) for key, value in self.baseline_programs.items()},
            candidate_programs={str(key): str(value) for key, value in self.grid_programs.items()},
            active_core_geometry="transpose_mcast: ceil(Mtiles32/3)=11 columns; ceil(Ntiles132/14 or88/9)=10 rows",
            padded_tile_work=dict(gate_before=32 * 132, gate_after=33 * 140, down_before=32 * 88, down_after=33 * 90),
            input_memory="DRAM interleaved in both variants",
        )
        self.set_enabled(True)

    def _grid_program(self, block, per_n, sub_w):
        ttnn = self.ttnn
        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(11, 10),
            in0_block_w=block,
            out_subblock_h=3,
            out_subblock_w=sub_w,
            out_block_h=3,
            out_block_w=per_n,
            per_core_M=3,
            per_core_N=per_n,
            transpose_mcast=True,
            fused_activation=None,
            fuse_batch=False,
        )

    def set_enabled(self, enabled):
        self.grid_enabled = enabled
        self.enabled = False  # The parent linear wrapper must keep input in DRAM.
        self.programs = self.grid_programs if enabled else self.baseline_programs


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    probe_hash = digest(__file__)
    try:
        with (
            patch.object(placement, "PlacementControl", SharedGridControl),
            patch.object(sys, "argv", [sys.argv[0], "--prefill-placement", "shared_l1", *sys.argv[1:]]),
        ):
            placement.main()
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report["shared_grid_probe_sha256"] = probe_hash
            report["prefill_placement_candidate"] = {"prefill_placement": "shared_grid110"}
            output.write_text(json.dumps(report, indent=2) + "\n")
        if digest(__file__) != probe_hash:
            raise RuntimeError("Shared grid probe changed during control")


if __name__ == "__main__":
    main()
