# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fit QKV input in L1 by reducing N blocking while retaining M4 and K16."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests import probe_optimized_prefill_qkv_l1 as original
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import digest


class InputPlacementN4(original.InputPlacement):
    def __init__(self, source, ttnn):
        self.baseline_programs = source.programs
        self.baseline_n_block = source.precision_policy["n_block"]
        self.candidate_programs = tuple(
            ttnn.MinimalMatmulConfig(
                M_block_size=program.M_block_size,
                K_block_size=program.K_block_size,
                N_block_size=4,
                subblock_h=program.subblock_h,
                subblock_w=program.subblock_w,
                compute_with_storage_grid_size=program.compute_with_storage_grid_size,
            )
            for program in source.programs
        )
        super().__init__(source, ttnn)
        self.observed["program_variants"] = dict(
            baseline=[str(program) for program in self.baseline_programs],
            candidate=[str(program) for program in self.candidate_programs],
        )

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        self._enabled = enabled
        self.source.programs = self.candidate_programs if enabled else self.baseline_programs
        self.source.precision_policy.update(
            n_block=4 if enabled else self.baseline_n_block,
            programs={str(index + 1): str(program) for index, program in enumerate(self.source.programs)},
        )


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    source_hash = digest(__file__)
    try:
        with patch.object(original, "InputPlacement", InputPlacementN4):
            original.main()
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report["qkv_input_l1_adaptation"] = dict(
                probe_sha256=source_hash,
                candidate="Mblock4, Kblock16, Nblock4, subblock1x4, grid11x8; inputL1; unchanged compute/dtypes/output",
                original_failure="M4N8 staticCB end1307648 overlaps liveL1 input start1114112",
                source="minimal_matmul_program_factory.cpp:312-358",
                circular_buffer_bytes=dict(
                    baseline_M4N8=dict(in0=524288, in1=278528, output=262144, intermediate=131072, total=1196032),
                    candidate_M4N4=dict(in0=524288, in1=139264, output=131072, intermediate=65536, total=860160),
                ),
                estimated_static_end=971776,
                estimated_margin_before_recorded_L1_input=142336,
                estimate_scope="Uses recorded original static-region base111616 and unchanged input allocation; actual retry proves legality",
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
        if digest(__file__) != source_hash:
            raise RuntimeError("N4 adaptation changed during the control")


if __name__ == "__main__":
    main()
