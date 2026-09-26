# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pair DRAM versus L1 input with identical M2/K16/N8 minimal QKV programs."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests import probe_optimized_prefill_qkv_l1 as original
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import describe, digest


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    source_hash = digest(__file__)
    prepared = []

    class FixedM2Placement(original.InputPlacement):
        def __init__(self, source, ttnn):
            source.programs = tuple(
                ttnn.MinimalMatmulConfig(
                    M_block_size=min(2, program.M_block_size),
                    K_block_size=program.K_block_size,
                    N_block_size=program.N_block_size,
                    subblock_h=program.subblock_h,
                    subblock_w=program.subblock_w,
                    compute_with_storage_grid_size=program.compute_with_storage_grid_size,
                )
                for program in source.programs
            )
            source.precision_policy.update(
                m_block="min(2, padded_M_tiles)",
                programs={str(index + 1): str(program) for index, program in enumerate(source.programs)},
            )
            prepared.append(describe(source))
            super().__init__(source, ttnn)
            self.observed["program_variants"] = dict(
                baseline=[str(program) for program in source.programs],
                candidate=[str(program) for program in source.programs],
            )

    try:
        with patch.object(original, "InputPlacement", FixedM2Placement):
            original.main()
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            if prepared:
                report["qkv_input_l1"]["metadata"]["projection"] = prepared[0]
            report["qkv_input_l1_adaptation"] = dict(
                probe_sha256=source_hash,
                baseline="DRAM input; M2 K16 N8 sub1x4 grid11x8",
                candidate="L1 input; identical M2 K16 N8 sub1x4 grid11x8",
                scope="Only placement toggles between paired calls; compute, dtype, weight and all programs are identical",
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
        if digest(__file__) != source_hash:
            raise RuntimeError("Matched M2 placement probe changed during the control")


if __name__ == "__main__":
    main()
