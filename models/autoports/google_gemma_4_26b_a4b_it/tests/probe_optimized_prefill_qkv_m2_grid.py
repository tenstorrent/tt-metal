# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pair 88/110 cores at matched M2/K16/N8 and copied L1 QKV input."""

import json
import sys
from pathlib import Path
from unittest.mock import patch

from models.autoports.google_gemma_4_26b_a4b_it.tests import probe_optimized_prefill_qkv_l1 as original
from models.autoports.google_gemma_4_26b_a4b_it.tests.probe_optimized_minimal_advice import digest


class InputPlacementGrid(original.InputPlacement):
    def __init__(self, source, ttnn):
        if str(source.compute.math_fidelity) != "MathFidelity.HiFi2":
            raise ValueError("This matched full-QKV control requires selected HiFi2")
        if not source.compute.fp32_dest_acc_en or source.compute.packer_l1_acc:
            raise ValueError("Expected selected FP32 destination without packer accumulation")
        if str(source.weight.dtype) != "DataType.BFLOAT8_B":
            raise ValueError("Expected selected BFP8 QKV weight")
        for program in source.programs:
            if (program.K_block_size, program.N_block_size, program.subblock_h, program.subblock_w) != (16, 8, 1, 4):
                raise ValueError("Expected selected K16/N8/subblock1x4")
        self.program_variants = tuple(
            tuple(
                ttnn.MinimalMatmulConfig(
                    M_block_size=min(2, program.M_block_size),
                    K_block_size=program.K_block_size,
                    N_block_size=program.N_block_size,
                    subblock_h=program.subblock_h,
                    subblock_w=program.subblock_w,
                    compute_with_storage_grid_size=ttnn.CoreCoord(11, grid_y),
                )
                for program in source.programs
            )
            for grid_y in (8, 10)
        )
        super().__init__(source, ttnn)
        self.observed["program_variants"] = dict(
            baseline=[str(program) for program in self.program_variants[0]],
            candidate=[str(program) for program in self.program_variants[1]],
        )

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        self._enabled = enabled
        self.source.programs = self.program_variants[int(enabled)]
        self.source.precision_policy.update(
            m_block="min(2, padded_M_tiles)",
            grid=(11, 10) if enabled else (11, 8),
            programs={str(index + 1): str(program) for index, program in enumerate(self.source.programs)},
        )

    def __call__(self, value, *, memory_config=None):
        before = str(value.memory_config())
        value = self.ttnn.to_memory_config(value, self.ttnn.L1_MEMORY_CONFIG)
        key = "candidate" if self.enabled else "baseline"
        if key not in self.observed:
            self.observed[key] = dict(
                input_shape=list(value.shape),
                input_dtype=str(value.dtype),
                input_before_memory=before,
                input_memory=str(value.memory_config()),
                requested_output_memory=str(memory_config),
                grid=(11, 10) if self.enabled else (11, 8),
            )
        return self.source(value, memory_config=memory_config)


def main():
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    source_hash = digest(__file__)
    try:
        with patch.object(original, "InputPlacement", InputPlacementGrid):
            original.main()
    finally:
        if output.exists():
            report = json.loads(output.read_text())
            report["qkv_input_l1"]["candidate"] = "Both copy input to L1; only grid11x8 versus11x10 differs"
            report["qkv_grid_control"] = dict(
                probe_sha256=source_hash,
                baseline="Mblock2/Kblock16/Nblock8/subblock1x4/grid11x8; copied FP32 inputL1",
                candidate="Mblock2/Kblock16/Nblock8/subblock1x4/grid11x10; copied FP32 inputL1",
                unchanged="HiFi2, FP32 destination, BFP8 weight, DRAM output, decode projection and input-copy boundary",
                advice_predicate="Increase grid if FLOPs>=65% and DRAM<65%; projected threshold336.082us at88 cores",
            )
            output.write_text(json.dumps(report, indent=2) + "\n")
        if digest(__file__) != source_hash:
            raise RuntimeError("Matched M2 grid probe changed during control")


if __name__ == "__main__":
    main()
