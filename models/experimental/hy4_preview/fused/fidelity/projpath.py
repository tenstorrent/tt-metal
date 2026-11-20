# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The projections piece's programs (fused/projections/program.py) with their compute kernel replaced by
fused/projections/kernels/proj_compute_hifi.cpp at ONE pass schedule for every unit, whatever the program was built
with: the instrument of tests/test_projpath.py. (The piece itself now takes ProjProgram(fidelity=...), with a
schedule per unit.)

    op = P.pre_program(device, w, fmt="bfp8")
    use_hifi(op, "hifi2")          # before the first call

Nothing of fused/projections is edited: the program descriptor it builds is given another kernel source."""

from __future__ import annotations

from pathlib import Path

from models.experimental.hy4_preview.fused.fidelity.lab import FIDELITIES

# the projections piece's own hifi_mm compute kernel (ProjProgram(fidelity=...) uses it too)
HIFI_COMPUTE = Path(__file__).parent.parent / "projections" / "kernels" / "proj_compute_hifi.cpp"


SPLIT = {"plain": 0, "finalize": 1, "merge": 2}


def use_hifi(op, fidelity="lofi", split="merge"):
    """Swap the compute kernel of a ProjProgram for the hifi_mm one. fidelity: lab.FIDELITIES; split: "merge"
    (DenseSplitAcc with the FP32 merge), "finalize" (DenseSplitAcc with custom_mm's merge, as the original) or
    "plain" (Dense)."""
    passes, step = FIDELITIES[fidelity]
    program, io = op.descriptor()
    kernels = []
    for k in program.kernels:
        if str(k.kernel_source).endswith("proj_compute.cpp"):
            k.kernel_source = str(HIFI_COMPUTE)
            k.defines = list(k.defines) + [
                ("HY4_FID_PASSES", str(passes)),
                ("HY4_FID_STEP", str(step)),
                ("HY4_FID_SPLIT", str(SPLIT[split])),
            ]
        kernels.append(k)
    program.kernels = kernels
    op._program = (program, io)
    return op
