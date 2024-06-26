# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import z3
from architecture import Arch


# Add set of general tensor contraints to solver
def apply_tensor_constraints(solver: z3.Solver, arch: Arch) -> None:
    # x = solver.append
    # solver.add(x > 3)

    print("adding general constraints")


def apply_memory_constraints(solver: z3.Solver, arch: Arch) -> None:
    pass


def apply_layout_constraints(solver: z3.Solver, arch: Arch) -> None:
    pass
