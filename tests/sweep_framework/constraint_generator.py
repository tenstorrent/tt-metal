# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from general_contraints import *


# Unused (Move to constraint based vector generation file)
def generate_vectors_z3(module_name, arch) -> z3.Model:
    test_module = importlib.import_module("sweeps." + module_name[:3])  # Macro this
    solver = z3.Solver()  # Probably want this in some test wrapper

    # Apply general purpose arch-specific constraints
    apply_tensor_constraints(solver, arch)
    apply_memory_constraints(solver, arch)
    apply_layout_constraints(solver, arch)

    # Apply op-specific constraints
    test_module.apply_op_specific_constraints(solver)

    solver.check()
    return solver.model()
