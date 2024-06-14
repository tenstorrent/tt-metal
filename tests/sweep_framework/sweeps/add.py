# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

# from z3 import *
import ttnn


parameters = {
    "batch_sizes": [(1,)],
    "height": [384, 1024],
    "width": [1024, 4096],
    "broadcast": [None, "h", "w", "hw"],
    "input_a_dtype": [ttnn.bfloat16],
    "input_b_dtype": [ttnn.bfloat16],
    "input_a_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "input_b_layout": [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT],
    "input_b_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "input_a_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
    "output_memory_config": [ttnn.DRAM_MEMORY_CONFIG],
}

# # Add op-specific test vector properties to be generated
# # Needs some global mapping so they can be constrained after being added in apply_op_specific_constraints
# def add_vars(solver: Solver):

#     pass

# def apply_op_specific_constraints(solver: z3.Solver):
#     # Add additional contraints for this test
#     solver.add(Int("batch_size") == 1)
#     # solver.add(Or(Int("input") == 384, Int("input") == 1024))
#     height_constraints = [384, 1024]
#     width_constraints = [1024, 4096]
#     height = Int("height")
#     width = Int("width")
#     solver.add(And(height >= height_constraints[0], height <= height_constraints[1]))
#     solver.add(And(width >= width_constraints[0], width <= width_constraints[1]))
#     pass


def run():
    # Run directives for this test, used by the runner, not by test input generation.
    pass
