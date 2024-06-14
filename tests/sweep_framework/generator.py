# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import z3
import importlib
import pathlib

from general_contraints import *
from architecture import str_to_arch
from permutations import *

SWEEPS_DIR = pathlib.Path(__file__).parent
SWEEP_SOURCES_DIR = SWEEPS_DIR / "sweeps"


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


def generate_vectors(module_name, arch):
    test_module = importlib.import_module("sweeps." + module_name[:3])  # Macro this
    parameters = test_module.parameters

    vectors = permutations(parameters)
    return vectors


# Perform any post-gen validation to the resulting vectors.
def validate_vectors(vectors) -> None:
    pass


# Output the individual test vectors from solver.model()
def export_test_vectors(vectors):
    # Perhaps we export with some sort of readable id, which can be passed to a runner to run specific sets of input vectors. (export seed as well for reproducability)
    print(vectors)
    pass


# Generate one or more sets of test vectors depending on module_name
def generate_tests(module_name, arch, output_dir):
    if not module_name:
        for file_name in sorted(SWEEP_SOURCES_DIR.glob("**/*.py")):
            vectors = generate_vectors(pathlib.Path(file_name).relative_to(SWEEP_SOURCES_DIR), arch)
            validate_vectors(vectors)
            export_test_vectors(vectors)
    else:
        vectors = generate_vectors(module_name, arch)
        validate_vectors(vectors)
        export_test_vectors(vectors)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Sweep Test Vector Generator",
        description="Generate test vector suites for the specified module.",
    )

    parser.add_argument("--module-name", required=False, help="Test Module Name, or all tests if omitted")
    parser.add_argument("--output-dir", required=True, help="Output Directory")
    parser.add_argument("--seed", required=False, default=0, help="Seed for random value generation")
    parser.add_argument(
        "--arch", required=True, choices=["grayskull", "wormhole", "wormhole_b0", "blackhole"], help="Output Directory"
    )

    args = parser.parse_args(sys.argv[1:])

    generate_tests(args.module_name, str_to_arch(args.arch), args.output_dir)
