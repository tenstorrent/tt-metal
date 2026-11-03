#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""NVFP4 entry point to the shared quantized-decoder harness (scripts/quant_harness.py).

Equivalent to `python scripts/quant_harness.py --format nvfp4 ...`.
"""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import quant_harness as _harness
from scripts.quant_harness import EvaluationError, compare, stage_timing, validate_junit  # noqa: F401

SPEC = _harness.FORMATS["nvfp4"]
KERNELS = SPEC.kernels
MANIFEST = SPEC.manifest
TRUSTED_FILES = list(SPEC.trusted_files)


def math_overlay(candidate, output, repeats):
    return _harness.math_overlay(SPEC, candidate, output, repeats)


def profile_metrics(stage, repetitions=1, pytest_repeats=1):
    return _harness.profile_metrics(SPEC, stage, repetitions, pytest_repeats)


if __name__ == "__main__":
    sys.exit(_harness.main(default_format="nvfp4"))
