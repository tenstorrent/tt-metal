# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The registry of Tuned values in the matmul default config selection and how each is computed.

Every field of a `struct Tuned` in the selector's policies must be listed here, with the calibration sweep and
the fit that compute it from the data in data/<arch>/. test_calibration.py recomputes each value and checks it
against the C++ source. A value whose fit doesn't exist yet is listed with `fit=None` and a note: it was set on
the OOB suite before calibration existed and is pending a sweep.
"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

REPO = Path(__file__).resolve().parents[6]
POLICIES = REPO / "ttnn/cpp/ttnn/operations/matmul/device/config/factory_blocking_source.hpp"


@dataclass
class Tuned:
    policy: str  # the C++ class whose `struct Tuned` holds the field
    field: str
    fit: Optional[Callable] = None  # fit(data_path) -> {"value": ..., "range": (lo, hi), ...}
    data: Optional[str] = None  # file name under data/<arch>/
    note: str = ""


REGISTRY = [
    Tuned("HeuristicBlocking", "max_in0_block_w", note="set on the OOB suite (8 against 16); pending a sweep"),
    Tuned("HeuristicBlocking", "large_block_tiles", note="set on the 2D K-depth sweeps of suite cases; pending"),
    Tuned("HeuristicBlocking", "large_block_in0_block_w", note="set with large_block_tiles; pending"),
    Tuned("HeuristicBlocking", "max_self_read_tiles_per_k_step", note="set on the OOB suite (8 against 4); pending"),
    Tuned("HeuristicFamily", "one_d_core_advantage", note="set on the OOB suite (1.25 to 2 alike); pending"),
]


def source_fields(path=POLICIES):
    """{(policy, field): default value} for every field of every `struct Tuned` in the policies header"""
    text = Path(path).read_text()
    fields = {}
    for cls in re.finditer(r"class (\w+) final : public \w+ \{(.*?)\n\};", text, re.S):
        tuned = re.search(r"struct Tuned \{(.*?)\n    \};", cls.group(2), re.S)
        if not tuned:
            continue
        for m in re.finditer(r"^\s*(?:double|float|u?int\d+_t|bool) (\w+) = ([^;]+);", tuned.group(1), re.M):
            fields[(cls.group(1), m.group(1))] = float(ast.literal_eval(m.group(2).strip()))
    return fields
