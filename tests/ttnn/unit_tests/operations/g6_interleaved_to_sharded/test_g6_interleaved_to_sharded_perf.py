# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf bench for g6_interleaved_to_sharded on the PERF FOCUS case (FLUX 58, p1).

Measurement only: run under `scripts/run_safe_pytest.sh --profile --no-py-zones`. Every test makes
REPEATS op calls of one variant and no readback, so the profiler CSV holds REPEATS GenericOp rows
per test, in parametrize order. The launch stagger across the 32 cores is noisy; compare medians. Correctness of each knob value is graded by test_g6_interleaved_to_sharded_knobs.py.

"ablate_reads" compiles the reader with G6I2S_ABLATE_READS: the NoC reads are dropped and the
launch, argument loading, address generation and barrier stay, which gives the sync-only floor.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from eval.golden_tests.g6_interleaved_to_sharded.feature_spec import FLUX_58
from eval.golden_tests.g6_interleaved_to_sharded.helpers import (
    build_memory_config,
    create_ttnn_input_tensor,
    make_torch_input,
)
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded_program_descriptor as descriptor

_ABLATE = (("G6I2S_ABLATE_READS", "1"),)
REPEATS = 5

_TWO_SPANS = {"LAST_ROW_FORM": "two_spans"}
_ONE_SPAN = {"LAST_ROW_FORM": "one_span"}
_ONE_READER = {"READER_WEIGHTS": (1,)}

VARIANTS = {
    "default": {},
    "two_spans-1reader": {**_TWO_SPANS, **_ONE_READER},
    "one_span-1reader": {**_ONE_SPAN, **_ONE_READER},
    "whole_page-1reader": {"LAST_ROW_FORM": "whole_page", **_ONE_READER},
    "one_span-1reader-noc1": {**_ONE_SPAN, **_ONE_READER, "READER_NOC": ttnn.NOC.NOC_1},
    "two_spans-split1:1": {**_TWO_SPANS, "READER_WEIGHTS": (1, 1)},
    "one_span-split1:1": {**_ONE_SPAN, "READER_WEIGHTS": (1, 1)},
    "one_span-split3:1": {**_ONE_SPAN, "READER_WEIGHTS": (3, 1)},
    "one_span-split1:1-noc1first": {**_ONE_SPAN, "READER_WEIGHTS": (1, 1), "READER_NOC": ttnn.NOC.NOC_1},
    "ablate-1reader": {**_ONE_READER, "READER_DEFINES": _ABLATE},
    "ablate-split1:1": {"READER_WEIGHTS": (1, 1), "READER_DEFINES": _ABLATE},
}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_perf_flux58(variant, device, monkeypatch):
    for name, value in VARIANTS[variant].items():
        monkeypatch.setattr(descriptor, name, value)

    torch.manual_seed(0)
    torch_input = make_torch_input(FLUX_58["shape"], FLUX_58["dtype"])
    ttnn_input = create_ttnn_input_tensor(
        torch_input,
        device,
        dtype=FLUX_58["dtype"],
        layout=FLUX_58["layout"],
        memory_config=build_memory_config(FLUX_58["in_memory"], device),
    )
    output_config = build_memory_config(FLUX_58["out_memory"], device)
    for _ in range(REPEATS):
        output = g6_interleaved_to_sharded(ttnn_input, output_config)
        ttnn.synchronize_device(device)
        assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        output.deallocate()
