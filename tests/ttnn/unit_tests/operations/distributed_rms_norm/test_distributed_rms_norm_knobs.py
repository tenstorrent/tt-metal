# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Knob coverage for distributed_rms_norm: the blocking / wave knobs at non-default values must stay correct (the
default knobs do not reach every path, e.g. multi-block x_depth = 2 or > 1 wave per block on every regime)."""

from __future__ import annotations

import pytest
import ttnn

from eval.golden_tests.distributed_rms_norm.conftest import _fabric_mesh_cache, mesh_device  # noqa: F401 (fixtures)
from ttnn.operations.distributed_rms_norm import distributed_rms_norm_program_descriptor as pd

import importlib.util
from pathlib import Path

# The acceptance test module (immutable spec) holds the reference and the case runner; load it by path (this test
# directory is not a package).
_spec = importlib.util.spec_from_file_location(
    "drms_acceptance", Path(__file__).with_name("test_distributed_rms_norm.py")
)
_acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acc)
LINEAR_AXES, needs_group, run_case = _acc.LINEAR_AXES, _acc.needs_group, _acc.run_case


@needs_group
@pytest.mark.parametrize("mesh_shape", [None])
@pytest.mark.parametrize(
    "min_blocks,num_waves", [(2, 2), (1, 1), (1, 4), (4, 2)], ids=["mb2-w2", "mb1-w1", "mb1-w4", "mb4-w2"]
)
@pytest.mark.parametrize(
    "shape,dtype",
    [((1, 1, 640, 1792), ttnn.bfloat16), ((1, 1, 4096, 1024), ttnn.float32), ((2, 4, 128, 512), ttnn.bfloat16)],
    ids=["640x1792", "4096x1024-fp32", "2x4x128x512"],
)
def test_knobs(mesh_shape, min_blocks, num_waves, shape, dtype, mesh_device, monkeypatch):
    monkeypatch.setattr(pd, "MIN_BLOCKS_PER_CORE", min_blocks)
    monkeypatch.setattr(pd, "NUM_WAVES", num_waves)
    run_case(mesh_device, shape, cluster_axis=LINEAR_AXES[0], dtype=dtype, gamma_dtype=dtype)
    run_case(mesh_device, shape, cluster_axis=LINEAR_AXES[0], dtype=dtype, has_gamma=False, seed=3)
