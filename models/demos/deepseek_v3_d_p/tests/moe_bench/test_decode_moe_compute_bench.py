# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Decode MoE benchmark: ttnn.experimental.moe_compute on one Blackhole Galaxy chip, one row per model shape and
decode batch.

The chip holds the experts it would own with expert parallelism over the whole Galaxy (E / 32), and receives the
tokens routed to it. With U decode users per chip (32 * U users per Galaxy) each local expert sees on average
32 * U * k / E tokens; a single chip running U tokens at top-k over its E / 32 local experts sees the same load.
compute_only=True isolates the expert FFN (tilize -> W0/W1 -> activation -> W2), the part with a roofline;
dispatch and combine are fabric ops.

Each moe_compute call is replayed MOE_BENCH_DECODE_REPEATS times (default 10) so the profiler sees warm, program-
cache-hit runs; roofline.py drops the first run of each call. The upstream single-card runner still validates the
final output, so every number comes from a correct run.

    python -m tracy -r -v -m pytest models/demos/deepseek_v3_d_p/tests/moe_bench/test_decode_moe_compute_bench.py
"""

import os

import pytest

import ttnn
from tracy import signpost
from models.demos.deepseek_v3_d_p.tests.moe_bench.moe_models import decode_users_per_chip, selected_models
from tests.ttnn.nightly.unit_tests.operations.experimental.test_moe_compute_single_card import (
    _run_moe_compute_single_card_test,
)
from ttnn.experimental.moe_compute_utils import auto_output_width_shard_dim, effective_matmul_ring_size
from ttnn.operations.ccl import MoEActivationFunction

_REPEATS = int(os.environ.get("MOE_BENCH_DECODE_REPEATS", "10"))
_FP32_DEST = os.environ.get("MOE_BENCH_DECODE_FP32_DEST", "0")


def _cases():
    cases = []
    for m in selected_models():
        for users in decode_users_per_chip():
            for fp32 in (int(x) for x in _FP32_DEST.split(",")):
                cases.append(pytest.param(m, users, bool(fp32), id=f"{m.name}-u{users}-{'fp32dest' if fp32 else 'bf16dest'}"))
    return cases


@pytest.fixture
def repeated_moe_compute(monkeypatch):
    """Replay every moe_compute call so the profiler records warm runs; return the last run's outputs."""
    original = ttnn.experimental.moe_compute

    def repeated(*args, **kwargs):
        for _ in range(_REPEATS - 1):
            outputs = original(*args, **kwargs)
            # Slots 3 and 4 share one L1 buffer; a supplied output tensor (slot 5) is the caller's.
            for slot in (0, 1, 2, 4):
                ttnn.deallocate(outputs[slot])
        return original(*args, **kwargs)

    monkeypatch.setattr(ttnn.experimental, "moe_compute", repeated)


@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 16384, "dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 500000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("model, users_per_chip, fp32_dest", _cases())
@pytest.mark.timeout(0)
def test_decode_moe_compute_bench(mesh_device, mesh_shape, model, users_per_chip, fp32_dest, repeated_moe_compute):
    experts_per_chip = model.experts_per_chip
    k = min(model.k, experts_per_chip)
    grid = mesh_device.compute_with_storage_grid_size()
    # roofline.py splits the profiler CSV on these and reads the case parameters back from them.
    tag = (
        f"moe_bench|decode|model={model.name}|H={model.hr}|N={model.inter}|E_local={experts_per_chip}|k={k}"
        f"|T={users_per_chip}|fp32_dest={int(fp32_dest)}|ring={effective_matmul_ring_size(mesh_device)}"
        f"|grid={grid.x}x{grid.y}|repeats={_REPEATS}"
    )
    signpost(header=f"{tag}|START")
    _run_moe_compute_single_card_test(
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        experts_per_device=experts_per_chip,
        tokens_per_device=users_per_chip,
        selected_experts_k=k,
        N=model.inter,
        hidden_size=model.hr,
        output_height_shard_dim=4,
        output_width_shard_dim=auto_output_width_shard_dim(model.hr),
        dtype=ttnn.bfloat16,
        activation_type=getattr(MoEActivationFunction, model.decode_act),
        activation_limit=model.decode_act_limit,
        compute_only=True,
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=fp32_dest,
    )
    signpost(header=f"{tag}|END")
