# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-expert fixed cost of custom_ops/geglu_routed_expert at the 26B shapes: the warm per-call time with 8 / 32 /
128 / 512 / 2048 rows per local expert (the 128-token dispatch prefill lands ~8 rows per expert; the 0419 trace
profile measured 672 us of kernel per layer against a ~290 us DRAM floor). A linear fit of time vs rows gives the
per-expert intercept. One chip; prints RESULT lines, asserts only correctness (PCC vs the torch golden)."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

import ttnn

_base = Path(__file__).resolve().parent / "test_geglu_routed_expert.py"
_spec = importlib.util.spec_from_file_location("geglu_base_test", _base)
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)

ROWS = [int(v) for v in os.environ.get("GEMMA4_GEGLU_ROWS", "8,32,128,512,2048").split(",")]
# GEMMA4_GEGLU_SHAPE: "block" = the 8x4 Galaxy dispatch shape (4 local experts, 4096 rows each); "4x1" = the QB2
# column's small-row instance (32 local experts, <= 256 rows each). GEMMA4_GEGLU_WEIGHTS: bf4 | bf8 (production
# prefill experts are bf8). GEMMA4_GEGLU_ACTIVE: how many local experts carry rows (the rest count 0 and are
# skipped), default all.
SHAPES = {
    "block": _m.GEMMA_26B,
    "4x1": dict(emb=2816, hidden=704, E=32, max_tokens=256),
}
SHAPE_NAME = os.environ.get("GEMMA4_GEGLU_SHAPE", "block")
SHAPE = dict(SHAPES[SHAPE_NAME])
if os.environ.get("GEMMA4_GEGLU_HIDDEN"):  # intermediate-size override: scales weight bytes and per_core_N, not rounds
    SHAPE["hidden"] = int(os.environ["GEMMA4_GEGLU_HIDDEN"])
WEIGHTS = {"bf4": ttnn.bfloat4_b, "bf8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16}[
    os.environ.get("GEMMA4_GEGLU_WEIGHTS", "bf4")
]
ACTIVE = int(os.environ.get("GEMMA4_GEGLU_ACTIVE", SHAPE["E"]))


@pytest.mark.skipif(not _m.is_blackhole(), reason="GegluRoutedExpert assumes the Blackhole compute grid")
@pytest.mark.parametrize("mesh_device, device_params", _m.MESHES[:1], indirect=True)
def test_geglu_routed_expert_rows(mesh_device, device_params):
    _m.skip_unless_supported(mesh_device)
    D = mesh_device.get_num_devices()
    iters = int(os.environ.get("GEMMA4_GEGLU_TIMING_ITERS", "10"))
    times = []
    tag = (
        f"shape={SHAPE_NAME} hidden={SHAPE['hidden']} weights={os.environ.get('GEMMA4_GEGLU_WEIGHTS', 'bf4')} "
        f"active={ACTIVE}/{SHAPE['E']} "
        f"M_max={os.environ.get('GEMMA4_GEGLU_PER_CORE_M_MAX', 'default')} w_gu={os.environ.get('GEMMA4_GEGLU_IN0_BLOCK_W', 'default')}"
    )
    for n in ROWS:
        assert n <= SHAPE["max_tokens"], (n, SHAPE)
        counts = [[n if e < ACTIVE else 0 for e in range(SHAPE["E"])] for _ in range(D)]
        fx = _m._Fixture(
            mesh_device, **SHAPE, counts_per_chip=counts, x_layout="row_major", seed=3, weights_dtype=WEIGHTS
        )
        op = fx.custom_op()
        custom = fx.per_chip(fx.run_custom(op))
        plan = next(iter(op._plans.values()))  # the plan that call built (one shape per op instance here)
        pcc = _m._check(fx, custom, f"rows{n}")
        t = _m._time_ms(mesh_device, lambda: fx.run_custom(op), iters)
        times.append((n, t))
        print(
            f"RESULT geglu_rows {tag} rows_per_expert={n} warm_ms={t:.3f} per_active_expert_us={t / ACTIVE * 1e3:.1f} "
            f"pcc={pcc:.5f} plan: per_core_M={plan.get('per_core_M')} in0_block_w_gu={plan.get('in0_block_w_gu')} "
            f"num_blocks_gu={plan.get('num_blocks_gu')} num_blocks_d={plan.get('num_blocks_d')}"
        )
    if len(times) >= 2:
        xs = torch.tensor([float(n) for n, _ in times])
        ys = torch.tensor([t for _, t in times])
        A = torch.stack([xs, torch.ones_like(xs)], dim=1)
        slope, intercept = torch.linalg.lstsq(A, ys.unsqueeze(1)).solution.flatten().tolist()
        print(
            f"RESULT geglu_rows fit {tag}: per_row_us={slope * 1e3:.3f} intercept_ms={intercept:.3f} "
            f"(per_active_expert_us={intercept / ACTIVE * 1e3:.1f})"
        )
