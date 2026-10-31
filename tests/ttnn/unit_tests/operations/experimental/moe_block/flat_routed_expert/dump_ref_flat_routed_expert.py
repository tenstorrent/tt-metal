# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reference cases for ttnn.experimental.moe_block.flat_routed_expert, produced by the Python-tree op
(ttnn.bringup.flat_routed_expert via FlatRoutedExpert in mstaletovic/moe-trim). Run in that tree's environment on one
Blackhole chip of a Galaxy:

    MOE_BLOCK_REFS=<refs root> pytest -q <this file>        (MOE_BLOCK_REFS: where to write; required)

Each case is one chip at the Galaxy 8x4 per-chip load (the flat expert runs the same program on every chip, chip by
chip, with no fabric): the seeded inputs (flat_cases.case_inputs), the plain op's y, the plan's layout facts, and the
reference's host time per call (untraced enqueue, after warm-up) for the C++ op's host-timing test. The weights are
not stored: flat_cases.weight_sets regenerates them from the seed, and meta records their digest.
"""

import importlib.util
import time
from pathlib import Path

import pytest
import torch

import ttnn

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


harness = _load("moe_block_ref_harness", HERE.parent / "ref_harness.py")
cases = _load("flat_cases", HERE / "flat_cases.py")
HOST_CALLS = 50


@pytest.mark.skipif(not harness.refs_available(), reason="set MOE_BLOCK_REFS (where to write the cases)")
@pytest.mark.timeout(3600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("case", cases.REF_CASES, ids=[cases.case_name(*c) for c in cases.REF_CASES])
def test_dump_ref_flat_routed_expert(device, case):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    model, chunk, y_row_major = case
    md, pc = cases.MODELS[model], cases.per_chip(model, "8x4", chunk)
    H, I, E, NG = md["H"], md["I"], pc["epc"], md["E"]
    wsets = cases.weight_sets(H, I)
    x, c_, r_, gids, spans = cases.case_inputs(model, chunk)
    layout = dict(rows_hint=round(pc["M"]))
    op = FlatRoutedExpert(
        device,
        [[wsets[e % cases.N_W] for e in range(E)]],
        m=chunk,
        H=H,
        I=I,
        gids=[gids],
        n_global=NG,
        act=md["act"],
        pin=1,
        layout=layout,
    )
    rm = lambda t, dt: ttnn.from_torch(  # noqa: E731
        t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    xd, cd, rd = rm(x, ttnn.bfloat16), rm(c_, ttnn.uint32), rm(r_, ttnn.uint32)
    y = op(xd, cd, rd, y_row_major=y_row_major)
    y_host = ttnn.to_torch(y)
    # host cost per call: untraced back-to-back enqueues after warm-up (program-cache hits)
    for _ in range(3):
        op(xd, cd, rd, y_row_major=y_row_major)
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(HOST_CALLS):
        op(xd, cd, rd, y_row_major=y_row_major)
    host_us = (time.perf_counter() - t0) / HOST_CALLS * 1e6
    ttnn.synchronize_device(device)

    p = dict(op.plan)
    plan = {k: p[k] for k in cases.PLAN_KEYS}
    plan.update({k: [list(c) for c in p[k]] for k in cases.PLAN_CORE_KEYS})
    per_chip = {
        (0, 0): {
            "x": x.to(torch.bfloat16),
            "counts": c_,
            "regions": r_,
            "global_expert_ids": torch.tensor(gids, dtype=torch.int32),
            "y": y_host,
        }
    }
    harness.save_case(
        "flat_routed_expert",
        cases.case_name(*case),
        (1, 1),
        per_chip,
        inputs=["x", "counts", "regions", "global_expert_ids"],
        outputs=["y"],
        seed=cases.W_SEED,
        model=model,
        config=dict(
            chunk=chunk,
            stage="8x4",
            H=H,
            I=I,
            E=E,
            NG=NG,
            m=chunk,
            act=md["act"],
            pin=1,
            wdtype="bf4",
            layout=layout,
            y_row_major=y_row_major,
            spans=spans,
        ),
        extra=dict(
            weights=dict(seed=cases.W_SEED, n_sets=cases.N_W, scale=cases.W_SCALE, sha256=cases.weights_digest(wsets)),
            plan=plan,
            host_us_per_call=host_us,
        ),
    )
