# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host cost of ttnn.experimental.moe_block.routed_experts_combine on a program-cache hit: N back-to-back untraced calls
after a warm-up, us a call for the enqueue (host until the call returns). A hit does the resources lookup, the hash and
the address patches only. Compare with the Python reference's experts + combine launch (k3 _experts_combine). Runs where
test_routed_experts_combine.py runs (the same skips).
"""

import time

import pytest

import ttnn

from test_routed_experts_combine import CASES, FLAT_INTEGRATED, OP, RM, U32, _resources, mb  # noqa: F401
from ref_harness import load_case, to_mesh

CALLS = 50


@pytest.mark.skipif(not FLAT_INTEGRATED, reason="SEAM(flat): the flat routed expert's builder is not integrated yet")
@pytest.mark.skipif(not CASES, reason="no reference cases (dump_ref_routed_experts_combine.py, $MOE_BLOCK_REFS)")
@pytest.mark.parametrize("case", CASES[:1] or ["none"])
@pytest.mark.parametrize("device_params", [dict(fabric_config=ttnn.FabricConfig.FABRIC_2D_TORUS_XY)], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_host_us_per_call(mesh_device, case):
    meta, ref = load_case(OP, case)
    if list(mesh_device.shape) != meta["mesh_shape"]:
        pytest.skip(f"case {case} needs a {meta['mesh_shape']} mesh")
    cfg = meta["config"]
    res = _resources(mesh_device, meta)
    try:
        plan = to_mesh(ref, "plan", mesh_device, dtype=U32, layout=RM)
        table = to_mesh(ref, "table", mesh_device, dtype=U32, layout=RM)
        combine = mb.CombineConfig()
        combine.n_acc = cfg["n_acc"]
        combine.arena_bytes = cfg["arena_bytes"]
        kw = dict(
            cells=[ttnn.CoreCoord(x, y) for x, y in cfg["cells"]],
            writers=[ttnn.CoreCoord(x, y) for x, y in cfg["writers"]],
            writer_words_offset=cfg["arena_bytes"],
            combine=combine,
        )
        mb.routed_experts_combine(plan, table, res, **kw)  # (the miss: compile)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(CALLS):
            mb.routed_experts_combine(plan, table, res, **kw)
        host = (time.perf_counter() - t0) / CALLS * 1e6
        ttnn.synchronize_device(mesh_device)
        print(f"routed_experts_combine {case}: {host:.1f} us a call (host, cache hit, {CALLS} calls)")
    finally:
        res.release()
