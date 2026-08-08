# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
#
"""Minimal repro for the BH traced untilize(sub_core_grids)+argmax corruption.

In the Qwen accuracy run (prefetcher / unfused-CCL path) the traced untilize writes the
sampling logits displaced by exactly 1824 tiles (3/8 of the row), so the on-device argmax
returns token+58368. Eagerly rerunning the same ops on the same input is correct, and the
gathered input is verified byte-perfect, so the corruption needs: program-cache reuse with a
DIFFERENT input address (compile vs capture) + trace capture/replay + sub_core_grids.

This test mirrors that: compile untilize+argmax on input A (eager), then capture a trace on
input B allocated at a different address, replay with fresh data in B, and verify argmax.
"""

import os

import pytest
import torch
from loguru import logger
import ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "trace_region_size": 102000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [(8, 4)],
    indirect=True,
)
def test_bh_untilize_argmax_trace(mesh_device):
    torch.manual_seed(1234)
    width = 155648
    batch = 32
    chunk = width // 8

    sub_core_grids = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
            ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
        ]
    )

    def make_input(seed_offset):
        torch.manual_seed(1234 + seed_offset)
        x = torch.rand(1, 1, batch, width) * 4.0 - 2.0
        expected = []
        positions = [198, 257, 11, 279, chunk + 5, 2 * chunk + 100, 3 * chunk + 7, 5 * chunk + 3000]
        for r in range(batch):
            pos = (positions[r % len(positions)] + seed_offset * 31) % width
            x[0, 0, r, pos] = 30.0 + r
            expected.append(pos)
        return x, expected

    def upload(x):
        return ttnn.from_torch(
            x,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def run(inp):
        x_unt = ttnn.untilize(inp, use_multicore=True, sub_core_grids=sub_core_grids)
        tok = ttnn.argmax(x_unt, dim=-1, keepdim=False, sub_core_grids=sub_core_grids)
        return x_unt, tok

    def check(tok, expected, label):
        got = ttnn.to_torch(ttnn.get_device_tensors(tok)[0]).reshape(-1).tolist()
        errs = [(r, expected[r], got[r]) for r in range(batch) if got[r] != expected[r]]
        logger.info(f"[{label}] mismatches: {len(errs)}")
        for r, exp, g in errs[:6]:
            logger.warning(f"[{label}] row {r}: expected {exp} got {g} (delta {g - exp} = {(g - exp) / chunk} chk)")
        return not errs

    # ---- compile run (eager) on input A ----
    x_a, exp_a = make_input(0)
    tt_a = upload(x_a)
    unt_a, tok_a = run(tt_a)
    ok = check(tok_a, exp_a, "compile-eager")
    ttnn.deallocate(unt_a)
    ttnn.deallocate(tok_a)
    ttnn.deallocate(tt_a)
    assert ok, "eager compile run wrong — different bug"

    # Shift the allocator so input B lands at a different address than A.
    pad = ttnn.from_torch(
        torch.zeros(1, 1, 32, 4096),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # ---- trace capture on input B (program-cache hit, new addresses) ----
    x_b, exp_b = make_input(1)
    tt_b = upload(x_b)
    logger.info(f"input A vs B addresses differ; B addr {tt_b.buffer_address():#x}")

    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    unt_b, tok_b = run(tt_b)
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    host_b = ttnn.from_torch(
        x_b,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
    )

    n_replays = int(os.environ.get("QWEN_UNTILIZE_TEST_ITERS", "5"))
    all_pass = True
    for it in range(n_replays):
        x_i, exp_i = make_input(2 + it)
        host_i = ttnn.from_torch(
            x_i,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )
        ttnn.copy_host_to_device_tensor(host_i, tt_b)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        if not check(tok_b, exp_i, f"trace-replay-{it}"):
            all_pass = False
            # diagnose: readback untilize output
            u = ttnn.to_torch(ttnn.get_device_tensors(unt_b)[0]).float()[0, 0, 0, :]
            ref = torch.from_numpy(ttnn.to_torch(ttnn.get_device_tensors(tt_b)[0]).float()[0, 0, 0, :].numpy())
            max_diff = (u - ref).abs().max().item()
            logger.warning(f"replay {it}: max|untilized - input| = {max_diff}")

    ttnn.release_trace(mesh_device, trace_id)
    assert all_pass, "traced untilize+argmax returned wrong indices"
