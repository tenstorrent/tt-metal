# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the chain program of a chained MTP verify (spec_chain.ChainInputs.body), eager and replayed from a trace on
fresh inputs, writes the verify step buffer and the GDN ctrl pages the host stages for the same block
(test_spec_chain_host._chain_case: random predecessors and chained verifies, held rows included); and the replay time.

    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_spec_chain.py
"""

import random
import time

import pytest
import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_accept_host import ROWS
from models.demos.blackhole.qwen36.tests.test_spec_chain_host import _chain_case, _ctrl_words
from models.demos.blackhole.qwen36.tests.test_spec_step import _host, _parametrize, _rep
from models.demos.blackhole.qwen36.tt.spec_chain import ChainInputs
from models.demos.blackhole.qwen36.tt.spec_step import drafter_pack_rows, pack_width


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("B", [1, 8])
def test_the_chain_trace_writes_the_host_block_and_ctrl_pages(mesh_device, B):
    K, groups, nv = 3, 3, 6
    T, R = K + 1, B * (K + 1)
    words = _ctrl_words(B, nv)
    rng = random.Random(B)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def copy_in(host, dev, dtype):
        h = ttnn.from_torch(torch.as_tensor(host).to(torch.int32), dtype=dtype, mesh_mapper=rep)
        ttnn.copy_host_to_device_tensor(h, dev)

    def stage(case):
        copy_in(case.ids, ids_dev, ttnn.uint32)
        copy_in(case.pack, pack_dev, ttnn.uint32)
        for d, dev in zip(case.drafts, draft_devs):
            copy_in(d.reshape(1, 1, B, 1), dev, ttnn.uint32)
        chain.stage(case.overlay, case.cand)

    def check(case):
        assert torch.equal(_host(step_dev).to(torch.int64), case.step.to(torch.int64)), "step buffer"
        for k, (dev, want) in enumerate(zip(ctrl_devs, case.ctrls)):
            assert torch.equal(_host(dev).to(torch.int64), want.to(torch.int64)), f"ctrl page {k}"

    step_dev = _rep(mesh_device, torch.zeros(9, pack_width(R), dtype=torch.int32), ttnn.uint32)
    pack_dev = _rep(mesh_device, torch.zeros(drafter_pack_rows(K), pack_width(R), dtype=torch.int32), ttnn.uint32)
    ctrl_devs = [_rep(mesh_device, torch.zeros(1, words, dtype=torch.int32), ttnn.int32) for _ in range(groups)]
    ids_dev = _rep(mesh_device, torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
    draft_devs = [_rep(mesh_device, torch.zeros(1, 1, B, 1, dtype=torch.int32), ttnn.uint32) for _ in range(K)]
    chain = ChainInputs(mesh_device, B, K, ROWS, step_dev, pack_dev, ctrl_devs, nv)
    chain.bind(ids_dev, draft_devs)
    case = _chain_case(rng, B, K, groups, nv)
    stage(case)
    chain.body()
    check(case)
    mesh_device.set_program_cache_misses_allowed(False)
    try:
        with trace_allocation_tracker.corruptible_allocation_scope(mesh_device):
            trace = ttnn.begin_trace_capture(mesh_device, cq_id=0)
            chain.body()
            ttnn.end_trace_capture(mesh_device, trace, cq_id=0)
    finally:
        mesh_device.set_program_cache_misses_allowed(True)
    for _ in range(20):
        case = _chain_case(rng, B, K, groups, nv)
        stage(case)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        check(case)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(50):
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[chain trace] B={B} K={K}: {(time.perf_counter() - t0) / 50 * 1e6:.1f} us per replay")
    ttnn.release_trace(mesh_device, trace)
    chain.release()
    for t in (step_dev, pack_dev, ids_dev, *ctrl_devs, *draft_devs):
        ttnn.deallocate(t)
