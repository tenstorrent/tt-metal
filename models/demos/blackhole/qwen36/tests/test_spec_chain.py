# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device: the chain program of a chained MTP verify (spec_chain.ChainInputs.body, the kernel
kernels/chain_inputs.cpp).

* test_the_chain_trace_writes_the_host_block_and_ctrl_pages: eager and replayed from a trace on fresh inputs, the
  kernel writes the verify step buffer and the GDN ctrl pages the host stages for the same block
  (test_spec_chain_host._chain_case: random predecessors and chained verifies, held rows included, every other one
  with rows that continue another predecessor row), and with a device sampler continues drawn users from their draws
  and writes the sampler seeds of each user's count; and the replay time.
* test_the_chain_kernel_writes_the_op_graph_words_and_replays_faster: the kernel and the op graph it replaced
  (tests/spec_op_graphs.py OpGraphChain) on the same inputs, every count of the predecessor, rows that stay and rows
  that move: every word of the step buffer, the ctrl pages and the seeds is equal (outputs filled with a marker
  before each run), eager and replayed from traces; the kernel's trace replays faster.

    MESH_DEVICE=P150x8 pytest models/demos/blackhole/qwen36/tests/test_spec_chain.py
"""

import random
import time
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_accept_host import ROWS
from models.demos.blackhole.qwen36.tests.test_spec_chain_host import _chain_case, _ctrl_words
from models.demos.blackhole.qwen36.tests.test_spec_step import _capture, _host, _parametrize, _rep, _replay_us
from models.demos.blackhole.qwen36.tt.spec_chain import ChainInputs
from models.demos.blackhole.qwen36.tt.spec_step import drafter_pack_rows, pack_width


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("B", [1, 8])
def test_the_chain_trace_writes_the_host_block_and_ctrl_pages(mesh_device, B, sampled):
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
        if sampler is not None:
            copy_in(case.draws[..., :R], sampler.draws, ttnn.uint32)
        chain.stage(case.overlay, case.cand, case.seed_cand)

    def check(case):
        assert torch.equal(_host(step_dev).to(torch.int64), case.step.to(torch.int64)), "step buffer"
        for k, (dev, want) in enumerate(zip(ctrl_devs, case.ctrls)):
            assert torch.equal(_host(dev).to(torch.int64), want.to(torch.int64)), f"ctrl page {k}"
        if sampler is not None:
            # Any change of a seed would change a draw.
            assert torch.equal(_host(sampler.seeds).reshape(-1)[:R].to(torch.int64), case.seeds), "sampler seeds"

    step_dev = _rep(mesh_device, torch.zeros(9, pack_width(R), dtype=torch.int32), ttnn.uint32)
    pack_dev = _rep(mesh_device, torch.zeros(drafter_pack_rows(K), pack_width(R), dtype=torch.int32), ttnn.uint32)
    ctrl_devs = [_rep(mesh_device, torch.zeros(1, words, dtype=torch.int32), ttnn.int32) for _ in range(groups)]
    ids_dev = _rep(mesh_device, torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
    draft_devs = [_rep(mesh_device, torch.zeros(1, 1, B, 1, dtype=torch.int32), ttnn.uint32) for _ in range(K)]
    sampler = None
    if sampled:
        # The buffers of spec_sample.VerifySampler the chain program reads and writes.
        sampler = SimpleNamespace(
            draws=_rep(mesh_device, torch.zeros(1, 1, 1, R, dtype=torch.int32), ttnn.uint32),
            seeds=_rep(mesh_device, torch.zeros(R, dtype=torch.int32), ttnn.uint32),
        )
    chain = ChainInputs(mesh_device, B, K, ROWS, step_dev, pack_dev, ctrl_devs, nv, sampler=sampler)
    chain.bind(ids_dev, draft_devs)
    case = _chain_case(rng, B, K, groups, nv, sampled)
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
    for i in range(20):
        case = _chain_case(rng, B, K, groups, nv, sampled, moved=i % 2 == 1)
        stage(case)
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
        check(case)
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    for _ in range(50):
        ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    logger.info(
        f"[chain trace] B={B} K={K} sampler={sampled}: {(time.perf_counter() - t0) / 50 * 1e6:.1f} us per replay"
    )
    ttnn.release_trace(mesh_device, trace)
    chain.release()
    for t in (
        step_dev,
        pack_dev,
        ids_dev,
        *ctrl_devs,
        *draft_devs,
        *([] if sampler is None else vars(sampler).values()),
    ):
        ttnn.deallocate(t)


@torch.no_grad()
@_parametrize
@pytest.mark.parametrize("sampled", [False, True], ids=["greedy", "sampled"])
@pytest.mark.parametrize("B", [1, 8])
def test_the_chain_kernel_writes_the_op_graph_words_and_replays_faster(mesh_device, B, sampled):
    from models.demos.blackhole.qwen36.tests.spec_op_graphs import OpGraphChain

    K, groups, nv = 3, 3, 6
    T, R, W = K + 1, B * (K + 1), pack_width(B * (K + 1))
    words = _ctrl_words(B, nv)
    rng = random.Random(100 + B + sampled)
    cases, seen = [], set()
    while len(cases) < 22 or seen != set(range(T)):
        cases.append(_chain_case(rng, B, K, groups, nv, sampled, moved=len(cases) % 2 == 1))
        seen |= set(cases[-1].m)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)

    def copy_in(host, dev, dtype):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.as_tensor(host).to(torch.int32), dtype=dtype, mesh_mapper=rep), dev
        )

    def stage(case):
        copy_in(case.ids, ids_dev, ttnn.uint32)
        copy_in(case.pack, pack_dev, ttnn.uint32)
        for d, dev in zip(case.drafts, draft_devs):
            copy_in(d.reshape(1, 1, B, 1), dev, ttnn.uint32)
        if sampled:
            copy_in(case.draws[..., :R], draws_dev, ttnn.uint32)
        chain.stage(case.overlay, case.cand, case.seed_cand)
        ref.stage(case.cand, case.seed_cand)

    def run(outs, go):
        """Fill ``outs`` (step, ctrl pages, seeds) with markers, run ``go``, return what it wrote."""
        step, ctrls, seeds = outs
        copy_in(torch.full((9, W), 0x5EEDBEEF), step, ttnn.uint32)
        for c in ctrls:
            copy_in(torch.full((1, words), -7), c, ttnn.int32)
        if seeds is not None:
            copy_in(torch.full((R,), 0x5EEDBEEF), seeds, ttnn.uint32)
        go()
        got = [_host(step).to(torch.int64)] + [_host(c).to(torch.int64) for c in ctrls]
        return got + ([] if seeds is None else [_host(seeds).reshape(-1)[:R].to(torch.int64)])

    def check(case, got, want):
        assert all(torch.equal(g, w) for g, w in zip(got, want)) and len(got) == len(want), "kernel == op graph"
        host = [case.step.to(torch.int64)] + [c.to(torch.int64) for c in case.ctrls]
        host += [] if case.seeds is None else [case.seeds]
        assert all(torch.equal(g, h) for g, h in zip(got, host)), "kernel == host block"

    def outputs():
        step = _rep(mesh_device, torch.zeros(9, W, dtype=torch.int32), ttnn.uint32)
        ctrls = [_rep(mesh_device, torch.zeros(1, words, dtype=torch.int32), ttnn.int32) for _ in range(groups)]
        seeds = _rep(mesh_device, torch.zeros(R, dtype=torch.int32), ttnn.uint32) if sampled else None
        return step, ctrls, seeds

    outs, ref_outs = outputs(), outputs()
    pack_dev = _rep(mesh_device, torch.zeros(drafter_pack_rows(K), W, dtype=torch.int32), ttnn.uint32)
    ids_dev = _rep(mesh_device, torch.zeros(1, 1, 1, 32, dtype=torch.int32), ttnn.uint32)
    draft_devs = [_rep(mesh_device, torch.zeros(1, 1, B, 1, dtype=torch.int32), ttnn.uint32) for _ in range(K)]
    draws_dev = _rep(mesh_device, torch.zeros(1, 1, 1, R, dtype=torch.int32), ttnn.uint32) if sampled else None
    sampler = SimpleNamespace(draws=draws_dev, seeds=outs[2]) if sampled else None
    chain = ChainInputs(mesh_device, B, K, ROWS, outs[0], pack_dev, outs[1], nv, sampler=sampler)
    chain.bind(ids_dev, draft_devs)
    ref = OpGraphChain(chain, ROWS, step=ref_outs[0], ctrls=ref_outs[1], seeds=ref_outs[2])
    stage(cases[0])
    check(cases[0], run(outs, chain.body), run(ref_outs, ref.body))
    traces = [_capture(mesh_device, chain.body), _capture(mesh_device, ref.body)]
    replay = lambda trace: lambda: ttnn.execute_trace(mesh_device, trace, cq_id=0, blocking=True)
    for case in cases[1:]:
        stage(case)
        check(case, run(outs, replay(traces[0])), run(ref_outs, replay(traces[1])))
    kernel_us, graph_us = (_replay_us(mesh_device, t) for t in traces)
    logger.info(
        f"[chain trace] B={B} K={K} sampler={sampled}: kernel {kernel_us:.1f} us, op graph {graph_us:.1f} us per replay"
    )
    assert kernel_us < graph_us
    for t in traces:
        ttnn.release_trace(mesh_device, t)
    chain.release()
    ref.release()
    for t in (
        pack_dev,
        ids_dev,
        *draft_devs,
        draws_dev,
        outs[0],
        *outs[1],
        outs[2],
        ref_outs[0],
        *ref_outs[1],
        ref_outs[2],
    ):
        if t is not None:
            ttnn.deallocate(t)
