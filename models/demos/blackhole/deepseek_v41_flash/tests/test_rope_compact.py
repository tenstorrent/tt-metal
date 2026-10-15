# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Compact RoPE tables (tt/rope_compact.py) against the full-width tables: every tensor of the step-state dict built by ``DSV41PagedStepState.build`` must be bit-identical
(window / ratio-2 / ratio-4 kinds, with the indexer rows, T = 4 and 16, random positions incl. 0 and the last), and the DRAM they take must shrink by ~10x.
Also prints RC lines: step-state build time (traced, 20 builds in one trace) and the free DRAM per bank after building the three kinds at 131072 and 1048576 positions."""

import os
import time
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.attention import DSV41Attention, rope_tables
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import DSV41PagedStepState

P = 4096


def stub_attn(md, T, ratio, max_pos):
    a = types.SimpleNamespace()
    a.mesh_device, a.T, a.rows, a.cols, a.ratio = md, T, 4, 8, ratio
    ang = torch.rand(max_pos, 32) * 6.0
    a.cos_tab, a.sin_tab = rope_tables(torch.polar(torch.ones(max_pos, 32), ang))
    a._rope_inputs = types.MethodType(DSV41Attention._rope_inputs, a)
    return a


def dev_pos(md, pos):
    return ttnn.from_torch(
        pos.to(torch.int32),
        device=md,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def host(t, dev=0):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[dev])


def traced_us(md, fn, rep=20, n=20):
    def body():
        for _ in range(rep):
            fn()

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    dt = (time.perf_counter() - t) / n / rep
    ttnn.release_trace(md, tid)
    return dt * 1e6


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 100_000_000}, id="plain")],
    indirect=True,
)
@torch.no_grad()
def test_rope_compact_bit_identical(mesh_device):
    md = mesh_device
    torch.manual_seed(0)
    for ratio in (0, 2, 4):
        for T in (4, 16):
            a = stub_attn(md, T, ratio, P)
            os.environ["DSV41_ROPE_COMPACT"] = "0"
            full = DSV41PagedStepState(a, max_pos=P, with_indexer=ratio > 0, per_user_valid=ratio > 0)
            os.environ["DSV41_ROPE_COMPACT"] = "1"
            comp = DSV41PagedStepState(a, max_pos=P, with_indexer=ratio > 0, per_user_valid=ratio > 0)
            assert full.rope is None and comp.rope is not None
            pos = torch.randint(0, P, (T,))
            pos[0], pos[-1] = 0, P - 1
            dp = dev_pos(md, pos)
            sf, sc = full.build(dp), comp.build(dp)
            assert sorted(sf) == sorted(sc), (sorted(sf), sorted(sc))
            n = 0
            for k in sf:
                if isinstance(sf[k], ttnn.Tensor):
                    for dev in (0, 17, 31):
                        x, y = host(sf[k], dev), host(sc[k], dev)
                        assert x.shape == y.shape and torch.equal(
                            x.view(torch.int16) if x.dtype == torch.bfloat16 else x,
                            y.view(torch.int16) if y.dtype == torch.bfloat16 else y,
                        ), f"ratio {ratio} T {T} key {k} dev {dev}"
                    n += 1
            print(f"RC bit-identical ratio {ratio} T {T}: {n} tensors x 3 devices OK", flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [pytest.param({"l1_small_size": 16384, "trace_region_size": 100_000_000}, id="plain")],
    indirect=True,
)
@torch.no_grad()
def test_rope_compact_cost(mesh_device):
    """build time of the step states (the three kinds) and the DRAM of the tables, compact vs full."""
    md = mesh_device
    T = 16
    kinds = [(0, False), (2, True), (4, True)]
    for mode in ("0", "1"):
        os.environ["DSV41_ROPE_COMPACT"] = mode
        sts = []
        for ratio, idx in kinds:
            a = stub_attn(md, T, ratio, 8192)
            sts.append(DSV41PagedStepState(a, max_pos=8192, with_indexer=idx, per_user_valid=idx))
        pos = dev_pos(md, torch.randint(0, 8192, (T,)))
        for s in sts:
            s.build(pos)  # eager pass creates the constants
        us = traced_us(md, lambda: [s.build(pos) for s in sts])
        print(f"RC compact={mode}: build of the 3 kinds = {us:.1f} us per step", flush=True)
    # DRAM: bytes per position
    for mode, per_kind in (("full", (2048, 2048 + 512, 2048 + 512)), ("compact", (256, 256, 256))):
        tot = sum(per_kind)
        print(
            f"RC {mode}: {tot} B/position/chip -> {tot * 131072 / 2**20:.0f} MiB at 131072, {tot * 1048576 / 2**30:.2f} GiB at 1M",
            flush=True,
        )
