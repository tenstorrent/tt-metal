# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DSV41_PF_POS_DEV on the device: the full-width RoPE tables / window mask derived in ``DynCtx.derive`` from the compact uploads must equal the host-built, uploaded tables of the default path bit for
bit on every device of the (4, 8) mesh, for ragged per-slot chunk starts (U = 2 slots per mesh row), idle-slot style repeated starts and start 0."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_prefill_pos_dev_cpu import FakeAttn
from models.demos.blackhole.deepseek_v41_flash.tt import prefill_dyn as PD


def bits(t):
    return ttnn.to_torch(t).to(torch.bfloat16).view(torch.int16) if t.dtype != ttnn.float32 else ttnn.to_torch(t)


def per_device(t):
    return [ttnn.to_torch(x).to(torch.float32).view(torch.int32) if x.dtype == ttnn.float32 else ttnn.to_torch(x).to(torch.bfloat16).view(torch.int16) for x in ttnn.get_device_tensors(t)]


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="plain")], indirect=True)
@torch.no_grad()
def test_pos_dev_tables(mesh_device):
    md = mesh_device
    C, S, U = 512, 4096, 2
    src = {False: FakeAttn(S, 1), True: FakeAttn(S, 2)}
    ctxs = {}
    for flag in ("0", "1"):
        os.environ["DSV41_PF_POS_DEV"] = flag
        ctxs[flag] = PD.DynCtx(md, C, S, {0, 1, 2}, src, per_row=True, U=U)
    os.environ.pop("DSV41_PF_POS_DEV")
    off, on = ctxs["0"], ctxs["1"]
    assert not off.pos_dev and on.pos_dev
    nslot = on.nslot
    for starts in (
        [0] * nslot,
        [512 * ((k * 3) % 8) for k in range(nslot)],
        [0, 512, 1024, 1536, 2048, 2560, 3072, 3584],
        [3584] * nslot,
        [0, 3584, 0, 512, 3584, 3584, 0, 1024],
    ):
        off.update(starts)
        on.update(starts)
        on.derive()
        for k in off.tabs:
            for i, (a, b) in enumerate(zip(off.tabs[k], on.tabs[k])):
                for d, (x, y) in enumerate(zip(per_device(a), per_device(b))):
                    if not torch.equal(x, y):
                        nz = (x != y).nonzero()
                        print(f"DIFF tab kind={k} idx={i} dev={d} shape={tuple(x.shape)} / {tuple(y.shape)} n={len(nz)} first={nz[:3].tolist()} off={x[tuple(nz[0])] if len(nz) else None} on={y[tuple(nz[0])] if len(nz) else None}", flush=True)
                    assert torch.equal(x, y), (k, starts)
        for r in off.lat_tabs:
            for a, b in zip(off.lat_tabs[r], on.lat_tabs[r]):
                for x, y in zip(per_device(a), per_device(b)):
                    assert torch.equal(x, y), (r, starts)
        for x, y in zip(per_device(off.win_mask), per_device(on.win_mask)):
            assert torch.equal(x, y), starts
        for r in off.lim:
            for a, b in ((off.lim[r], on.lim[r]), (off.off[r], on.off[r])):
                for x, y in zip(per_device(a), per_device(b)):
                    assert torch.equal(x, y)
        print("POS_DEV equal for starts", starts, flush=True)


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [pytest.param({"l1_small_size": 16384}, id="plain")], indirect=True)
@torch.no_grad()
def test_pos_dev_update_cost(mesh_device):
    """per-window host cost of ``DynCtx.update`` (host build + recorded device copies, as ``record_window``) flag off / on, 8 slots with ragged starts; prints POSDEV_BENCH lines (ms per window)."""
    import time

    from models.demos.blackhole.deepseek_v41_flash.tt.h2d import recording

    md = mesh_device
    C, S, U = 512, 1048576 + 512, 2
    src = {False: FakeAttn(S, 1), True: FakeAttn(S, 2)}
    res = {}
    for flag in ("0", "1"):
        os.environ["DSV41_PF_POS_DEV"] = flag
        ctx = PD.DynCtx(md, C, S, {0, 1, 2}, src, per_row=True, U=U)
        ts = []
        for it in range(12):
            starts = [512 * (it * 37 + k * 11) % 1048064 for k in range(ctx.nslot)]
            t0 = time.perf_counter()
            with recording() as ops:
                ctx.update(starts)
            ts.append(time.perf_counter() - t0)
        res[flag] = sum(ts[2:]) / len(ts[2:]) * 1e3
        nbytes = sum(h.volume() * 2 for h, _ in ops)  # (approx. bf16 element bytes of the recorded uploads)
        print(f"POSDEV_BENCH flag={flag} update {res[flag]:.2f} ms/window, {len(ops)} uploads, ~{nbytes / 2**20:.1f} MiB", flush=True)
    os.environ.pop("DSV41_PF_POS_DEV")
