# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The layer program's two cross-chip steps with no compute between them (fused/layer_common/transport.py): the
broadcast X1' of [8, 6144] BF16 from the root chip into every chip's hub buffer, and the reduce X6 of every chip's
partial output to the root, landing as F on the cores of the root's iHC site.

Rungs, smallest first (fused/README.md; each alone and with -x on its first run):

    TT_METAL_MOCK_CLUSTER_DESC_PATH=<blackhole_galaxy.yaml> pytest <this file> -k "galaxy and exact"       (the mock runs whole-mesh programs only)
    TT_METAL_OPERATION_TIMEOUT_SECONDS=15 pytest <this file> -x -k "solo and exact"
    ... -k "pair_row and exact", "pair_col and exact", "row and exact", "col and exact", "galaxy and exact"
    pytest <this file> -x -k latency

exact: three rounds on fresh data; x bit-equal on every chip, F the exact sum (the partials are small integers, so
every Float32 add and both roundings to BFloat16 are exact), every word at rest afterwards.
latency: the program as a trace, 20 replays and one synchronize, 1 and 8 calls per trace; stage times from the
cores' stamps (kernels built with timing). Log lines start with HY4_MK8."""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.common import tracing
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.tt import prefill_bridge as pb

MESH = (8, 4)
TRACE_REGION = 32 << 20
PAYLOAD = 98304  # [8, 6144] BF16
# the cores of the MLP layer program (layer_mlp/PLAN.md section 2)
HUB, SENDER = (5, 9), (6, 9)
REDUCE = [(5, 6), (5, 7), (5, 8), (6, 8)]
ROOT_REDUCE = [(4, 6), (4, 7), (4, 8), (4, 9)]
SITE_ORIGIN, SITE_COLS, SITE_ROWS = (5, 0), 2, 8  # the root's iHC site cores: F lands there
CYCLE_NS = 1e3 / 1350  # the cores' wall clock counts AICLK cycles (1350 MHz), as fused/moe/tests/test_rate.py

galaxy = pytest.mark.parametrize(
    "device_params, mesh_device",
    [(dict(pb.device_params(), trace_region_size=TRACE_REGION), MESH)],
    indirect=True,
    ids=["8x4"],
)

RUNGS = {
    "solo": [(0, 0)],
    "pair_row": [(0, 0), (0, 1)],
    "pair_col": [(0, 0), (1, 0)],
    "row": [(0, c) for c in range(4)],
    "col": [(r, 0) for r in range(8)],
    "galaxy_small": None,  # one packet per multicast branch: 4096 B on 2 reduce cores
    "galaxy": None,
}
SMALL = {"galaxy_small": (4096, 2)}  # rung -> (payload bytes, reduce cores)


def site_cores():
    return [(SITE_ORIGIN[0] + x, SITE_ORIGIN[1] + y) for y in range(SITE_ROWS) for x in range(SITE_COLS)]


def site_set():
    x, y = SITE_ORIGIN
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x + SITE_COLS - 1, y + SITE_ROWS - 1))]
    )


def bf16_host(data):
    return ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)


def write_per_chip(mesh_device, tensor, data):
    """data [chips, rows, 256] -> the hub buffer ``tensor`` of every chip."""
    shards = [bf16_host(data[i]) for i in range(data.shape[0])]
    ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, mesh_device.shape), tensor)


def read_chip(tensor, chip):
    return ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip]).float()


class Setup:
    def __init__(self, mesh_device, rung, timing=False, wait_limit=30_000_000):
        self.mesh = mesh_device
        self.topo = tp.Topology(tuple(mesh_device.shape), (0, 0), RUNGS[rung])
        self.payload, parts = SMALL.get(rung, (PAYLOAD, len(REDUCE)))
        self.t = tp.Transport(
            mesh_device,
            self.topo,
            HUB,
            SENDER,
            REDUCE[:parts],
            self.payload,
            root_reduce_cores=ROOT_REDUCE[:parts],
            timing=timing,
            wait_limit=wait_limit,
        )
        self.source = tp.core_buffer(mesh_device, [HUB], self.payload)
        self.partial = tp.core_buffer(mesh_device, [HUB], self.payload)
        sites = site_cores()
        per = len(sites) // parts
        self.f = tp.core_buffer(
            mesh_device, site_set(), self.payload // len(sites), width=self.payload // len(sites) // 2
        )
        self.targets = [sites[k * per : (k + 1) * per] for k in range(parts)]
        self.io, self.desc = self.t.xfer_program(self.source, self.partial, self.targets, tp.address(self.f))
        self.chips = mesh_device.get_num_devices()

    def set(self, seed):
        g = torch.Generator().manual_seed(seed)
        x = (torch.randn(self.payload // 512, 256, generator=g) * 3).to(torch.bfloat16).float()
        partial = torch.randint(-3, 4, (self.chips, self.payload // 512, 256), generator=g).float()
        write_per_chip(self.mesh, self.source, x[None].repeat(self.chips, 1, 1))
        write_per_chip(self.mesh, self.partial, partial)
        ttnn.synchronize_device(self.mesh)
        return x, partial

    def run(self):
        if tp.is_mock():
            ttnn.experimental.prepare_generic_op(self.io, self.desc)
        else:
            ttnn.generic_op(self.io, self.desc)

    def read_f(self):
        return read_chip(self.f, self.topo.chip(self.topo.root)).reshape(-1, 256)

    def check(self, x, partial, label):
        status = self.t.status()
        assert not status, f"{label}: words not at rest: {status}"
        wrong_x = [
            c for c in self.topo.chips if not torch.equal(read_chip(self.t.x, self.topo.chip(c)).reshape(-1, 256), x)
        ]
        want = sum(partial[self.topo.chip(c)] for c in self.topo.chips)
        got = self.read_f()
        diff = int((got != want).sum())
        logger.info(
            f"HY4_MK8 xfer {label}: {len(self.topo.chips)} chips, x bit-equal on {len(self.topo.chips) - len(wrong_x)} of "
            f"them, F elements wrong {diff} of {want.numel()} (max abs err {float((got - want).abs().max()):.3g})"
        )
        assert not wrong_x, f"{label}: x differs on {wrong_x}"
        assert diff == 0, f"{label}: F is not the sum of the partials"


@galaxy
@pytest.mark.parametrize("rung", list(RUNGS), ids=list(RUNGS))
def test_exact(mesh_device, rung):
    if tp.is_mock() and RUNGS[rung] is not None and not os.environ.get("HY4_XFER_MOCK_ALL"):
        pytest.skip("the mock runs whole-mesh programs only")
    s = Setup(mesh_device, rung)
    logger.info(
        f"HY4_MK8 xfer {rung}: L1 per core for words, staging and landings {s.t.l1_bytes()} B, chunk {s.t.chunk}, "
        f"routes of the broadcast {len(s.topo.x_routes())}, allocated per core "
        f"{ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).total_bytes_allocated_per_bank}"
    )
    if tp.is_mock():
        s.run()
        return
    for round_ in range(3):
        x, partial = s.set(100 * round_ + len(rung))
        s.run()
        ttnn.synchronize_device(mesh_device)
        try:
            s.check(x, partial, f"{rung} round {round_}")
        except AssertionError:
            logger.info(f"HY4_MK8 xfer {rung} round {round_} FAILED: {s.t.status()}")
            raise


def us(ticks):
    return (ticks & 0xFFFFFFFF) * CYCLE_NS / 1e3


@galaxy
@pytest.mark.parametrize("rung", ["pair_row", "row", "col", "galaxy"])
def test_latency(mesh_device, rung):
    if tp.is_mock():  # the build with stamps compiles; nothing is timed
        Setup(mesh_device, rung, timing=True).run()
        return
    out = {}
    for timing in (False, True):
        s = Setup(mesh_device, rung, timing=timing)
        x, partial = s.set(7)
        s.run()
        ttnn.synchronize_device(mesh_device)
        s.check(x, partial, f"{rung} before the trace")
        for calls in (1, 8):
            r = tracing.traced_ms(mesh_device, s.run, calls=calls)
            out[(timing, calls)] = r["replay_ms"] * 1e3
            logger.info(
                f"HY4_MK8 xfer latency {rung} timing={timing} {calls} call(s) per trace: {r['replay_ms'] * 1e3:.1f} us per "
                f"call (rounds {[round(v * 1e3, 1) for v in r['rounds_ms']]})"
            )
        s.check(x, partial, f"{rung} after the traces")
        if timing:
            # stamps of the last replay; the root's are on one clock (its hub and its first reduce core start within
            # the same launch). A chip that is not the root starts its next replay before the root does, so its own
            # "x arrived" includes waiting for the root: not shown.
            hub = s.t.stamps(lambda c: HUB)[s.topo.root]
            red = s.t.stamps(lambda c: s.t.reduce_cores(c)[0] if s.topo.is_head(c) else None)[s.topo.root]
            x_sent, s1, s2, f = us(hub[1] - hub[0]), us(red[1] - red[0]), us(red[2] - red[0]), us(red[3] - red[0])
            logger.info(
                f"HY4_MK8 xfer stages {rung} (us from the root's kernel start, {CYCLE_NS:.3f} ns a tick): X1' sent and "
                f"the root's own copy counted {x_sent:.1f}; X6 stage 1 landed on the root {s1:.1f}, stage 2 landed "
                f"{s2:.1f}, F written {f:.1f}  =>  X1' {x_sent:.1f} us (send side), X6 {f - x_sent:.1f} us (from the "
                f"root's x to F, the other chips' x arrival included), both {f:.1f} us; traced "
                f"{out[(False, 8)]:.1f} us per call without stamps"
            )
        for t in (s.source, s.partial, s.f, s.t.x):
            ttnn.deallocate(t)
        s.t.release()
