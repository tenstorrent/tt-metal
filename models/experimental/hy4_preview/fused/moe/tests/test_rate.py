# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DRAM read rate of one chip's routed experts while computing (fused/moe/program.py): the program captured as a
trace and replayed 20 times back to back with one synchronize, for unions of 1 to 64 experts (TP) and 1 to 6 whole
experts (EP), with the reader placement varied. Weights are random (the time does not depend on them).

Per case the log has one HY4_MK5 line: ms per replay, bytes of expert weights read, GB/s of the whole program, and
the marginal GB/s between the smallest and the largest union of the sweep (which leaves out the program's fixed
cost: launch, the x multicast, the tail).

Sweeps (SWEEPS; HY4_RATE_CASES selects by name, comma separated) vary the union, the cores per bank, the banks, the
placement mode and the streamers' ring, and run the variants that show what bounds the rate:
    noread  the streamers issue no DRAM reads (compute on whatever the ring holds)
    nomath  the matmuls are skipped (reads, the hand-overs and the packs remain)
    timing  every core reports its cycles from the release to its first hand-over and to its end
Board only."""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.fused.moe import layout as L
from models.experimental.hy4_preview.fused.moe.program import MOCK, RoutedExperts

REPLAYS = 20
TRACE_REGION = 64 << 20
PEAK_GBS, TTNN_GBS = 512.0, 200.0

# name: (share, experts in DRAM, unions, RoutedExperts kwargs, variants)
SWEEPS = {
    "tp32": ("tp32", 64, [1, 8, 16, 32, 64], {}, ()),
    "tp32_g8d4_first": ("tp32", 64, [1, 8, 16, 32, 64], {"gate_up_per_bank": 8, "down_per_bank": 4}, ()),
    # the finishers: with neither reads nor matmuls the first sweep took as long, so the hand-over bounds it
    "tp32_f12": ("tp32", 64, [1, 8, 64], {"finishers": 12}, ()),
    "tp32_f16": ("tp32", 64, [1, 8, 16, 32, 64], {"finishers": 16}, ()),
    "tp32_f22": ("tp32", 64, [1, 8, 64], {"finishers": 22}, ()),
    "tp32_f16_noread_nomath": ("tp32", 64, [8, 64], {"finishers": 16}, ("noread", "nomath")),
    "tp32_f16_nomath": ("tp32", 64, [8, 64], {"finishers": 16}, ("nomath",)),
    "tp32_f16_noread": ("tp32", 64, [8, 64], {"finishers": 16}, ("noread",)),
    "tp32_f16_timing": ("tp32", 64, [8, 64], {"finishers": 16}, ("timing",)),
    "tp8_f16": ("tp8", 64, [1, 8, 16, 32, 64], {"finishers": 16}, ()),
    "ep_f16": ("ep", 8, [1, 2, 3, 4, 5, 6], {"max_slots": 8, "finishers": 16}, ()),
    "tp32_launch": ("tp32", 64, [1], {"finishers": 16}, ("launch",)),
    "tp32_launch_sem5": ("tp32", 64, [1], {"finishers": 16, "dummy_semaphores": 5}, ("launch",)),
    "tp32_sem5": ("tp32", 64, [1, 64], {"finishers": 16, "dummy_semaphores": 5}, ()),
    # launch cost against the shape of the core sets: every role as whole rows, whatever the banks
    # bank-local and rectangular (program.py place_rect): 6 gate|up and 4 down cores a bank
    "tp32_launch_rect": ("tp32", 64, [1], {"mode": "rect", "down_per_bank": 4}, ("launch",)),
    "tp32_rect": ("tp32", 64, [1, 8, 16, 32, 64], {"mode": "rect", "down_per_bank": 4}, ()),
    "tp32_rect_f9": ("tp32", 64, [8, 32, 64], {"mode": "rect", "down_per_bank": 4, "finishers": 9}, ()),
    "tp8_rect": ("tp8", 64, [1, 8, 64], {"mode": "rect", "down_per_bank": 4}, ()),
    "ep_rect": ("ep", 8, [1, 6], {"max_slots": 8, "mode": "rect", "down_per_bank": 4}, ()),
    "tp32_launch_block": ("tp32", 64, [1], {"mode": "block"}, ("launch",)),
    "tp32_block": ("tp32", 64, [1, 8, 64], {"mode": "block"}, ()),
    "tp8": ("tp8", 64, [1, 8, 16, 32, 64], {}, ()),
    "ep": ("ep", 8, [1, 2, 3, 4, 5, 6], {"max_slots": 8}, ()),
    # what bounds it: no DRAM reads, no matmuls, per-core times
    "tp32_noread": ("tp32", 64, [8, 64], {}, ("noread",)),
    "tp32_nomath": ("tp32", 64, [8, 64], {}, ("nomath",)),
    "tp32_noread_nomath": ("tp32", 64, [8, 64], {}, ("noread", "nomath")),
    "tp32_timing": ("tp32", 64, [8, 64], {}, ("timing",)),
    "tp8_noread": ("tp8", 64, [8, 64], {}, ("noread",)),
    "tp8_nomath": ("tp8", 64, [8, 64], {}, ("nomath",)),
    "tp8_timing": ("tp8", 64, [8, 64], {}, ("timing",)),
    "ep_noread": ("ep", 8, [1, 6], {"max_slots": 8}, ("noread",)),
    "ep_nomath": ("ep", 8, [1, 6], {"max_slots": 8}, ("nomath",)),
    "ep_timing": ("ep", 8, [1, 6], {"max_slots": 8}, ("timing",)),
    # placement: cores per bank, which banks, how the cores are chosen, the streamers' ring
    "tp32_g4d6": ("tp32", 64, [8, 64], {"gate_up_per_bank": 4}, ()),
    "tp32_g5d6": ("tp32", 64, [8, 64], {"gate_up_per_bank": 5}, ()),
    "tp32_g6d3": ("tp32", 64, [8, 64], {"down_per_bank": 3}, ()),
    "tp32_g8d3": ("tp32", 64, [8, 64], {"gate_up_per_bank": 8, "down_per_bank": 3}, ()),
    "tp32_g9d3": ("tp32", 64, [8, 64], {"gate_up_per_bank": 9, "down_per_bank": 3}, ()),
    "tp32_g8d4": ("tp32", 64, [8, 64], {"gate_up_per_bank": 8, "down_per_bank": 4}, ()),
    "tp32_g8d4_timing": ("tp32", 64, [64], {"gate_up_per_bank": 8, "down_per_bank": 4}, ("timing",)),
    "tp8_g8d4": ("tp8", 64, [8, 64], {"gate_up_per_bank": 8, "down_per_bank": 4}, ()),
    "ep_g8d4": ("ep", 8, [1, 6], {"max_slots": 8, "gate_up_per_bank": 8, "down_per_bank": 4}, ()),
    "tp32_row": ("tp32", 64, [8, 64], {"mode": "row"}, ()),
    "tp32_any": ("tp32", 64, [8, 64], {"mode": "any"}, ()),
    "tp32_west": ("tp32", 64, [8, 64], {"banks": [0, 1, 2, 3]}, ()),
    "tp32_east": ("tp32", 64, [8, 64], {"banks": [4, 5, 6, 7]}, ()),
    "tp32_bank0": ("tp32", 64, [8, 64], {"banks": [0]}, ()),
    "tp32_bank5": ("tp32", 64, [8, 64], {"banks": [5]}, ()),
    "tp32_bank5_nomath": ("tp32", 64, [8, 64], {"banks": [5]}, ("nomath",)),
    "tp32_ring42": ("tp32", 64, [8, 64], {"gate_up_ring": (4, 2)}, ()),
    "tp32_ring63": ("tp32", 64, [8, 64], {"gate_up_ring": (6, 3)}, ()),
    "tp8_g8d3": ("tp8", 64, [8, 64], {"gate_up_per_bank": 8, "down_per_bank": 3}, ()),
}
SHARES = {"tp32": L.tp_share(32), "tp8": L.tp_share(8), "ep": L.ep_share()}
VARIANTS = {
    "noread": ("HY4_NO_READ", "1"),
    "nomath": ("HY4_NO_MATH", "1"),
    "timing": ("HY4_TIMING", "1"),
    "launch": ("HY4_LAUNCH_ONLY", "1"),  # every kernel returns at once: what launching the program costs
}
CYCLE_NS = 1e3 / 1350  # the wall clock counts AICLK cycles (1350 MHz)

_selected = [s for s in os.environ.get("HY4_RATE_CASES", ",".join(SWEEPS)).split(",") if s]


def _streams(device, share, experts):
    g = torch.Generator().manual_seed(0)
    I = share.channels
    gate = torch.randn(experts, I, L.HIDDEN, generator=g) * 0.006
    up = torch.randn(experts, I, L.HIDDEN, generator=g) * 0.006
    down = torch.randn(experts, L.HIDDEN, I, generator=g) * 0.006
    return (
        L.to_device_stream(device, L.gate_up_stream(gate, up, share)),
        L.to_device_stream(device, L.down_stream(down, share)),
    )


def replay_ms(device, fn, replays=REPLAYS):
    """ms per replay of fn() captured as a trace: `replays` replays back to back, one synchronize."""
    fn()  # compile, and the program cache
    ttnn.synchronize_device(device)
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    fn()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    ttnn.synchronize_device(device)
    start = time.perf_counter()
    for _ in range(replays):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - start) * 1e3 / replays
    ttnn.release_trace(device, trace_id)
    return ms


@pytest.mark.parametrize("device_params", [{"trace_region_size": TRACE_REGION}], indirect=True)
@pytest.mark.parametrize("sweep", _selected)
def test_rate(device, sweep):
    if MOCK:
        pytest.skip("timing needs the board")
    share_name, experts, unions, kwargs, _variant = SWEEPS[sweep]
    share = SHARES[share_name]
    defines = [VARIANTS[v] for v in _variant]
    op = RoutedExperts(device, share, defines=defines, **kwargs)
    w_gu, w_dn = _streams(device, share, experts)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(L.ROWS, L.HIDDEN, generator=g)
    tag = sweep
    logger.info(f"HY4_MK5 rate {tag} placement {op.placement.describe()} hub {op.hub} finishers {op.fin}")
    points = []
    for U in unions:
        ids = torch.randperm(experts, generator=g)[:U].tolist()
        w = torch.full((L.ROWS, U), 2.827 / 8)
        op.set_inputs(x, ids, w)
        ms = replay_ms(device, lambda: op(w_gu, w_dn))
        _, status = op.read_output()
        nbytes = op.bytes_read(U)
        gbs = nbytes / (ms * 1e-3) / 1e9
        points.append((U, ms, nbytes))
        logger.info(
            f"HY4_MK5 rate {tag} U={U}: {ms * 1e3:.1f} us per replay, {nbytes / 1e6:.2f} MB read, "
            f"{gbs:.1f} GB/s whole program, status {[hex(v) for v in status[1:6]]}"
        )
        if "launch" not in _variant:
            assert status[1:6] == [0, 0, 0, 0, 0], f"a core timed out: {[hex(v) for v in status[:6]]}"
        if "timing" in _variant:
            t = op.read_timing()
            us = lambda cycles: cycles * CYCLE_NS / 1e3
            for role in ("gate_up", "down"):
                per_bank = {b: f"{us(max(c[3] for c in cores)):.0f}" for b, cores in t[role].items()}
                first = {b: f"{us(min(c[2] for c in cores)):.0f}" for b, cores in t[role].items()}
                logger.info(f"HY4_MK5 timing {tag} U={U} {role}: us to the end, slowest core per bank {per_bank}")
                logger.info(f"HY4_MK5 timing {tag} U={U} {role}: us to the first hand-over, fastest per bank {first}")
            logger.info(
                f"HY4_MK5 timing {tag} U={U} finishers: us to first / end "
                f"{[(round(us(c[2])), round(us(c[3]))) for c in t['finish']]}; hub cycles {t['hub']}"
            )
            worst = max(
                (c[3], role, b, c[0], c[1]) for role in ("gate_up", "down") for b in t[role] for c in t[role][b]
            )
            logger.info(
                f"HY4_MK5 timing {tag} U={U}: last core {worst[1]} bank {worst[2]} at {worst[3:]}: {us(worst[0]):.0f} us"
            )
    (u0, ms0, b0), (u1, ms1, b1) = points[0], points[-1]
    if ms1 > ms0:
        marginal = (b1 - b0) / ((ms1 - ms0) * 1e-3) / 1e9
        logger.info(
            f"HY4_MK5 rate {tag} marginal U={u0}->{u1}: {marginal:.1f} GB/s "
            f"({(ms1 - ms0) * 1e3 / (u1 - u0):.2f} us per extra expert; chip peak {PEAK_GBS:.0f}, "
            f"ttnn matmul {TTNN_GBS:.0f} GB/s)"
        )
