# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DeviceSampler stages on one Galaxy row (TP=8): checks each chip's candidates and normalizer against the host, then
times each stage, captured in its own trace and replayed back to back. Run with pytest under the Galaxy lock."""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.mimo_v2_6_flash.tests.common import GALAXY_DEVICE_PARAMS, dp_submeshes
from models.experimental.mimo_v2_6_flash.tt.common import MeshPlan
from models.experimental.mimo_v2_6_flash.tt.device_sampler import DeviceSampler

VOCAB = 152576
B = 32
USERS = 4
REPLAYS = 30


def timed(plan, name, fn):
    fn()  # compile
    ttnn.synchronize_device(plan.mesh_device)
    trace = ttnn.begin_trace_capture(plan.mesh_device, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(plan.mesh_device, trace, cq_id=0)
    ttnn.synchronize_device(plan.mesh_device)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(plan.mesh_device, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(plan.mesh_device)
    ms = 1e3 * (time.perf_counter() - t0) / REPLAYS
    logger.info(f"STAGE {name}: {ms:.3f} ms")
    ttnn.release_trace(plan.mesh_device, trace)
    return out


def raw_words(t, chip=0):
    """A one-tile uint32 TILE tensor's page as stored (faces in order), int64 [1024]."""
    x = ttnn.to_torch(ttnn.get_device_tensors(t)[chip]).to(torch.int64).reshape(32, 32) & 0xFFFFFFFF
    return torch.cat([x[r : r + 16, c : c + 16].reshape(-1) for r in (0, 16) for c in (0, 16)])


@pytest.mark.parametrize("device_params", [{**GALAXY_DEVICE_PARAMS, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_sampler_stages(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    vl = VOCAB // plan.tp
    s = DeviceSampler(plan, vl, USERS, B)
    torch.manual_seed(0)
    logits = torch.randn(1, 1, B, VOCAB) * torch.tensor([3.0, 6.0, 1.0, 2.0] + [1.0] * (B - USERS)).reshape(1, 1, -1, 1)
    logits[0, 0, 0, 70000] = logits[0, 0, 0, 3000] = logits[0, 0, 0].max() + 1.0
    logits_tt = ttnn.from_torch(
        logits,
        device=plan.mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=plan.shard(3),
    )
    temps = torch.tensor([0.6, 1.0, 0.8, 1.5])
    host, _ = s.host_params(temps, torch.tensor([-1, -1, -1, -1]), torch.tensor([0.95] * USERS))
    s.set_params(host)

    # Each chip's sorted list and normalizer against the host, eagerly before any trace.
    s._scan(logits_tt)
    s._chip_merge()
    for ch in range(plan.tp):
        words = raw_words(s.chip_cands, ch)
        for u in range(USERS):
            row = logits[0, 0, u, ch * vl : (ch + 1) * vl].double()
            vals, idx = torch.sort(row, descending=True, stable=True)
            got = words[u * 64 : u * 64 + 64].reshape(32, 2)
            assert torch.equal(
                got[:, 1], idx[:32]
            ), f"chip {ch} user {u}: ids {got[:8, 1].tolist()} != {idx[:8].tolist()}"
            host_s = float(torch.exp((row - vals[0]) / float(temps[u])).sum())
            dev_s = int(words[512 + u * 4] + (words[512 + u * 4 + 1] << 32)) / 2**30
            assert abs(dev_s - host_s) <= 2e-6 * host_s, f"chip {ch} user {u}: normalizer {dev_s} != {host_s}"
            if ch < 2:
                logger.info(f"DUMP chip {ch} user {u}: normalizer device {dev_s:.7f} host {host_s:.7f}")

    timed(plan, "scan", lambda: s._scan(logits_tt))
    timed(plan, "chip_merge", lambda: s._chip_merge())
    cands = timed(plan, "gather_cands", lambda: plan.all_gather(s.chip_cands, dim=-1))
    timed(plan, "merge", lambda: s._merge(cands, None))
    timed(plan, "full", lambda: s(logits_tt))
    s64 = DeviceSampler(plan, vl, USERS, B, scan_cores=64)
    s64.set_params(host)
    timed(plan, "scan64", lambda: s64._scan(logits_tt))
    timed(plan, "chip_merge64", lambda: s64._chip_merge())
    timed(plan, "full64", lambda: s64(logits_tt))
