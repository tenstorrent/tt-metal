# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Times each stage of DeviceSampler on one Galaxy row (TP=8): every stage is captured in its own trace and replayed
back to back. Also dumps the stages' outputs for a row with tied maxima. Run with pytest under the Galaxy lock."""

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


@pytest.mark.parametrize("device_params", [{**GALAXY_DEVICE_PARAMS, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_sampler_stages(mesh_device):
    plan = MeshPlan(dp_submeshes(mesh_device)[0])
    vl = VOCAB // plan.tp
    s = DeviceSampler(plan, vl, 4, B)
    torch.manual_seed(0)
    logits = torch.randn(1, 1, B, VOCAB) * 3
    logits[0, 0, 0, 70000] = logits[0, 0, 0, 3000] = logits[0, 0, 0].max() + 1.0
    logits_tt = ttnn.from_torch(
        logits,
        device=plan.mesh_device,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=plan.shard(3),
    )
    hosts, _ = s.host_params(
        torch.tensor([0.6, 1.0, 1.0, 1.0]), torch.tensor([-1, -1, -1, -1]), torch.tensor([0.95] * 4)
    )
    s.set_params(hosts)

    # Row 0 intermediates against the host, eagerly before any trace.
    s._scan(logits_tt)
    s._chip_merge()
    sums = s._normalizer(logits_tt)
    cands = plan.all_gather(s.chip_cands, dim=-1)
    sums_all = plan.all_gather(sums, dim=-1)
    s._merge(cands, sums_all, None)
    t = 0.6
    chips = logits[0, 0, 0].double().reshape(plan.tp, vl)
    dev_sums = ttnn.to_torch(ttnn.get_device_tensors(sums_all)[0]).float()  # [1, 1, 32, tp * 32]
    dev_max = [ttnn.to_torch(d).float()[0, 0, :4, 0].tolist() for d in ttnn.get_device_tensors(s.chip_max)]
    for ch in range(plan.tp):
        m = chips[ch].max()
        host = torch.exp((chips[ch] - m) / t).sum()
        logger.info(
            f"DUMP chip {ch}: max {float(m):.5f} dev max {dev_max[ch][0]:.5f} host S {float(host):.6f} device S {float(dev_sums[0, 0, 0, ch * 32]):.6f}"
        )
    tok = ttnn.to_torch(ttnn.get_device_tensors(s.next_tokens)[0])
    logger.info(f"DUMP tokens {tok.reshape(-1)[:4].tolist()}")
    host0 = logits[0, 0, :, :vl].double()
    m0 = host0.max(dim=-1, keepdim=True).values
    x = ttnn.subtract(logits_tt, s.chip_max)
    hx = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).double()[0, 0]
    logger.info(
        f"DUMP sub rows0-5 max err {[(hx[r] - (host0[r] - (m0[r] if r < 4 else 0))).abs().max().item() for r in range(6)]}"
    )
    inv = ttnn.to_torch(ttnn.get_device_tensors(s.inv_temp)[0]).double()[0, 0, :6, 0]
    logger.info(f"DUMP inv_temp {inv.tolist()}")
    y = ttnn.multiply(x, s.inv_temp)
    hy = ttnn.to_torch(ttnn.get_device_tensors(y)[0]).double()[0, 0]
    logger.info(f"DUMP mul row0 {hy[0, :4].tolist()} host {((host0[0, :4] - m0[0]) / 0.6).tolist()}")
    e = ttnn.exp(y)
    he = ttnn.to_torch(ttnn.get_device_tensors(e)[0]).double()[0, 0]
    logger.info(f"DUMP exp row0 sum {he[0].sum().item()} row1 {he[1].sum().item()} row5 {he[5].sum().item()}")
    ss = ttnn.sum(e, dim=-1, keepdim=True)
    hs = ttnn.to_torch(ttnn.get_device_tensors(ss)[0]).double()[0, 0, :6, 0]
    logger.info(f"DUMP sum {hs.tolist()}")

    logger.info(f"STAGE grid {plan.mesh_device.compute_with_storage_grid_size()}")
    timed(plan, "scan", lambda: s._scan(logits_tt))
    s64 = DeviceSampler(plan, vl, 4, B, scan_cores=64)
    timed(plan, "scan64", lambda: s64._scan(logits_tt))
    timed(plan, "chip_merge64", lambda: s64._chip_merge())
    timed(plan, "chip_merge", lambda: s._chip_merge())
    sums = timed(plan, "normalizer", lambda: s._normalizer(logits_tt))
    timed(plan, "norm_sub_mul", lambda: ttnn.multiply(ttnn.subtract(logits_tt, s.chip_max), s.inv_temp))
    e = ttnn.exp(logits_tt)
    timed(plan, "norm_exp", lambda: ttnn.exp(logits_tt))
    timed(plan, "norm_sum_w", lambda: ttnn.sum(e, dim=-1, keepdim=True))
    eb = ttnn.typecast(e, ttnn.bfloat16)
    timed(plan, "norm_sum_w_bf16", lambda: ttnn.sum(eb, dim=-1, keepdim=True))
    for name in ("moreh_sum",):
        f = getattr(ttnn, name, None)
        if f is not None:
            try:
                timed(plan, "norm_" + name, lambda: f(e, dim=3, keepdim=True))
            except Exception as ex:  # noqa: BLE001 - probe only
                logger.info(f"STAGE norm_{name}: unsupported ({str(ex)[:120]})")
    cands = timed(plan, "gather_cands", lambda: plan.all_gather(s.chip_cands, dim=-1))
    sums_all = timed(plan, "gather_sums", lambda: plan.all_gather(sums, dim=-1))
    timed(plan, "merge", lambda: s._merge(cands, sums_all, None))
    timed(plan, "full", lambda: s(logits_tt))
