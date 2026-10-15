# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Column-sharded compact RoPE tables (DSV41_ROPE_SHARD=1, tt/rope_compact.py) against the full-width and the replicated compact tables: every tensor of the step-state dict built by
``DSV41PagedStepState.build`` must be bit-identical (window / ratio-2 / ratio-4 kinds, T = 4 and 16, positions incl. 0, the last and the column block boundaries). Prints RS lines:
the step-state build time (traced) of the three variants and the table bytes per position per chip."""

import os

import pytest
import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tests.test_rope_compact import dev_pos, host, stub_attn, traced_us
from models.demos.blackhole.deepseek_v41_flash.tt.paged_attention import DSV41PagedStepState
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

P = 4096


def attn_with_ccl(md, T, ratio, max_pos, cfg, ccl):
    a = stub_attn(md, T, ratio, max_pos)
    a.mesh_config, a.ccl = cfg, ccl
    return a


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_rope_shard_bit_identical_and_cost(mesh_device):
    md = mesh_device
    cfg, ccl = mesh_4x8(), CCLManager(md, num_links=2, topology=ttnn.Topology.Ring)
    torch.manual_seed(0)
    Pc = P // 8
    edge = [0, P - 1, Pc - 1, Pc, 7 * Pc - 1, 7 * Pc]
    times = {}
    for ratio in (0, 2, 4):
        for T in (4, 16):
            a = attn_with_ccl(md, T, ratio, P, cfg, ccl)
            kw = dict(max_pos=P, with_indexer=ratio > 0, per_user_valid=ratio > 0)
            os.environ["DSV41_ROPE_COMPACT"], os.environ["DSV41_ROPE_SHARD"] = "0", "0"
            full = DSV41PagedStepState(a, **kw)
            os.environ["DSV41_ROPE_COMPACT"], os.environ["DSV41_ROPE_SHARD"] = "1", "0"
            rep = DSV41PagedStepState(a, **kw)
            os.environ["DSV41_ROPE_COMPACT"], os.environ["DSV41_ROPE_SHARD"] = "1", "1"
            shd = DSV41PagedStepState(a, **kw)
            assert full.rope is None and rep.rope is not None and not rep.rope.sharded and shd.rope.sharded
            pos = torch.randint(0, P, (T,))
            pos[: min(T, len(edge))] = torch.tensor(edge[:T])
            dp = dev_pos(md, pos)
            sf, sr, ss = full.build(dp), rep.build(dp), shd.build(dp)
            ttnn.synchronize_device(md)
            assert sorted(sf) == sorted(sr) == sorted(ss)
            n = 0
            for k in sf:
                if isinstance(sf[k], ttnn.Tensor):
                    for dev in (0, 9, 17, 31):
                        x = host(sf[k], dev)
                        for name, st in (("replicated", sr), ("sharded", ss)):
                            y = host(st[k], dev)
                            vx, vy = (x.view(torch.int16), y.view(torch.int16)) if x.dtype == torch.bfloat16 else (x, y)
                            assert x.shape == y.shape and torch.equal(
                                vx, vy
                            ), f"{name} ratio {ratio} T {T} key {k} dev {dev}"
                    n += 1
            print(f"RS bit-identical ratio {ratio} T {T}: {n} tensors x 4 devices, replicated + sharded OK", flush=True)
            if T == 16:
                times[(ratio, "full")] = (full, dp)
                times[(ratio, "replicated")] = (rep, dp)
                times[(ratio, "sharded")] = (shd, dp)
    for variant in ("full", "replicated", "sharded"):
        sts = [times[(r, variant)][0] for r in (0, 2, 4)]
        dp = times[(0, variant)][1]
        us = traced_us(md, lambda: [s.build(dp) for s in sts])
        print(f"RS build of the 3 kinds ({variant}): {us:.1f} us per step", flush=True)
    for name, b in (("full", 7168), ("replicated compact", 768), ("sharded compact", 96)):
        print(
            f"RS {name}: {b} B/position/chip -> {b * 1048576 / 2**30:.3f} GiB at 1M, {b * 524288 / 2**30:.3f} GiB at 524288",
            flush=True,
        )
