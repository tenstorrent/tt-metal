# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Stress + determinism: flat_routed_expert on the whole LoudBox mesh (2x4, all 8 chips) for GLM_STRESS_ITERS mesh
calls (default 2,000,000), the model's call (C++ op, capacity 8192, indexed into T = 5120 gathered tokens,
clamped_silu, bfp4, row-major bf16 y, fp32 down). Each chip runs its own 36 of 288 experts.

Routing cycles over patterns, different on every chip: balanced 128 / 160 / 512, model-like skewed (Dirichlet 0.8 over
~5120 rows), spiky (empty experts, 1-31 row experts, a few at 1-2k), all tiny (1-3 rows, half empty), one expert at the
full capacity. Every output is compared on device, bit-exact over its active rows, against the first output of the
same pattern (a sticky marker, read every GLM_STRESS_CHECK calls): any difference is a non-determinism. Every 7th call
a matmul runs in between; every 5000 calls the L1 allocation shifts (the op's per-launch arena moves). Hangs: the
dispatch timeout of scripts/run_safe_pytest.sh.

  BRINGUP_SPEC=models/demos/glm53_flash_d_p_lb/bringup/spec.yaml TT_METAL_PINNED_MEMORY_CACHE_LIMIT_BYTES=0 \\
  scripts/run_safe_pytest.sh --run-all --no-precompile models/demos/glm53_flash_d_p_lb/tests/test_flat_stress.py -s"""

import os
import time

import pytest
import torch
from loguru import logger

from models.demos.common.bringup.testing.harness import mesh_parametrize

E, H, I, T, CAP = 36, 4096, 2048, 5120, 8192
ITERS = int(os.environ.get("GLM_STRESS_ITERS", "2000000"))
CHECK = int(os.environ.get("GLM_STRESS_CHECK", "5000"))


def _pattern(kind, g):
    if kind.startswith("bal"):
        return torch.full((E,), int(kind[3:]), dtype=torch.long)
    if kind == "skew":
        p = torch.distributions.Dirichlet(torch.ones(E) * 0.8).sample()
        c = torch.floor(p * 5120).long()
        c[torch.argmax(p)] += 5120 - int(c.sum())
        return c
    if kind == "spiky":
        c = torch.randint(1, 32, (E,), generator=g)
        c[torch.randperm(E, generator=g)[: E // 3]] = 0
        c[torch.randperm(E, generator=g)[:3]] = torch.randint(1024, 2049, (3,), generator=g)
        return c
    if kind == "tiny":
        c = torch.randint(1, 4, (E,), generator=g)
        c[torch.randperm(E, generator=g)[: E // 2]] = 0
        return c
    if kind == "cap":
        c = torch.zeros(E, dtype=torch.long)
        c[int(torch.randint(0, E, (1,), generator=g))] = CAP
        return c
    raise ValueError(kind)


PATTERNS = ("bal128", "bal160", "skew", "spiky", "tiny", "cap", "bal512")


@pytest.mark.timeout(24 * 3600)
@mesh_parametrize
def test_flat_stress(mesh_device):
    from ttnn.bringup.flat_routed_expert_ttnn.flat_expert import FlatRoutedExpert

    import ttnn

    n_dev = mesh_device.get_num_devices()
    ng = E * n_dev
    gids = [[d * E + e for e in range(E)] for d in range(n_dev)]
    torch.manual_seed(0)
    w = [(torch.randn(H, I) * 0.02, torch.randn(H, I) * 0.02, torch.randn(I, H) * 0.02) for _ in range(E)]
    op = FlatRoutedExpert(
        mesh_device, [w] * n_dev, m=CAP, H=H, I=I, gids=gids, n_global=ng, wdtype="bf4", act="clamped_silu", pin=1
    )
    del w
    shard = lambda ts, dt, layout=ttnn.ROW_MAJOR_LAYOUT: ttnn.from_torch(  # noqa: E731
        torch.stack(ts),
        dtype=dt,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    x = ttnn.reshape(shard([torch.randn(T, H).to(torch.bfloat16) * 0.3 for _ in range(n_dev)], ttnn.bfloat16), (T, H))

    g = torch.Generator().manual_seed(1)
    cases = []
    for kind in PATTERNS:
        per = [_pattern(kind, g) for _ in range(n_dev)]
        pads = [((c + 31) // 32) * 32 for c in per]
        rows = max(int(p.sum()) for p in pads)
        cnt, reg, tix, msk = [], [], [], []
        for d in range(n_dev):
            off = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), pads[d][:-1]]), 0)
            c_ = torch.zeros(1, ng, dtype=torch.int32)
            r_ = torch.zeros(1, ng, dtype=torch.int32)
            c_[0, gids[d]] = per[d].int()
            r_[0, gids[d]] = off.int()
            m_ = torch.zeros(rows, H)
            for e in range(E):
                m_[int(off[e]) : int(off[e]) + int(per[d][e])] = 1.0
            cnt.append(c_)
            reg.append(r_)
            tix.append(torch.randint(0, T, (1, rows), dtype=torch.int32, generator=g))
            msk.append(m_)
        case = dict(
            kind=kind,
            rows=rows,
            counts=ttnn.reshape(shard(cnt, ttnn.uint32), (1, ng)),
            regions=ttnn.reshape(shard(reg, ttnn.uint32), (1, ng)),
            tidx=ttnn.reshape(shard(tix, ttnn.uint32), (1, rows)),
            mask=ttnn.reshape(shard(msk, ttnn.bfloat16, ttnn.TILE_LAYOUT), (rows, H)),
        )
        y = op(x, case["counts"], case["regions"], token_index=case["tidx"], y_row_major=True, down_fp32=True)
        case["gold"] = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
        ttnn.deallocate(y)
        cases.append(case)
        logger.info(f"[stress] {kind}: rows {rows}, per-chip routed {[int(c.sum()) for c in per]}")
    ttnn.synchronize_device(mesh_device)
    # the checker must see a difference: the first pattern's gold against itself doubled (active rows are nonzero)
    c0 = cases[0]
    probe = ttnn.max(
        ttnn.multiply(ttnn.ne(c0["gold"], ttnn.multiply(c0["gold"], 2.0), dtype=ttnn.bfloat16), c0["mask"])
    )
    assert min(float(ttnn.to_torch(t).item()) for t in ttnn.get_device_tensors(probe)) == 1.0, "checker is blind"

    a = ttnn.from_torch(
        torch.randn(1, 1, 1024, 4096),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    b = ttnn.from_torch(
        torch.randn(1, 1, 4096, 2048),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    pinned = []
    marker = None
    t0 = time.time()
    for it in range(ITERS):
        c = cases[it % len(cases)]
        y = op(x, c["counts"], c["regions"], token_index=c["tidx"], y_row_major=True, down_fp32=True)
        ne = ttnn.ne(c["gold"], ttnn.to_layout(y, ttnn.TILE_LAYOUT), dtype=ttnn.bfloat16)
        mk = ttnn.max(ttnn.multiply(ne, c["mask"]))
        marker = mk if marker is None else ttnn.maximum(marker, mk)
        ttnn.deallocate(y)
        ttnn.deallocate(ne)
        if it % 7 == 6:
            ttnn.deallocate(ttnn.matmul(a, b))
        if it % 5000 == 4999:  # shift the L1 allocation: the next launches' arena sits elsewhere
            if len(pinned) == 8:
                for p in pinned:
                    ttnn.deallocate(p)
                pinned = []
            pinned.append(
                ttnn.from_torch(
                    torch.zeros(32 * 110 // 16, 32),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=mesh_device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                )
            )
        if (it + 1) % CHECK == 0 or it + 1 == ITERS:
            vals = [float(ttnn.to_torch(t).item()) for t in ttnn.get_device_tensors(marker)]
            dt = time.time() - t0
            logger.info(
                f"[stress] {it + 1:,} / {ITERS:,} calls, {(it + 1) / dt:.0f} calls/s, {dt / 60:.1f} min, "
                f"mismatch marker per chip {vals}"
            )
            assert max(vals) == 0.0, f"non-deterministic output by call {it + 1}: per-chip marker {vals}"
            ttnn.deallocate(marker)
            marker = None
    logger.info(f"[stress] PASS: {ITERS:,} mesh calls ({ITERS * n_dev:,} chip calls), bit-exact, no hang")
