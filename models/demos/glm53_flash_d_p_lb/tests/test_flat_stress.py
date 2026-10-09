# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Determinism + stress: flat_routed_expert on the whole LoudBox mesh (2x4, all 8 chips), GLM_STRESS_ITERS mesh calls
(default 2,000,000), the model's call (C++ op, capacity 8192, indexed into T = 5120 tokens, clamped_silu, bfp4, row-major
bf16 y, fp32 down). Each chip runs its own 36 of 288 experts.

Accuracy regimes (x / h tile formats): bfp8/bfp8, bfp8 x + bf16 h, bf16 x + bfp8 h, bf16/bf16; one op per regime, built
up front; the regime switches every GLM_STRESS_BLOCK calls (default 10,000), cycling. Within a regime the calls cycle
over routing patterns, different on every chip: balanced 128 / 160 / 512, model-like skewed, spiky (empty, 1-31 row and
1-2k row experts), all tiny, one expert at the full capacity.

Everything stays on device (as tests/nightly/blackhole/sdpa/test_ring_joint_sdpa.py): the first output of each
(regime, pattern) is that pair's reference; every later output of the pair is compared bit-exactly with it over the
routed rows (ne * routed-row mask -> max) into a per-chip sticky marker (each chip reduces only its own data, nothing
crosses devices). The 8 markers are read back once per block (before each regime switch): any nonzero is a
non-determinism, reported with its regime. Every 7th call a matmul runs in between; every 5000 calls the L1
allocation shifts (the op's per-launch arena moves). Hangs: the dispatch timeout of scripts/run_safe_pytest.sh.

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
BLOCK = int(os.environ.get("GLM_STRESS_BLOCK", "10000"))
REGIMES = ((False, False), (False, True), (True, False), (True, True))  # (x_bf16, h_bf16)
PATTERNS = ("bal128", "bal160", "skew", "spiky", "tiny", "cap", "bal512")


def _rname(r):
    return f"x {'bf16' if r[0] else 'bfp8'} / h {'bf16' if r[1] else 'bfp8'}"


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
    ops = {}
    for r in REGIMES:
        ops[r] = FlatRoutedExpert(
            mesh_device,
            [w] * n_dev,
            m=CAP,
            H=H,
            I=I,
            gids=gids,
            n_global=ng,
            wdtype="bf4",
            act="clamped_silu",
            pin=1,
            x_bf16=r[0],
            h_bf16=r[1],
        )
        logger.info(f"[stress] built {_rname(r)}")
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
        cases.append(
            dict(
                kind=kind,
                counts=ttnn.reshape(shard(cnt, ttnn.uint32), (1, ng)),
                regions=ttnn.reshape(shard(reg, ttnn.uint32), (1, ng)),
                tidx=ttnn.reshape(shard(tix, ttnn.uint32), (1, rows)),
                mask=ttnn.reshape(shard(msk, ttnn.bfloat16, ttnn.TILE_LAYOUT), (rows, H)),
            )
        )
        logger.info(f"[stress] pattern {kind}: rows {rows}, per-chip routed {[int(c.sum()) for c in per]}")

    def call(r, c):
        return ops[r](x, c["counts"], c["regions"], token_index=c["tidx"], y_row_major=True, down_fp32=True)

    # the checker must see a difference: an output against itself doubled (its routed rows are nonzero)
    c0 = cases[0]
    y0 = ttnn.to_layout(call(REGIMES[0], c0), ttnn.TILE_LAYOUT)
    probe = ttnn.max(ttnn.multiply(ttnn.ne(y0, ttnn.multiply(y0, 2.0), dtype=ttnn.bfloat16), c0["mask"]))
    assert min(float(ttnn.to_torch(t).item()) for t in ttnn.get_device_tensors(probe)) == 1.0, "checker is blind"
    ttnn.deallocate(y0)

    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    a = ttnn.from_torch(
        torch.randn(1, 1, 1024, 4096), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    b = ttnn.from_torch(
        torch.randn(1, 1, 4096, 2048), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=rep
    )
    gold = {}  # (regime, pattern) -> the pair's first output (TILE), kept on device
    pinned, marker, n_checked = [], None, 0
    t0 = time.time()
    for it in range(ITERS):
        r = REGIMES[(it // BLOCK) % len(REGIMES)]
        ci = it % len(cases)
        c = cases[ci]
        y = ttnn.to_layout(call(r, c), ttnn.TILE_LAYOUT)
        if (r, ci) not in gold:
            gold[(r, ci)] = y
        else:
            ne = ttnn.ne(gold[(r, ci)], y, dtype=ttnn.bfloat16)
            mk = ttnn.max(ttnn.multiply(ne, c["mask"]))
            marker = mk if marker is None else ttnn.maximum(marker, mk)
            ttnn.deallocate(ne)
            ttnn.deallocate(y)
            n_checked += 1
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
                    mesh_mapper=rep,
                )
            )
        if (it + 1) % BLOCK == 0 or it + 1 == ITERS:
            vals = (
                [float(ttnn.to_torch(t).item()) for t in ttnn.get_device_tensors(marker)] if marker is not None else []
            )
            dt = time.time() - t0
            logger.info(
                f"[stress] {it + 1:,} / {ITERS:,} calls (block {_rname(r)}), {n_checked:,} compared, "
                f"{(it + 1) / dt:.0f} calls/s, {dt / 60:.1f} min, mismatch marker per chip {vals}"
            )
            assert not vals or max(vals) == 0.0, f"non-deterministic output in regime {_rname(r)}: per chip {vals}"
            if marker is not None:
                ttnn.deallocate(marker)
            marker = None
    logger.info(
        f"[stress] PASS: {ITERS:,} mesh calls ({ITERS * n_dev:,} chip calls), {n_checked:,} compared bit-exact over "
        f"{len(REGIMES)} regimes x {len(PATTERNS)} patterns, no hang"
    )
