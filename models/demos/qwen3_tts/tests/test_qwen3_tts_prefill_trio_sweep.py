# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The three prefill items left after the matmul work: RMSNorm, SiLU-mul, gate/up at m=128.

Each test emits exactly ONE op code per launch (LayerNorm / BinaryNg / Matmul), so the
three sequences can be aligned in the CSV independently and positional drift between
arms of different op counts is impossible. Score with se_-style grouping:

    python -m tracy -p -v -r --op-support-count 100000 \
        -m pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_prefill_trio_sweep.py
    python models/demos/qwen3_tts/tests/prefill_trio_report.py

**1. RMSNorm.** `decoder_layer._build_sharded_rmsnorm_configs` derives
``block_w = (dim/num_cores)/TILE`` and ``subblock_w = largest divisor of block_w <= 4``,
and `ln_num_cores` is picked as the LARGEST core count dividing dim_tiles
(`decoder_layer.py:148`) — 64 for hidden=2048. That drives block_w to 1 and subblock_w to
1: the same "most cores, thinnest block" shape that cost the matmuls 34 us each. Fewer
cores means a fatter block. Prefill runs four of these per layer for 43 us.

**2. SiLU-mul.** 20 us at m=64 against an 8.2 us roofline (2.36 MB of traffic). Writing
the output sharded was already measured and rejected (19.7 -> 32.6 us). Untried: the
INPUT core count. gate/up now hands it a 32-core width shard, and the op reports 64
cores, so input and output granularity disagree.

**3. gate/up at m=128.** 111 us at 39 % of peak, the worst matmul efficiency left. On the
1D path in0 is sharded along K across the cores, so 32 cores caps ``in0_block_w`` at
64/32 = 2 — and fewer cores measured worse (c16 136 us, c8 221 us). The 2D path splits K
along ``grid_y`` instead, which reaches in0_block_w = 8 while still using 32 cores. One
2D arm was tried before (8x4, auto subblock) at 143 us; this sweeps the subblock and
in0_block_w space the helper never explored, plus the 1D subblocks.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import ttnn

TILE = 32
HIDDEN, INTER = 2048, 6144
REPS = 4
MANIFEST = "generated/prefill_trio_manifest.json"
_ARMS: list = []


@pytest.fixture(scope="module")
def device():
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1), l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    os.makedirs("generated", exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump({"reps": REPS, "arms": _ARMS}, f, indent=2)
    print(f"\nwrote {MANIFEST}: {len(_ARMS)} arms x {REPS} reps")


def _kcfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _width_sharded(m, dim, cores, cg):
    cols = min(cg.x, cores)
    while cores % cols:
        cols -= 1
    rows = cores // cols
    assert rows <= cg.y
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (m, dim // cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _record(group, label, extra, ref, got):
    """Norms/muls are not bit-exact across block sizes (reduction order changes), so score
    relative RMS against the shipped arm rather than demanding equality."""
    rel = None
    if ref is not None:
        a, b = got.to(torch.float32), ref.to(torch.float32)
        rel = float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp(min=1e-12))
    rec = {"group": group, "label": label, "reps": REPS, "rel_rms": rel}
    rec.update(extra)
    _ARMS.append(rec)
    print(f"  {label:44s} -> ran" + (f"  rel_rms {rel*100:.4f} %" if rel is not None else "  (reference)"))


# ───────────────────────── 1. prefill RMSNorm ─────────────────────────
@pytest.mark.parametrize("m", [64, 128], ids=["m64", "m128"])
def test_prefill_rmsnorm_cores(device, m):
    from models.demos.qwen3_tts.tt.decoder_layer import _build_sharded_rmsnorm_configs

    cg = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    x = torch.randn(1, 1, m, HIDDEN, dtype=torch.bfloat16)
    w = ttnn.from_torch(
        torch.randn(1, 1, HIDDEN // TILE, TILE, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    kc = _kcfg()
    dim_tiles = HIDDEN // TILE
    print(f"\n### RMSNorm m={m} dim={HIDDEN} (dim_tiles={dim_tiles}); shipped = 64 cores, block_w=1")
    ref = None
    for cores in [c for c in (64, 32, 16, 8, 4) if dim_tiles % c == 0]:
        try:
            in_mc, pc = _build_sharded_rmsnorm_configs(device, HIDDEN, cores, m=m)
        except Exception as e:
            print(f"  cores={cores:<3d} -> CONFIG REFUSED {str(e).splitlines()[0][:70]}")
            continue
        xt = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in_mc)
        label = f"sharded c{cores} block_w={pc.block_w} subblock_w={pc.subblock_w}"
        try:
            o = ttnn.rms_norm(
                xt, epsilon=1e-6, weight=w, compute_kernel_config=kc, memory_config=in_mc, program_config=pc
            )
            ttnn.synchronize_device(device)
            got = ttnn.to_torch(o).reshape(1, 1, m, HIDDEN)
            ttnn.deallocate(o)
        except Exception as e:
            why = next((l.strip() for l in str(e).splitlines() if "TT_FATAL" in l), str(e)[:90])
            print(f"  {label:44s} -> REFUSED {why[:90]}")
            ttnn.deallocate(xt)
            continue
        if ref is None:
            ref = got.clone()
        for _ in range(REPS):
            o = ttnn.rms_norm(
                xt, epsilon=1e-6, weight=w, compute_kernel_config=kc, memory_config=in_mc, program_config=pc
            )
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        _record("rmsnorm", f"m{m} {label}", {"m": m, "cores": cores}, ref if cores != 64 else None, got)
        ttnn.deallocate(xt)
    ttnn.deallocate(w)


# ───────────────────────── 2. prefill SiLU-mul ─────────────────────────
@pytest.mark.parametrize("m", [64, 128], ids=["m64", "m128"])
def test_prefill_silu_mul_cores(device, m):
    cg = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    g = torch.randn(1, 1, m, INTER, dtype=torch.bfloat16)
    u = torch.randn(1, 1, m, INTER, dtype=torch.bfloat16)
    print(f"\n### SiLU-mul m={m} N={INTER}; shipped = inputs sharded c32, output L1-interleaved")
    ref = None
    arms = [
        ("in c32 -> interleaved", 32, None),
        ("in c64 -> interleaved", 64, None),
        ("in c16 -> interleaved", 16, None),
        ("in c64 -> sharded c64", 64, 64),
        ("in c48 -> interleaved", 48, None),
        ("interleaved -> interleaved", None, None),
    ]
    for label, in_cores, out_cores in arms:
        try:
            in_mc = _width_sharded(m, INTER, in_cores, cg) if in_cores else ttnn.L1_MEMORY_CONFIG
            out_mc = _width_sharded(m, INTER, out_cores, cg) if out_cores else ttnn.L1_MEMORY_CONFIG
            gt = ttnn.from_torch(g, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in_mc)
            ut = ttnn.from_torch(u, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in_mc)
        except Exception as e:
            print(f"  {label:44s} -> ALLOC REFUSED {str(e).splitlines()[0][:70]}")
            continue
        try:
            o = ttnn.mul(gt, ut, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=out_mc)
            ttnn.synchronize_device(device)
            got = ttnn.to_torch(o).reshape(1, 1, m, INTER)
            ttnn.deallocate(o)
        except Exception as e:
            why = next((l.strip() for l in str(e).splitlines() if "TT_FATAL" in l), str(e)[:90])
            print(f"  {label:44s} -> REFUSED {why[:90]}")
            ttnn.deallocate(gt)
            ttnn.deallocate(ut)
            continue
        if ref is None:
            ref = got.clone()
        for _ in range(REPS):
            o = ttnn.mul(gt, ut, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=out_mc)
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        _record("silumul", f"m{m} {label}", {"m": m}, None if label.startswith("in c32") else ref, got)
        ttnn.deallocate(gt)
        ttnn.deallocate(ut)


# ───────────────────────── 3. gate/up at m=128 ─────────────────────────
def test_prefill_gate_up_m128_subblocks(device):
    from models.demos.qwen3_tts.tt.dram_sharded_matmul import width_sharded_l1_memcfg

    m, K, N = 128, HIDDEN, INTER
    cg = device.compute_with_storage_grid_size()
    kc = _kcfg()
    m_t, k_t, n_t = m // TILE, K // TILE, N // TILE
    torch.manual_seed(0)
    w = ttnn.from_torch(
        torch.randn(1, 1, K, N, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    act = torch.randn(1, 1, m, K, dtype=torch.bfloat16)
    a_il = ttnn.from_torch(
        act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    print(f"\n### gate/up m={m} K={K} N={N} (m_t={m_t} k_t={k_t} n_t={n_t}); shipped = 1D c32 ibw=2 sb=(1,3)")
    ref = None

    def _run(label, pc, a, extra):
        nonlocal ref
        try:
            o = ttnn.linear(a, w, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=kc)
            ttnn.synchronize_device(device)
            got = ttnn.to_torch(o).reshape(1, 1, m, N)
            ttnn.deallocate(o)
        except Exception as e:
            why = next((l.strip() for l in str(e).splitlines() if "TT_FATAL" in l or "must" in l), str(e)[:90])
            print(f"  {label:44s} -> REFUSED {why[:90]}")
            return
        if ref is None:
            ref = got.clone()
        for _ in range(REPS):
            o = ttnn.linear(a, w, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=kc)
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        _record("gateup128", label, extra, None if extra.get("shipped") else ref, got)

    # 1D subblock search at 32 cores (in0 sharded, as the model now does)
    a_sh = ttnn.to_memory_config(a_il, width_sharded_l1_memcfg(m_t, k_t, 8, 4))
    for sh, sw in [(1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (4, 1)]:
        if sh * sw > 4 or (n_t // 32) % sw or m_t % sh:
            continue
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            in0_block_w=k_t // 32,
            out_subblock_h=sh,
            out_subblock_w=sw,
            per_core_M=m_t,
            per_core_N=n_t // 32,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        _run(f"1D c32 ibw=2 sb=({sh},{sw})", pc, a_sh, {"shipped": (sh, sw) == (1, 3)})

    # 2D search: grid_y splits K, so in0_block_w can reach 8 at 32 cores
    for gx, gy in [(8, 4), (8, 2), (8, 1), (6, 4), (4, 4), (8, 4)]:
        if n_t % gx or m_t % gy or k_t % gy:
            continue
        pcm, pcn = m_t // gy, n_t // gx
        for ibw in [b for b in (1, 2, 4, 8) if (k_t // gy) % b == 0]:
            for sh, sw in [(1, 1), (1, 2), (1, 4), (2, 2), (4, 1)]:
                if sh * sw > 4 or pcn % sw or pcm % sh:
                    continue
                pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(gx, gy),
                    in0_block_w=ibw,
                    out_subblock_h=sh,
                    out_subblock_w=sw,
                    per_core_M=pcm,
                    per_core_N=pcn,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=True,
                )
                _run(f"2D {gx}x{gy} ibw={ibw} sb=({sh},{sw})", pc, a_il, {})
    ttnn.deallocate(a_sh)
    ttnn.deallocate(a_il)
    ttnn.deallocate(w)
