# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Three Talker-prefill items the earlier sweeps left unmeasured.

Re-profiling HEAD (29/24/24 ops for buckets 32/64/128) leaves three layout/config gaps
that no previous sweep covered. Each is worth **-1 op per layer** or a fidelity-free
matmul/SDPA win, and each is a pure layout or program-config change:

**1. SiLU-mul -> sharded c32 output.** `prefill_trio_sweep` tried six arms but only ONE
with a sharded output, `in c64 -> sharded c64`, which needs gate/up on 64 cores and so
lost to gate/up's own 32-core optimum. The arm that fits what the model actually ships —
gate/up writing 32-core width-sharded (`_PREFILL_DOWN_SHARD_IN0`) — was never run. If
`in c32 -> sharded c32` is no slower than `in c32 -> interleaved`, then the SiLU-mul
already writes `down`'s in0 spec and the `InterleavedToSharded` after it (op 21: 4.4 us
at m=64, 7.5 us at m=128) disappears.

**2. o_proj with a width-sharded in0 at 16 cores.** `prefill_mm_sweep` swept sharded-in0
only for `cores in (32, 64)` (it required `k_tiles % cores == 0` for BOTH loops), but the
config `_PREFILL_WO` actually ships for m=64 is **c16** — so the sharded-in0 variant of the
winning arm was never measured. NLPConcatHeads writes 16-core width-sharded and o_proj
then takes interleaved, which is what op 13 (`ShardedToInterleaved`, 2.9/4.1 us) pays for.

**3. Masked-prefill SDPA k_chunk.** PERF_NOTES 6.5, never attempted: the masked prefill
config is `q_chunk=64 / k_chunk=64`, so kv=352 becomes 6 chunks over 384 padded rows where
352 is one exact chunk (11 tiles). The same reasoning already took decode SDPA 82 -> 38 us.

    python -m tracy -p -v -r --op-support-count 100000 \
        -m pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_prefill_gaps_sweep.py
    python models/demos/qwen3_tts/tests/prefill_gaps_report.py
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tt.dram_sharded_matmul import width_sharded_l1_memcfg
from models.demos.qwen3_tts.tt.linear_1d_program_config import make_linear_1d_program_config

TILE = 32
HIDDEN, INTER = 2048, 6144
HEADS, KV_HEADS, HEAD_DIM = 16, 8, 128
REPS = 4
MANIFEST = "generated/prefill_gaps_manifest.json"
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


def _kcfg(fidelity=ttnn.MathFidelity.LoFi):
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
    )


def _width_sharded(m, dim, cores, cg):
    """Width-shard `m x dim` over `cores`, laid out row-major on the compute grid."""
    cols = min(cg.x, cores)
    while cores % cols:
        cols -= 1
    rows = cores // cols
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (m, dim // cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


def _record(group, tag, extra, ref, got):
    """Score against the shipped arm: layout changes must be bit-exact, so report the
    max abs diff rather than a PCC that would hide a real reorder."""
    rec = {"group": group, "tag": tag, "reps": REPS}
    rec.update(extra)
    if ref is not None:
        rec["max_abs_diff"] = float((got.float() - ref.float()).abs().max())
    _ARMS.append(rec)
    d = f" maxdiff={rec['max_abs_diff']:.3e}" if ref is not None else " (reference)"
    print(f"  {tag:44s} -> ran{d}")


# ───────────────────── 1. SiLU-mul output grid at 32 cores ─────────────────────
@pytest.mark.parametrize("m", [64, 128], ids=["m64", "m128"])
def test_prefill_silu_mul_out_c32(device, m):
    cg = device.compute_with_storage_grid_size()
    torch.manual_seed(0)
    g = torch.randn(1, 1, m, INTER, dtype=torch.bfloat16)
    u = torch.randn(1, 1, m, INTER, dtype=torch.bfloat16)
    in_mc = _width_sharded(m, INTER, 32, cg)  # what gate/up ships today
    print(f"\n### SiLU-mul m={m} N={INTER}; shipped = in c32 -> L1-interleaved")

    ref = None
    for label, out_mc in [
        ("in c32 -> interleaved (shipped)", ttnn.L1_MEMORY_CONFIG),
        ("in c32 -> sharded c32", _width_sharded(m, INTER, 32, cg)),
    ]:
        gt = ttnn.from_torch(g, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in_mc)
        ut = ttnn.from_torch(u, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in_mc)
        try:
            o = ttnn.mul(gt, ut, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=out_mc)
            ttnn.synchronize_device(device)
            got = ttnn.to_torch(o)
            ttnn.deallocate(o)
        except Exception as e:
            print(f"  {label:44s} -> REFUSED {str(e).splitlines()[0][:70]}")
            ttnn.deallocate(gt), ttnn.deallocate(ut)
            continue
        for _ in range(REPS):
            o = ttnn.mul(gt, ut, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=out_mc)
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        _record("silumul", f"m{m} {label}", {"m": m}, ref, got)
        if ref is None:
            ref = got
        ttnn.deallocate(gt)
        ttnn.deallocate(ut)


# ───────────────────── 2. o_proj sharded-in0 at the shipped grid ─────────────────────
@pytest.mark.parametrize("m,cores", [(64, 16), (128, 32)], ids=["m64_c16", "m128_c32"])
def test_prefill_o_proj_sharded_in0(device, m, cores):
    cg = device.compute_with_storage_grid_size()
    gx, gy = cg.x, cg.y
    K = N = HIDDEN
    k_tiles = K // TILE
    torch.manual_seed(0)
    w = torch.randn(K, N, dtype=torch.bfloat16)
    act = torch.randn(1, 1, m, K, dtype=torch.bfloat16)
    w_tt = ttnn.from_torch(
        w.unsqueeze(0).unsqueeze(0).contiguous(),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pc = make_linear_1d_program_config(m, K, N, gx, gy, True, num_cores=cores)
    print(f"\n### o_proj m={m} K={K} N={N} at c{cores} (ibw={pc.in0_block_w}); k_tiles={k_tiles}")

    ref = None
    arms = [("interleaved in0 (shipped)", ttnn.L1_MEMORY_CONFIG)]
    if k_tiles % cores == 0:
        arms.append((f"sharded in0 c{cores}", width_sharded_l1_memcfg(m // TILE, k_tiles, gx, max(1, cores // gx))))
    else:
        print(f"  sharded in0 c{cores} -> SKIPPED: k_tiles={k_tiles} not divisible by {cores}")

    for label, in_mc in arms:
        try:
            a = ttnn.from_torch(act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in_mc)
        except Exception as e:
            print(f"  {label:44s} -> INPUT REFUSED {str(e).splitlines()[0][:70]}")
            continue
        try:
            o = ttnn.linear(
                a, w_tt, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_kcfg()
            )
            ttnn.synchronize_device(device)
            got = ttnn.to_torch(o)
            ttnn.deallocate(o)
        except Exception as e:
            msg = next((l.strip() for l in str(e).splitlines() if "TT_FATAL" in l or "must" in l), "")
            print(f"  {label:44s} -> REFUSED {msg[:70]}")
            ttnn.deallocate(a)
            continue
        for _ in range(REPS):
            o = ttnn.linear(
                a, w_tt, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=_kcfg()
            )
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        _record("oproj", f"m{m} c{cores} {label}", {"m": m, "cores": cores}, ref, got)
        if ref is None:
            ref = got
        ttnn.deallocate(a)
    ttnn.deallocate(w_tt)


# ───────────────────── 3. masked-prefill SDPA k_chunk (PERF_NOTES 6.5) ─────────────────────
@pytest.mark.parametrize("m,kv", [(64, 352), (128, 416)], ids=["m64_kv352", "m128_kv416"])
def test_prefill_sdpa_k_chunk(device, m, kv):
    """Sq = bucket, Sk = kv_max, explicit padding mask — the shipped masked-prefill call."""
    torch.manual_seed(0)
    q = ttnn.from_torch(
        torch.randn(1, HEADS, m, HEAD_DIM, dtype=torch.bfloat16),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    kt, vt = [
        ttnn.from_torch(
            torch.randn(1, KV_HEADS, kv, HEAD_DIM, dtype=torch.bfloat16),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    ]
    mask = torch.zeros(1, 1, m, kv, dtype=torch.bfloat16)
    mask[..., m:] = -1e4  # pad columns beyond the written cache
    mt = ttnn.from_torch(
        mask, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print(f"\n### masked prefill SDPA m={m} kv={kv}; shipped = q_chunk=64 k_chunk=64")

    ref = None
    for k_chunk in [64, 128, 192, kv // 2, kv]:
        if k_chunk % TILE or k_chunk > kv:
            print(f"  k_chunk={k_chunk:<4d} -> SKIPPED (not a tile multiple / > kv)")
            continue
        pc = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=64, k_chunk_size=k_chunk
        )
        label = f"q64 k{k_chunk}" + (" (shipped)" if k_chunk == 64 else "")

        def _call():
            return ttnn.transformer.scaled_dot_product_attention(
                q,
                kt,
                vt,
                attn_mask=mt,
                is_causal=False,
                scale=1.0,
                compute_kernel_config=_kcfg(ttnn.MathFidelity.HiFi4),
                program_config=pc,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )

        try:
            o = _call()
            ttnn.synchronize_device(device)
            got = ttnn.to_torch(o)
            ttnn.deallocate(o)
        except Exception as e:
            msg = next((l.strip() for l in str(e).splitlines() if "TT_FATAL" in l or "must" in l), "")
            print(f"  {label:44s} -> REFUSED {msg[:70]}")
            continue
        for _ in range(REPS):
            o = _call()
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        _record("sdpa", f"m{m} kv{kv} {label}", {"m": m, "kv": kv, "k_chunk": k_chunk}, ref, got)
        if ref is None:
            ref = got

    for t in (q, kt, vt, mt):
        ttnn.deallocate(t)
