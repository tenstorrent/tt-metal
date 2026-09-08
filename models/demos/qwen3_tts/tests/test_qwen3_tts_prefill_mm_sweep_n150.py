# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sweep the N150 Talker PREFILL matmul program configs at bfp8, buckets 64 and 128.

The demo pads 61 tokens to bucket 64, so this is the graph behind its 21.0 ms prefill.
The window (607 us/layer at bucket 64) is 65 % matmul, and `mlp.py`'s N150 full-grid
choice for gate/up was swept at **bf16** ("210 GB/s, 73 % of DRAM peak") — the same
stale-config situation that cost 34 us/matmul in decode. At bfp8 the block report shows
prefill gate/up at 121 GB/s (42 %) and o_proj at 91 GB/s (31 %).

Prefill M is 2 or 4 tiles, so the DRAM-sharded kernel is unavailable (it needs M=1 tile);
the levers are num_cores, in0_block_w and out_subblock on the 1D-mcast config, plus the
2D-mcast config used elsewhere for M>1 tile.

    python -m tracy -p -v -r --op-support-count 100000 \
        -m pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_prefill_mm_sweep_n150.py
    python models/demos/qwen3_tts/tests/prefill_mm_sweep_report.py
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tt.dram_sharded_matmul import width_sharded_l1_memcfg
from models.demos.qwen3_tts.tt.linear_1d_program_config import (
    find_2d_mcast_grid,
    make_linear_1d_program_config,
    make_linear_2d_program_config,
)

TILE = 32
REPS = 4
MANIFEST = "generated/prefill_mm_manifest.json"

# (name, K, N) for the Talker at TP=1. QKV N is the unpadded 4096: prefill >32 uses the
# plain 1D path, so there is no DRAM-bank padding here.
SHAPES = [
    ("gate_up", 2048, 6144),
    ("down", 6144, 2048),
    ("qkv", 2048, 4096),
    ("o_proj", 2048, 2048),
]
BUCKETS = [64, 128]
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
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


@pytest.mark.parametrize("m", BUCKETS, ids=[f"m{b}" for b in BUCKETS])
@pytest.mark.parametrize("shape", SHAPES, ids=[s[0] for s in SHAPES])
def test_prefill_mm_sweep(device, shape, m):
    name, K, N = shape
    cg = device.compute_with_storage_grid_size()
    gx, gy = cg.x, cg.y
    kcfg = _kcfg()
    k_tiles, n_tiles, m_tiles = K // TILE, N // TILE, m // TILE

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
    # Shipped in0: L1-interleaved (attention/mlp S2I before the 1D prefill matmul).
    act_il = ttnn.from_torch(
        act, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    shipped_cores = gx * gy  # make_linear_1d_program_config default = whole grid

    print(f"\n### {name}  M={m} K={K} N={N}  (k_tiles={k_tiles} n_tiles={n_tiles})")

    def _run(label, pc, a, extra=None):
        try:
            probe = ttnn.linear(
                a, w_tt, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=kcfg
            )
            ttnn.synchronize_device(device)
            ttnn.deallocate(probe)
        except Exception as e:
            msg = next((l.strip() for l in str(e).splitlines() if "TT_FATAL" in l or "must" in l), "")
            print(f"  {label:38s} -> REFUSED {msg[:80]}")
            return
        for _ in range(REPS):
            o = ttnn.linear(a, w_tt, program_config=pc, memory_config=ttnn.L1_MEMORY_CONFIG, compute_kernel_config=kcfg)
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        rec = {
            "tag": f"{name}:m{m}:{label}",
            "shape": name,
            "m": m,
            "K": K,
            "N": N,
            "label": label,
            "reps": REPS,
            "shipped": label.startswith("1D c64 auto"),
        }
        rec.update(extra or {})
        _ARMS.append(rec)
        print(f"  {label:38s} -> ran")

    # --- 1D mcast, interleaved in0, core sweep (auto subblock) ---
    for cores in sorted({c for c in (8, 16, 32, 64) if n_tiles % c == 0}):
        pc = make_linear_1d_program_config(m, K, N, gx, gy, True, num_cores=cores)
        _run(
            f"1D c{cores} auto ibw={pc.in0_block_w} sb=({pc.out_subblock_h},{pc.out_subblock_w})",
            pc,
            act_il,
            {"cores": cores, "in0_block_w": int(pc.in0_block_w)},
        )

    # --- 1D mcast with a width-sharded in0 (skips the S2I the model does today) ---
    for cores in sorted({c for c in (32, 64) if n_tiles % c == 0 and k_tiles % c == 0}):
        try:
            a_sh = ttnn.to_memory_config(act_il, width_sharded_l1_memcfg(m_tiles, k_tiles, gx, cores // gx))
        except Exception:
            continue
        pc = make_linear_1d_program_config(m, K, N, gx, gy, True, num_cores=cores)
        _run(f"1D c{cores} sharded-in0 ibw={pc.in0_block_w}", pc, a_sh, {"cores": cores})

    # --- 2D mcast (the config used for M>1-tile speaker TDNNs) ---
    g2x, g2y = find_2d_mcast_grid(m, K, N, gx, gy)
    try:
        pc2 = make_linear_2d_program_config(m, K, N, g2x, g2y, True)
        _run(f"2D {g2x}x{g2y} ibw={pc2.in0_block_w}", pc2, act_il, {"cores": g2x * g2y})
    except Exception as e:
        print(f"  2D {g2x}x{g2y} -> build REFUSED: {str(e).splitlines()[0][:70]}")

    ttnn.deallocate(act_il)
    ttnn.deallocate(w_tt)
