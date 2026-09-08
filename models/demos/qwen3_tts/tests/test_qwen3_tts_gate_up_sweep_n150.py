# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sweep the N150 Talker decode gate/up matmul: 1D-mcast vs DRAM-sharded, at bfp8.

The shipped N150 decode path runs gate/up as 1D-mcast on the full 64-core grid
(``MLP._n150_gate_up_1d``). That grid was swept at **bf16**, where it measured
210 GB/s. With ``QWEN3_TTS_BF8_WEIGHTS`` now default ON the block report shows it
at 92 us / 137 GB/s (47.7 % of the 288 GB/s wall) while ``down_proj`` — a
DRAM-sharded matmul streaming MORE weight — hits 61 us / 230 GB/s. This sweep asks
whether the grid choice simply outlived the dtype it was tuned for.

MUST run under the device profiler, which only dumps a CSV under ``python -m tracy``:

    python -m tracy -p -v -r --op-support-count 100000 \
        -m pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_gate_up_sweep_n150.py

The test writes ``generated/gate_up_sweep_manifest.json`` naming every arm in the
exact order it ran, ``REPS`` matmuls each. Align it to the CSV with:

    python models/demos/qwen3_tts/tests/gate_up_sweep_report.py
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
    build_dram_sharded_weight,
    dram_sharded_program_config,
    find_grid_k_n,
    pad_n_for_dram_align,
    width_sharded_l1_memcfg,
)
from models.demos.qwen3_tts.tt.linear_1d_program_config import make_linear_1d_program_config

TILE = 32
_TRACE_REGION = 50_000_000

# Talker decode. hidden=2048, intermediate=6144; N is per-chip (intermediate // tp_size),
# so N150/TP=1 sees 6144 and N300/TP=2 sees 3072 — different shapes, measure each.
M, K, INTERMEDIATE = 32, 2048, 6144
REPS = 4  # matmul launches per arm; the report drops the first (compile/cold) one.
MANIFEST = "generated/gate_up_sweep_manifest.json"


@pytest.fixture(scope="module")
def device():
    shape = {"N150": (1, 1), "N300": (1, 2)}[os.environ.get("MESH_DEVICE", "N150")]
    if shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), l1_small_size=32768, trace_region_size=_TRACE_REGION)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    if shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _kcfg():
    # Exactly MLP.compute_kernel_config.
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def test_gate_up_grid_sweep(device):
    from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size

    tp = get_tp_size(device) if device.__class__.__name__ == "MeshDevice" else 1
    N = INTERMEDIATE // tp
    print(f"\n[sku] tp_size={tp} -> per-chip N={N}")

    grid = device.compute_with_storage_grid_size()
    gx, gy = grid.x, grid.y
    dram_cores = device.dram_grid_size().x
    kcfg = _kcfg()

    n_padded = pad_n_for_dram_align(N, dram_cores)
    k_tiles, n_tiles = K // TILE, n_padded // TILE
    shipped_rows, shipped_cols = find_grid_k_n(k_tiles, n_tiles, max_rows=gy, max_cols=gx)
    shipped_cores = shipped_rows * shipped_cols

    print(f"\n### Talker gate/up  M={M} K={K} N={N} (pad {n_padded}), bfloat8_b, LoFi + fp32_acc")
    print(f"    grid {gx}x{gy}, {dram_cores} DRAM banks, k_tiles={k_tiles}, n_tiles={n_tiles}")
    # What the code picks TODAY for this SKU: N150/TP=1 took 1D on the full grid
    # (_n150_gate_up_1d); TP>1 already took DRAM-sharded on find_grid_k_n's grid.
    shipped_arm = "1d" if tp == 1 else "dram"
    print(
        f"    find_grid_k_n -> {shipped_rows}x{shipped_cols} = {shipped_cores} cores"
        f"; reference arm for this SKU = {shipped_arm} @ {shipped_cores}"
    )

    torch.manual_seed(0)
    w_t = torch.randn(K, N, dtype=torch.bfloat16)
    act_t = torch.randn(1, 1, M, K, dtype=torch.bfloat16)

    # 1D-mcast reads a plain DRAM-interleaved [1,1,K,N] (MLP.gate_proj);
    # the DRAM-sharded arm reads a bank-width-sharded weight.
    w_1d = ttnn.from_torch(
        w_t.unsqueeze(0).unsqueeze(0).contiguous(),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    w_ds, _k, _n_pad = build_dram_sharded_weight(w_t, device, dtype=ttnn.bfloat8_b)
    assert _n_pad == n_padded

    cores_list = [c for c in range(1, gx * gy + 1) if k_tiles % c == 0 and n_tiles % c == 0]

    def _grid_for(cores):
        for rows in range(1, gy + 1):
            if cores % rows == 0 and cores // rows <= gx:
                return rows, cores // rows
        return None

    manifest = []
    for cores in cores_list:
        g = _grid_for(cores)
        if g is None:
            continue
        rows, cols = g
        in0_mc = width_sharded_l1_memcfg(1, k_tiles, cols, rows)
        out_mc = width_sharded_l1_memcfg(1, n_tiles, cols, rows)
        act = ttnn.from_torch(act_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in0_mc)

        for arm in ("1d", "dram"):
            try:
                if arm == "1d":
                    pc = make_linear_1d_program_config(1, K, N, gx, gy, True, num_cores=cores)
                    w = w_1d
                else:
                    pc = dram_sharded_program_config(M, K, n_padded, num_cores=cores)
                    w = w_ds
                ibw, pcn = pc.in0_block_w, pc.per_core_N

                # One untimed launch to compile + populate the program cache, so every
                # rep the report reads is steady state.
                probe = ttnn.linear(act, w, program_config=pc, memory_config=out_mc, compute_kernel_config=kcfg)
                ttnn.synchronize_device(device)
                ttnn.deallocate(probe)
            except Exception as e:
                print(f"  SKIP {arm} c{cores} ({rows}x{cols}): {type(e).__name__}: {str(e)[:90]}")
                continue

            for _ in range(REPS):
                out = ttnn.linear(act, w, program_config=pc, memory_config=out_mc, compute_kernel_config=kcfg)
                ttnn.synchronize_device(device)
                ttnn.deallocate(out)

            manifest.append(
                {
                    "arm": arm,
                    "cores": cores,
                    "grid": f"{rows}x{cols}",
                    "in0_block_w": int(ibw),
                    "per_core_N": int(pcn),
                    "reps": REPS,
                    "shipped": bool(arm == shipped_arm and cores == shipped_cores),
                }
            )
            print(f"  ran {arm:4s} c{cores:<2d} ({rows}x{cols})  in0_block_w={ibw} per_core_N={pcn}")

        ttnn.deallocate(act)

    os.makedirs("generated", exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump({"M": M, "K": K, "N": N, "tp": tp, "n_padded": n_padded, "arms": manifest}, f, indent=2)
    print(f"\nwrote {MANIFEST}: {len(manifest)} arms x {REPS} reps = {len(manifest) * REPS} matmuls")
    print("now run: python models/demos/qwen3_tts/tests/gate_up_sweep_report.py")
