# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Sweep the two Talker DECODE matmuls that were never swept: wqkv and wo.

gate/up was fixed by noticing that ``find_grid_k_n`` returns the LARGEST core count
dividing both K_tiles and N_tiles, which MINIMISES ``in0_block_w`` — free at bf16,
expensive at bfp8. wqkv and wo sit on the same helper, and it hands them the opposite
problem: because ``pad_n_for_dram_align`` pads to a multiple of ``TILE*12 = 384``, the
gcd collapses.

    wqkv  K_tiles=64, N 4096 -> pad 4224 -> N_tiles=132, gcd(64,132) =  4 cores
    wo    K_tiles=64, N 2048 -> pad 2304 -> N_tiles= 72, gcd(64, 72) =  8 cores

Four cores for a 8.6 MB weight read is the whole grid sitting idle. The lever nobody
pulled is the PAD ITSELF: padding wqkv to 4608 (=384*12) gives N_tiles=144 and
gcd(64,144)=16 cores, at the cost of 9 % more weight bytes. This sweep prices that
trade for every legal (pad, cores) pair, and scores GB/s against each arm's OWN byte
count so a bigger pad cannot win by cheating the metric.

    python -m tracy -p -v -r --op-support-count 100000 \
        -m pytest -s -q models/demos/qwen3_tts/tests/test_qwen3_tts_decode_mm_sweep_n150.py
    python models/demos/qwen3_tts/tests/decode_mm_sweep_report.py
"""

from __future__ import annotations

import json
import os

import pytest
import torch

import ttnn
from models.demos.qwen3_tts.tt.dram_sharded_matmul import (
    dram_sharded_program_config,
    dram_sharded_weight_memcfg,
    find_grid_k_n,
    pad_n_for_dram_align,
    width_sharded_l1_memcfg,
)
from models.demos.qwen3_tts.tt.linear_1d_program_config import make_linear_1d_program_config

TILE = 32
M = 32  # decode: 1 tile row
REPS = 4
MANIFEST = "generated/decode_mm_manifest.json"

# (name, K, N_full_tp1). N is per-chip, so TP=2 halves it; the fixture derives that.
SHAPES = [
    ("wqkv", 2048, 4096),
    ("wo", 2048, 2048),
]
_ARMS: list = []


@pytest.fixture(scope="module")
def device():
    shape = {"N150": (1, 1), "N300": (1, 2)}[os.environ.get("MESH_DEVICE", "N150")]
    if shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    d = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), l1_small_size=32768)
    d.enable_program_cache()
    yield d
    ttnn.close_mesh_device(d)
    if shape != (1, 1):
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    os.makedirs("generated", exist_ok=True)
    with open(MANIFEST, "w") as f:
        json.dump({"M": M, "reps": REPS, "arms": _ARMS}, f, indent=2)
    print(f"\nwrote {MANIFEST}: {len(_ARMS)} arms x {REPS} reps")


def _kcfg():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


def _build_weight(w_kn: torch.Tensor, n_padded: int, device, dtype):
    k, n = w_kn.shape
    if n_padded != n:
        w_kn = torch.cat([w_kn, torch.zeros(k, n_padded - n, dtype=w_kn.dtype)], dim=1)
    return ttnn.from_torch(
        w_kn.unsqueeze(0).unsqueeze(0).contiguous(),
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=dram_sharded_weight_memcfg(k, n_padded, device),
    )


@pytest.mark.parametrize("shape", SHAPES, ids=[s[0] for s in SHAPES])
def test_decode_mm_sweep(device, shape):
    from models.demos.qwen3_tts.tt.mesh_utils import get_tp_size

    name, K, N_full = shape
    tp = get_tp_size(device) if device.__class__.__name__ == "MeshDevice" else 1
    # wqkv is column-parallel (N splits); wo is row-parallel (K splits).
    N = N_full // tp if name == "wqkv" else N_full
    K_local = K // tp if name == "wo" else K

    cg = device.compute_with_storage_grid_size()
    gx, gy = cg.x, cg.y
    dram_cores = device.dram_grid_size().x
    align = TILE * dram_cores
    kcfg = _kcfg()
    k_tiles = K_local // TILE

    ship_pad = pad_n_for_dram_align(N, dram_cores)
    ship_rows, ship_cols = find_grid_k_n(k_tiles, ship_pad // TILE, max_rows=gy, max_cols=gx)
    ship_cores = ship_rows * ship_cols
    print(f"\n### {name}  M={M} K={K_local} N={N}  tp={tp}  grid {gx}x{gy}, {dram_cores} banks")
    print(f"    SHIPPED: pad {N}->{ship_pad} (n_tiles={ship_pad//TILE}), find_grid_k_n -> {ship_cores} cores")

    torch.manual_seed(0)
    w_t = torch.randn(K_local, N, dtype=torch.bfloat16)
    act_t = torch.randn(1, 1, M, K_local, dtype=torch.bfloat16)

    # Pad candidates: every DRAM-aligned width from the shipped one up to +40 % bytes.
    pads = [p for p in range(ship_pad, int(N * 1.4) + align, align) if p % align == 0]

    def _run(label, pc, a, w, out_mc, n_pad, cores, extra=None):
        try:
            probe = ttnn.linear(a, w, program_config=pc, memory_config=out_mc, compute_kernel_config=kcfg)
            ttnn.synchronize_device(device)
            got_cores = probe.memory_config().shard_spec.num_cores() if probe.memory_config().is_sharded() else 0
            ttnn.deallocate(probe)
        except Exception as e:
            msg = next((l.strip() for l in str(e).splitlines() if "TT_FATAL" in l or "must" in l), str(e)[:80])
            print(f"  {label:42s} -> REFUSED {msg[:88]}")
            return
        for _ in range(REPS):
            o = ttnn.linear(a, w, program_config=pc, memory_config=out_mc, compute_kernel_config=kcfg)
            ttnn.synchronize_device(device)
            ttnn.deallocate(o)
        rec = {
            "tag": f"{name}:{label}",
            "shape": name,
            "label": label,
            "K": K_local,
            "N": N,
            "n_padded": n_pad,
            "cores": cores,
            "out_cores": got_cores,
            "in0_block_w": int(getattr(pc, "in0_block_w", 0)),
            "reps": REPS,
            "shipped": bool(extra and extra.get("shipped")),
        }
        _ARMS.append(rec)
        print(f"  {label:42s} -> ran (out shard on {got_cores} cores)")

    for n_pad in pads:
        n_tiles = n_pad // TILE
        w_tt = _build_weight(w_t.clone(), n_pad, device, ttnn.bfloat8_b)
        cand = [c for c in range(1, gx * gy + 1) if k_tiles % c == 0 and n_tiles % c == 0]
        print(f"  -- pad {n_pad} (n_tiles={n_tiles}, +{100*(n_pad/N-1):.1f} % bytes): legal cores {cand}")
        for cores in cand:
            rows = next((r for r in range(1, gy + 1) if cores % r == 0 and cores // r <= gx), None)
            if rows is None:
                continue
            cols = cores // rows
            in0_mc = width_sharded_l1_memcfg(1, k_tiles, cols, rows)
            out_mc = width_sharded_l1_memcfg(1, n_tiles, cols, rows)
            act = ttnn.from_torch(
                act_t, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=in0_mc
            )
            pc = dram_sharded_program_config(M, K_local, n_pad, num_cores=cores)
            _run(
                f"dram pad{n_pad} c{cores} ibw={pc.in0_block_w}",
                pc,
                act,
                w_tt,
                out_mc,
                n_pad,
                cores,
                {"shipped": n_pad == ship_pad and cores == ship_cores},
            )
            ttnn.deallocate(act)
        ttnn.deallocate(w_tt)

    # 1D-mcast on the UNPADDED weight: no pad waste at all, but in0 comes from DRAM.
    w_1d = ttnn.from_torch(
        w_t.unsqueeze(0).unsqueeze(0).contiguous(),
        device=device,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    n_tiles_real = N // TILE
    for cores in [c for c in (8, 16, 32, 64) if n_tiles_real % c == 0 and k_tiles % c == 0]:
        rows = next((r for r in range(1, gy + 1) if cores % r == 0 and cores // r <= gx), None)
        if rows is None:
            continue
        cols = cores // rows
        act = ttnn.from_torch(
            act_t,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=width_sharded_l1_memcfg(1, k_tiles, cols, rows),
        )
        pc = make_linear_1d_program_config(M, K_local, N, gx, gy, True, num_cores=cores)
        _run(
            f"1d nopad c{cores} ibw={pc.in0_block_w}",
            pc,
            act,
            w_1d,
            width_sharded_l1_memcfg(1, n_tiles_real, cols, rows),
            N,
            cores,
        )
        ttnn.deallocate(act)
    ttnn.deallocate(w_1d)
