# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""MOE_BLOCK_SRC (owner 2026-11-06): one layer's MoE on the carried block and on Marko's reworked block (main), real
weights, the same random x, each in its own test (one block a process's L1).

Not a gate: a check before the model runs the new block, for BREADCRUMBS. Per side:
  MOEMAIN_TIME  synced wall a call (median of N_CALLS after a warm call)
  the output saved to generated/deepseek_v41_flash_d_p/moe_block_main/L<layer>_<src>.pt; the main side compares with the
  carried side's file when it is there:
  MOEMAIN_CMP   rel L2 main vs carried, PCC, worst row, values that differ

    PYTHONPATH=$PWD scripts/run_safe_pytest.sh --run-all --no-precompile \\
        models/demos/deepseek_v41_flash_d_p/tt/perf_moe_block_main.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing.harness import spec
from models.demos.deepseek_v41_flash_d_p.bringup import hooks
from models.demos.deepseek_v41_flash_d_p.tt.perf_experts import _weighted_params

N_CALLS = 5
OUT = Path("generated/deepseek_v41_flash_d_p/moe_block_main")


@pytest.mark.timeout(7200)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
@pytest.mark.parametrize("src", ["carried", "main"])
@pytest.mark.parametrize("layer", [0], ids=["L00"])
def test_moe_block_src(mesh_device, layer, src, monkeypatch):
    from models.demos.deepseek_v41_flash_d_p.tt.ccl import dealloc
    from models.demos.deepseek_v41_flash_d_p.tt.common import own_rows
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir
    from models.demos.deepseek_v41_flash_d_p.tt.layout import own_rows_to_device, own_rows_to_host
    from models.demos.deepseek_v41_flash_d_p.tt.moe_block_adapter import build_moe_block_layer
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_BLOCK_SRC", src)
    settings.refresh()
    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    mesh = mesh_device
    own = own_rows()
    assert own, "the check runs the model's default (OWN_ROWS=on)"
    x = torch.randn(chunk, cfg.dim, generator=torch.Generator().manual_seed(layer)).to(torch.bfloat16)
    xd = own_rows_to_device(mesh, x, dtype=ttnn.bfloat16)
    t0 = time.time()
    moe = build_moe_block_layer(mesh, ck, cfg, layer, chunk, cache_dir=expert_cache_dir(S), own_rows=own)
    t_build = time.time() - t0
    y = moe(xd)
    got = own_rows_to_host(mesh, y).float()
    dealloc(y)
    again = moe(xd)
    det = int((own_rows_to_host(mesh, again).float() != got).sum().item())
    dealloc(again)
    ttnn.synchronize_device(mesh)
    ts = []
    for _ in range(N_CALLS):
        t0 = time.perf_counter()
        yy = moe(xd)
        ttnn.synchronize_device(mesh)
        ts.append((time.perf_counter() - t0) * 1e3)
        dealloc(yy)
    ts.sort()
    print(
        f"MOEMAIN_TIME L{layer:02d} src={src} build {t_build:.1f} s, call synced wall median {ts[len(ts) // 2]:.3f} ms "
        f"(min {ts[0]:.3f}); twice on the same x: {det} values differ"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(got, OUT / f"L{layer:02d}_{src}.pt")
    other = OUT / f"L{layer:02d}_carried.pt"
    if src == "main" and other.exists():
        ref = torch.load(other).double()
        d = got.double() - ref
        rel = (d.norm() / ref.norm()).item()
        pcc = torch.corrcoef(torch.stack([got.double().flatten(), ref.flatten()]))[0, 1].item()
        row = (d.norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-30)).max().item()
        print(
            f"MOEMAIN_CMP L{layer:02d} main vs carried: rel={rel:.5f} pcc={pcc:.6f} worst_row={row:.4f} "
            f"differ={int((d != 0).sum().item())}/{d.numel()}"
        )
        assert pcc > 0.99, f"main vs carried PCC {pcc}"


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("device_params", [_weighted_params()], indirect=True, ids=["box_payload"])
@pytest.mark.parametrize("mesh_device", [tuple(spec().mesh)], indirect=True, ids=["x".join(map(str, spec().mesh))])
def test_build_cache(mesh_device, monkeypatch):
    """MOE_BLOCK_SRC=main's routed-expert weight cache for every layer (host dequant + [N, K] pack ~2.5 min a layer
    on a miss), so a model run loads it in seconds: MOEMAIN_CACHE L<layer> <seconds> per layer."""
    from models.demos.deepseek_v41_flash_d_p.tt import moe_block_main_adapter as A
    from models.demos.deepseek_v41_flash_d_p.tt.common import own_rows
    from models.demos.deepseek_v41_flash_d_p.tt.experts import expert_cache_dir
    from models.demos.deepseek_v41_flash_d_p.tt.settings import settings

    monkeypatch.setenv("DS41_MOE_BLOCK_SRC", "main")
    settings.refresh()
    S = spec()
    ck, cfg = hooks._checkpoint(S), hooks._config(S)
    chunk = hooks._chunk(S)
    for layer in range(int(cfg.n_layers)):
        t0 = time.time()
        moe = A.TtMoEBlockMainLayer(
            mesh_device, ck, cfg, layer, chunk, cache_dir=expert_cache_dir(S), own_rows=own_rows()
        )
        for t in (moe.gidx, moe.w_gu, moe.w_d):
            ttnn.deallocate(t)
        print(f"MOEMAIN_CACHE L{layer:02d} {time.time() - t0:.1f} s", flush=True)
