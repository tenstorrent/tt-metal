# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEFINITIVE token-0 cause test (single chip): does the COMPOSITE routed expert (ttnn.linear(down)
transient + bf16->bf8 in-place insert) develop garbage at output channels ~4128-4143 when DRAM is
POISONED (huge bf8 pattern in freed DRAM = what the full-model footprint provides), while the FUSED
swiglu_oai expert stays clean? composite=garbage & fused=clean => the composite expert-output stale read
is PROVEN as the source AND the fused kernel is the fix. composite=clean => bug is footprint-specific.
Reuses the known-good single-chip fixture (correct fabric config)."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.tt.moe.tt_routed_expert import TtRoutedExpert
from models.demos.minimax_m3.tt.experts_throughput.composite_routed_expert import CompositeRoutedExpert

from .test_swigluoai_routed_expert import SINGLE_CHIP_MESH_PARAMS

EMB, HID, NTOK = 6144, 3072, 256  # M3 dims; 256 tok -> fused fits L1 (no cap needed)


def _idx(mesh, vals):
    return ttnn.from_torch(torch.tensor(vals, dtype=torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, dtype=ttnn.uint32)


def _poison(mesh):
    junk = [ttnn.from_torch(torch.full((1, 1, 8192, EMB), 3.0e38), device=mesh, dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) for _ in range(6)]
    for t in junk:
        t.deallocate(True)


def _run(mesh, which, w, x_t, do_poison):
    x = ttnn.from_torch(x_t, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh), layout=ttnn.TILE_LAYOUT, device=mesh, dtype=ttnn.bfloat8_b)
    common = dict(mesh_device=mesh, experts_per_chip=1, global_expert_idx_table=_idx(mesh, [0]),
                  emb_dim=EMB, hidden_dim=HID, max_tokens=NTOK, torch_weights=[w],
                  activations_dtype=ttnn.bfloat8_b, weights_dtype=ttnn.bfloat4_b)
    expert = (CompositeRoutedExpert(**common, swiglu_limit=7.0, alpha=1.702) if which == "composite"
              else TtRoutedExpert(**common, swiglu_oai=True))
    if do_poison:
        _poison(mesh)
    out = expert(x, _idx(mesh, [NTOK]), _idx(mesh, [0]))
    o = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[:NTOK].float().reshape(NTOK, EMB)
    n_big = int((o.abs() > 1e30).sum())
    logger.info(f"[{which} poison={int(do_poison)}] absmax={o.abs().max():.3e} finite={bool(torch.isfinite(o).all())} "
                f"block[4128:4144]_absmax={o[:,4128:4144].abs().max():.3e} #>1e30={n_big}")
    return o.abs().max().item(), bool(torch.isfinite(o).all()), n_big


@pytest.mark.skipif(not is_blackhole(), reason="unified_routed_expert op is Blackhole-only")
@pytest.mark.parametrize("mesh_device, device_params", SINGLE_CHIP_MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_poison_composite_vs_fused(mesh_device, device_params):
    torch.manual_seed(0)
    w = {"gate_proj": torch.randn(HID, EMB) * 0.08, "up_proj": torch.randn(HID, EMB) * 0.08, "down_proj": torch.randn(EMB, HID) * 0.05}
    x_t = torch.randn(NTOK, EMB) * 1.0
    _run(mesh_device, "composite", w, x_t, False)
    cmax, cfin, cbig = _run(mesh_device, "composite", w, x_t, True)
    _run(mesh_device, "fused", w, x_t, False)
    fmax, ffin, fbig = _run(mesh_device, "fused", w, x_t, True)
    logger.info(f">>> composite-poison: absmax={cmax:.3e} big={cbig} | fused-poison: absmax={fmax:.3e} big={fbig}")
    # Not an assert on the bug repro (we want to OBSERVE); just guard the fused stays clean.
    assert ffin and fbig == 0, f"fused expert produced garbage under poison: absmax={fmax:.3e} big={fbig}"
