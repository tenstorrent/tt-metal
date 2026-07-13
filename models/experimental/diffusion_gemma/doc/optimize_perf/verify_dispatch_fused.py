# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness check: DG_MOE_DISPATCH_FUSED disp/comb vs the impl path (no checkpoint needed).

Builds random dense routings (exactly top_k nonzero per token, the router contract) INCLUDING
capacity-overflow cases, runs both ``_build_capacity_dispatch_impl`` and
``_build_capacity_dispatch_fused``, and asserts the kept columns [0:EC] of disp AND comb are
byte-for-byte identical. The fused path only removes ops that do not change those columns (see
``sparse_moe._dispatch_fused_enabled``), so any nonzero diff is a bug.

Run:  TT_METAL_HOME=/home/zni/tt-metal PYTHONPATH=/home/zni/tt-metal \
      python models/experimental/diffusion_gemma/doc/optimize_perf/verify_dispatch_fused.py
"""

from __future__ import annotations

import torch
import ttnn

from models.experimental.diffusion_gemma.tt.sparse_moe import (
    _build_capacity_dispatch_fused,
    _build_capacity_dispatch_fused2,
    _build_capacity_dispatch_impl,
)


def _make_routing(S, E, top_k, skew_expert=None, seed=0):
    """Dense routing [1,1,S,E], exactly top_k positive entries per token (rest exact 0).

    skew_expert: if set, force every token to route to that expert (its top), driving heavy
    capacity overflow for the fixed-C dispatch.
    """
    g = torch.Generator().manual_seed(seed)
    routing = torch.zeros(1, 1, S, E, dtype=torch.float32)
    for t in range(S):
        if skew_expert is not None:
            experts = torch.tensor([skew_expert] + torch.randperm(E, generator=g)[: top_k - 1].tolist())
            experts = torch.unique(experts)
            while experts.numel() < top_k:
                extra = torch.randint(0, E, (1,), generator=g)
                experts = torch.unique(torch.cat([experts, extra]))
            experts = experts[:top_k]
        else:
            experts = torch.randperm(E, generator=g)[:top_k]
        w = torch.softmax(torch.rand(top_k, generator=g), dim=0) + 0.05  # strictly positive
        routing[0, 0, t, experts] = w.to(torch.float32)
    return routing


def _to_host0(t):
    dev = t.device()
    if dev is not None and hasattr(dev, "get_num_devices") and dev.get_num_devices() > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).float()
    return ttnn.to_torch(t).float()


def _run_case(mesh, name, S, E, C, top_k, skew_expert=None, seed=0):
    routing_host = _make_routing(S, E, top_k, skew_expert=skew_expert, seed=seed)
    mapper = ttnn.ReplicateTensorToMesh(mesh) if hasattr(mesh, "shape") else None
    routing = ttnn.from_torch(
        routing_host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper
    )

    disp_i, comb_i = _build_capacity_dispatch_impl(routing, E, C, top_k)
    disp_f, comb_f = _build_capacity_dispatch_fused(routing, E, C, top_k)
    disp_g, comb_g = _build_capacity_dispatch_fused2(routing, E, C, top_k)

    di, ci = _to_host0(disp_i), _to_host0(comb_i)
    df, cf = _to_host0(disp_f), _to_host0(comb_f)
    dg, cg = _to_host0(disp_g), _to_host0(comb_g)
    routing.deallocate(True)
    for t in (disp_i, comb_i, disp_f, comb_f, disp_g, comb_g):
        t.deallocate(True)

    disp_max = (di - df).abs().max().item()
    comb_max = (ci - cf).abs().max().item()
    disp2_max = (di - dg).abs().max().item()
    comb2_max = (ci - cg).abs().max().item()
    n_dispatched = int(di.sum().item())  # kept assignments (post-overflow-drop)
    ok = disp_max == 0.0 and comb_max == 0.0 and disp2_max == 0.0 and comb2_max == 0.0
    print(
        f"[{name}] S={S} E={E} C={C} k={top_k} skew={skew_expert} "
        f"fused[disp={disp_max} comb={comb_max}] fused2[disp={disp2_max} comb={comb2_max}] "
        f"kept_assignments={n_dispatched} {'OK' if ok else 'MISMATCH'}"
    )
    return ok


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 4))
    all_ok = True
    try:
        # Production shape, no overflow expected.
        all_ok &= _run_case(mesh, "prod", S=256, E=128, C=32, top_k=8, seed=1)
        all_ok &= _run_case(mesh, "prod2", S=256, E=128, C=32, top_k=8, seed=7)
        # Heavy capacity overflow (all tokens skewed to one expert -> >C on it).
        all_ok &= _run_case(mesh, "overflow-e0", S=256, E=64, C=32, top_k=8, skew_expert=0, seed=2)
        all_ok &= _run_case(mesh, "overflow-e5", S=256, E=64, C=32, top_k=8, skew_expert=5, seed=3)
        # Extreme: top_k == E so every expert gets all S tokens (max overflow).
        all_ok &= _run_case(mesh, "allexperts", S=256, E=8, C=32, top_k=8, seed=4)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    print("DISPATCH_FUSED_BITEXACT " + ("PASS" if all_ok else "FAIL"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
