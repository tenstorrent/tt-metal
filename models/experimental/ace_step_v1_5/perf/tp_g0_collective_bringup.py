# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TP gate **G0**: confirm ttnn collective signatures + a shard→gather→readback round-trip on
the real BH_QB 2×2 mesh, so ``ttnn_impl/tp_config.py`` wrappers are grounded before Phase 2.

Run (device must be free):
    ACE_STEP_TP=on python models/experimental/ace_step_v1_5/perf/tp_g0_collective_bringup.py

It is intentionally defensive: it prints the real ``all_reduce`` / ``all_gather`` docstrings,
then tries the tp_config wrappers; on a signature mismatch it probes a few documented
alternatives and reports which one round-trips at PCC≈1.0. Nothing here mutates model code.
"""

from __future__ import annotations

import os
import traceback

import torch

import ttnn
from models.experimental.ace_step_v1_5.ttnn_impl import tp_config as tp
from models.experimental.ace_step_v1_5.utils.tt_device import open_dit_device


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    if a.numel() != b.numel():
        return float("nan")
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item()
    if denom == 0:
        return 1.0 if a.norm().item() == 0 and b.norm().item() == 0 else 0.0
    return float((a @ b).item() / denom)


def _banner(msg: str) -> None:
    print(f"\n=== [G0] {msg} ===", flush=True)


def main() -> int:
    os.environ.setdefault("ACE_STEP_TP", "on")
    _banner("collective docstrings (record the real signatures)")
    for name in ("all_reduce", "all_gather", "reduce_scatter"):
        fn = getattr(ttnn, name, None)
        doc = (getattr(fn, "__doc__", "") or "").strip().splitlines()
        print(f"- ttnn.{name}: {doc[0] if doc else '(no docstring)'}", flush=True)

    _banner("open BH_QB 2x2 mesh (fabric FABRIC_1D for CCL)")
    # CCL collectives need the fabric context up BEFORE the mesh opens
    # (TT_FATAL control_plane.cpp: fabric_context_ != nullptr otherwise).
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    dev = open_dit_device(ttnn, mesh_sku="BH_QB", num_command_queues=1)
    try:
        cfg = tp.resolve_tp_config(dev)
        print(f"TPConfig: enabled={cfg.enabled} axis={cfg.axis} shape={cfg.mesh_shape} degree={cfg.degree}", flush=True)
        assert cfg.enabled and cfg.degree > 1, "TP must resolve enabled on the mesh (ACE_STEP_TP=on)"

        # ---- Test A: column-parallel weight shard -> all_gather -> readback round-trips ----
        _banner("A: shard(dim=-1) -> tp_all_gather -> readback  (expect PCC≈1.0)")
        # A tile-friendly [1,1,64,128] weight; shard the last dim across the TP axis.
        w = torch.randn(1, 1, 64, 128, dtype=torch.float32)
        shard_dim = 3
        mapper = tp.tp_weight_mesh_mapper(dev, shard_dim=shard_dim, cfg=cfg)
        print(f"mapper = {type(mapper).__name__ if mapper is not None else None}", flush=True)
        w_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=mapper)
        try:
            w_back = tp.tp_read_sharded_to_torch(w_tt, dev, shard_dim=shard_dim, cfg=cfg)
            print(f"round-trip shape {tuple(w_back.shape)}  PCC={_pcc(w, w_back):.5f}", flush=True)
        except Exception:
            print("tp_read_sharded_to_torch FAILED — signature likely off:", flush=True)
            traceback.print_exc()

        # ---- Test B: all_reduce sum semantics (replicated ones -> degree) ----
        _banner("B: all_reduce(replicated ones) (expect each element == degree along TP axis)")
        ones = torch.ones(1, 1, 32, 32, dtype=torch.float32)
        rep = tp._replicate_mapper(dev)
        o_tt = ttnn.from_torch(ones, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, mesh_mapper=rep)
        try:
            red = tp.tp_all_reduce(o_tt, dev, cfg=cfg)
            red_host = tp.tp_read_sharded_to_torch(red, dev, shard_dim=shard_dim, cfg=cfg)
            print(f"all_reduce(ones) sample={red_host.flatten()[:4].tolist()} (want ~{cfg.degree})", flush=True)
        except Exception:
            print("tp_all_reduce FAILED — probing alternative signatures:", flush=True)
            traceback.print_exc()
            _probe_all_reduce(dev, o_tt, cfg)

        _banner("G0 complete — see PCC/signature notes above; update tp_config.py accordingly")
        return 0
    finally:
        try:
            from models.experimental.ace_step_v1_5.utils.tt_device import close_ace_step_device

            close_ace_step_device(ttnn, dev)
        except Exception:
            try:
                ttnn.close_device(dev)
            except Exception:
                pass
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


def _probe_all_reduce(dev, tensor, cfg) -> None:
    """Try documented all_reduce call shapes so we can fix the wrapper."""
    attempts = [
        ("cluster_axis+mesh_device", lambda: ttnn.all_reduce(tensor, cluster_axis=cfg.axis, mesh_device=dev)),
        (
            "math_op=sum",
            lambda: ttnn.all_reduce(tensor, ttnn.ReduceType.Sum, cluster_axis=cfg.axis, mesh_device=dev)
            if hasattr(ttnn, "ReduceType")
            else None,
        ),
        ("positional dim", lambda: ttnn.all_reduce(tensor, cfg.axis)),
    ]
    for label, fn in attempts:
        try:
            r = fn()
            print(f"  probe OK: {label} -> {type(r).__name__}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  probe FAIL: {label}: {exc}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
