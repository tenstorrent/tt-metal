# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Feasibility probe for TRUE 4-way TP on a 2x2 mesh: does an all-4-chip collective
(cluster_axis=None + ShardTensorToMesh) work under FABRIC_1D, or is FABRIC_2D needed?

Run: python models/experimental/ace_step_v1_5/perf/tp_g0b_4way.py
"""

from __future__ import annotations

import traceback

import torch

import ttnn
from models.experimental.ace_step_v1_5.utils.tt_device import close_ace_step_device, open_dit_device


def _try(fabric_name):
    print(f"\n=== [G0b] fabric={fabric_name} ===", flush=True)
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, fabric_name))
    dev = open_dit_device(ttnn, mesh_sku="BH_QB", num_command_queues=1)
    try:
        n = dev.get_num_devices()
        # 4-way shard of last dim across all n devices, then all_gather back.
        w = torch.randn(1, 1, 32, 32 * n, dtype=torch.float32)
        w_tt = ttnn.from_torch(
            w,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=ttnn.ShardTensorToMesh(dev, dim=3),
        )
        try:
            g = ttnn.all_gather(w_tt, 3)  # cluster_axis omitted -> all devices
            gh = ttnn.to_torch(ttnn.get_device_tensors(g)[0]).to(torch.float32)
            ok_shape = tuple(gh.shape) == tuple(w.shape)
            print(
                f"  all_gather(all devices): out shape {tuple(gh.shape)} (want {tuple(w.shape)}) ok={ok_shape}",
                flush=True,
            )
        except Exception:
            print("  all_gather(all devices) FAILED:", flush=True)
            traceback.print_exc()
        # all_reduce over all devices on replicated ones -> each == n
        ones = ttnn.from_torch(
            torch.ones(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=dev,
            mesh_mapper=ttnn.ShardTensor2dMesh(dev, dims=(None, None), mesh_shape=tuple(int(x) for x in dev.shape)),
        )
        try:
            r = ttnn.all_reduce(ones)  # cluster_axis omitted -> all devices
            rh = ttnn.to_torch(ttnn.get_device_tensors(r)[0]).to(torch.float32)
            print(f"  all_reduce(all devices) sample={rh.flatten()[:4].tolist()} (want ~{n})", flush=True)
        except Exception:
            print("  all_reduce(all devices) FAILED:", flush=True)
            traceback.print_exc()
    finally:
        close_ace_step_device(ttnn, dev)
        try:
            ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
        except Exception:
            pass


def main() -> int:
    for fabric in ("FABRIC_1D", "FABRIC_2D"):
        try:
            _try(fabric)
        except Exception:
            print(f"  open with {fabric} FAILED:", flush=True)
            traceback.print_exc()
    print("\n[G0b] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
