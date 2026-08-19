# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which (rows, cols) submeshes of this galaxy can be opened with FABRIC_1D?

test_mla.run_model hardcodes sp_axis=0 / tp_axis=1, so SP=8 with TP=1 is only expressible as the
column shape (8, 1) -- (1, 8) is SP=1/TP=8, the inverse. That shape failed fabric router sync, so
probe the shapes directly to separate "column submeshes are unsupported" from a transient fault.
"""

import sys

import ttnn

SHAPES = [(8, 4), (1, 8), (8, 1), (4, 1), (2, 1), (1, 4), (4, 4)]


def probe(rows: int, cols: int, reliability: str = "STRICT_INIT") -> tuple[bool, str]:
    mesh_device = None
    try:
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_1D,
            getattr(ttnn.FabricReliabilityMode, reliability),
            None,
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
        )
        mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))
        return True, f"opened, shape={tuple(mesh_device.shape)}"
    except Exception as exc:  # noqa: BLE001 - probing, any failure is a result
        first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        return False, first[:150]
    finally:
        if mesh_device is not None:
            ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main() -> int:
    # One shape per process: opening several meshes in a single process leaves fabric state behind and
    # every probe after the first fails regardless of shape, which reads as "only (8,4) works".
    if len(sys.argv) in (3, 4):
        rows, cols = int(sys.argv[1]), int(sys.argv[2])
        reliability = sys.argv[3] if len(sys.argv) == 4 else "STRICT_INIT"
        ok, msg = probe(rows, cols, reliability)
        print(f"[probe] ({rows},{cols}) {reliability:14s} -> {'OK ' if ok else 'FAIL'}  {msg}", flush=True)
        return 0 if ok else 1

    results = []
    for rows, cols in SHAPES:
        ok, msg = probe(rows, cols)
        results.append((rows, cols, ok, msg))
        print(f"[probe] ({rows},{cols}) -> {'OK ' if ok else 'FAIL'}  {msg}", flush=True)

    print("\n=== summary ===")
    for rows, cols, ok, msg in results:
        print(f"  ({rows},{cols}){'':4s} {'OK' if ok else 'FAIL':4s}  {msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
