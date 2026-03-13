"""
Minimal repro: FABRIC_1D crash on 1x16 single-mesh dual T3K.

When opening a 1x16 BigMesh on dual T3K without explicitly setting the
fabric config to FABRIC_2D, the framework auto-promotes DISABLED → FABRIC_1D
(device_manager.cpp). This causes:

    TT_FATAL: No links available from (M0, D3) to (M0, D0)

Run with the 1x16 BigMesh rank bindings:

    tt-run \
        --rank-binding tests/tt_metal/distributed/config/dual_t3k_1x16_experimental_bigmesh_rank_bindings.yaml \
        python tests/scripts/multihost/repro_dual_t3k_1x16_fabric_1d_crash.py

To verify the fix, pass --fabric-2d:

    tt-run \
        --rank-binding tests/tt_metal/distributed/config/dual_t3k_1x16_experimental_bigmesh_rank_bindings.yaml \
        python tests/scripts/multihost/repro_dual_t3k_1x16_fabric_1d_crash.py --fabric-2d
"""

import argparse
import ttnn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fabric-2d",
        action="store_true",
        help="Set FABRIC_2D before opening the mesh device (the fix).",
    )
    args = parser.parse_args()

    if args.fabric_2d:
        print("Setting fabric config to FABRIC_2D before open_mesh_device...")
        ttnn.set_fabric_config(
            ttnn.FabricConfig.FABRIC_2D,
            ttnn.FabricReliabilityMode.STRICT_INIT,
        )
    else:
        print(
            "Not setting fabric config — will default to FABRIC_1D on open_mesh_device.\n"
            "Expect: TT_FATAL: No links available from (M0, D3) to (M0, D0)"
        )

    mesh_shape = ttnn.MeshShape(1, 16)
    print(f"Opening mesh device with shape {mesh_shape} ...")

    mesh_device = ttnn.open_mesh_device(
        mesh_shape=mesh_shape,
        dispatch_core_config=ttnn.DispatchCoreConfig(),
    )

    print(f"Mesh device opened successfully with {mesh_device.get_num_devices()} devices.")
    ttnn.close_mesh_device(mesh_device)
    print("Mesh device closed.")


if __name__ == "__main__":
    main()
