#!/usr/bin/env python3
"""
Open a TTNN mesh with MeshShape(1,2) (N300) and print diagnostics.

Usage:
  python3 models/experimental/minicpm_o_2_6/tt/open_mesh_n300.py

This script tries to be conservative: it queries available devices first and
skips opening the mesh if there are not enough devices. It prints detailed
diagnostics useful to debug `std::bad_alloc` during TTNN initialization.
"""
import traceback
import os
import sys


def main():
    try:
        import ttnn
    except Exception as e:
        print("Failed to import ttnn:", e)
        traceback.print_exc()
        return 1

    try:
        print("TTNN version:", getattr(ttnn, "__version__", "unknown"))
        num_devices = ttnn.get_num_devices()
        device_ids = ttnn.get_device_ids()
        pcie_ids = ttnn.get_pcie_device_ids()
        print("Available devices (logical):", num_devices)
        print("Device IDs:", device_ids)
        print("PCIe device IDs:", pcie_ids)

        # Desired mesh shape for N300: (1,2)
        mesh_shape = ttnn.MeshShape(1, 2)
        # MeshShape may implement indexing or mesh_size()/dims() depending on ttnn version.
        try:
            r = mesh_shape[0]
            c = mesh_shape[1]
            total_needed = r * c
            print(f"Requested mesh shape: {r} x {c} = {total_needed} devices")
        except Exception:
            try:
                total_needed = mesh_shape.mesh_size()
                print(f"Requested mesh shape (mesh_size): {total_needed} devices")
            except Exception:
                print("Requested mesh shape: MeshShape(1,2) (unable to introspect rows/cols)")
                total_needed = 2

        if not ttnn.using_distributed_env() and total_needed > num_devices:
            print(f"Not enough devices available: need {total_needed}, have {num_devices}. Aborting mesh open.")
            return 2

        # Optionally set fabric config minimal (match conftest default behavior)
        try:
            # Only enable fabric if environment requests it
            fabric_env = os.environ.get("ENABLE_FABRIC", "0") == "1"
            if fabric_env:
                print("Enabling fabric config (STRICT_INIT)")
                ttnn.set_fabric_config(ttnn.FabricConfig.ENABLED, ttnn.FabricReliabilityMode.STRICT_INIT)
        except Exception as e:
            print("Warning: failed to set fabric config:", e)

        # Open mesh device with conservative parameters
        updated_device_params = {}
        # Keep debug flags low; user can override via env variables if needed
        try:
            print("Opening mesh device...")
            mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)
        except Exception as e:
            print("Failed to open mesh device:", e)
            traceback.print_exc()
            return 3

        try:
            print("Mesh opened. Num devices:", mesh_device.get_num_devices())
            try:
                print("Submeshes:", [str(s) for s in mesh_device.get_submeshes()])
            except Exception:
                pass
            # Print a few diagnostic calls that are cheap
            try:
                coords = [mesh_device.index_to_coord(i) for i in range(mesh_device.get_num_devices())]
                print("Device coords:", coords)
            except Exception:
                pass

        finally:
            # Close opened mesh device(s)
            try:
                for sub in mesh_device.get_submeshes():
                    ttnn.close_mesh_device(sub)
            except Exception:
                pass
            try:
                ttnn.close_mesh_device(mesh_device)
            except Exception:
                pass

        print("Mesh closed cleanly.")
        return 0

    except Exception as e:
        print("Unhandled error during diagnostics:", e)
        traceback.print_exc()
        return 4


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)


"""
Open a TTNN mesh with MeshShape(1,2) (N300) and print diagnostics.

Usage:
  python3 models/experimental/minicpm_o_2_6/tt/open_mesh_n300.py

This script tries to be conservative: it queries available devices first and
skips opening the mesh if there are not enough devices. It prints detailed
diagnostics useful to debug `std::bad_alloc` during TTNN initialization.
"""
import traceback
import os
import sys


def main():
    try:
        import ttnn
    except Exception as e:
        print("Failed to import ttnn:", e)
        traceback.print_exc()
        return 1

    try:
        print("TTNN version:", getattr(ttnn, "__version__", "unknown"))
        num_devices = ttnn.get_num_devices()
        device_ids = ttnn.get_device_ids()
        pcie_ids = ttnn.get_pcie_device_ids()
        print("Available devices (logical):", num_devices)
        print("Device IDs:", device_ids)
        print("PCIe device IDs:", pcie_ids)

        # Desired mesh shape for N300: (1,2)
        mesh_shape = ttnn.MeshShape(1, 2)
        # MeshShape may implement indexing or mesh_size()/dims() depending on ttnn version.
        try:
            r = mesh_shape[0]
            c = mesh_shape[1]
            total_needed = r * c
            print(f"Requested mesh shape: {r} x {c} = {total_needed} devices")
        except Exception:
            try:
                total_needed = mesh_shape.mesh_size()
                print(f"Requested mesh shape (mesh_size): {total_needed} devices")
            except Exception:
                print("Requested mesh shape: MeshShape(1,2) (unable to introspect rows/cols)")
                total_needed = 2

        if not ttnn.using_distributed_env() and total_needed > num_devices:
            print(f"Not enough devices available: need {total_needed}, have {num_devices}. Aborting mesh open.")
            return 2

        # Optionally set fabric config minimal (match conftest default behavior)
        try:
            # Only enable fabric if environment requests it
            fabric_env = os.environ.get("ENABLE_FABRIC", "0") == "1"
            if fabric_env:
                print("Enabling fabric config (STRICT_INIT)")
                ttnn.set_fabric_config(ttnn.FabricConfig.ENABLED, ttnn.FabricReliabilityMode.STRICT_INIT)
        except Exception as e:
            print("Warning: failed to set fabric config:", e)

        # Open mesh device with conservative parameters
        updated_device_params = {}
        # Keep debug flags low; user can override via env variables if needed
        try:
            print("Opening mesh device...")
            mesh_device = ttnn.open_mesh_device(mesh_shape=mesh_shape, **updated_device_params)
        except Exception as e:
            print("Failed to open mesh device:", e)
            traceback.print_exc()
            return 3

        try:
            print("Mesh opened. Num devices:", mesh_device.get_num_devices())
            try:
                print("Submeshes:", [str(s) for s in mesh_device.get_submeshes()])
            except Exception:
                pass
            # Print a few diagnostic calls that are cheap
            try:
                coords = [mesh_device.index_to_coord(i) for i in range(mesh_device.get_num_devices())]
                print("Device coords:", coords)
            except Exception:
                pass

        finally:
            # Close opened mesh device(s)
            try:
                for sub in mesh_device.get_submeshes():
                    ttnn.close_mesh_device(sub)
            except Exception:
                pass
            try:
                ttnn.close_mesh_device(mesh_device)
            except Exception:
                pass

        print("Mesh closed cleanly.")
        return 0

    except Exception as e:
        print("Unhandled error during diagnostics:", e)
        traceback.print_exc()
        return 4


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
