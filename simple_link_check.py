#!/usr/bin/env python3
import os
import sys
import argparse
import ttnn
import torch
import yaml

# Silence ttnn logger noise
os.environ["TT_METAL_LOG_LEVEL"] = "error"
os.environ["TT_METAL_CONSOLE_LEVEL"] = "error"


def make_mesh_shape(rows, cols):
    """Create a mesh shape, trying MeshShape first, falling back to tuple"""
    try:
        if hasattr(ttnn, "MeshShape"):
            return ttnn.MeshShape(rows, cols)
        else:
            # Fallback to tuple if MeshShape not available
            return (rows, cols)
    except:
        return (rows, cols)


# --- Available mesh configs ---
MESH_CONFIGS = [
    {
        "name": "P150 x4 (2x2 mesh)",
        "descriptor": "./tt_metal/fabric/mesh_graph_descriptors/p150_x4_mesh_graph_descriptor.yaml",
        "shape": make_mesh_shape(2, 2),
    },
    {
        "name": "P150 x2 (1x2 mesh)",
        "descriptor": "./tt_metal/fabric/mesh_graph_descriptors/p150_x2_mesh_graph_descriptor.yaml",
        "shape": make_mesh_shape(1, 2),
    },
    {
        "name": "P150 single device",
        "descriptor": "./tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.yaml",
        "shape": make_mesh_shape(1, 1),
    },
]


def infer_mesh_shape_from_yaml(yaml_path):
    """Infer mesh shape from cluster descriptor YAML"""
    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Try to get chip count from various fields
        num_chips = 0

        # Check 'arch' field (common in topology tool output)
        if "arch" in data and data["arch"]:
            num_chips = len(data["arch"])
        # Check 'chips' field (if it's not empty)
        elif "chips" in data and data["chips"]:
            chips = data["chips"]
            num_chips = len(chips)

            # Try to infer shape from chip locations if available
            if chips and isinstance(chips, dict) and "location" in list(chips.values())[0]:
                max_x = max(chip["location"]["x"] for chip in chips.values())
                max_y = max(chip["location"]["y"] for chip in chips.values())
                rows = max_y + 1
                cols = max_x + 1
                print(f"   Inferred mesh shape from locations: {rows}x{cols}")
                return make_mesh_shape(rows, cols)

        # Fallback: guess based on number of chips
        if num_chips == 0:
            print(f"   Warning: No chips found in YAML")
            return None
        elif num_chips == 1:
            print(f"   Found {num_chips} chip, using 1x1 mesh")
            return make_mesh_shape(1, 1)
        elif num_chips == 2:
            print(f"   Found {num_chips} chips, using 1x2 mesh")
            return make_mesh_shape(1, 2)
        elif num_chips == 4:
            print(f"   Found {num_chips} chips, using 2x2 mesh")
            return make_mesh_shape(2, 2)
        elif num_chips == 8:
            print(f"   Found {num_chips} chips, using 2x4 mesh")
            return make_mesh_shape(2, 4)
        else:
            print(f"   Warning: {num_chips} chips found, defaulting to 1x{num_chips} mesh")
            return make_mesh_shape(1, num_chips)

    except Exception as e:
        print(f"   Error reading YAML: {e}")
        import traceback

        traceback.print_exc()
        return None


def detect_and_test_mesh():
    print("Smart Ethernet Link Detection")
    print("=" * 45)

    for config in MESH_CONFIGS:
        print(f"\nTrying: {config['name']}")
        try:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = config["descriptor"]
            mesh_device = ttnn.open_mesh_device(mesh_shape=config["shape"])
            device_ids = mesh_device.get_device_ids()
            mesh_shape = mesh_device.shape

            print(f"  ✅ SUCCESS! Opened mesh with:")
            print(f"     Shape: {mesh_shape}")
            print(f"     Devices: {len(device_ids)}")
            print(f"     Device IDs: {device_ids}")
            print(f"     Device coordinate mapping:")
            for r in range(mesh_shape[0]):
                for c in range(mesh_shape[1]):
                    try:
                        if hasattr(ttnn, "MeshCoordinate"):
                            did = mesh_device.get_device_id(ttnn.MeshCoordinate(r, c))
                        else:
                            did = mesh_device.get_device_id((r, c))
                        print(f"       ({r}, {c}) -> Device {did}")
                    except Exception as e:
                        print(f"       ({r}, {c}) -> Unable to get device ID: {e}")

            ttnn.close_mesh_device(mesh_device)
            return config

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            try:
                ttnn.close_mesh_device(mesh_device)
            except:
                pass

    print("\n❌ No working mesh configuration found!")
    return None


def test_ethernet(config):
    print("\n" + "=" * 45)
    print("TESTING ETHERNET CONNECTIVITY")
    print(f"Using: {config['name']}")
    print("=" * 45)

    # Use the appropriate environment variable based on descriptor type
    if "is_cluster_descriptor" in config and config["is_cluster_descriptor"]:
        # Physical cluster descriptor (old format)
        os.environ["TT_METAL_CLUSTER_DESC_PATH"] = config["descriptor"]
        # Use standard T3K mesh graph descriptor for the logical topology
        os.environ[
            "TT_MESH_GRAPH_DESC_PATH"
        ] = "./tt_metal/fabric/mesh_graph_descriptors/t3k_mesh_graph_descriptor.yaml"
    else:
        # Mesh graph descriptor (new format)
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = config["descriptor"]

    mesh_device = ttnn.open_mesh_device(mesh_shape=config["shape"])
    device_ids = mesh_device.get_device_ids()
    mesh_shape = mesh_device.shape

    print("✅ Mesh opened successfully")
    print(f"   Available devices: {device_ids}")
    print(f"   Mesh topology: {mesh_shape[0]}x{mesh_shape[1]}")

    if len(device_ids) > 1:
        print("\n📡 Testing inter-device ethernet connectivity...")
        print("   Creating test tensor...")
        test_data = torch.ones([1, 1, 32, 32], dtype=torch.bfloat16)
        ttnn_tensor = ttnn.from_torch(
            test_data, ttnn.bfloat16, device=mesh_device, mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0)
        )
        print("   Testing inter-device operations...")
        result_tensor = ttnn.add(ttnn_tensor, ttnn_tensor)
        result_data = ttnn.to_torch(result_tensor)
        if torch.allclose(result_data, test_data * 2, rtol=1e-2):
            print("   ✅ Inter-device communication test PASSED!")
        else:
            print("   ❌ Inter-device communication test FAILED!")
    else:
        print("   ℹ️  Single device configuration - no inter-device test needed")

    ttnn.close_mesh_device(mesh_device)
    return True


def main():
    # ---------------- CLI FLAGS ----------------
    parser = argparse.ArgumentParser(
        description="Smart Ethernet Link Detection for TT-Metal",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Auto-detect mesh configuration
  %(prog)s --yaml my_cluster.yaml            # Use custom cluster descriptor
  %(prog)s --yaml my_cluster.yaml --shape 2,2  # Override mesh shape
  %(prog)s --descriptor path.yaml --shape 1,2  # Use mesh graph descriptor
        """,
    )
    parser.add_argument(
        "--yaml", type=str, help="Path to cluster descriptor YAML (from topology tool). Shape will be auto-inferred."
    )
    parser.add_argument("--descriptor", type=str, help="Path to mesh graph descriptor YAML. Requires --shape if used.")
    parser.add_argument(
        "--shape",
        type=str,
        help="Mesh shape as 'rows,cols' (e.g. '2,2'). Auto-inferred for --yaml, required for --descriptor.",
    )
    args = parser.parse_args()

    if args.yaml and args.descriptor:
        print("❌ Error: Cannot use both --yaml and --descriptor. Choose one.")
        sys.exit(1)

    if args.yaml:
        # Use custom cluster descriptor YAML
        print(f"Using custom cluster descriptor: {args.yaml}")
        if not os.path.exists(args.yaml):
            print(f"❌ Error: File not found: {args.yaml}")
            sys.exit(1)

        # Set as cluster descriptor for TTNN
        os.environ["TT_METAL_CLUSTER_DESC_PATH"] = args.yaml

        # Infer or use provided shape
        if args.shape:
            rows, cols = map(int, args.shape.split(","))
            shape = make_mesh_shape(rows, cols)
            print(f"   Using provided shape: {shape}")
        else:
            print("   Attempting to infer mesh shape from YAML...")
            shape = infer_mesh_shape_from_yaml(args.yaml)
            if shape is None:
                print("❌ Error: Could not infer mesh shape. Please provide --shape rows,cols")
                sys.exit(1)

        config = {
            "name": f"Custom cluster ({args.yaml})",
            "descriptor": args.yaml,
            "shape": shape,
            "is_cluster_descriptor": True,  # Old format (physical topology)
        }
    elif args.descriptor:
        # Use mesh graph descriptor
        if not args.shape:
            print("❌ Error: --descriptor requires --shape to be specified")
            sys.exit(1)
        rows, cols = map(int, args.shape.split(","))
        shape = make_mesh_shape(rows, cols)
        config = {"name": "Custom mesh", "descriptor": args.descriptor, "shape": shape}
    else:
        # Auto-detect
        print("No custom YAML provided, attempting auto-detection...")
        config = detect_and_test_mesh()

    if not config:
        print("\n❌ No valid mesh found. Exiting.")
        print("\nTry:")
        print("  1. Generate topology: ./build/tools/umd/topology -f my_cluster.yaml")
        print("  2. Use custom YAML: ./simple_link_check.py --yaml my_cluster.yaml")
        sys.exit(1)

    print("\n🎉 BEST CONFIGURATION FOUND:")
    print(f"   {config['name']}")
    print(f"   Descriptor: {config['descriptor']}")
    print(f"   Shape: {config['shape']}")

    success = test_ethernet(config)

    print("\n" + "=" * 45)
    print("FINAL RESULT:")
    if success:
        print("🎉 Ethernet links are working correctly!")
        print("   Your TT-Metal mesh is ready for multi-device workloads")
    else:
        print("⚠️  Ethernet connectivity issues detected")
        print("   Check physical connections and mesh descriptor configuration")


if __name__ == "__main__":
    main()
