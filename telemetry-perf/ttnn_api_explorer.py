#!/usr/bin/env python3
"""
Explore the ttnn API to find how to use devices on this system.
"""

import sys

print("Exploring ttnn API...")

# Import ttnn
try:
    import ttnn

    print("✓ ttnn imported successfully")
except ImportError as e:
    print(f"✗ Failed to import ttnn: {e}")
    sys.exit(1)

# Get all attributes
attrs = dir(ttnn)
print(f"\nTotal attributes in ttnn: {len(attrs)}")

# Look for device-related
print("\n=== Device-related attributes ===")
device_attrs = [a for a in attrs if "device" in a.lower()]
for attr in device_attrs[:20]:
    print(f"  - {attr}")

# Look for common operations
print("\n=== Common operations ===")
ops = ["add", "multiply", "matmul", "from_torch", "to_torch", "open", "close", "create"]
for op in ops:
    matching = [a for a in attrs if op in a.lower()]
    if matching:
        print(f"{op}: {matching[:5]}")

# Check for classes
print("\n=== Classes/Types ===")
classes = [a for a in attrs if a[0].isupper() and not a.startswith("_")]
print(f"Found {len(classes)} classes/types")
for cls in classes[:20]:
    print(f"  - {cls}")

# Check if there's a device manager or context
print("\n=== Checking for device management ===")

# Try tt_lib (older API)
try:
    import tt_lib

    print("✓ tt_lib is available")
    tt_attrs = dir(tt_lib)
    device_funcs = [a for a in tt_attrs if "device" in a.lower()]
    print(f"  tt_lib device functions: {device_funcs[:10]}")
except ImportError:
    print("✗ tt_lib not available")

# Try tt_eager (another possible API)
try:
    import tt_eager

    print("✓ tt_eager is available")
    eager_attrs = dir(tt_eager)
    device_funcs = [a for a in eager_attrs if "device" in a.lower()]
    print(f"  tt_eager device functions: {device_funcs[:10]}")
except ImportError:
    print("✗ tt_eager not available")

# Check for a global device or mesh
print("\n=== Checking for global device state ===")
try:
    # Sometimes devices are managed globally
    import tt_metal as ttm

    print("✓ tt_metal is available")
    ttm_attrs = dir(ttm)
    device_funcs = [a for a in ttm_attrs if "device" in a.lower() or "Device" in a]
    print(f"  tt_metal device functions: {device_funcs[:10]}")

    # Try to create a device
    if "CreateDevice" in ttm_attrs:
        print("  Found CreateDevice in tt_metal")
    if "Device" in ttm_attrs:
        print("  Found Device class in tt_metal")

except ImportError:
    print("✗ tt_metal not available")

# Check environment
print("\n=== Environment Check ===")
import os

print(f"TT_METAL_HOME: {os.environ.get('TT_METAL_HOME', 'NOT SET')}")
print(f"PYTHONPATH includes tt-metal: {any('tt-metal' in p for p in sys.path)}")

# Try to find example usage
print("\n=== Looking for example files ===")
import glob
import os

tt_metal_home = os.environ.get("TT_METAL_HOME", "/localdev/kkfernandez/tt-metal")
example_patterns = [
    f"{tt_metal_home}/tests/ttnn/unit_tests/**/*.py",
    f"{tt_metal_home}/models/demos/**/*.py",
    f"{tt_metal_home}/tests/tt_eager/**/*.py",
]

for pattern in example_patterns:
    files = glob.glob(pattern, recursive=True)
    if files:
        print(f"\nFound {len(files)} files matching {pattern}")
        # Look for device usage in first file
        with open(files[0], "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if "device" in line.lower() and ("=" in line or "open" in line or "create" in line):
                    print(f"  Example from {os.path.basename(files[0])} line {i+1}:")
                    print(f"    {line.strip()}")
                    if i > 0:
                        print(f"    Previous: {lines[i-1].strip()}")
                    break
        break

print("\n=== Trying direct device creation ===")

# Method 1: Try using tt_metal directly
try:
    import tt_metal as ttm

    device_id = 0
    device = ttm.CreateDevice(device_id)
    print(f"✓ SUCCESS: tt_metal.CreateDevice({device_id}) works!")
    print(f"  Device type: {type(device)}")
    ttm.CloseDevice(device)
except Exception as e:
    print(f"✗ tt_metal.CreateDevice failed: {e}")

# Method 2: Try ttnn with specific import
try:
    from ttnn import experimental as ttnn_exp

    device = ttnn_exp.open_device(device_id=0)
    print(f"✓ SUCCESS: ttnn.experimental.open_device works!")
    ttnn_exp.close_device(device)
except Exception as e:
    print(f"✗ ttnn.experimental failed: {e}")

# Method 3: Check for mesh device
try:
    if hasattr(ttnn, "open_mesh_device"):
        print("✓ Found ttnn.open_mesh_device - this system may use mesh API")
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
        print(f"  Mesh device created: {type(mesh)}")
        ttnn.close_mesh_device(mesh)
except Exception as e:
    print(f"✗ Mesh device failed: {e}")

print("\n=== Summary ===")
print("Run a simple test to see what actually works:")
print(
    """
# Test script:
import tt_metal as ttm
device = ttm.CreateDevice(0)
print(f"Device created: {device}")
ttm.CloseDevice(device)
"""
)
