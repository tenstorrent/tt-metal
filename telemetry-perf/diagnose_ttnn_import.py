#!/usr/bin/env python3
"""
Diagnose how ttnn module works on this system.
"""

import sys

sys.path.insert(0, "/localdev/kkfernandez/tt-metal")

# Try different imports
print("=== Trying different ttnn imports ===")

# Method 1: Direct ttnn
try:
    import ttnn

    print(f"✓ import ttnn works")
    print(f"  Type: {type(ttnn)}")
    print(f"  File: {getattr(ttnn, '__file__', 'N/A')}")
    print(f"  Attributes ({len(dir(ttnn))}): {dir(ttnn)[:20]}")
except Exception as e:
    print(f"✗ import ttnn failed: {e}")

# Method 2: Check if ttnn is a lazy loader
try:
    import ttnn

    # Force lazy loading
    _ = ttnn.CreateDevice
    print(f"✓ ttnn.CreateDevice accessible after import")
except AttributeError as e:
    print(f"✗ ttnn.CreateDevice not available: {e}")
except Exception as e:
    print(f"✗ Other error: {e}")

# Method 3: Check ttnn submodules
try:
    import ttnn

    submodules = [attr for attr in dir(ttnn) if not attr.startswith("_")]
    print(f"  Non-private attributes: {submodules}")

    # Try to access each
    for attr in submodules[:5]:
        try:
            obj = getattr(ttnn, attr)
            print(f"    {attr}: {type(obj)}")
        except:
            pass
except Exception as e:
    print(f"✗ Error checking submodules: {e}")

# Method 4: Check pytest environment
print("\n=== Checking if pytest sets up ttnn ===")
try:
    # Maybe pytest does some magic
    import pytest

    print("✓ pytest available")
except:
    print("✗ pytest not available")

# Method 5: Look for working test example
print("\n=== Looking for working test example ===")
import glob

test_files = glob.glob("/localdev/kkfernandez/tt-metal/tests/ttnn/unit_tests/operations/*.py")[:3]
for tf in test_files:
    print(f"\nChecking {tf}...")
    with open(tf, "r") as f:
        lines = f.readlines()[:30]
        for i, line in enumerate(lines):
            if "CreateDevice" in line or "close_device" in line:
                print(f"  Line {i+1}: {line.strip()}")

# Method 6: Check if we can run a test directly
print("\n=== Trying to run a simple test ===")
import subprocess

result = subprocess.run(
    ["pytest", "tests/ttnn/unit_tests/operations/test_add.py::test_add", "-v", "--collect-only"],
    cwd="/localdev/kkfernandez/tt-metal",
    capture_output=True,
    text=True,
    timeout=10,
)
print("Pytest collect output (first 500 chars):")
print(result.stdout[:500])
if result.stderr:
    print("Errors:")
    print(result.stderr[:500])

# Method 7: Try importing from ttnn internals
print("\n=== Checking ttnn internals ===")
try:
    import ttnn
    import inspect

    # Get the module path
    if hasattr(ttnn, "__path__"):
        print(f"ttnn is a package at: {ttnn.__path__}")

    # Check for submodules
    if hasattr(ttnn, "__all__"):
        print(f"ttnn.__all__: {ttnn.__all__}")

    # Try to find CreateDevice in any submodule
    print("\nSearching for CreateDevice...")
    for name in dir(ttnn):
        if not name.startswith("_"):
            try:
                attr = getattr(ttnn, name)
                if hasattr(attr, "CreateDevice"):
                    print(f"  Found CreateDevice in ttnn.{name}")
                if inspect.ismodule(attr):
                    sub_attrs = dir(attr)
                    if "CreateDevice" in sub_attrs:
                        print(f"  Found CreateDevice in ttnn.{name}")
                        print(f"    Usage: ttnn.{name}.CreateDevice")
            except:
                pass

except Exception as e:
    print(f"Error: {e}")

print("\n=== Summary ===")
print(
    """
If ttnn doesn't have CreateDevice directly, you may need to:
1. Run tests through pytest (it may set up the environment)
2. Use a different import path
3. Check if there's an initialization step required
"""
)
