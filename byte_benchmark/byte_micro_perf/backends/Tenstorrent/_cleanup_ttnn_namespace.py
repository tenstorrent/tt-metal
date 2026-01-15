"""
Helper module to clean up ttnn's namespace pollution.
Import this BEFORE importing from core.ops to ensure clean imports.
"""
import sys

# Remove ttnn's core module from sys.modules if present
if "core" in sys.modules:
    core_module = sys.modules["core"]
    if hasattr(core_module, "__file__"):
        core_file = str(core_module.__file__)
        if "ttnn" in core_file:
            del sys.modules["core"]
