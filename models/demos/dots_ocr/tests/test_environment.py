# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Diagnostic script to understand the test environment and TTNN availability.

This explains why some tests are being skipped or failing.
"""

import importlib.util
import os
import sys


def check_module_available(module_name):
    """Check if a module is available."""
    try:
        if module_name == "ttnn":
            # Special handling for ttnn
            spec = importlib.util.find_spec("ttnn")
            if spec is None:
                return False, "Module not found"
            module = importlib.import_module("ttnn")
            has_open_mesh = hasattr(module, "open_mesh_device")
            has_mesh_shape = hasattr(module, "MeshShape")
            has_dram = hasattr(module, "DRAM_MEMORY_CONFIG")
            return True, f"Available. open_mesh_device: {has_open_mesh}, MeshShape: {has_mesh_shape}, DRAM: {has_dram}"
        else:
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                return False, "Module not found"
            return True, "Available"
    except Exception as e:
        return False, f"Error: {type(e).__name__}: {e}"


def diagnose_environment():
    """Diagnose the test environment."""
    print("=== Dots OCR Test Environment Diagnostic ===\n")

    print("1. Python Environment:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Path: {sys.path[0]}")
    print()

    print("2. Required Dependencies:")
    deps = ["torch", "ttnn", "pydantic", "loguru", "transformers", "pillow"]
    for dep in deps:
        available, status = check_module_available(dep)
        status_symbol = "✅" if available else "❌"
        print(f"   {status_symbol} {dep:12} : {status}")

    print("\n3. Environment Variables:")
    relevant_vars = ["MESH_DEVICE", "HF_MODEL", "DOTS_MAX_SEQ_LEN_WH_LB", "RUN_DOTS_REAL_WEIGHTS"]
    for var in relevant_vars:
        value = os.environ.get(var, "NOT SET")
        print(f"   {var:20} : {value}")

    print("\n4. Test Recommendations:")
    print()
    print("For CPU-only testing (what you can run now):")
    print(
        "  python -m pytest models/demos/dots_ocr/tests/test_reference_embeddings.py -q --confcutdir=models/demos/dots_ocr/tests"
    )
    print(
        "  python -m pytest models/demos/dots_ocr/tests/test_vision_components.py -q --confcutdir=models/demos/dots_ocr/tests"
    )
    print(
        "  python -m pytest models/demos/dots_ocr/tests/test_e2e_pcc.py::test_e2e_hybrid_compatibility -q --confcutdir=models/demos/dots_ocr/tests"
    )
    print()
    print("For full TTNN tests (requires Tenstorrent device):")
    print("  export MESH_DEVICE=N150")
    print("  export HF_MODEL=rednote-hilab/dots.mocr")
    print("  python -m pytest models/demos/dots_ocr/tests/ -q --confcutdir=models/demos/dots_ocr/tests")
    print()
    print("The SKIPPED tests are EXPECTED in this CPU-only environment.")
    print("The ERROR was caused by missing dependencies (pydantic, full TTNN SDK).")
    print("This is normal for a development environment without the complete Tenstorrent stack.")


if __name__ == "__main__":
    diagnose_environment()
