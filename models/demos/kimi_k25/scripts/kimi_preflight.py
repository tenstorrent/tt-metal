#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""kimi_preflight.py — Kimi K2.5 hardware readiness checker.

Run this *before* launching the M5 hardware gate test to catch configuration
problems early.  No hardware is touched — only environment, imports, and
config validation are checked.

Usage::

    cd tt-metal
    python models/demos/kimi_k25/scripts/kimi_preflight.py
    python models/demos/kimi_k25/scripts/kimi_preflight.py --mesh-device T3K
    python models/demos/kimi_k25/scripts/kimi_preflight.py --verbose

Exit codes:
    0  All critical checks passed — safe to run hardware tests.
    1  One or more critical checks failed — fix before proceeding.
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PASS = "\033[32m✓\033[0m"
_FAIL = "\033[31m✗\033[0m"
_WARN = "\033[33m⚠\033[0m"
_INFO = "\033[36mℹ\033[0m"


def _ok(msg: str) -> None:
    print(f"  {_PASS}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_FAIL}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_WARN}  {msg}")


def _info(msg: str) -> None:
    print(f"  {_INFO}  {msg}")


# ---------------------------------------------------------------------------
# Check functions — each returns True (pass) or False (fail)
# ---------------------------------------------------------------------------


def check_python_version() -> bool:
    """Python ≥ 3.9 required."""
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 9:
        _ok(f"Python {major}.{minor} (≥3.9 required)")
        return True
    _fail(f"Python {major}.{minor} — need ≥ 3.9")
    return False


def check_tenstorrent_devices() -> bool:
    """At least one /dev/tenstorrent/{N} device must be present."""
    dev_root = "/dev/tenstorrent"
    if not os.path.exists(dev_root):
        _fail(f"{dev_root} not present — no Tenstorrent hardware detected")
        return False
    try:
        devices = [d for d in os.listdir(dev_root) if d.isdigit()]
    except PermissionError:
        _warn(f"{dev_root} exists but not readable (permission issue)")
        return False
    if not devices:
        _fail(f"{dev_root} exists but no numeric device nodes found")
        return False
    _ok(f"Tenstorrent devices: {', '.join(sorted(devices, key=int))} (in {dev_root})")
    return True


def check_mesh_device_env(requested: str | None) -> bool:
    """MESH_DEVICE env var is set to a supported topology."""
    supported = {"TG", "T3K", "DUAL", "QUAD", "N300", "N150"}
    env_val = os.environ.get("MESH_DEVICE", "")

    # If user passed --mesh-device, honour it (also set env for subprocess)
    if requested:
        os.environ["MESH_DEVICE"] = requested
        env_val = requested

    if not env_val:
        _fail("MESH_DEVICE env var not set.  Export it before running tests:")
        _info("  export MESH_DEVICE=TG     # Galaxy (4×8 = 32 chips)")
        _info("  export MESH_DEVICE=T3K    # T3K (1×8 = 8 chips — viable alternative)")
        return False

    if env_val not in supported:
        _warn(f"MESH_DEVICE={env_val!r} — unrecognised topology (supported: {', '.join(sorted(supported))})")
        return False

    if env_val in {"N150", "N300"}:
        _fail(
            f"MESH_DEVICE={env_val} — NOT viable for Kimi K2.5.\n"
            "    N150 requires ~33.8 GB DRAM (only 12 GB available).\n"
            "    N300 requires ~16.9 GB DRAM (only 12 GB available).\n"
            "    Use TG (Galaxy) or T3K instead."
        )
        return False

    _ok(f"MESH_DEVICE={env_val}")
    return True


def check_kimi_hf_model_env(verbose: bool) -> bool:
    """KIMI_HF_MODEL env var — optional but recommended for real-weight tests."""
    env_val = os.environ.get("KIMI_HF_MODEL", "")
    if not env_val:
        _warn("KIMI_HF_MODEL not set — real-weight tests will be skipped.")
        _info("  Random-weight smoke test (test_forward_pass) does NOT require this.")
        _info("  Set KIMI_HF_MODEL=/workspace/extra/Kimi-K2.5 for PCC tests.")
        return True  # Not a hard failure — smoke tests work without real weights.

    if not os.path.isdir(env_val):
        _fail(f"KIMI_HF_MODEL={env_val!r} — directory does not exist")
        return False

    # Check for index file
    index = os.path.join(env_val, "model.safetensors.index.json")
    if os.path.exists(index):
        _ok(f"KIMI_HF_MODEL={env_val!r} (safetensors index found)")
    else:
        _warn(f"KIMI_HF_MODEL={env_val!r} — directory exists but index not found")

    if verbose:
        try:
            shard_count = len([f for f in os.listdir(env_val) if f.endswith(".safetensors")])
            _info(f"  Safetensors shards: {shard_count} (expected 64 for full Kimi K2.5)")
        except Exception:
            pass

    return True


def check_kimi_imports() -> bool:
    """Import kimi_k25 Python modules — validates the package is importable."""
    # Add tt-metal root to path if not already there (useful for direct script invocation).
    # Path: scripts/ -> kimi_k25/ -> demos/ -> models/ -> tt-metal/ (5 levels up)
    _repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    )
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)

    # Modules that may depend on loguru/ttnn (dev-container only deps)
    _DEV_CONTAINER_DEPS = frozenset({"loguru", "ttnn", "torch_ttnn"})

    modules = [
        ("models.demos.kimi_k25.utils.config_adapter", "KimiK25Config"),
        ("models.demos.kimi_k25.utils.int4_dequantize", "dequantize_int4_weight"),
        ("models.demos.kimi_k25.utils.weight_loader", "KimiLazyStateDict"),
        ("models.demos.kimi_k25.tt.kimi_model", "KimiGenerator"),
    ]
    all_ok = True
    for mod_name, symbol in modules:
        try:
            mod = __import__(mod_name, fromlist=[symbol])
            getattr(mod, symbol)
            _ok(f"import {mod_name}.{symbol}")
        except ImportError as exc:
            missing = str(exc).split("'")[-2] if "'" in str(exc) else str(exc)
            if missing in _DEV_CONTAINER_DEPS:
                _warn(
                    f"import {mod_name}: missing '{missing}' — OK outside dev container; "
                    "will work in tt-metal dev container"
                )
                # Not a hard failure — only the dev container has all deps installed
            else:
                _fail(f"import {mod_name}: {exc}")
                all_ok = False
        except AttributeError as exc:
            _fail(f"{mod_name}.{symbol} missing: {exc}")
            all_ok = False
    return all_ok


def check_config_validation() -> bool:
    """KimiK25Config.from_fixture() succeeds and validates all reference values."""
    try:
        from models.demos.kimi_k25.utils.config_adapter import KimiK25Config

        cfg = KimiK25Config.from_fixture()
        # Spot-check critical values (use actual attribute names from KimiK25Config)
        assert cfg.n_routed_experts == 384, f"experts={cfg.n_routed_experts} (expected 384)"
        assert cfg.num_attention_heads == 64, f"heads={cfg.num_attention_heads} (expected 64)"
        assert cfg.vocab_size == 163840, f"vocab={cfg.vocab_size} (expected 163840)"
        assert cfg.n_group == 1, f"n_group={cfg.n_group} (expected 1)"
        qc = cfg.quantization_config
        assert "weight_block_size" in qc, "quantization_config missing weight_block_size"
        _ok(f"KimiK25Config: {cfg.n_routed_experts} experts, {cfg.num_attention_heads} heads, vocab={cfg.vocab_size}")
        return True
    except Exception as exc:
        _fail(f"KimiK25Config.from_fixture() failed: {exc}")
        return False


def check_ttnn_importable() -> bool:
    """ttnn is importable (required for actual hardware tests)."""
    try:
        import ttnn  # noqa: F401

        _ok("ttnn importable")
        return True
    except ImportError as exc:
        _warn(f"ttnn not importable: {exc}")
        _info("  ttnn is required to run hardware tests.")
        _info("  Use the tt-metal dev container: source build/python_env/activate")
        return False  # Not hard-failing — just a warning for HW runs


def check_torch_available() -> bool:
    """PyTorch is available — needed by some test classes (TestKimiFullModelReference)."""
    try:
        import torch  # noqa: F401

        _ok(f"PyTorch {torch.__version__} available")
        return True
    except ImportError:
        _warn("PyTorch not available — TestKimiFullModelReference tests will be skipped")
        _info("  Install in dev container: pip install torch")
        return True  # Not a hard failure


# ---------------------------------------------------------------------------
# Test command printer
# ---------------------------------------------------------------------------

_HW_COMMANDS = {
    "TG": (
        "MESH_DEVICE=TG pytest models/demos/kimi_k25/tests/test_kimi_generate.py \\\n"
        "  -k 'test_forward_pass or test_pcc_correctness_random_weights' \\\n"
        "  -v 2>&1 | tee /tmp/kimi_m5_hw.log"
    ),
    "T3K": (
        "MESH_DEVICE=T3K pytest models/demos/kimi_k25/tests/test_kimi_generate.py \\\n"
        "  -k 'test_forward_pass[mode_decode_seq_1_batch_32]' \\\n"
        "  -v 2>&1 | tee /tmp/kimi_t3k_forward.log"
    ),
    "DUAL": (
        "MESH_DEVICE=DUAL pytest models/demos/kimi_k25/tests/test_kimi_generate.py \\\n"
        "  -k 'test_forward_pass' -v 2>&1 | tee /tmp/kimi_dual_forward.log"
    ),
    "QUAD": (
        "MESH_DEVICE=QUAD pytest models/demos/kimi_k25/tests/test_kimi_generate.py \\\n"
        "  -k 'test_forward_pass' -v 2>&1 | tee /tmp/kimi_quad_forward.log"
    ),
}

_CPU_COMMAND = (
    "# CPU-only tests (no hardware needed — import + config + mock-device):\n"
    "pytest models/demos/kimi_k25/tests/ \\\n"
    "  -k 'not Hardware and not TG and not test_forward_pass' \\\n"
    "  -v 2>&1 | tee /tmp/kimi_cpu.log"
)


def print_commands(mesh_device: str | None) -> None:
    print()
    print("=" * 66)
    print("  RECOMMENDED COMMANDS")
    print("=" * 66)
    print()

    # CPU first
    print("  [CPU-only — always safe, no hardware needed]")
    print(f"  {_CPU_COMMAND}")
    print()

    # Hardware
    print("  [Hardware — run after CPU tests pass]")
    if mesh_device and mesh_device in _HW_COMMANDS:
        print(f"  cd tt-metal && git checkout brain/kimi")
        print(f"  {_HW_COMMANDS[mesh_device]}")
    else:
        # Show TG (preferred) then T3K (alternative)
        print(f"  # Galaxy (preferred):")
        print(f"  cd tt-metal && git checkout brain/kimi")
        print(f"  {_HW_COMMANDS['TG']}")
        print()
        print(f"  # T3K (alternative if TG unavailable):")
        print(f"  cd tt-metal && git checkout brain/kimi")
        print(f"  {_HW_COMMANDS['T3K']}")
    print()
    print("  Log triage tips:")
    print("    grep -E 'PASSED|FAILED|ERROR' /tmp/kimi_m5_hw.log")
    print("    grep -i 'nan\\|inf\\|assert' /tmp/kimi_m5_hw.log | head -20")
    print()
    print("  Share log with BrAIn (paste output or /tmp/*.log path).")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kimi K2.5 hardware readiness preflight checker.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python models/demos/kimi_k25/scripts/kimi_preflight.py\n"
            "  python models/demos/kimi_k25/scripts/kimi_preflight.py --mesh-device T3K\n"
            "  python models/demos/kimi_k25/scripts/kimi_preflight.py --verbose\n"
        ),
    )
    parser.add_argument(
        "--mesh-device",
        metavar="TOPOLOGY",
        help="Override MESH_DEVICE env var (e.g. TG, T3K, DUAL).",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show extra detail.")
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color codes (useful for CI logs).",
    )
    args = parser.parse_args()

    if args.no_color:
        global _PASS, _FAIL, _WARN, _INFO
        _PASS = "[PASS]"
        _FAIL = "[FAIL]"
        _WARN = "[WARN]"
        _INFO = "[INFO]"

    print()
    print("=" * 66)
    print("  Kimi K2.5 — Hardware Readiness Preflight")
    print("=" * 66)
    print()

    # Critical checks (failures block hardware test)
    critical_failures: list[str] = []

    print("  [Environment]")
    if not check_python_version():
        critical_failures.append("Python version")

    has_hw = check_tenstorrent_devices()
    # Not counting as critical — could be running on headnode
    if not has_hw:
        print(f"  {_WARN}  No Tenstorrent devices (OK if on headnode; needed on compute node)")

    mesh_ok = check_mesh_device_env(args.mesh_device)
    if not mesh_ok:
        critical_failures.append("MESH_DEVICE")

    check_kimi_hf_model_env(args.verbose)  # warning only

    print()
    print("  [Kimi K2.5 package]")
    if not check_kimi_imports():
        critical_failures.append("kimi module imports")

    if not check_config_validation():
        critical_failures.append("KimiK25Config validation")

    print()
    print("  [Runtime]")
    ttnn_ok = check_ttnn_importable()
    if not ttnn_ok:
        critical_failures.append("ttnn not importable (hardware tests will fail)")

    check_torch_available()

    # Summary
    print()
    print("=" * 66)
    if critical_failures:
        print(f"  {_FAIL}  PREFLIGHT FAILED — fix before running hardware tests:")
        for item in critical_failures:
            print(f"        • {item}")
        print("=" * 66)
        print_commands(args.mesh_device or os.environ.get("MESH_DEVICE"))
        return 1
    else:
        mesh = args.mesh_device or os.environ.get("MESH_DEVICE", "?")
        print(f"  {_PASS}  PREFLIGHT PASSED — ready to run hardware tests (MESH_DEVICE={mesh})")
        print("=" * 66)
        print_commands(args.mesh_device or os.environ.get("MESH_DEVICE"))
        return 0


if __name__ == "__main__":
    sys.exit(main())
