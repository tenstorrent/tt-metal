# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Fresh stock capture for installed package tests; caller owns all locks.

No compiler, fitter, build, checkout mutation, golden fixture or qualification
decision lives here. Prepared binary/build-flag and physical-device identity
are explicit caller responsibilities; equal git bases alone do not prove them.
Wheel installations may supply TTPOLY_TEST_STOCK_PYTHON and
TTPOLY_TEST_STOCK_BUILD_DIR; source-tree paths remain the defaults.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from ttpoly.ttmetal_public_tests import MANIFEST_PATH


def _inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe installed package path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("installed package path escapes package root")
    return path


def upstream_base_selection(stock_root, revision, operation, architecture):
    """Resolve the committed upstream implementation; never waive JIT checks."""
    from ttpoly.ttmetal_public_tests import resolve_test_entry

    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("upstream comparison requires an explicit full base revision")
    git = ["git", "-C", str(stock_root)]
    if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != revision:
        raise ValueError("stock HEAD differs from upstream comparison base")
    if subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("upstream stock has dirty tracked sources")
    paths = subprocess.check_output([*git, "ls-tree", "--name-only", "HEAD", "--", MANIFEST_PATH], text=True)
    if not paths.strip():
        return None
    manifest = json.loads(subprocess.check_output([*git, "show", "HEAD:" + MANIFEST_PATH], text=True))
    if manifest.get("schema") != 2 or manifest.get("test_runtime") != "packaged_canonical_runtime":
        raise ValueError("unsupported committed upstream package manifest")
    operation_entry = manifest["operations"].get(operation)
    if operation_entry is None or architecture not in manifest["architectures"]:
        return None
    if "targets" in operation_entry and architecture not in operation_entry["targets"]:
        return None
    entry = resolve_test_entry(manifest, operation, architecture)
    defines = entry["expected_defines"]
    config = defines["TT_POLY_SELECTED_CONFIG_HEADER"].strip('"')
    config_path = _inside(stock_root, config)
    cpp = subprocess.check_output([*git, "show", "HEAD:" + config], text=True)
    if config_path.read_text() != cpp:
        raise ValueError("upstream selected configuration differs from committed base")
    metadata = _inside(stock_root, entry["directory"] + "/kernel.cpp")
    if metadata.read_text() != cpp:
        raise ValueError("upstream package configuration metadata is stale")
    return cpp, defines


def capture_stock_baseline(
    *,
    package_root,
    stock_root,
    operation,
    output_root,
    chip,
    architecture,
    visible_devices,
    caller_holds_locks=False,
    prepared_builds_verified=False,
    upstream_base_revision=None,
):
    from ttpoly.bf16_scoring import _load_compiled_semantics
    from ttpoly.ttmetal_public_invocation import PublicInvocation
    from ttpoly.ttmetal_dispatch import verify_public_stock_dispatch
    from ttpoly.ttmetal_public_tests import resolve_test_entry, prepared_execution_paths

    if not caller_holds_locks or not prepared_builds_verified:
        raise ValueError("caller must hold both build-reader/device locks and verify prepared binary/build contracts")
    target = {"blackhole": "bh", "wormhole_b0": "wh"}.get(architecture)
    if target is None or type(chip) is not int or chip < 0 or not visible_devices.strip():
        raise ValueError("explicit supported architecture, chip and nonempty device mapping required")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", operation):
        raise ValueError("invalid operation")
    package_root, stock_root = Path(package_root).resolve(strict=True), Path(stock_root).resolve(strict=True)
    if package_root == stock_root:
        raise ValueError("distinct installed package and prepared stock root required")
    python, build = prepared_execution_paths(stock_root, "stock")
    manifest = json.loads((package_root / MANIFEST_PATH).read_text())
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != 2
        or manifest.get("test_runtime") != "packaged_canonical_runtime"
    ):
        raise ValueError("installed portable package contract is unavailable")
    support = _inside(package_root, manifest["support_directory"])
    if Path(__file__).resolve().parents[1] != support:
        raise ValueError("run the baseline helper from this package's installed support directory")
    entry = resolve_test_entry(manifest, operation, architecture)
    data = _inside(package_root, entry["directory"])
    semantics = _load_compiled_semantics(data / "semantic.json", activation=operation, precision="bf16")
    io = semantics.semantic_io_contract
    if semantics.target.architecture != target or semantics.semantic_profile != "ttnn":
        raise ValueError("baseline requires matching installed BF16 TTNN target semantics")
    invocation_data = json.loads((data / "invocation.json").read_text())
    invocation = PublicInvocation(**invocation_data)
    if tuple(invocation.tensor_roles) != io.tensor_input_roles:
        raise ValueError("public input roles disagree with the semantic IO contract")
    expected_scope = "unit_gradient_one" if io.fuse_grad else "unary"
    if entry.get("gradient_scope") != expected_scope:
        raise ValueError("package gradient scope disagrees with typed IO")
    revisions = [
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, text=True, capture_output=True
        ).stdout.strip()
        for root in (package_root, stock_root)
    ]
    if upstream_base_revision is None and revisions[0] != revisions[1]:
        raise ValueError("candidate package and stock must share the same TT-Metal base revision")
    base_selection = None
    if upstream_base_revision is not None:
        if revisions[1] != upstream_base_revision:
            raise ValueError("stock base differs from upstream comparison base")
        git = ["git", "-C", str(package_root)]
        ancestor = subprocess.run(
            [*git, "merge-base", "--is-ancestor", upstream_base_revision, "HEAD"], capture_output=True
        )
        if ancestor.returncode:
            raise ValueError("candidate is not descended from upstream comparison base")
        clean = subprocess.run(
            [*git, "status", "--porcelain", "--untracked-files=no"], check=True, text=True, capture_output=True
        )
        if clean.stdout.strip():
            raise ValueError("upstream candidate has dirty tracked sources")
        base_selection = upstream_base_selection(stock_root, upstream_base_revision, operation, architecture)
    output = Path(output_root).resolve() / operation
    output.mkdir(parents=True, exist_ok=False)
    for name in ("semantic.json", "invocation.json"):
        (output / name).write_bytes((data / name).read_bytes())
    cache = output / "stock-cache"
    env = dict(
        os.environ,
        TT_VISIBLE_DEVICES=visible_devices,
        TT_METAL_HOME=str(stock_root),
        TT_METAL_RUNTIME_ROOT=str(stock_root),
        TT_METAL_BUILD_DIR=str(build),
        TT_METAL_CACHE=str(cache),
        PYTHONPATH=f"{stock_root}:{stock_root / 'ttnn'}:{support}",
    )
    for name in (
        "TT_METAL_DEVICE_HOME",
        "TT_METAL_COMPILE_HOME",
        "TT_METAL_INSTALL_ROOT",
        "TT_METAL_JIT_SERVER_CACHE_ROOT",
    ):
        env.pop(name, None)
    command = [
        str(python),
        "-m",
        "ttpoly.public_test_baseline",
        "--capture-leg",
        str(stock_root),
        str(output),
        str(chip),
        architecture,
    ]
    with (output / "stock.log").open("w") as log:
        subprocess.run(command, cwd=stock_root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    if (output / "stock.bf16").stat().st_size != 131072:
        raise ValueError("stock capture does not contain exactly 65536 BF16 words")
    if not list(cache.rglob("defines_generated.h")):
        raise ValueError("stock capture has no retained JIT dispatch evidence")
    if base_selection is None:
        verify_public_stock_dispatch(cache)
    else:
        from ttpoly.ttmetal_dispatch import verify_public_jit_selection

        cpp, defines = base_selection
        verify_public_jit_selection(cache, cpp, defines, root=stock_root)
    if io.fuse_grad:
        expected = {
            "io_contract": io.to_dict(),
            "auxiliary_inputs": [
                {
                    "tensor_index": io.gradient_input_index,
                    "role": "incoming_gradient",
                    "encoding": "constant_bfloat16",
                    "raw_bits": "0x3f80",
                    "value": 1.0,
                    "count": 65536,
                }
            ],
        }
        record = json.loads((output / "stock.inputs.json").read_text())
        unit = (
            record.get("auxiliary_inputs", [None])[0]
            if isinstance(record, dict) and record.get("auxiliary_inputs")
            else None
        )
        if (
            record != expected
            or not isinstance(unit, dict)
            or type(unit.get("count")) is not int
            or type(unit.get("tensor_index")) is not int
            or isinstance(unit.get("value"), bool)
        ):
            raise ValueError("stock capture lacks the exact completed unit-gradient attestation")
    report = {
        "operation": operation,
        "status": "fresh_stock_capture_complete",
        "count": 65536,
        "architecture": architecture,
        "chip": chip,
        "visible_devices": visible_devices,
        "gradient_scope": expected_scope,
        "pr_ready": False,
        "build_contract": (
            "upstream_base_and_candidate_ancestry_checked; binary_flags_and_physical_mapping_caller_verified"
            if upstream_base_revision is not None
            else "same_base_revision_checked; binary_flags_and_physical_mapping_caller_verified"
        ),
        "baseline_scope": "actual_upstream_base" if upstream_base_revision is not None else "native_stock",
    }
    (output / "stock-capture.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("package-root", "stock-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--operation", required=True)
    parser.add_argument("--chip", type=int, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--visible-devices", required=True)
    parser.add_argument("--caller-holds-locks", action="store_true")
    parser.add_argument("--prepared-builds-verified", action="store_true")
    parser.add_argument("--upstream-base-revision")
    args = parser.parse_args(argv)
    try:
        result = capture_stock_baseline(**vars(args))
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(2, str(error) + "\n")
    print(
        f"{result['operation']}: fresh stock capture 65536 BF16 encodings; {result['gradient_scope']}; not qualification"
    )
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 6 and sys.argv[1] == "--capture-leg":
        from types import SimpleNamespace
        import ttnn

        if ttnn.get_arch_name() != sys.argv[5]:
            raise RuntimeError("actual device architecture differs from package contract")
        from ttpoly.ttmetal_public_capture import leg

        stock, output = Path(sys.argv[2]), Path(sys.argv[3])
        leg(
            SimpleNamespace(
                root=stock, leg=output / "invocation.json", chip=int(sys.argv[4]), out_bf16_bin=output / "stock.bf16"
            )
        )
    else:
        raise SystemExit(main())
