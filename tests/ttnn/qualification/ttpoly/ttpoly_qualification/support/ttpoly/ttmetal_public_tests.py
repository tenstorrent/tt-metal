# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Packaged canonical capture/scoring tests, without a fitter checkout.

Independent compiler-to-installed-source qualification remains a preflight duty;
these tests validate typed invocation, selected JIT configuration and accuracy.
"""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess


TEST_ROOT = "tests/ttnn/qualification/ttpoly"
MANIFEST_PATH = TEST_ROOT + "/ttpoly_qualification/manifest.json"
SUPPORT_ROOT = TEST_ROOT + "/ttpoly_qualification/support"


def _json(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def package_test_files(root: Path, programs: dict) -> dict[str, bytes]:
    """Generate executable tests from the very same compiled program objects."""
    from ttpoly.ttmetal_public_invocation import public_invocation
    from ttpoly.ttmetal_llk_lifecycle import lifecycle_defines, backward_lifecycle_defines
    from ttpoly.public_test_support import qualification_support_files

    if not programs:
        raise ValueError("package tests require at least one program")
    targets = {program.execution_abi.architecture for program in programs.values()}
    if len(targets) != 1 or not targets <= {"bh", "wh"}:
        raise ValueError("generated tests require one supported compiled architecture")
    target = targets.pop()
    architecture = {"bh": "blackhole", "wh": "wormhole_b0"}[target]
    files = {SUPPORT_ROOT + "/" + path: content for path, content in qualification_support_files(programs).items()}
    files[SUPPORT_ROOT + "/ttpoly/ttmetal_public_tests.py"] = Path(__file__).read_bytes()
    files[SUPPORT_ROOT + "/ttpoly/public_test_baseline.py"] = (
        Path(__file__).with_name("public_test_baseline.py").read_bytes()
    )
    entries = {}
    for operation, program in sorted(programs.items()):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", operation):
            raise ValueError("unsafe operation identifier")
        if (program.execution_abi.precision, program.execution_abi.compliance) != ("bf16", "ttnn"):
            raise ValueError("generated tests require the BF16 TTNN contract")
        if (
            target == "wh"
            and getattr(getattr(program.execution_abi, "resources", None), "llk_target_capability", None)
            != "wh_log_square_decoded_nonfinite"
        ):
            raise ValueError("generated WH tests require a supported compiled target capability")
        if not program.scoring_semantics or program.scoring_semantics.get("activation") != operation:
            raise ValueError("test operation differs from compiled semantics")
        invocation = public_invocation(root, operation, program)
        directory = TEST_ROOT + "/ttpoly_qualification/" + operation
        lower = backward_lifecycle_defines if program.execution_abi.io_contract.fuse_grad else lifecycle_defines
        entry = {
            "test": TEST_ROOT + f"/test_{operation}_bf16_exhaustive.py",
            "directory": directory,
            "source_paths": [source.path for source in program.sources],
            "expected_defines": lower(program),
            "gradient_scope": "unit_gradient_one" if program.execution_abi.io_contract.fuse_grad else "unary",
        }
        files[directory + "/semantic.json"] = program.files["semantic.json"]
        files[directory + "/invocation.json"] = _json(asdict(invocation))
        files[directory + "/kernel.cpp"] = program.kernel_cpp.encode()
        files[
            entry["test"]
        ] = f'''# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Exhaustive BF16 test using the packaged canonical reference."""
from pathlib import Path
import sys


def test_{operation}_bf16_exhaustive(tmp_path):
    candidate = Path(__file__).resolve().parents[{len(Path(TEST_ROOT).parts)}]
    runtime = (candidate / {json.dumps(SUPPORT_ROOT)}).resolve(strict=True)
    if not runtime.is_relative_to(candidate):
        raise RuntimeError("packaged runtime escapes candidate root")
    sys.path.insert(0, str(runtime))
    from ttpoly import ttmetal_public_tests

    if not Path(ttmetal_public_tests.__file__).resolve().is_relative_to(runtime):
        raise RuntimeError("wrong packaged canonical runtime imported")
    ttmetal_public_tests.run_package_test({json.dumps(operation)}, tmp_path, candidate_root=candidate, support_root=runtime)
'''.encode()
        entries[operation] = entry
    files[MANIFEST_PATH] = _json(
        {
            "schema": 2,
            "test_runtime": "packaged_canonical_runtime",
            "support_directory": SUPPORT_ROOT,
            "external_fitter_required": False,
            "standalone_ci_ready": False,
            "architectures": [architecture],
            "scope": "exhaustive_bf16_accuracy_only",
            "operations": entries,
        }
    )
    return files


def package_test_files_for_targets(root: Path, programs_by_target: dict) -> dict[str, bytes]:
    """One installed runtime/test per operation, with explicit target data.

    Single-target callers retain the existing manifest and file layout. A
    combined package does not imply either target has passed qualification.
    """
    if not programs_by_target or not set(programs_by_target) <= {"bh", "wh"}:
        raise ValueError("package tests require supported explicit targets")
    for target, programs in programs_by_target.items():
        if not programs or any(program.execution_abi.architecture != target for program in programs.values()):
            raise ValueError("program architecture differs from package target")
    if len(programs_by_target) == 1:
        return package_test_files(root, next(iter(programs_by_target.values())))
    from ttpoly.ttmetal_unary_factory import public_unary_target_program

    files, entries, architectures = {}, {}, []
    manifest = None
    for target in ("bh", "wh"):
        programs = {
            operation: public_unary_target_program(operation, program)
            for operation, program in programs_by_target[target].items()
        }
        target_files = package_test_files(root, programs)
        manifest = json.loads(target_files.pop(MANIFEST_PATH))
        architecture = manifest["architectures"][0]
        architectures.append(architecture)
        for operation, entry in manifest["operations"].items():
            previous = entry["directory"]
            entry["directory"] = TEST_ROOT + "/ttpoly_qualification/" + architecture + "/" + operation
            for name in ("semantic.json", "invocation.json", "kernel.cpp"):
                target_files[entry["directory"] + "/" + name] = target_files.pop(previous + "/" + name)
            shared = entries.setdefault(operation, {"test": entry["test"], "targets": {}})
            shared["targets"][architecture] = entry
        for path, content in target_files.items():
            if path in files and files[path] != content:
                raise ValueError("target packages disagree on shared test support: " + path)
            files[path] = content
    manifest["architectures"] = architectures
    manifest["operations"] = entries
    files[MANIFEST_PATH] = _json(manifest)
    return files


def resolve_test_entry(manifest: dict, operation: str, architecture: str) -> dict:
    """Resolve only the explicitly requested compiled target; never fall back."""
    architectures = manifest.get("architectures")
    if (
        not isinstance(architectures, list)
        or not architectures
        or any(value not in {"blackhole", "wormhole_b0"} for value in architectures)
        or len(set(architectures)) != len(architectures)
        or architecture not in architectures
    ):
        raise ValueError("architecture unavailable: requested architecture differs from compiled package")
    entry = manifest.get("operations", {}).get(operation)
    if not isinstance(entry, dict):
        raise ValueError("operation unavailable in compiled package")
    if "targets" not in entry:
        if architectures != [architecture]:
            raise ValueError("combined package requires explicit operation targets")
        return entry
    targets = entry["targets"]
    if not isinstance(targets, dict) or not set(targets) <= set(architectures):
        raise ValueError("invalid operation targets")
    selected = targets.get(architecture)
    if not isinstance(selected, dict) or selected.get("test") != entry.get("test"):
        raise ValueError("operation unavailable for requested architecture")
    return selected


def _required_directory(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(name + " is required")
    path = Path(value).resolve(strict=True)
    if not path.is_dir():
        raise ValueError(name + " must name a directory")
    return path


def _installed_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("unsafe installed package path")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("installed package path escapes candidate root")
    return path


def _require_installed_runtime(candidate: Path) -> Path:
    runtime = _installed_path(candidate, SUPPORT_ROOT)
    if not Path(__file__).resolve().is_relative_to(runtime):
        raise ValueError("wrong installed canonical runtime imported")
    return runtime


def prepared_execution_paths(root: Path, arm: str) -> tuple[Path, Path]:
    """Resolve prepared source-tree or explicit wheel-CI execution paths.

    Paths select an existing environment; they do not attest build compatibility.
    Keep a venv interpreter's symlink spelling so Python discovers that venv.
    """
    if arm not in {"candidate", "stock"}:
        raise ValueError("prepared execution arm must be candidate or stock")
    prefix = "TTPOLY_TEST_" + arm.upper()
    paths = []
    for name, default, executable in (
        (prefix + "_PYTHON", root / "python_env/bin/python", True),
        (prefix + "_BUILD_DIR", root / "build_Release", False),
    ):
        value = os.environ.get(name)
        path = Path(value) if value is not None else default
        if value is not None and (not value.strip() or not path.is_absolute()):
            raise ValueError(name + " must be an explicit absolute path")
        if executable:
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(name + " must name an existing executable Python file")
            paths.append(path)
        else:
            if not path.is_dir():
                raise ValueError(name + " must name an existing build directory")
            paths.append(path.resolve(strict=True))
    return tuple(paths)


def run_package_test(operation: str, output: Path, *, candidate_root=None, support_root=None) -> dict:
    """Execute fresh candidate outputs and score both arms with one oracle.

    The qualification caller owns build-reader/device locks and provides a fresh
    stock capture for this source/build campaign. This test checks matching typed
    invocation/semantics, not historical provenance or elapsed-time performance.
    Wheel CI may explicitly supply TTPOLY_TEST_CANDIDATE_PYTHON and
    TTPOLY_TEST_CANDIDATE_BUILD_DIR; compatibility remains a caller obligation.
    """
    candidate = (
        Path(candidate_root).resolve(strict=True)
        if candidate_root is not None
        else _required_directory("TTPOLY_TEST_CANDIDATE_ROOT")
    )
    installed_runtime = _installed_path(candidate, SUPPORT_ROOT)
    runtime = Path(support_root).resolve(strict=True) if support_root is not None else installed_runtime
    stock_root = _required_directory("TTPOLY_TEST_STOCK_CAPTURE_ROOT")
    if runtime != installed_runtime:
        raise ValueError("wrong installed canonical runtime location")
    manifest = json.loads(_installed_path(candidate, MANIFEST_PATH).read_text())
    if (
        manifest.get("schema") != 2
        or manifest.get("standalone_ci_ready") is not False
        or manifest.get("test_runtime") != "packaged_canonical_runtime"
        or manifest.get("external_fitter_required") is not False
        or manifest.get("support_directory") != SUPPORT_ROOT
    ):
        raise ValueError("invalid qualification-only test manifest")
    architecture = os.environ.get("TTPOLY_TEST_ARCH")
    entry = resolve_test_entry(manifest, operation, architecture)
    if not _installed_path(candidate, entry["test"]).is_file():
        raise ValueError("generated package test is missing")
    data = _installed_path(candidate, entry["directory"])
    kernel = _installed_path(candidate, str((data / "kernel.cpp").relative_to(candidate)))
    stock = stock_root / operation
    for name in ("semantic.json", "invocation.json"):
        _installed_path(candidate, str((data / name).relative_to(candidate)))
        if (data / name).read_bytes() != (stock / name).read_bytes():
            raise ValueError("stock capture has different typed " + name)
    if (stock / "stock.bf16").stat().st_size != 131072:
        raise ValueError("stock capture lacks all 65536 BF16 outputs")
    from ttpoly.ttmetal_dispatch import verify_public_jit_selection

    # Check installed dependencies, not a tautological source-vs-itself digest.
    # Compiler-source equality is independently checked by qualification preflight.
    for relative in entry["source_paths"]:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe installed source path")
        resolved = (candidate / path).resolve(strict=True)
        if not resolved.is_relative_to(candidate) or not resolved.is_file():
            raise ValueError("missing installed shared source")
    python, build = prepared_execution_paths(candidate, "candidate")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("candidate.bf16", "stock.bf16", "invocation.json", "semantic.json"):
        if (output / name).exists():
            raise ValueError("qualification test requires fresh output files")
    for name in ("semantic.json", "invocation.json"):
        shutil.copyfile(data / name, output / name)
    shutil.copyfile(stock / "stock.bf16", output / "stock.bf16")
    if entry["gradient_scope"] == "unit_gradient_one":
        shutil.copyfile(stock / "stock.inputs.json", output / "stock.inputs.json")
    chip = int(os.environ.get("TTPOLY_TEST_CHIP", "0"))
    if chip < 0:
        raise ValueError("negative device index")
    cache = output / "candidate-cache"
    if cache.exists():
        raise ValueError("qualification test requires a fresh candidate JIT cache")
    env = dict(
        os.environ,
        TT_METAL_HOME=str(candidate),
        TT_METAL_RUNTIME_ROOT=str(candidate),
        TT_METAL_BUILD_DIR=str(build),
        TT_METAL_CACHE=str(cache),
        PYTHONPATH=f"{candidate}:{candidate / 'ttnn'}:{runtime}",
    )
    for name in (
        "TT_METAL_DEVICE_HOME",
        "TT_METAL_COMPILE_HOME",
        "TT_METAL_INSTALL_ROOT",
        "TT_METAL_JIT_SERVER_CACHE_ROOT",
    ):
        env.pop(name, None)
    command = [str(python), "-m", "ttpoly.ttmetal_public_tests", str(candidate), str(output), str(chip), architecture]
    with (output / "candidate.log").open("w") as log:
        subprocess.run(command, cwd=candidate, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    verify_public_jit_selection(cache, kernel.read_text(), entry["expected_defines"], root=candidate)
    from ttpoly.public_accuracy import score_operation
    from ttpoly.groundtruth.spec_context import activation_spec_root
    from ttpoly.groundtruth.reference import require_mpmath_available

    require_mpmath_available()
    with activation_spec_root(runtime / "activations"):
        result = score_operation(output, operation)
    (output / "accuracy.json").write_bytes(_json(result))
    require_accuracy_pass(result)
    return result


def require_accuracy_pass(result: dict) -> None:
    """One strict accuracy predicate for tests and their qualification caller."""
    ours, baseline = result["candidate"], result["stock"]
    ulp = float(ours["max_pure_ulp"])
    baseline_ulp = float(baseline["max_pure_ulp"])
    if (
        result.get("status") != "accuracy_scored"
        or any(type(arm.get("count")) is not int or arm["count"] != 65536 for arm in (ours, baseline))
        or any(isinstance(arm["max_pure_ulp"], bool) for arm in (ours, baseline))
        or any(type(ours.get(name)) is not int for name in ("class_mismatches", "invalid_finite_reference_outputs"))
        or math.isnan(baseline_ulp)
        or baseline_ulp < 0
        or not math.isfinite(ulp)
        or ulp < 0
        or ulp >= 1.0
        or ulp > baseline_ulp
        or ours["class_mismatches"]
        or ours["invalid_finite_reference_outputs"]
        or ours["conformance_status"] != "declared_class_pass"
    ):
        raise AssertionError("generated package failed exhaustive accuracy; see accuracy.json")


if __name__ == "__main__":
    import sys
    from types import SimpleNamespace

    root, output = Path(sys.argv[1]).resolve(strict=True), Path(sys.argv[2])
    _require_installed_runtime(root)
    import ttnn

    if ttnn.get_arch_name() != sys.argv[4]:
        raise RuntimeError("actual device architecture differs from test contract")
    from ttpoly.ttmetal_public_capture import leg

    leg(
        SimpleNamespace(
            root=root, leg=output / "invocation.json", chip=int(sys.argv[3]), out_bf16_bin=output / "candidate.bf16"
        )
    )
