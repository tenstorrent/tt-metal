#!/usr/bin/env python3
"""Run the reproducible, hardware-free SFPI validation selftest suite.

This is the single CI/local entry point for host tooling.  ``formal_campaign``
is the current compiler-admission path.  ``prove_all`` is retained only as a
historical pin-59 coverage reproducer and is always invoked with ``--no-live``
here.  Real trace capture, pinned historical instruments, and silicon are
integration checks, not silently weakened host tests.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIREMENTS = HERE / "requirements-formal.txt"


@dataclass(frozen=True)
class Check:
    name: str
    command: tuple[str, ...]
    group: str
    timeout: int = 300


def python_check(
    name: str,
    script: str,
    *args: str,
    group: str = "current",
    timeout: int = 300,
) -> Check:
    return Check(name, (sys.executable, str(HERE / script), *args), group, timeout)


CHECKS = (
    python_check("formal campaign runner", "selftest_formal_campaign.py"),
    # The solver check exercises all modeled instructions and can take several
    # minutes on an otherwise idle CI host.  Keep a bound, but do not mistake
    # a normal comprehensive run for a hung test.
    python_check("formal instruction model", "selftest_formal_equiv.py", timeout=900),
    python_check(
        "GELU asymptote host model",
        "selftest_gelu_appx_asymptote.py",
    ),
    python_check("staged variant identity", "selftest_staged_variant.py"),
    python_check("binary stream accounting", "selftest_binary_stream.py"),
    python_check("fp32 stream accounting", "selftest_fp32_stream.py"),
    python_check(
        "three-way golden portable core",
        "selftest_threeway_golden.py",
        "--portable",
    ),
    python_check("campaign fail-closed gates", "selftest_campaign_fail_closed.py"),
    python_check("exhaustive campaign plan", "selftest_exhaustive_campaign_plan.py"),
    python_check("galaxy numeric admission", "selftest_galaxy_numeric_admission.py"),
    python_check("sweep admission claim filter", "selftest_sweep_admission_report.py"),
    python_check("galaxy shard/combiner", "selftest_galaxy_shard.py"),
    python_check("leg-store tools", "selftest_leg_store_tools.py"),
    python_check("tri-campaign manifest", "selftest_tri_campaign_manifest.py"),
    Check(
        "core-budget wrapper",
        ("bash", str(HERE / "selftest_with_cores.sh")),
        "host utility",
    ),
    Check(
        "campaign shell syntax",
        ("bash", "-n", str(HERE / "galaxy_shard.sh")),
        "host utility",
    ),
    python_check(
        "prove_all routing/census (historical, no live instruments)",
        "selftest_prove_all.py",
        "--no-live",
        group="historical pin-59",
    ),
)


INTEGRATION_SKIPS = (
    "current formal trace capture/proof: requires installed SFPI compiler and instrumented simulator",
    "full three-way harness-oracle parity: requires the installed TT-LLK Python/toolchain environment",
    "historical prove_all live leg: requires archived pin-59 instruments",
    "device anchors and silicon streamers: require Tenstorrent hardware",
)


def dependency_versions() -> dict[str, str]:
    pins = {}
    for raw_line in REQUIREMENTS.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "==" not in line:
            raise SystemExit(f"host requirement is not exactly pinned: {line}")
        package, expected = line.split("==", 1)
        pins[package.strip()] = expected.strip()

    versions = {}
    missing = []
    mismatched = []
    for package, expected in pins.items():
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
            continue
        if versions[package] != expected:
            mismatched.append(f"{package}=={versions[package]} (need {expected})")
    if missing or mismatched:
        command = f"{sys.executable} -m pip install -r {REQUIREMENTS}"
        problems = []
        if missing:
            problems.append("missing: " + ", ".join(missing))
        if mismatched:
            problems.append("version mismatch: " + ", ".join(mismatched))
        raise SystemExit(
            "pinned host dependencies unsatisfied ("
            + "; ".join(problems)
            + f")\nbootstrap with: {command}"
        )
    return versions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="omit the explicitly historical prove_all unit check",
    )
    parser.add_argument(
        "--list", action="store_true", help="list checks without running"
    )
    args = parser.parse_args(argv)

    selected = [
        check
        for check in CHECKS
        if not (args.current_only and check.group == "historical pin-59")
    ]
    if args.list:
        for check in selected:
            print(f"{check.group:18s} {check.name}")
        return 0

    versions = dependency_versions()
    print(
        "SFPI host selftests: "
        + ", ".join(f"{name}={version}" for name, version in versions.items()),
        flush=True,
    )
    failures = []
    started = time.monotonic()
    for index, check in enumerate(selected, 1):
        print(
            f"\n[{index}/{len(selected)}] {check.group}: {check.name}",
            flush=True,
        )
        try:
            result = subprocess.run(
                check.command,
                cwd=HERE,
                timeout=check.timeout,
            )
        except subprocess.TimeoutExpired:
            failures.append((check.name, "timeout"))
            print(f"FAIL: exceeded {check.timeout}s", flush=True)
            continue
        if result.returncode:
            failures.append((check.name, f"exit {result.returncode}"))
            print(f"FAIL: exit {result.returncode}", flush=True)
        else:
            print("PASS", flush=True)

    print("\nIntegration checks intentionally not run by the host suite:")
    for reason in INTEGRATION_SKIPS:
        print(f"  SKIP: {reason}")
    elapsed = time.monotonic() - started
    if failures:
        print(f"\nHOST SELFTESTS FAILED in {elapsed:.1f}s:")
        for name, reason in failures:
            print(f"  {name}: {reason}")
        return 1
    print(f"\nHOST SELFTESTS PASS: {len(selected)} checks in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
