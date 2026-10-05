#!/usr/bin/env python3
"""Fast, hardware-free checks for tri_campaign_manifest.py."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
TOOL = HERE / "tri_campaign_manifest.py"
HEAD_A = "a" * 40
HEAD_B = "b" * 40
COMPILER = "c" * 64
HEADER = (
    "op\tcategory\tfull_space\tstate\ta_selected_sem_node\tselected_flags\t"
    "b_baseline_sem_node\tbaseline_flags\tc_baseline_hand_node\tc_baseline_flags\n"
)


def profile(op: str, selected: str, *, a: str = "sem", b: str = "sem",
            baseline: str = "-mbase", cflags: str = "-mbase") -> str:
    return f"{op}\tunary_32_exhaustive\t4294967296\tGAP\t{a}\t{selected}\t{b}\t{baseline}\thand\t{cflags}\n"


def identity(op: str, salt: str) -> str:
    fields = [hashlib.sha256(f"{salt}-{index}".encode()).hexdigest() for index in range(6)]
    return "\t".join([op, *fields]) + "\n"


def fixture(root: Path) -> dict[str, Path]:
    paths = {name: root / name for name in ("roster", "profiles", "idmap", "search")}
    paths["roster"].write_text("zeta\nalpha\n")
    paths["profiles"].write_text(
        HEADER + profile("zeta", "-mz") + profile("alpha", "-ma")
    )
    paths["idmap"].write_text(
        identity("unused", "u") + identity("zeta", "z") + identity("alpha", "a")
    )
    paths["search"].write_text(json.dumps({
        "settings": {"baseline_flags": "-mbase"},
        "operations": {
            op: {"proposal": {
                "selection": {"flags": flags},
                "frozen_baseline_flags": "-mbase",
            }} for op, flags in (("alpha", "-ma"), ("zeta", "-mz"), ("unused", "-mu"))
        },
    }))
    return paths


def run(paths: dict[str, Path], out: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([
        sys.executable, str(TOOL),
        "--roster", str(paths["roster"]),
        "--profiles", str(paths["profiles"]),
        "--idmap", str(paths["idmap"]),
        "--search", str(paths["search"]),
        "--producer-tt-metal-head", HEAD_A,
        "--runner-tt-metal-head", HEAD_B,
        "--compiler-sha256", COMPILER,
        "--out", str(out), *extra,
    ], text=True, capture_output=True)


def reject(mutator, expected: str) -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        paths = fixture(root)
        mutator(paths)
        result = run(paths, root / "out")
        assert result.returncode == 2, result
        assert expected in result.stderr, result.stderr
        assert not (root / "out").exists(), "validation failure left an output directory"


def assert_assembler_contract(manifest: dict, root: Path) -> None:
    """Mirror assemble_search_validation.py's Galaxy manifest authority join."""
    flags = (root / "flags.tsv").read_text().splitlines()
    idmap = (root / "idmap.tsv").read_text().splitlines()
    assert manifest.get("schema_version") == 1
    assert manifest.get("status") == "READY"
    assert manifest.get("eligible_ops") == len(flags) == len(idmap)
    declared = {
        "search_sha256": root.parent / "search",
        "flags_tsv_sha256": root / "flags.tsv",
        "idmap_sha256": root / "idmap.tsv",
    }
    for field, path in declared.items():
        value = manifest.get(field)
        assert isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
        assert value == hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        paths = fixture(root)
        out = root / "out"
        result = run(paths, out, "--sfpi-head", "d" * 40, "--sfpi-gcc-head", "e" * 40)
        assert result.returncode == 0, result.stderr
        assert (out / "flags.tsv").read_text() == "alpha\t-ma\nzeta\t-mz\n"
        assert (out / "idmap.tsv").read_text() == identity("alpha", "a") + identity("zeta", "z")
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["schema"] == "tt-llk-tri-campaign-v1"
        assert manifest["ops"] == ["alpha", "zeta"]
        assert manifest["eligible_ops"] == 2
        assert manifest["identities"]["compiler_sha256"] == COMPILER
        assert len(manifest["roster_sha256"]) == 64
        assert len(manifest["profiles_sha256"]) == 64
        assert_assembler_contract(manifest, out)
        again = run(paths, out)
        assert again.returncode == 2 and "already exists" in again.stderr

    reject(lambda p: p["roster"].write_text("alpha\nalpha\n"), "duplicate roster op")
    reject(lambda p: p["roster"].write_text("alpha\n"), "extra=['zeta']")
    reject(lambda p: p["profiles"].write_text(HEADER + profile("alpha", "-ma")), "missing=['zeta']")
    reject(lambda p: p["profiles"].write_text(HEADER + profile("alpha", "-ma") + profile("alpha", "-ma")), "duplicate profile op")
    reject(lambda p: p["profiles"].write_text(HEADER + profile("alpha", "-ma", a="one", b="two") + profile("zeta", "-mz")), "A/B semantic nodes differ")
    reject(lambda p: p["profiles"].write_text(HEADER + profile("alpha", "-ma", cflags="-mother") + profile("zeta", "-mz")), "B/C baseline flags differ")
    reject(lambda p: p["idmap"].write_text(identity("alpha", "a")), "missing roster ops")
    reject(lambda p: p["idmap"].write_text("alpha\tbad\tbad\tbad\tbad\tbad\tbad\n" + identity("zeta", "z")), "invalid SHA-256")
    reject(lambda p: p["profiles"].write_text(HEADER + profile("alpha", "-wrong") + profile("zeta", "-mz")), "selected flags disagree")

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        paths = fixture(root)
        result = run(paths, root / "out", "--sfpi-head", "NOT-A-HASH")
        assert result.returncode == 2 and "sfpi head" in result.stderr
        assert not (root / "out").exists()

    print("tri campaign manifest selftest: PASS")


if __name__ == "__main__":
    main()
