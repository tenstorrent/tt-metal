# OpenMP NOP-injection pytest plugin (space-efficient, xdist-safe).
#
# For every test case (when OMP_NOP=1):
#   1. Baseline: compile (DEFAULT) + device run; capture TestConfig.
#   2. Snapshot ELFs into a per-nodeid work dir.
#   3. OpenMP `ttnop batch` 1..100 into that private dir.
#   4. Re-run each count via a unique temporary variant_id.
#       Keep ONLY failing sets under OMP_NOP_OUT/fails/.
#   5. Wipe this item's private work dir + temp variant dirs;
#
# Failures kept under OMP_NOP_OUT/fails/

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_CFG = {}
_IN_SWEEP = False


def _parse_counts() -> list[int]:
    spec = os.environ.get("NOP_COUNTS", "1-100").strip()
    counts: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            counts.extend(range(int(lo), int(hi) + 1))
        else:
            counts.append(int(part))
    return counts


def _ttnop_bin() -> Path:
    here = Path(__file__).resolve().parent / "ttnop" / "ttnop"
    return Path(os.environ.get("TTNOP", str(here)))


def _item_key(item) -> str:
    return hashlib.sha1(item.nodeid.encode()).hexdigest()[:16]


def _out_base() -> Path:
    return Path(os.environ.get("OMP_NOP_OUT", "/tmp/tt-llk-build/nop_injector"))


def _record_fail(item, text: str) -> None:
    """Append one line to a single summary.log (no per-test log files)."""
    out = _out_base()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.log", "a", buffering=1) as f:
        f.write(f"{item.nodeid}\t{text}")


def _components_present(elf_dir: Path) -> dict[str, bool]:
    return {e: (elf_dir / f"{e}.elf").is_file() for e in ("unpack", "math", "pack")}


def _copy_elf_set(src: Path, dst: Path, have: dict[str, bool]) -> None:
    """Copy present ELFs; raise a clear error if a claimed file is missing."""
    dst.mkdir(parents=True, exist_ok=True)
    for e, ok in have.items():
        if not ok:
            continue
        s = src / f"{e}.elf"
        if not s.is_file():
            raise FileNotFoundError(
                f"expected {e}.elf under {src} (race or incomplete snapshot)"
            )
        shutil.copy2(s, dst / f"{e}.elf")


def _rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _move_dir(src: Path, dst: Path) -> None:
    """Move src -> dst; fall back to copy+rm if src vanished mid-flight."""
    if not src.exists():
        raise FileNotFoundError(f"cannot move missing dir: {src}")
    _rm_tree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
    except FileNotFoundError:
        if src.exists():
            shutil.copytree(src, dst)
            _rm_tree(src)
        else:
            raise


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if os.environ.get("OMP_NOP") != "1":
        yield
        return

    global _IN_SWEEP
    if _IN_SWEEP:
        yield
        return

    from helpers.test_config import BuildMode, TestConfig

    if TestConfig.BUILD_MODE == BuildMode.PRODUCE:
        yield
        return

    _CFG.clear()
    orig_run, orig_ref = TestConfig.run, TestConfig.run_elf_files

    def cap(orig):
        def wrapped(self, *a, **k):
            _CFG.setdefault("cfg", self)
            return orig(self, *a, **k)

        return wrapped

    TestConfig.run = cap(orig_run)
    TestConfig.run_elf_files = cap(orig_ref)
    try:
        outcome = yield
    finally:
        TestConfig.run = orig_run
        TestConfig.run_elf_files = orig_ref

    if outcome.excinfo is not None:
        return
    if "cfg" not in _CFG:
        return

    _IN_SWEEP = True
    try:
        _sweep(item)
    finally:
        _IN_SWEEP = False


def _keep_elfs() -> bool:
    return os.environ.get("OMP_NOP_KEEP", "").strip() in ("1", "true", "yes")


def _sweep(item) -> None:
    from helpers.test_config import TestConfig

    cfg = _CFG["cfg"]
    thread = os.environ.get("NOP_THREAD", "math")
    counts = _parse_counts()
    keep = _keep_elfs()
    ttnop = _ttnop_bin()
    if not ttnop.is_file():
        raise RuntimeError(
            f"ttnop binary not found: {ttnop}\n"
            "Build it with `make` in tests/python_tests/ttnop/."
        )

    key = _item_key(item)
    vdir = TestConfig.ARTEFACTS_DIR / cfg.test_name / cfg.variant_id / "elf"
    if not (vdir / f"{thread}.elf").is_file():
        return

    have = _components_present(vdir)
    out_base = _out_base()
    # Per-nodeid paths: many different cases share the same variant_id hash.
    work = out_base / "work" / key
    bk = work / "bk"
    batch_root = work / "batch"
    fail_root = out_base / "fails" / key

    _rm_tree(work)
    try:
        _copy_elf_set(vdir, bk, have)
    except FileNotFoundError:
        _rm_tree(work)
        return

    batch_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ttnop),
        "batch",
        "--base-dir",
        str(bk),
        "--out-root",
        str(batch_root),
        "--thread",
        thread,
        "--counts",
        ",".join(str(c) for c in counts),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        _record_fail(
            item,
            f"BATCH-ERR\trc={r.returncode}\t{(r.stderr or '').strip()[:200]}\n",
        )
        if not keep:
            _rm_tree(work)
        return

    for n in counts:
        src = batch_root / f"n{n}"
        if not (src / f"{thread}.elf").is_file():
            _record_fail(item, f"n{n}\tFAIL-ERR\tmissing perturbed ELF set\n")
            continue

        # Unique temp variant so we never clobber the shared baseline elf dir.
        per_suffix = f"__omp_{key}_n{n}"
        try:
            _run_perturbed(item, src, have, per_suffix)
            if not keep:
                _rm_tree(src)
        except Exception as ex:  # noqa: BLE001
            s = str(ex)
            if "Timeout" in s or "TIMED OUT" in s:
                tag = "FAIL-TIMEOUT"
            else:
                tag = "FAIL-MISMATCH"
            if keep:
                _record_fail(item, f"n{n}\t{tag}\t{s[:120]}\n")
            else:
                try:
                    _move_dir(src, fail_root / f"n{n}")
                except FileNotFoundError as move_ex:
                    _record_fail(item, f"n{n}\tFAIL-ERR\tkeep-fail move: {move_ex}\n")
                _record_fail(item, f"n{n}\t{tag}\t{s[:120]}\n")
        finally:
            if not keep:
                _rm_tree(
                    TestConfig.ARTEFACTS_DIR
                    / cfg.test_name
                    / f"{cfg.variant_id}{per_suffix}"
                )

    if keep:
        print(
            f"[omp_nop] kept ELFs: {work} " f"(bk=baseline, batch/n<count>=perturbed)",
            flush=True,
        )
    else:
        _rm_tree(work)
    TestConfig.LAST_LOADED_ELFS = None


def _run_perturbed(item, elf_src: Path, have: dict[str, bool], per_suffix: str) -> None:
    """Re-invoke the test loading ELFs from a unique temporary variant dir."""
    from helpers.test_config import TestConfig

    orig_run = TestConfig.run
    done = {"ok": False}

    def wrapped(self, *a, **k):
        if not done["ok"]:
            self.prepare()
            per_id = f"{self.variant_id}{per_suffix}"
            dest = TestConfig.ARTEFACTS_DIR / self.test_name / per_id / "elf"
            _copy_elf_set(elf_src, dest, have)
            self.variant_id = per_id
            TestConfig.LAST_LOADED_ELFS = None
            done["ok"] = True
        return orig_run(self, *a, **k)

    TestConfig.run = wrapped
    try:
        item.runtest()
    finally:
        TestConfig.run = orig_run
        TestConfig.LAST_LOADED_ELFS = None
