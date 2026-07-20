"""Shared paths, keys, counts, and ELF helpers for the NOP injector."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

# Match run_nop_injector.sh default when env is unset.
DEFAULT_NOP_THREAD = "math"
DEFAULT_OUT = "/tmp/tt-llk-nop/injector"
DEFAULT_COUNTS_SPEC = "1-100"
_VALID_THREADS = frozenset({"unpack", "math", "pack"})


def item_key(nodeid: str) -> str:
    """Stable short id for a pytest nodeid (shared by prepare / batch / consume)."""
    return hashlib.sha1(nodeid.encode()).hexdigest()[:16]


def item_key_from_item(item) -> str:
    return item_key(item.nodeid)


def out_base() -> Path:
    """Injector root: work/, fails/, summary.log."""
    return Path(os.environ.get("OPEN_MP_NOP_OUT", DEFAULT_OUT))


def work_dir(key: str) -> Path:
    """Per-case workspace: base_elfs/, batch/nN/, meta.json."""
    return out_base() / "work" / key


def fails_dir(key: str) -> Path:
    """Retained ELF sets for failing NOP counts."""
    return out_base() / "fails" / key


def phase() -> str:
    return os.environ.get("OPEN_MP_NOP_PHASE", "").strip().lower()


def keep_elfs() -> bool:
    return os.environ.get("OPEN_MP_NOP_KEEP", "").strip() in ("1", "true", "yes")


def nop_thread() -> str:
    """Which TRISC ELF to patch; always lowercase (files are math.elf, not MATH.elf)."""
    thread = os.environ.get("NOP_THREAD", DEFAULT_NOP_THREAD).strip().lower()
    if thread not in _VALID_THREADS:
        raise ValueError(
            f"NOP_THREAD={thread!r}; expected one of {sorted(_VALID_THREADS)}"
        )
    return thread


def parse_counts(spec: str | None = None) -> list[int]:
    """Parse NOP_COUNTS: CSV ints and/or ranges (e.g. '1,2,4' or '1-100')."""
    if spec is None:
        spec = os.environ.get("NOP_COUNTS", DEFAULT_COUNTS_SPEC)
    spec = spec.strip()
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


def load_case_list(path: Path | str) -> list[dict]:
    """Load batch output JSON: [{nodeid, key, work}, ...]."""
    entries = json.loads(Path(path).read_text())
    if not isinstance(entries, list):
        raise RuntimeError(f"case-list must be a JSON list: {path}")
    return entries


def elfs_present(elf_dir: Path) -> dict[str, bool]:
    """Which of unpack/math/pack.elf exist under ``elf_dir``."""
    return {e: (elf_dir / f"{e}.elf").is_file() for e in ("unpack", "math", "pack")}


def copy_elf_set(src: Path, dst: Path, present: dict[str, bool]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for e, ok in present.items():
        if not ok:
            continue
        s = src / f"{e}.elf"
        if not s.is_file():
            raise FileNotFoundError(
                f"expected {e}.elf under {src} (race or incomplete snapshot)"
            )
        shutil.copy2(s, dst / f"{e}.elf")


def link_elf_set(src: Path, dst: Path, present: dict[str, bool]) -> None:
    """Hardlink ELFs from src into dst"""
    dst.mkdir(parents=True, exist_ok=True)
    for e, ok in present.items():
        if not ok:
            continue
        s = src / f"{e}.elf"
        if not s.is_file():
            raise FileNotFoundError(
                f"expected {e}.elf under {src} (race or incomplete snapshot)"
            )
        d = dst / f"{e}.elf"
        d.unlink(missing_ok=True)
        os.link(s, d)


def rm_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def move_dir(src: Path, dst: Path) -> None:
    """Move src  → dst (used to store failing batch nums)."""
    if not src.exists():
        raise FileNotFoundError(f"cannot move missing dir: {src}")
    rm_tree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
    except FileNotFoundError:
        if src.exists():
            shutil.copytree(src, dst)
            rm_tree(src)
        else:
            raise


def record_fail(nodeid: str, text: str) -> None:
    """Append a failure line to OPEN_MP_NOP_OUT/summary.log."""
    out = out_base()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.log", "a", buffering=1) as f:
        f.write(f"{nodeid}\t{text}")


def exc_value(excinfo):
    """pytest hookwrapper excinfo may be ExceptionInfo or a (type, value, tb) tuple."""
    if excinfo is None:
        return None
    if isinstance(excinfo, tuple):
        return excinfo[1] if len(excinfo) > 1 else None
    return getattr(excinfo, "value", None)


def exc_typename(excinfo) -> str:
    if excinfo is None:
        return ""
    if isinstance(excinfo, tuple):
        t = excinfo[0] if excinfo else None
        return t.__name__ if t is not None else ""
    return getattr(excinfo, "typename", "") or ""


def is_compile_skip(excinfo) -> bool:
    """PRODUCE mode ends in pytest.skip(SKIP_JUST_FOR_COMPILE_MARKER)."""
    if excinfo is None:
        return False
    from helpers.test_config import TestConfig

    marker = getattr(TestConfig, "SKIP_JUST_FOR_COMPILE_MARKER", "") or ""
    val = str(exc_value(excinfo) or "")
    if marker and marker in val:
        return True
    return exc_typename(excinfo) in ("Skipped",) and (
        "compile" in val.lower() or (marker and marker in val)
    )
