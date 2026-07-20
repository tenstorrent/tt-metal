"""Shared paths, keys, counts, and ELF helpers for the NOP injector."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

# Match run_nop_injector.sh default when env is unset.
DEFAULT_NOP_THREAD = "math"
DEFAULT_OUT = "/tmp/tt-llk-nop/injector"
DEFAULT_ARTEFACTS = "/tmp/tt-llk-build"
DEFAULT_COUNTS_SPEC = "1-100"
_VALID_THREADS = frozenset({"unpack", "math", "pack"})
_KEY_RE = re.compile(r"^[0-9a-f]{16}$")


def item_key(nodeid: str) -> str:
    """Stable short id for a pytest nodeid (shared by prepare / batch / consume)."""
    return hashlib.sha1(nodeid.encode()).hexdigest()[:16]


def item_key_from_item(item) -> str:
    return item_key(item.nodeid)


def out_base() -> Path:
    """Injector root: work/, fails/, summary.log."""
    return Path(os.environ.get("OPEN_MP_NOP_OUT", DEFAULT_OUT)).expanduser().resolve()


def artefacts_base() -> Path:
    """Harness compile/load tree (TestConfig.ARTEFACTS_DIR default)."""
    try:
        from helpers.test_config import TestConfig

        art = getattr(TestConfig, "ARTEFACTS_DIR", None)
        if art is not None:
            return Path(art).expanduser().resolve()
    except Exception:
        pass
    return Path(DEFAULT_ARTEFACTS).resolve()


def allowed_roots() -> tuple[Path, ...]:
    """Filesystem roots this injector may read/write under."""
    return (
        out_base(),
        artefacts_base(),
        Path(tempfile.gettempdir()).resolve(),
    )


def require_under(path: Path | str, *extra_roots: Path | str) -> Path:
    """Resolve ``path`` and require it stays under an allowed root (path-traversal guard)."""
    resolved = Path(path).expanduser().resolve()
    roots = [*(allowed_roots()), *(Path(r).expanduser().resolve() for r in extra_roots)]
    for root in roots:
        if resolved == root or resolved.is_relative_to(root):
            return resolved
    raise ValueError(f"path escapes allowed roots {roots}: {resolved}")


def work_dir(key: str) -> Path:
    """Per-case workspace: base_elfs/, batch/nN/, meta.json."""
    if not _KEY_RE.fullmatch(key):
        raise ValueError(f"invalid item key: {key!r}")
    return require_under(out_base() / "work" / key)


def fails_dir(key: str) -> Path:
    """Retained ELF sets for failing NOP counts."""
    if not _KEY_RE.fullmatch(key):
        raise ValueError(f"invalid item key: {key!r}")
    return require_under(out_base() / "fails" / key)


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
            if not lo.isdigit() or not hi.isdigit():
                raise ValueError(f"invalid NOP count range: {part!r}")
            counts.extend(range(int(lo), int(hi) + 1))
        else:
            if not part.isdigit():
                raise ValueError(f"invalid NOP count: {part!r}")
            counts.append(int(part))
    if not counts:
        raise ValueError("NOP counts list is empty")
    return counts


def counts_csv(spec: str | None = None) -> str:
    """Canonical CSV of validated NOP counts (safe for argv)."""
    return ",".join(str(c) for c in parse_counts(spec))


def load_case_list(path: Path | str) -> list[dict]:
    """Load batch output JSON: [{nodeid, key, work}, ...]."""
    safe = require_under(path)
    entries = json.loads(safe.read_text())
    if not isinstance(entries, list):
        raise RuntimeError(f"case-list must be a JSON list: {safe}")
    return entries


def elfs_present(elf_dir: Path) -> dict[str, bool]:
    """Which of unpack/math/pack.elf exist under ``elf_dir``."""
    elf_dir = require_under(elf_dir)
    return {e: (elf_dir / f"{e}.elf").is_file() for e in ("unpack", "math", "pack")}


def _iter_present_elfs(src: Path, present: dict[str, bool]):
    """Yield (name, src_path) for each present ELF that exists under src."""
    src = require_under(src)
    for e, ok in present.items():
        if not ok:
            continue
        if e not in _VALID_THREADS:
            raise ValueError(f"invalid ELF name: {e!r}")
        s = require_under(src / f"{e}.elf")
        if not s.is_file():
            raise FileNotFoundError(
                f"expected {e}.elf under {src} (race or incomplete snapshot)"
            )
        yield e, s


def copy_elf_set(src: Path, dst: Path, present: dict[str, bool]) -> None:
    dst = require_under(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for e, s in _iter_present_elfs(src, present):
        shutil.copy2(s, require_under(dst / f"{e}.elf"))


def link_elf_set(src: Path, dst: Path, present: dict[str, bool]) -> None:
    """Hardlink ELFs from src into dst."""
    dst = require_under(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for e, s in _iter_present_elfs(src, present):
        d = require_under(dst / f"{e}.elf")
        d.unlink(missing_ok=True)
        os.link(s, d)


def rm_tree(path: Path) -> None:
    safe = require_under(path)
    if safe.exists():
        shutil.rmtree(safe, ignore_errors=True)


def move_dir(src: Path, dst: Path) -> None:
    """Move src → dst (used to store failing batch nums)."""
    src = require_under(src)
    dst = require_under(dst)
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
