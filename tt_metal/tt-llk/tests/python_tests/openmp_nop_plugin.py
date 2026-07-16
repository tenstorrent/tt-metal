# OpenMP NOP-injection pytest plugin (host OpenMP is NOT here — see run_nop_injector.sh).
#
# OPENMP_NOP=1 and OPENMP_NOP_PHASE=
#   prepare  — compile-only (PRODUCE): snapshot base ELFs → work/<key>/bk + meta.json
#   consume  — device runs: expand via OPENMP_NOP_CHUNK_MANIFEST into one item per NOP
#              count; delete-on-pass / keep-on-fail
#
# ttnop batch (OpenMP) is invoked by the shell between prepare and consume.

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest
from _pytest.python import Function

_CFG: dict = {}


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


def _item_key(item) -> str:
    return hashlib.sha1(item.nodeid.encode()).hexdigest()[:16]


def _out_base() -> Path:
    return Path(os.environ.get("OPENMP_NOP_OUT", "/tmp/tt-llk-nop/injector"))


def _phase() -> str:
    return os.environ.get("OPENMP_NOP_PHASE", "").strip().lower()


def _keep_elfs() -> bool:
    return os.environ.get("OPENMP_NOP_KEEP", "").strip() in ("1", "true", "yes")


def _record_fail(nodeid: str, text: str) -> None:
    out = _out_base()
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.log", "a", buffering=1) as f:
        f.write(f"{nodeid}\t{text}")


def _components_present(elf_dir: Path) -> dict[str, bool]:
    return {e: (elf_dir / f"{e}.elf").is_file() for e in ("unpack", "math", "pack")}


def _copy_elf_set(src: Path, dst: Path, have: dict[str, bool]) -> None:
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


def _elf_fingerprint(elf_path: Path) -> str:
    """Short identity for log lines: size + md5 prefix (proves which ELF was loaded)."""
    if not elf_path.is_file():
        return "missing"
    data = elf_path.read_bytes()
    return f"size={len(data)} md5={hashlib.md5(data).hexdigest()[:12]}"


def _move_dir(src: Path, dst: Path) -> None:
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


def _exc_value(excinfo):
    """pytest hookwrapper excinfo may be ExceptionInfo or a (type, value, tb) tuple."""
    if excinfo is None:
        return None
    if isinstance(excinfo, tuple):
        return excinfo[1] if len(excinfo) > 1 else None
    return getattr(excinfo, "value", None)


def _exc_typename(excinfo) -> str:
    if excinfo is None:
        return ""
    if isinstance(excinfo, tuple):
        t = excinfo[0] if excinfo else None
        return t.__name__ if t is not None else ""
    return getattr(excinfo, "typename", "") or ""


def _is_compile_skip(excinfo) -> bool:
    """PRODUCE mode ends in pytest.skip(SKIP_JUST_FOR_COMPILE_MARKER)."""
    if excinfo is None:
        return False
    from helpers.test_config import TestConfig

    marker = getattr(TestConfig, "SKIP_JUST_FOR_COMPILE_MARKER", "") or ""
    val = str(_exc_value(excinfo) or "")
    if marker and marker in val:
        return True
    return _exc_typename(excinfo) in ("Skipped",) and (
        "compile" in val.lower() or (marker and marker in val)
    )


def _prepare_snapshot(item, cfg) -> None:
    from helpers.test_config import TestConfig

    thread = os.environ.get("NOP_THREAD", "math")
    counts = _parse_counts()
    key = _item_key(item)
    vdir = TestConfig.ARTEFACTS_DIR / cfg.test_name / cfg.variant_id / "elf"
    if not (vdir / f"{thread}.elf").is_file():
        raise RuntimeError(
            f"prepare: base ELF not found: {vdir / f'{thread}.elf'}\n"
            "Compile-producer must materialize unpack/math/pack.elf first."
        )

    have = _components_present(vdir)
    work = _out_base() / "work" / key
    bk = work / "bk"
    _rm_tree(work)
    _copy_elf_set(vdir, bk, have)

    meta = {
        "nodeid": item.nodeid,
        "key": key,
        "test_name": cfg.test_name,
        "variant_id": cfg.variant_id,
        "thread": thread,
        "counts": counts,
        "have": have,
    }
    (work / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"[openmp_nop] prepare ok key={key} bk={bk} "
        f"counts={counts[0]}..{counts[-1]} ({len(counts)})",
        flush=True,
    )


def _suffix_for_count(s: str, n: int) -> str:
    """Unique name/nodeid for one NOP count (avoids xdist teardown collisions)."""
    if s.endswith("]"):
        return f"{s[:-1]}-n{n}]"
    return f"{s}[n{n}]"


def _clone_item_for_count(item, n: int, work: Path, meta: dict):
    """
    Build a fresh Function node for one NOP count.

    Do NOT use copy.copy(item): shallow copies share Session/fixture finalizer
    state and blow up under xdist (AssertionError in runner teardown).
    """
    kwargs = {
        "name": _suffix_for_count(item.name, n),
        "callobj": item.obj,
        "fixtureinfo": item._fixtureinfo,
        "originalname": getattr(item, "originalname", None) or item.name,
    }
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        kwargs["callspec"] = callspec
    keywords = getattr(item, "keywords", None)
    if keywords is not None:
        try:
            kwargs["keywords"] = dict(keywords)
        except Exception:  # noqa: BLE001
            pass

    ni = Function.from_parent(item.parent, **kwargs)
    ni.own_markers = list(getattr(item, "own_markers", []) or [])
    ni._omp_nop_count = n
    ni._omp_work = work
    ni._omp_meta = meta
    ni._omp_base_nodeid = item.nodeid
    object.__setattr__(ni, "_nodeid", _suffix_for_count(item.nodeid, n))
    return ni


def pytest_collection_modifyitems(session, config, items):
    """Consume: expand selected nodeid(s) into one fresh item per NOP count.

    Requires OPENMP_NOP_CHUNK_MANIFEST = JSON list of
      {"nodeid": "...", "key": "...", "work": "/path/to/work/<key>"}
    """
    if os.environ.get("OPENMP_NOP") != "1" or _phase() != "consume":
        return

    manifest_path = os.environ.get("OPENMP_NOP_CHUNK_MANIFEST", "").strip()
    if not manifest_path:
        raise RuntimeError("consume: set OPENMP_NOP_CHUNK_MANIFEST")
    mp = Path(manifest_path)
    if not mp.is_file():
        raise RuntimeError(f"consume: missing chunk manifest {mp}")
    entries = json.loads(mp.read_text())
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"consume: chunk manifest empty/invalid: {mp}")
    by_nodeid = {e["nodeid"]: e for e in entries}

    selected = [i for i in items if i.nodeid in by_nodeid]
    # Fallback when pytest rewrites a lone nodeid slightly vs the manifest.
    if not selected and len(items) == 1 and len(by_nodeid) == 1:
        selected = list(items)
    if not selected:
        raise RuntimeError(
            f"consume: no collected items match chunk manifest "
            f"({len(by_nodeid)} nodeid(s); collected {len(items)} item(s))"
        )

    new_items = []
    for item in selected:
        entry = by_nodeid.get(item.nodeid) or next(iter(by_nodeid.values()))
        work = Path(entry["work"])
        meta_path = work / "meta.json"
        if not meta_path.is_file():
            raise RuntimeError(f"consume: missing {meta_path}")
        meta = json.loads(meta_path.read_text())
        counts = meta.get("counts") or _parse_counts()
        for n in counts:
            new_items.append(_clone_item_for_count(item, n, work, meta))
    items[:] = new_items
    print(
        f"[openmp_nop] consume: {len(selected)} case(s) → {len(new_items)} "
        f"count-item(s)",
        flush=True,
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if os.environ.get("OPENMP_NOP") != "1":
        yield
        return

    phase = _phase()
    if phase not in ("prepare", "consume"):
        yield
        return

    from helpers.test_config import TestConfig

    if phase == "prepare":
        _CFG.clear()
        orig_run = TestConfig.run

        def cap(self, *a, **k):
            _CFG.setdefault("cfg", self)
            return orig_run(self, *a, **k)

        TestConfig.run = cap
        try:
            outcome = yield
        finally:
            TestConfig.run = orig_run

        if "cfg" not in _CFG:
            if outcome.excinfo is not None and not _is_compile_skip(outcome.excinfo):
                return
            raise RuntimeError(
                "prepare: TestConfig.run was never called; cannot snapshot ELFs"
            )
        if outcome.excinfo is not None and not _is_compile_skip(outcome.excinfo):
            return
        _prepare_snapshot(item, _CFG["cfg"])
        return

    # --- consume ---
    n = getattr(item, "_omp_nop_count", None)
    work: Path = getattr(item, "_omp_work", None)
    meta: dict = getattr(item, "_omp_meta", None)
    if n is None or work is None or meta is None:
        raise RuntimeError(
            "consume: item missing _omp_nop_count/_omp_work/_omp_meta "
            "(collection_modifyitems did not expand counts)"
        )

    thread = meta["thread"]
    have = meta["have"]
    key = meta["key"]
    test_name = meta["test_name"]
    variant_id = meta["variant_id"]
    base_nodeid = getattr(item, "_omp_base_nodeid", meta["nodeid"])
    keep = _keep_elfs()

    src = work / "batch" / f"n{n}"
    if not (src / f"{thread}.elf").is_file():
        message = f"n{n}\tFAIL-ERR\tmissing perturbed ELF set at {src}"
        _record_fail(f"{base_nodeid}::n{n}", f"{message}\n")
        raise RuntimeError(message)

    elf_fp = _elf_fingerprint(src / f"{thread}.elf")
    print(
        f"[openmp_nop] n={n} LOAD {thread}.elf {elf_fp}  path={src / f'{thread}.elf'}",
        flush=True,
    )

    per_suffix = f"__omp_{key}_n{n}"
    fail_root = _out_base() / "fails" / key
    orig_run = TestConfig.run
    done = {"ok": False}

    def wrapped(self, *a, **k):
        if not done["ok"]:
            self.prepare()
            # Prefer meta.variant_id so we do not depend on re-hash matching prepare.
            per_id = f"{variant_id}{per_suffix}"
            dest = TestConfig.ARTEFACTS_DIR / self.test_name / per_id / "elf"
            _copy_elf_set(src, dest, have)
            self.variant_id = per_id
            TestConfig.LAST_LOADED_ELFS = None
            done["ok"] = True
        return orig_run(self, *a, **k)

    TestConfig.run = wrapped
    try:
        outcome = yield
    finally:
        TestConfig.run = orig_run
        TestConfig.LAST_LOADED_ELFS = None
        if not keep:
            _rm_tree(TestConfig.ARTEFACTS_DIR / test_name / f"{variant_id}{per_suffix}")

    report_id = f"{base_nodeid}::n{n}"
    if outcome.excinfo is None:
        print(f"[openmp_nop] n={n} PASS  {elf_fp}", flush=True)
        if not keep:
            _rm_tree(src)
        return

    err = _exc_value(outcome.excinfo)
    s = str(err) if err is not None else ""
    tag = "FAIL-TIMEOUT" if ("Timeout" in s or "TIMED OUT" in s) else "FAIL-MISMATCH"
    print(f"[openmp_nop] n={n} {tag}  {elf_fp}  {s[:120]}", flush=True)
    if not keep:
        try:
            _move_dir(src, fail_root / f"n{n}")
        except FileNotFoundError as move_ex:
            _record_fail(report_id, f"n{n}\tFAIL-ERR\tkeep-fail move: {move_ex}\n")
    _record_fail(report_id, f"n{n}\t{tag}\t{s[:120]}\n")
