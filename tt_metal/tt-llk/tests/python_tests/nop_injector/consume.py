"""pytest plugin: consume phase — expand × NOP counts and run on device.

Flow (A = injector, B = harness):
  prepare:  compile → /tmp/tt-llk-build/..., snapshot → /tmp/tt-llk-nop/injector/work/<key>/base_elfs/
  batch:    /tmp/tt-llk-nop/injector/work/<key>/ only (ttnop injects NOPs → batch/nN/)
  consume:  batch/nN/ → (hardlink) → /tmp/tt-llk-build/<test>/<variant>/elf/
            → device load → delete that temp variant under /tmp/tt-llk-build/

Per selected case:
  1) collection: clone one pytest item per NOP count (n1, n2, …)
  2) run: hardlink batch/nN/ ELFs into TestConfig's expected variant path, run on device
  3) pass → delete batch/nN; fail → move batch/nN to fails/ and append summary.log

Does not call ttnop (OpenMP batch already wrote work/<key>/batch/n<N>/).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from _pytest.python import Function
from helpers.test_config import TestConfig
from nop_injector.helper import (
    exc_value,
    fails_dir,
    keep_elfs,
    link_elf_set,
    load_case_list,
    move_dir,
    parse_counts,
    phase,
    record_fail,
    rm_tree,
)


def _suffix_for_count(s: str, n: int) -> str:
    """Make a unique pytest name/nodeid for NOP count"""
    # Parametrized ids end in ']'; insert "-nN" before the closing bracket.
    if s.endswith("]"):
        return f"{s[:-1]}-n{n}]"
    return f"{s}[n{n}]"


def _clone_item_for_count(item, n: int, work: Path, meta: dict):
    """Create a new Function item for one NOP count and attach work/meta for the run hook."""
    # Rebuild kwargs pytest needs to construct a sibling Function node.
    kwargs = {
        "name": _suffix_for_count(item.name, n),
        "callobj": item.obj,
        "fixtureinfo": item._fixtureinfo,
        "originalname": getattr(item, "originalname", None) or item.name,
    }
    callspec = getattr(item, "callspec", None)
    if callspec is not None:
        kwargs["callspec"] = callspec

    # Build a new pytest test node from the parent test (without NOP)
    ni = Function.from_parent(item.parent, **kwargs)
    ni.own_markers = list(getattr(item, "own_markers", []) or [])
    ni._open_mp_nop_count = n
    ni._open_mp_work = work
    ni._open_mp_meta = meta
    ni._open_mp_base_nodeid = item.nodeid  # original id for summary.log
    # Force unique nodeid so xdist schedules on each count separately.
    object.__setattr__(ni, "_nodeid", _suffix_for_count(item.nodeid, n))
    return ni


def pytest_collection_modifyitems(session, config, items):
    """Replace collected cases with one item per NOP count for cases in the case-list JSON."""
    # No-op unless the shell set consume phase.
    if phase() != "consume":
        return

    # Case list = cases that finished prepare + ttnop batch ({nodeid, key, work}).
    case_list_path = os.environ.get("OPEN_MP_NOP_CASE_LIST", "").strip()
    if not case_list_path:
        raise RuntimeError("consume: set OPEN_MP_NOP_CASE_LIST")
    mp = Path(case_list_path)
    if not mp.is_file():
        raise RuntimeError(f"consume: missing case-list file {mp}")
    entries = load_case_list(mp)
    if not entries:
        raise RuntimeError(f"consume: case-list file empty: {mp}")
    by_nodeid = {e["nodeid"]: e for e in entries}

    # Keep only collected items that appear in the case list (exact nodeid match).
    selected = [i for i in items if i.nodeid in by_nodeid]
    if not selected:
        raise RuntimeError(
            f"consume: no collected items match case list "
            f"({len(by_nodeid)} nodeid(s); collected {len(items)} item(s))"
        )

    # For each selected case read prepare meta, expand to one clone per NOP count.
    new_items = []
    for item in selected:
        entry = by_nodeid[item.nodeid]
        work = Path(entry["work"])
        meta_path = work / "meta.json"
        if not meta_path.is_file():
            raise RuntimeError(f"consume: missing {meta_path}")
        meta = json.loads(meta_path.read_text())
        counts = meta.get("counts") or parse_counts()
        for n in counts:
            new_items.append(_clone_item_for_count(item, n, work, meta))
    # Swap the session item list so pytest only runs the expanded count-items.
    items[:] = new_items
    print(
        f"[nop_injector] consume: {len(selected)} case(s) → {len(new_items)} "
        f"count-item(s)",
        flush=True,
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Around each count-item: install batch/nN ELFs, run device test, keep/wipe on result."""
    if phase() != "consume":
        yield
        return

    n = getattr(item, "_open_mp_nop_count", None)
    work: Path = getattr(item, "_open_mp_work", None)
    meta: dict = getattr(item, "_open_mp_meta", None)
    if n is None or work is None or meta is None:
        raise RuntimeError(
            "consume: item missing _open_mp_nop_count/_open_mp_work/_open_mp_meta "
        )

    # extract values from prepare-side meta.json
    thread = meta["thread"]
    present = meta["elfs_present"]
    key = meta["key"]
    variant_id = meta["variant_id"]
    base_nodeid = item._open_mp_base_nodeid  # for summary.log grouping (no -nN)
    keep = keep_elfs()

    # Perturbed ELF set produced by ttnop batch for this count.
    src = work / "batch" / f"n{n}"
    if not (src / f"{thread}.elf").is_file():
        message = f"n{n}\tFAIL-ERR\tmissing perturbed ELF set at {src}"
        record_fail(f"{base_nodeid}::n{n}", f"{message}\n")
        raise RuntimeError(message)

    # Unique variant dir under ARTEFACTS so counts/xdist workers don't clobber each other.
    # ARTEFACTS is the default area where TestConfig.run loads elfs from. We hardlink the elfs
    # from the tt-llk-nop to ARTEFACTS, so the pytest harness can find them and run them.
    per_suffix = f"__open_mp_{key}_n{n}"
    orig_run = TestConfig.run
    # Some tests call TestConfig.run() more than once per item, swap ELFs only on first call.
    done = {"ok": False}

    def wrapped(self, *a, **k):
        """hardlink batch/nN into harness path, then call real TestConfig.run."""
        if not done["ok"]:
            per_id = f"{variant_id}{per_suffix}"
            dest = TestConfig.ARTEFACTS_DIR / self.test_name / per_id / "elf"
            link_elf_set(src, dest, present)
            self.variant_id = per_id
            self._prepared = True
            TestConfig.LAST_LOADED_ELFS = (
                None  # force device reload, to ensure that test is not cached
            )
            done["ok"] = True
        return orig_run(self, *a, **k)

    TestConfig.run = wrapped
    try:
        outcome = yield
    finally:
        TestConfig.run = orig_run
        TestConfig.LAST_LOADED_ELFS = None
        # delete the temporary ARTEFACTS variant dir unless KEEP is on.
        if not keep:
            rm_tree(
                TestConfig.ARTEFACTS_DIR
                / meta["test_name"]
                / f"{variant_id}{per_suffix}"
            )

    # Report / cleanup under OPEN_MP_NOP_OUT (batch tree + summary.log).
    report_id = f"{base_nodeid}::n{n}"
    if outcome.excinfo is None:
        if not keep:
            rm_tree(src)  # delete successful batch/nN
        return

    # Classify failure and retain the perturbed ELFs under fails/<key>/nN/.
    s = str(exc_value(outcome.excinfo) or "")
    tag = "FAIL-TIMEOUT" if ("Timeout" in s or "TIMED OUT" in s) else "FAIL-MISMATCH"
    print(f"[nop_injector] n={n} {tag}  {s[:120]}", flush=True)
    if not keep:
        try:
            move_dir(src, fails_dir(key) / f"n{n}")
        except FileNotFoundError as move_ex:
            record_fail(report_id, f"n{n}\tFAIL-ERR\tkeep-fail move: {move_ex}\n")
    record_fail(report_id, f"n{n}\t{tag}\t{s[:120]}\n")
