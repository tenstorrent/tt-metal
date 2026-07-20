"""pytest plugin: prepare phase — compile-producer snapshot of base ELFs.

Load with:  -p nop_injector.prepare
Requires:   OPEN_MP_NOP_PHASE=prepare

Writes work/<key>/base_elfs/{unpack,math,pack}.elf + meta.json.
Does not call ttnop (OpenMP batch runs between prepare and consume).
"""

from __future__ import annotations

import json

import pytest
from nop_injector.helper import (
    copy_elf_set,
    elfs_present,
    is_compile_skip,
    item_key_from_item,
    nop_thread,
    parse_counts,
    phase,
    rm_tree,
    work_dir,
)

_CFG: dict = {}


def _prepare_snapshot(item, cfg) -> None:
    from helpers.test_config import TestConfig

    thread = nop_thread()
    counts = parse_counts()  # 1-100 NOPs
    key = item_key_from_item(item)

    # directory where the compiled base elf is located
    variant_elf_dir = TestConfig.ARTEFACTS_DIR / cfg.test_name / cfg.variant_id / "elf"
    if not (variant_elf_dir / f"{thread}.elf").is_file():
        raise RuntimeError(
            f"prepare: base ELF not found: {variant_elf_dir / f'{thread}.elf'}\n"
            "Compile-producer must materialize unpack/math/pack.elf first."
        )

    # dict of bools for each elf if present
    present = elfs_present(variant_elf_dir)

    # create directory work/key
    work = work_dir(key)
    base_elfs = work / "base_elfs"
    rm_tree(work)

    # copy ELF files from /tmp/tt--llk-build/../elf.. -> work/key/base_elfs
    copy_elf_set(variant_elf_dir, base_elfs, present)

    meta = {
        "nodeid": item.nodeid,
        "key": key,
        "test_name": cfg.test_name,
        "variant_id": cfg.variant_id,
        "thread": thread,
        "counts": counts,
        "elfs_present": present,
    }
    (work / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if phase() != "prepare":
        yield
        return

    from helpers.test_config import TestConfig

    _CFG.clear()
    orig_run = TestConfig.run

    # when test calls TestConfig.run(), save that cfg, then run normally
    def cap(self, *a, **k):
        _CFG.setdefault("cfg", self)
        return orig_run(self, *a, **k)

    TestConfig.run = cap
    try:
        outcome = yield
    finally:
        TestConfig.run = orig_run

    if "cfg" not in _CFG:
        if outcome.excinfo is not None and not is_compile_skip(outcome.excinfo):
            return
        raise RuntimeError(
            "prepare: TestConfig.run was never called; cannot snapshot ELFs"
        )
    if outcome.excinfo is not None and not is_compile_skip(outcome.excinfo):
        return
    _prepare_snapshot(item, _CFG["cfg"])
