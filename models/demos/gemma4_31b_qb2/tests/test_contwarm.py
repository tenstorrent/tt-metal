# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host tests for tt/contwarm.py (update 4 item B): the marker's life and placement (critic 05fcb318 requirement 1,
65b2de78 N1 / N2, LEAD_DECISIONS_PKG P102). No device, no ttnn: the warm is a callable that writes fake JIT files.
The fixture mirrors this host's kernel cache: <root>/tt-metal-cache (pch only, no kernels/) and
<root>/tt-metal-cache2778076107680524023/kernels/..."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("contwarm", HERE.parent / "tt" / "contwarm.py")
CW = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(CW)

KEY = "tt-metal-cache2778076107680524023"


@pytest.fixture
def env(tmp_path, monkeypatch):
    pkg = tmp_path / "pkg"
    (pkg / "tt").mkdir(parents=True)
    (pkg / "tt" / "a.py").write_text("x = 1\n")
    (pkg / "resident_m5_11").mkdir()
    (pkg / "resident_m5_11" / "k.cpp").write_text("// k\n")
    root = tmp_path / "cache"
    (root / "tt-metal-cache" / "pch").mkdir(parents=True)
    (root / "tt-metal-cache" / "pch" / "p.pch").write_text("p")
    (root / KEY / "kernels" / "old").mkdir(parents=True)
    (root / KEY / "kernels" / "old" / "o.elf").write_text("o")
    monkeypatch.setenv("TT_METAL_CACHE", str(root))
    monkeypatch.delenv(CW.ENV, raising=False)
    return pkg, root


def warm_writing(root, n, key=KEY, tag="w"):
    def warm():
        d = root / key / "kernels" / f"sdpa_{tag}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"k{i}.elf").write_text("x")
        return 7
    return warm


def run(pkg, warm):
    return CW.run("batch", "resident_m5_11", warm, lambda: None, log=lambda *_: None, pkg_dir=pkg)


def marker_file(root, pkg):
    return root / KEY / "gemma4_contwarm" / CW.marker_name("batch", CW.code_hash("resident_m5_11", pkg))


def test_first_boot_warms_and_marks_in_the_build_dir(env):
    pkg, root = env
    r = run(pkg, warm_writing(root, 5))
    assert r["warmed"] and r["passes"] == 7 and r["new_files"] == 5
    m = json.loads(marker_file(root, pkg).read_text())
    assert len(m["jit"]) == 5 and all((root / KEY / j).exists() for j in m["jit"])
    assert not (root / "tt-metal-cache" / "gemma4_contwarm").exists()


def test_second_boot_skips(env):
    pkg, root = env
    run(pkg, warm_writing(root, 5))
    r = run(pkg, warm_writing(root, 5, tag="again"))
    assert not r["warmed"] and "marker" in r["why"] and 0 <= r["check_s"] < 10


def test_deleted_build_dir_rewarms(env):
    pkg, root = env
    run(pkg, warm_writing(root, 5))
    import shutil
    shutil.rmtree(root / KEY)
    (root / KEY / "kernels").mkdir(parents=True)
    r = run(pkg, warm_writing(root, 3))
    assert r["warmed"] and marker_file(root, pkg).exists()


def test_removed_jit_output_rewarms(env):
    pkg, root = env
    run(pkg, warm_writing(root, 5))
    m = json.loads(marker_file(root, pkg).read_text())
    os.remove(root / KEY / m["jit"][0])
    r = run(pkg, warm_writing(root, 0))
    assert r["warmed"]


@pytest.mark.parametrize("content", ["", "{\"profile\": \"batch\"", "not json", "[]"])
def test_truncated_or_unparsable_marker_rewarms(env, content):
    pkg, root = env
    run(pkg, warm_writing(root, 5))
    marker_file(root, pkg).write_text(content)
    r = run(pkg, warm_writing(root, 0))
    assert r["warmed"]


def test_zero_gain_warm_still_marks_the_single_kernel_cache(env):
    pkg, root = env
    r = run(pkg, warm_writing(root, 0))
    assert r["warmed"] and r["new_files"] == 0 and r["marker"] == str(marker_file(root, pkg))
    assert json.loads(marker_file(root, pkg).read_text())["jit"] == []
    assert not run(pkg, warm_writing(root, 0))["warmed"]


def test_two_candidate_build_dirs_write_no_marker(env):
    pkg, root = env
    other = "tt-metal-cache999"
    (root / other / "kernels").mkdir(parents=True)

    def warm():
        warm_writing(root, 2)()
        warm_writing(root, 2, key=other)()
        return 1
    r = run(pkg, warm)
    assert r["warmed"] and r["marker"] is None and "2 build dirs" in r["marker_place"]


def test_killed_warm_writes_nothing(env):
    pkg, root = env

    def warm():
        warm_writing(root, 3)()
        raise KeyboardInterrupt("killed mid-warm")
    with pytest.raises(KeyboardInterrupt):
        run(pkg, warm)
    assert not marker_file(root, pkg).exists()


def test_off_never_warms_force_always_does(env, monkeypatch):
    pkg, root = env
    monkeypatch.setenv(CW.ENV, "off")
    assert not run(pkg, warm_writing(root, 5))["warmed"]
    monkeypatch.setenv(CW.ENV, "auto")
    run(pkg, warm_writing(root, 5))
    monkeypatch.setenv(CW.ENV, "force")
    assert run(pkg, warm_writing(root, 1, tag="f"))["warmed"]
    monkeypatch.setenv(CW.ENV, "bogus")
    with pytest.raises(ValueError):
        run(pkg, warm_writing(root, 1))


def test_code_change_rewarms(env):
    pkg, root = env
    run(pkg, warm_writing(root, 5))
    (pkg / "tt" / "a.py").write_text("x = 2\n")
    assert run(pkg, warm_writing(root, 0))["warmed"]
