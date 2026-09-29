# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The planar-concat extension builds itself into a cache dir on import; nothing here needs AVX2 or g++."""

import os
import subprocess

from models.tt_dit.utils import planar_concat as pc


def test_cache_dir_keys_on_sources_and_abi(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_DIT_PLANAR_CONCAT_CACHE", str(tmp_path))
    d1 = pc._cache_dir()
    assert d1.startswith(str(tmp_path))
    monkeypatch.setattr(pc, "_SOURCES", pc._SOURCES[1:])  # a different source set -> a different key
    assert pc._cache_dir() != d1


def test_build_is_invoked_once_into_the_cache_and_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_DIT_PLANAR_CONCAT_CACHE", str(tmp_path))
    monkeypatch.setattr(pc, "_host_can_build", lambda: True)
    calls = []

    def fake_run(cmd, env=None, **kw):
        calls.append(cmd)
        open(os.path.join(env["BUILD_DIR"], "_planar_concat.fake.so"), "wb").close()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(pc.subprocess if hasattr(pc, "subprocess") else subprocess, "run", fake_run)
    so = pc._build_into_cache()
    assert so and so.endswith("_planar_concat.fake.so") and os.path.dirname(so) == pc._cache_dir()
    assert len(calls) == 1 and calls[0][-1].endswith("build.sh")
    assert pc._build_into_cache() == so  # second import: cached, no rebuild
    assert len(calls) == 1


def test_build_failure_falls_back_quietly(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_DIT_PLANAR_CONCAT_CACHE", str(tmp_path))
    monkeypatch.setattr(pc, "_host_can_build", lambda: True)
    monkeypatch.setattr(
        subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "g++: error: no such file")
    )
    logged = []
    monkeypatch.setattr(pc, "_log", logged.append)
    assert pc._build_into_cache() is None
    assert logged and "torch fallback" in logged[0]


def test_build_disabled_by_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TT_DIT_PLANAR_CONCAT_CACHE", str(tmp_path))
    monkeypatch.setenv("TT_DIT_PLANAR_CONCAT_BUILD", "0")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not build")))
    assert pc._build_into_cache() is None


def test_hand_built_so_takes_precedence(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "_BUILD_DIR", str(tmp_path))
    open(tmp_path / "_planar_concat.hand.so", "wb").close()
    monkeypatch.setattr(pc, "_build_into_cache", lambda: (_ for _ in ()).throw(AssertionError("must not build")))
    monkeypatch.setattr(pc, "_load_so", lambda path: path)
    assert pc._try_load_extension().endswith("_planar_concat.hand.so")
