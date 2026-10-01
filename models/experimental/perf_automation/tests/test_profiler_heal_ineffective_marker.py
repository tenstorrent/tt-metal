# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The "heal ineffective" verdict must be tied to the library it was measured against.

profiler_heal writes a ``.perfauto_heal_ineffective`` sentinel when patch+rebuild fails to
land the orphan-skip marker in the loaded ``libtt_metal.so``, so later runs skip a known
3-minute round trip. The sentinel used to live only on the build DIRECTORY. But worktrees
share one build tree, so a verdict recorded from a throwaway worktree (or from before the
library was rebuilt) would permanently disable healing for every future run -- the profiler
patch never got a second chance, and mesh profiling kept crashing on orphan markers.

The verdict is now keyed to the loaded library's identity (path/size/mtime). It is honored
only while that exact library is still in place, and a verdict against a since-changed or
vanished library is stale: it is ignored AND cleared so the heal is attempted again.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _heal():
    from agent import profiler_heal

    return profiler_heal


def _make_build(tmp_path):
    build = tmp_path / "build_Release"
    build.mkdir()
    return build


def _make_lib(tmp_path, name="libtt_metal.so", data=b"stock"):
    lib = tmp_path / name
    lib.write_bytes(data)
    return lib


def test_verdict_is_honored_for_the_same_library(tmp_path):
    h = _heal()
    build = _make_build(tmp_path)
    lib = _make_lib(tmp_path)
    h._mark_heal_ineffective(build, lib, "marker absent after ninja rebuild")
    # Same library, unchanged -> the known-failing rebuild is correctly skipped.
    assert h._heal_marked_ineffective(build, lib) is True
    assert (build / h._INEFFECTIVE_SENTINEL).is_file()


def test_verdict_is_stale_when_the_library_changed(tmp_path):
    h = _heal()
    build = _make_build(tmp_path)
    lib = _make_lib(tmp_path, data=b"old")
    h._mark_heal_ineffective(build, lib, "marker absent after ninja rebuild")
    # The library gets rebuilt/replaced -> its identity changes -> verdict is stale.
    lib.write_bytes(b"a-different-and-longer-library-image")
    assert h._heal_marked_ineffective(build, lib) is False
    # And the stale sentinel is cleared, so the heal retries once against the new library.
    assert not (build / h._INEFFECTIVE_SENTINEL).is_file()


def test_verdict_from_another_build_tree_does_not_stick(tmp_path):
    """The exact shape of the bug: a verdict written against a now-gone worktree library
    must not block healing of the library that is actually loaded now."""
    h = _heal()
    build = _make_build(tmp_path)
    gone = tmp_path / "worktree" / "build_Release" / "lib" / "libtt_metal.so"
    gone.parent.mkdir(parents=True)
    gone.write_bytes(b"worktree-image")
    h._mark_heal_ineffective(build, gone, "marker absent after ninja rebuild")
    gone.unlink()  # the throwaway worktree is cleaned up
    current = _make_lib(tmp_path)
    assert h._heal_marked_ineffective(build, current) is False


def test_legacy_plaintext_sentinel_is_treated_as_stale(tmp_path):
    """A sentinel written by the OLD (identity-less) code has a free-text first line that
    cannot match any real identity, so it is stale and cleared rather than sticking."""
    h = _heal()
    build = _make_build(tmp_path)
    (build / h._INEFFECTIVE_SENTINEL).write_text("marker absent from /tmp/gone/libtt_metal.so after ninja rebuild")
    lib = _make_lib(tmp_path)
    assert h._heal_marked_ineffective(build, lib) is False
    assert not (build / h._INEFFECTIVE_SENTINEL).is_file()


def test_no_sentinel_means_not_ineffective(tmp_path):
    h = _heal()
    build = _make_build(tmp_path)
    lib = _make_lib(tmp_path)
    assert h._heal_marked_ineffective(build, lib) is False


def test_lib_identity_distinguishes_size_and_absence(tmp_path):
    h = _heal()
    lib = _make_lib(tmp_path, data=b"abc")
    ident = h._lib_identity(lib)
    assert h._lib_identity(None) == "none"
    lib.write_bytes(b"abcd")  # size changes
    assert h._lib_identity(lib) != ident
