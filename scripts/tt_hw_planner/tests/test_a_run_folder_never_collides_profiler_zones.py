# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A run folder whose path makes two device-profiler zone labels share a 16-bit hash is never used.

WH Galaxy, 2026-09-28: in /tmp/tt_hw_planner_qwen_image_edit_1790623639 tt-metal threw "Source
location hashes are colliding" for the two "QKT@V MM+Pack" zones of the SDPA streaming kernel, every
later profiler read failed, and half the baseline's ops had no device time. The folder is part of
each label (__FILE__), so the tool now checks a folder before it creates a checkout there.
"""

from pathlib import Path

import pytest

from scripts.tt_hw_planner import profiler_zone_names as z
from scripts.tt_hw_planner.discovery import REPO_ROOT

_INCIDENT_ROOT = "/tmp/tt_hw_planner_qwen_image_edit_1790623639"
_TAIL = "/ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp"


def _consts():
    c = z._hash_constants(Path(REPO_ROOT).resolve())
    if c is None:
        pytest.skip("tt-metal's profiler.cpp is not in this checkout")
    return c


def test_the_definitions_are_read_from_tt_metal_itself():
    repo = Path(REPO_ROOT).resolve()
    if not (repo / z._PROFILER_HEADER).is_file():
        pytest.skip("tt-metal sources are not in this checkout")
    assert z._label_suffix(repo) and z._hash_constants(repo)
    assert z._zone_sites(repo), "no device zone found in the kernel trees"


def test_the_hash_is_tt_metals():
    """The exact pair from the incident: equal under that folder, different under another."""
    c = _consts()
    a = "QKT@V MM+Pack,%s%s,1680,KERNEL_PROFILER"
    b = "QKT@V MM+Pack,%s%s,1789,KERNEL_PROFILER"
    assert z.hash16(a % (_INCIDENT_ROOT, _TAIL), c) == z.hash16(b % (_INCIDENT_ROOT, _TAIL), c) == 42956
    other = "/home/somewhere/else"
    assert z.hash16(a % (other, _TAIL), c) != z.hash16(b % (other, _TAIL), c)


def _mini_repo(tmp_path: Path) -> Path:
    """A tt-metal-shaped tree: the real header and profiler.cpp lines, one kernel with two zones."""
    r = tmp_path / "repo"
    (r / z._PROFILER_HEADER).parent.mkdir(parents=True)
    (r / z._PROFILER_HEADER).write_text(
        "#define Stringize(L) #L\n"
        '#define PROFILER_MSG __FILE__ "," $Line ",KERNEL_PROFILER"\n'
        '#define PROFILER_MSG_NAME(name) name "," PROFILER_MSG\n'
        "#define DeviceZoneScopedN(name) \\\n    DO_PRAGMA(message(PROFILER_MSG_NAME(name)));\n"
    )
    (r / z._PROFILER_IMPL).parent.mkdir(parents=True)
    (r / z._PROFILER_IMPL).write_text(
        "uint32_t hash32CT(const char* str, size_t n, uint32_t basis) {\n"
        "    return n == 0 ? basis : hash32CT(str + 1, n - 1, (basis ^ str[0]) * UINT32_C(16777619));\n}\n"
        "uint16_t hash16CT(const std::string& str) {\n"
        "    uint32_t res = hash32CT(str.c_str(), str.length(), UINT32_C(2166136261));\n}\n"
    )
    k = r / "ttnn" / "kern.hpp"
    k.parent.mkdir(parents=True)
    k.write_text(
        "#define MaybeZone(EN, name) DeviceZoneScopedN(name)\n"
        '    MaybeZone(true, "Same");\n'
        '    DeviceZoneScopedN("Other");\n'
        '    MaybeZone(true, "Same");\n'
    )
    return r


def test_zones_are_found_through_wrapper_macros(tmp_path):
    r = _mini_repo(tmp_path)
    assert sorted(z._zone_sites(r.resolve())) == [
        ("Other", "ttnn/kern.hpp", 3),
        ("Same", "ttnn/kern.hpp", 2),
        ("Same", "ttnn/kern.hpp", 4),
    ]


def test_a_colliding_folder_is_reported_and_a_clean_one_is_not(tmp_path, monkeypatch):
    r = _mini_repo(tmp_path).resolve()
    monkeypatch.setattr(z, "hash16", lambda label, c: 7 if "/bad/" in label else hash(label) & 0xFFFFFF)
    assert z.colliding_labels([tmp_path / "good"], r) == []
    assert z.colliding_labels([tmp_path / "bad"], r), "every label under /bad/ shares one hash"


def test_the_next_name_is_taken_when_the_first_collides(tmp_path, monkeypatch):
    r = _mini_repo(tmp_path).resolve()
    first = tmp_path / "run_1"
    monkeypatch.setattr(z, "colliding_labels", lambda roots, repo: [("a", "b")] if Path(roots[0]) == first else [])
    assert z.collision_free(first, r) == tmp_path / "run_1_1"


def test_an_existing_folder_is_skipped(tmp_path, monkeypatch):
    r = _mini_repo(tmp_path).resolve()
    (tmp_path / "run_2").mkdir()
    monkeypatch.setattr(z, "colliding_labels", lambda roots, repo: [])
    assert z.collision_free(tmp_path / "run_2", r) == tmp_path / "run_2_1"


def test_unreadable_definitions_do_not_block_a_run(tmp_path, capsys):
    empty = tmp_path / "not_tt_metal"
    empty.mkdir()
    assert z.colliding_labels([tmp_path / "x"], empty) is None
    assert z.collision_free(tmp_path / "x", empty) == tmp_path / "x"
    assert "was not checked" in capsys.readouterr().err


def test_both_folder_makers_use_it():
    import inspect

    from scripts.tt_hw_planner import worktree
    from scripts.tt_hw_planner._cli_helpers import agent_worktree_pool as pool

    assert "collision_free(path, REPO_ROOT)" in inspect.getsource(worktree.create)
    assert "collision_free(" in inspect.getsource(pool.AgentWorktreePool._unique_path)
