# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The PR gate and the merge gate must split the suite the same way.

Both gates judge the same code against the same thresholds. The split decides
which tests share a board and in which order, and a kernel's cycles depend on
what ran before it on that core. If the two gates split differently, a PR that
passes one can be flagged by the other for a reason that is not in the code.
"""

import pathlib
import re

import pytest
import yaml

_ROOT = pathlib.Path(__file__).parents[5]
_LISTS = _ROOT / "tests" / "pipeline_reorg"
_PR_GATE = _LISTS / "llk_perf_pr_gate_tests.yaml"
_MERGE_GATE = _LISTS / "llk_perf_merge_gate_tests.yaml"

pytestmark = pytest.mark.skipif(
    not (_PR_GATE.exists() and _MERGE_GATE.exists()),
    reason="the pipeline lists are not in this checkout",
)

ARCHES = ("blackhole", "wormhole")


def _entries(path):
    return yaml.safe_load(path.read_text()) or []


def _arch_of(name):
    for arch in ARCHES:
        if arch in name:
            return arch
    return None


def splits_per_arch(path):
    """``{arch: {split_group, ...}}`` the list asks for."""
    out = {}
    for entry in _entries(path):
        arch = _arch_of(str(entry.get("name") or ""))
        if arch:
            out.setdefault(arch, set()).add(entry["split_group"])
    return out


def split_counts_in_cmd(path):
    """``{arch: {total, ...}}`` -- the split count each command passes to the runner."""
    out = {}
    for entry in _entries(path):
        arch = _arch_of(str(entry.get("name") or ""))
        m = re.search(r"\{split_group\}\s+(\d+)", str(entry.get("cmd") or ""))
        if arch and m:
            out.setdefault(arch, set()).add(int(m.group(1)))
    return out


@pytest.mark.parametrize("arch", ARCHES)
def test_both_gates_use_the_same_number_of_groups(arch):
    assert splits_per_arch(_PR_GATE)[arch] == splits_per_arch(_MERGE_GATE)[arch]


@pytest.mark.parametrize("path", [_PR_GATE, _MERGE_GATE])
@pytest.mark.parametrize("arch", ARCHES)
def test_the_group_numbers_cover_the_split_with_no_gap(path, arch):
    groups = splits_per_arch(path)[arch]
    assert groups == set(range(1, len(groups) + 1))


@pytest.mark.parametrize("path", [_PR_GATE, _MERGE_GATE])
@pytest.mark.parametrize("arch", ARCHES)
def test_every_leg_tells_the_runner_the_same_total(path, arch):
    totals = split_counts_in_cmd(path)[arch]
    assert totals == {len(splits_per_arch(path)[arch])}
