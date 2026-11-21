"""Local tests for the compute-kernel API-ordering checker. No hardware required.

Every rule has a positive kernel (must be reported) and, where the rule can be
satisfied, a negative kernel (must stay silent).
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import check  # noqa: E402

KERNELS = os.path.join(HERE, "kernels")
CHECKER = check.Checker(check.load_rules(os.path.join(os.path.dirname(HERE), "rules.yaml")))


def hits(name):
    return CHECKER.check(os.path.join(KERNELS, name))


def rules(name):
    return [h[2] for h in hits(name)]


def test_missing_init_reported():
    assert "missing add_init/binary_tiles_init/add_reuse_dest_init()" in rules("pos_missing_init.cpp")


def test_binary_tiles_init_alias_accepted():
    assert hits("neg_binary_tiles_init.cpp") == []


def test_reconfig_invalidates_block_init():
    assert "missing tilize_init/tilize_init_short()" in rules("pos_reconfig_invalidates_tilize.cpp")


def test_clean_tilize_kernel_silent():
    assert hits("neg_clean_tilize.cpp") == []


def test_missing_uninit_reported():
    assert "never paired with tilize_uninit()" in rules("pos_missing_uninit.cpp")


def test_wrong_init_for_matmul_reported():
    assert "missing matmul_init/mm_init()" in rules("pos_matmul_mm_init_alias.cpp")


def test_every_positive_kernel_has_a_finding():
    for name in sorted(os.listdir(KERNELS)):
        if name.startswith("pos_"):
            assert hits(name), f"{name} was expected to produce a finding"
