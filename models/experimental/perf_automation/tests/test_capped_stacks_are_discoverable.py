"""Pin: a stack built under a LAYER CAP must still be discoverable.

The G6 block-stack gate builds the model shallow on purpose (a shallow build is
seconds) and then asks whether every declared section is visible to the walk.
The cap lived in emit_e2e's probe and the member floor lived in _op_sig_probe,
and neither knew about the other: the probe capped every stack at 2 blocks while
the walk kept only lists of 3+. Every capped stack was therefore invisible, the
gate reported "7 sections, 2 stacks discoverable" for a model whose structure was
perfectly visible, and `termination_check` could never return can_stop.

The floor is a guard against mistaking a pair of unrelated submodules for a
stack -- it is not meant to exceed the depth the caller actually built.
"""

from __future__ import annotations

import torch.nn as nn

from models.experimental.perf_automation.cc_optimize import _op_sig_probe as P


class _Blk(nn.Module):
    def forward(self, x):  # pragma: no cover - never executed
        return x


def _pipe(n: int) -> nn.Module:
    class _Pipe(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([_Blk() for _ in range(n)])

    return _Pipe()


def test_floor_never_exceeds_the_build_cap():
    assert P.stack_member_floor(2) == 2, "a cap of 2 must not be judged by a floor of 3"
    assert P.stack_member_floor(1) == 2, "never below _is_block_stack's own minimum of 2"


def test_floor_is_unchanged_without_a_cap():
    """Full-depth walks keep the original behaviour, so nothing that passes today changes."""
    assert P.stack_member_floor(None) == P.MIN_STACK_MEMBERS == 3
    assert P.stack_member_floor(0) == 3, "a meaningless cap falls back to the default"
    assert P.stack_member_floor("nonsense") == 3, "an unparseable cap falls back, never raises"
    assert P.stack_member_floor(8) == 3, "a cap above the floor does not raise the floor"


def test_a_capped_stack_is_invisible_at_the_default_floor():
    """The bug, pinned: this is what the gate saw."""
    assert len(P.find_all_stacks(_pipe(2))) == 0


def test_the_same_stack_is_found_when_the_floor_honours_the_cap():
    found = P.find_all_stacks(_pipe(2), min_members=P.stack_member_floor(2))
    assert len(found) == 1, "a stack built at the cap must be discoverable"
    assert found[0].path.endswith("layers")


def test_uncapped_discovery_is_untouched():
    """Same input, same answer, with and without the new parameter."""
    for n in (3, 5):
        assert len(P.find_all_stacks(_pipe(n))) == 1
        assert len(P.find_all_stacks(_pipe(n), min_members=P.stack_member_floor(2))) == 1


def test_a_pair_of_unrelated_modules_is_still_not_a_stack():
    """Relaxing the floor must not turn any two-element list into a stack;
    _is_block_stack still has to agree."""

    class _Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixed = nn.ModuleList([nn.Linear(2, 2), nn.GELU()])

    assert P.find_all_stacks(_Odd(), min_members=P.stack_member_floor(2)) == []


def test_the_probe_sizes_its_build_and_its_floor_from_one_value():
    """The cap and the floor drifted apart because they were two literals in two
    files. Pin that the probe renders both from the same constant."""
    import re

    from pathlib import Path

    src = Path("scripts/tt_hw_planner/commands/emit_e2e.py").read_text()
    tpl = re.search(r'_STACK_PROBE = """(.*?)"""', src, re.S).group(1)
    assert "layers={cap}" in tpl, "the probe build must be sized by the cap placeholder"
    assert "stack_member_floor({cap})" in tpl, "the floor must come from the SAME cap"
