# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The op-sig probe must emit per-block signposts for BOTH shapes a tt-metal model comes in.

WHY THIS EXISTS
    The signpost stream is what lets run.py size the tracy window from ONE all-layers probe: each
    op is attributed to the block it ran in, so the smallest depth holding every op type falls out
    directly. When no signposts arrive, run.py falls back to `_measure_cov`, which probes depth
    2/4/8/16 and diffs the op sets -- four extra device runs to recover what the signposts already
    knew.

    Measured on llama3_1_8b_p150 (2026-07-26): `full_blocks=0`, so the ladder ran to exhaustion and
    the run paid 5 probes instead of 1. The cause was shape, not the model: the detector looked only
    for `torch.nn.ModuleList` under `torch.nn.Module`, while llama's blocks are `LightweightModule`
    instances in a PLAIN PYTHON LIST -- the standard TTNN shape, since models/common/
    lightweightmodule.py exists specifically to avoid torch's per-call overhead.

These tests pin both shapes so the torch path cannot regress while the TTNN path is added.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_PROBE = Path(__file__).resolve().parents[1] / "cc_optimize" / "_op_sig_probe.py"


def _load_probe():
    """Import the probe module directly; it is a script, not a package member."""
    spec = importlib.util.spec_from_file_location("_op_sig_probe_under_test", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_finds_plain_python_list_of_same_typed_blocks():
    """The TTNN shape: blocks in a plain list, reached through an attribute of any name."""
    P = _load_probe()

    class Block:
        pass

    class Model:
        def __init__(self):
            self.layers = [Block() for _ in range(32)]

    stack = P._largest_repeated_stack(Model())
    assert stack is not None and len(stack) == 32


def test_prefers_the_largest_stack_and_ignores_short_or_mixed_lists():
    """A model holds many lists; only the repeated BLOCK stack is the right one.

    Same-typedness plus length is the whole signal -- no attribute-name list, so it works on a model
    that calls them `blocks`, `h`, or anything else.
    """
    P = _load_probe()

    class Block:
        pass

    class Other:
        pass

    class Model:
        def __init__(self):
            self.norms = [Block(), Block()]  # same-typed but SHORTER
            self.mixed = [Block(), Other(), Block()]  # longer but NOT same-typed
            self.blocks = [Block() for _ in range(8)]  # the real stack
            self.scalars = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # no __dict__ -> not blocks

    stack = P._largest_repeated_stack(Model())
    assert stack is not None and len(stack) == 8


def test_finds_a_stack_nested_below_the_root():
    """Real pipelines wrap the stack (generator -> model -> layers), so the walk must descend."""
    P = _load_probe()

    class Block:
        pass

    class Inner:
        def __init__(self):
            self.layers = [Block() for _ in range(4)]

    class Outer:
        def __init__(self):
            self.model = Inner()

    stack = P._largest_repeated_stack(Outer())
    assert stack is not None and len(stack) == 4


def test_no_stack_returns_none_not_a_guess():
    """A model with no repeated stack must yield None, so the caller falls back rather than tagging
    something arbitrary as 'the layers'."""
    P = _load_probe()

    class Flat:
        def __init__(self):
            self.a = 1
            self.b = "x"

    assert P._largest_repeated_stack(Flat()) is None


def test_cycles_do_not_hang_the_walk():
    """Pipelines hold back-references (block.parent = model); the walk must terminate."""
    P = _load_probe()

    class Block:
        pass

    class Model:
        def __init__(self):
            self.layers = [Block() for _ in range(3)]
            for b in self.layers:
                b.parent = self  # cycle

    stack = P._largest_repeated_stack(Model())
    assert stack is not None and len(stack) == 3


def test_tag_stack_indexes_every_block():
    P = _load_probe()

    class Block:
        pass

    blocks = [Block() for _ in range(5)]
    assert P._tag_stack(blocks) is True
    assert [getattr(b, P._BLOCK_TAG) for b in blocks] == [0, 1, 2, 3, 4]
    assert P._tag_stack([]) is False


def test_lightweightmodule_call_emits_one_signpost_per_block_entry():
    """END TO END for the TTNN shape: calling the stack must append PERF_BLOCK_SIGNPOST:<idx>.

    This is the assertion that would have failed before the fix -- llama's blocks never went through
    torch.nn.Module.__call__, so nothing was emitted and full_blocks came back 0.
    """
    P = _load_probe()
    from models.common.lightweightmodule import LightweightModule

    _saved = LightweightModule.__call__
    try:

        class Block(LightweightModule):
            def forward(self):
                return 1

        class Model(LightweightModule):
            def __init__(self):
                self.layers = [Block() for _ in range(6)]

            def forward(self):
                return [b() for b in self.layers]

        P._SEQ.clear()
        P._install_block_signposts()
        Model()()

        signposts = [t for t in P._SEQ if isinstance(t, str) and t.startswith(P._SIGNPOST_PREFIX)]
        assert signposts == ["%s%d" % (P._SIGNPOST_PREFIX, i) for i in range(6)]
    finally:
        LightweightModule.__call__ = _saved
        P._SEQ.clear()


def test_torch_shape_still_emits_signposts():
    """The torch path must not regress: xtts_v2 and other torch-shaped models rely on it."""
    P = _load_probe()
    import torch

    _saved_torch = torch.nn.Module.__call__
    from models.common.lightweightmodule import LightweightModule

    _saved_lw = LightweightModule.__call__
    try:

        class Block(torch.nn.Module):
            def forward(self, x):
                return x

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # >8 submodules so the tagger's "this is the model, not a leaf" gate fires, and the
                # BLOCK stack is the LARGEST ModuleList (a bigger unrelated list would legitimately
                # be picked instead -- length is the signal).
                self.layers = torch.nn.ModuleList([Block() for _ in range(10)])

            def forward(self, x):
                for b in self.layers:
                    x = b(x)
                return x

        P._SEQ.clear()
        P._install_block_signposts()
        Model()(torch.zeros(1))

        signposts = [t for t in P._SEQ if isinstance(t, str) and t.startswith(P._SIGNPOST_PREFIX)]
        assert signposts == ["%s%d" % (P._SIGNPOST_PREFIX, i) for i in range(10)]
    finally:
        torch.nn.Module.__call__ = _saved_torch
        LightweightModule.__call__ = _saved_lw
        P._SEQ.clear()
