# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Signposts must survive the call shape tt_transformers actually uses.

Finding the stack is not enough -- the walk has to be ROOTED somewhere it can see one. The hook is
``LightweightModule.__call__``, and tt_transformers never calls its top module that way:

    generator.py:265   tt_out_trace = self.model[model_id].ttnn_prefill_forward(...)   <- a METHOD
    model.py:909       x = layer(...)                                                  <- first __call__

so the first module the wrapper sees is ``layers[0]``. Walking DOWN from one block finds attention,
MLP and norms -- never a 48-element stack -- so nothing is tagged and no signpost is emitted. That is
why gemma3 reported no block signposts and coverage fell to the unverified floor of 2.

These tests drive the real wrapper through that exact shape, on CPU, with no device: the blocks are
real LightweightModule subclasses and the invocation mirrors model.py:909.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_PROBE = _PA / "cc_optimize" / "_op_sig_probe.py"
sys.path.insert(0, str(_PA))


def _load_probe():
    spec = importlib.util.spec_from_file_location("_op_sig_probe_call_shape", _PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lw():
    try:
        from models.common.lightweightmodule import LightweightModule
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"LightweightModule unavailable: {exc}")
    return LightweightModule


def _build(n_layers=48, wrap_in_list=True):
    """A tt_transformers-shaped model: blocks in a plain list, top module called by METHOD."""
    LightweightModule = _lw()

    class Sub(LightweightModule):
        def forward(self):
            return None

    class Block(LightweightModule):
        def __init__(self):
            self.attn = Sub()
            self.mlp = Sub()

        def forward(self):
            return None

    class Transformer(LightweightModule):
        def __init__(self):
            self.layers = [Block() for _ in range(n_layers)]

        def ttnn_prefill_forward(self):
            for layer in self.layers:
                layer()

    class Generator:
        def __init__(self, model):
            self.model = [model] if wrap_in_list else model

    t = Transformer()
    return Generator(t), t


def test_the_wrapper_is_rooted_at_a_leaf_block_not_the_model():
    """Pins the premise: what the hook first sees is one block, not the stack."""
    probe = _load_probe()
    _, t = _build()
    first_seen = t.layers[0]
    assert probe._largest_repeated_stack(first_seen) is None, (
        "walking down from a single block should find no stack -- if this ever succeeds the "
        "premise of _enclosing_stack has changed"
    )


def test_enclosing_stack_recovers_the_full_depth_from_one_block():
    probe = _load_probe()
    _, t = _build(48)
    found = probe._enclosing_stack(t.layers[0])
    assert found is not None and len(found) == 48


def test_find_stack_covers_both_directions():
    probe = _load_probe()
    _, t = _build(48)
    assert len(probe._find_stack(t)) == 48, "rooted at the model: downward walk"
    assert len(probe._find_stack(t.layers[0])) == 48, "rooted at a block: upward recovery"


def test_signposts_are_emitted_for_every_block_in_order():
    """End to end through the REAL wrapper, driven exactly as model.py:909 drives it."""
    probe = _load_probe()
    LightweightModule = _lw()
    orig = LightweightModule.__call__
    try:
        _, t = _build(48)
        probe._SEQ.clear()
        probe._install_block_signposts()
        t.ttnn_prefill_forward()
        posts = [s for s in probe._SEQ if isinstance(s, str) and s.startswith(probe._SIGNPOST_PREFIX)]
        idxs = [int(s.split(":")[1]) for s in posts]
    finally:
        LightweightModule.__call__ = orig
    assert idxs, "no block signposts emitted -- coverage falls back to the unverified floor"
    assert sorted(set(idxs)) == list(range(48)), f"expected blocks 0..47, got {sorted(set(idxs))[:5]}..."
    assert idxs == sorted(idxs), "signposts must arrive in execution order"


def test_run_py_derives_full_depth_from_those_signposts():
    """The consumer's view: run.py must accept the sequence and size the window from it."""
    probe = _load_probe()
    LightweightModule = _lw()
    orig = LightweightModule.__call__
    try:
        _, t = _build(6)
        probe._SEQ.clear()
        probe._install_block_signposts()
        t.ttnn_prefill_forward()
        seq = list(probe._SEQ)
    finally:
        LightweightModule.__call__ = orig

    spec = importlib.util.spec_from_file_location("cc_run_signpost", str(_PA / "cc_optimize" / "run.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    assert run._blocks_ran(seq) == 6, f"run.py saw {run._blocks_ran(seq)} blocks, not 6"
    first_block, _ = run._first_block_map(seq)
    assert isinstance(first_block, dict)


@pytest.mark.parametrize("n", [2, 4, 32, 48])
def test_depth_is_recovered_for_any_stack_size(n):
    probe = _load_probe()
    _, t = _build(n)
    assert len(probe._find_stack(t.layers[0])) == n


def test_an_inner_stack_does_not_beat_the_enclosing_one():
    """REGRESSION, found on the live gemma3 model and missed by every synthetic case here.

    _find_stack originally returned `down or up`. Rooted at layers[0] -- the root the wrapper
    actually gets -- the downward walk found a 9-element list of sub-modules INSIDE the block and
    returned it, so the enclosing 48-block stack was never looked for. On the real generator that
    printed `block0=9`: nine sub-modules would have been tagged as the decoder stack, attributing
    every op to the wrong depth. Both directions must be evaluated and the better kept.
    """
    probe = _load_probe()
    LightweightModule = _lw()

    class Leaf(LightweightModule):
        def forward(self):
            return None

    class FatBlock(LightweightModule):
        def __init__(self):
            self.experts = [Leaf() for _ in range(9)]

        def forward(self):
            return None

    class Model(LightweightModule):
        def __init__(self):
            self.layers = [FatBlock() for _ in range(48)]

    m = Model()
    inner = probe._largest_repeated_stack(m.layers[0])
    assert inner is not None and len(inner) == 9, "fixture must reproduce the inner-stack trap"
    assert len(probe._find_stack(m.layers[0])) == 48, "the inner stack displaced the enclosing one"
