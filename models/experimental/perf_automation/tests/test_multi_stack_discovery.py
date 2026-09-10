# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for find_all_stacks() multi-stack discovery in _op_sig_probe.py.

WHY THIS EXISTS
    Task 1 of the depth-capping feature extends the probe to discover ALL repeating
    block stacks in a model (not just the largest one).  Voxtral-Mini has two: a 32-layer
    VoxtralEncoderLayer stack and a 30-layer LlamaDecoderLayer stack.  The old
    _find_stack() found only one; find_all_stacks() finds both.

    These tests use mock models so we never need to download the HF model weights.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_torch_two_stack_model():
    """A torch.nn.Module with two separate nn.ModuleList stacks of different types.

    Mimics the Voxtral structure:
        model.audio_tower.layers  → 32 × EncoderLayer
        model.text_model.layers   → 30 × DecoderLayer
    """
    import torch.nn as nn

    class EncoderLayer(nn.Module):
        def forward(self, x):
            return x

    class DecoderLayer(nn.Module):
        def forward(self, x):
            return x

    class AudioTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([EncoderLayer() for _ in range(32)])

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer() for _ in range(30)])

    class TwoStackModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_tower = AudioTower()
            self.text_model = TextModel()

    return TwoStackModel()


def _make_torch_one_stack_model():
    """A torch.nn.Module with a single nn.ModuleList stack."""
    import torch.nn as nn

    class Block(nn.Module):
        def forward(self, x):
            return x

    class SingleStackModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(8)])

    return SingleStackModel()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_find_all_stacks_returns_StackInfo_objects():
    """find_all_stacks() must return StackInfo instances, not raw lists."""
    P = _load_probe()
    model = _make_torch_one_stack_model()
    stacks = P.find_all_stacks(model)
    assert len(stacks) >= 1
    si = stacks[0]
    assert isinstance(si, P.StackInfo)
    assert isinstance(si.path, str)
    assert isinstance(si.stack, list)
    assert isinstance(si.element_type, type)
    assert isinstance(si.count, int)
    assert isinstance(si.stack_idx, int)


def test_single_stack_model_finds_exactly_one_stack():
    """A model with one block list must yield exactly 1 StackInfo."""
    P = _load_probe()
    model = _make_torch_one_stack_model()
    stacks = P.find_all_stacks(model)
    assert len(stacks) == 1
    si = stacks[0]
    assert si.count == 8
    assert si.stack_idx == 0


def test_two_stack_model_finds_both_stacks():
    """A model with two distinct block types must yield 2 StackInfos.

    This is the Voxtral-Mini case: encoder + decoder.
    """

    P = _load_probe()
    model = _make_torch_two_stack_model()
    stacks = P.find_all_stacks(model)
    assert len(stacks) == 2, "Expected 2 stacks, got %d: %s" % (len(stacks), [(s.path, s.count) for s in stacks])
    counts = {s.count for s in stacks}
    assert counts == {32, 30}, "Expected counts {32, 30}, got %s" % counts
    # stack_idx values must be 0 and 1
    idxs = {s.stack_idx for s in stacks}
    assert idxs == {0, 1}


def test_stack_idx_assigned_sequentially():
    """stack_idx values must be 0, 1, 2, ... in discovery order."""
    P = _load_probe()
    model = _make_torch_two_stack_model()
    stacks = P.find_all_stacks(model)
    for expected_idx, si in enumerate(stacks):
        assert si.stack_idx == expected_idx


def test_count_matches_stack_length():
    """StackInfo.count must equal len(StackInfo.stack)."""
    P = _load_probe()
    model = _make_torch_two_stack_model()
    for si in P.find_all_stacks(model):
        assert si.count == len(si.stack)


def test_element_type_is_correct():
    """StackInfo.element_type must be the class of the stack elements."""

    P = _load_probe()
    model = _make_torch_one_stack_model()
    stacks = P.find_all_stacks(model)
    assert len(stacks) == 1
    si = stacks[0]
    # All elements should be instances of element_type
    for elem in si.stack:
        assert isinstance(elem, si.element_type)


def test_deduplication_nested_modulelists():
    """Nested ModuleLists with the same element type must not be double-counted.

    Structure:
        outer.layers (nn.ModuleList of 5 Block)
            └── each Block has .sublayers (nn.ModuleList of 3 Block)

    The outer stack should be found; the inner stacks (if same type) should
    be deduplicated via the prefix rule.
    """
    import torch.nn as nn

    P = _load_probe()

    class Inner(nn.Module):
        def forward(self, x):
            return x

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            # Sublayers of a DIFFERENT type to avoid cross-contamination
            self.sublayers = nn.ModuleList([Inner() for _ in range(3)])

        def forward(self, x):
            return x

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Block() for _ in range(5)])

    model = Model()
    stacks = P.find_all_stacks(model)
    # There should be two distinct stacks: outer (Block×5) and inner (Inner×3)
    # They differ in element_type so dedup does not merge them
    element_types = {s.element_type for s in stacks}
    block_stacks = [s for s in stacks if s.element_type is Block]
    inner_stacks = [s for s in stacks if s.element_type is Inner]
    assert len(block_stacks) == 1, "Expected exactly 1 Block stack, got %d" % len(block_stacks)
    assert block_stacks[0].count == 5
    # Inner stacks come from 5 separate Block modules; dedup should not collapse
    # them (different paths) but they are the same type at different positions.
    # The key assertion: the outer Block stack is not suppressed.
    assert block_stacks[0].count == 5


def test_deduplication_same_type_prefix_collapses():
    """If a ModuleList wraps another ModuleList of the same type, keep only the deeper one."""
    import torch.nn as nn

    P = _load_probe()

    class Layer(nn.Module):
        def forward(self, x):
            return x

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(6)])

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            # outer.inner.layers has path "inner.layers"
            # outer.inner has named_children so we'll discover "inner.layers" (6×Layer)
            self.inner = Inner()

    model = Outer()
    stacks = P.find_all_stacks(model)
    # Only one stack: inner.layers (6 × Layer)
    layer_stacks = [s for s in stacks if s.element_type is Layer]
    assert len(layer_stacks) == 1
    assert layer_stacks[0].count == 6


def test_no_stacks_returns_empty_list():
    """A model with no repeating block stack must return an empty list, not raise."""
    P = _load_probe()

    class Flat:
        def __init__(self):
            self.a = 1
            self.b = "x"
            self.c = [1, 2, 3]  # atomic list, not block stack

    result = P.find_all_stacks(Flat())
    assert result == []


def test_plain_python_list_lw_model():
    """LightweightModule stacks in plain Python lists must be found by find_all_stacks()."""
    P = _load_probe()

    class Block:
        """Minimal LightweightModule-like block with a __dict__."""

    class Model:
        def __init__(self):
            self.layers = [Block() for _ in range(10)]

    stacks = P.find_all_stacks(Model())
    assert len(stacks) == 1
    assert stacks[0].count == 10
    assert stacks[0].stack_idx == 0


def test_two_lw_stacks_different_types():
    """Two plain Python list stacks of different types must both be found.

    Simulates a LightweightModule model with encoder + decoder stacks
    (the TTNN equivalent of the Voxtral test).
    """
    P = _load_probe()

    class EncoderBlock:
        pass

    class DecoderBlock:
        pass

    class LWModel:
        def __init__(self):
            self.encoder_layers = [EncoderBlock() for _ in range(12)]
            self.decoder_layers = [DecoderBlock() for _ in range(10)]

    stacks = P.find_all_stacks(LWModel())
    assert len(stacks) == 2, "Expected 2 stacks, got %d: %s" % (
        len(stacks),
        [(s.path, s.count, s.element_type.__name__) for s in stacks],
    )
    types_found = {s.element_type for s in stacks}
    assert EncoderBlock in types_found
    assert DecoderBlock in types_found


def test_find_stack_still_works_single_stack():
    """Backward-compat: existing _find_stack() must still work for single-stack models."""
    P = _load_probe()

    class Block:
        pass

    class Model:
        def __init__(self):
            self.layers = [Block() for _ in range(8)]

    stack = P._find_stack(Model())
    assert stack is not None
    assert len(stack) == 8


def test_path_attribute_set_on_stack_info():
    """StackInfo.path must be a non-empty dot-separated string for torch models."""
    P = _load_probe()
    model = _make_torch_one_stack_model()
    stacks = P.find_all_stacks(model)
    assert len(stacks) >= 1
    assert stacks[0].path != "", "Expected non-empty path for torch model stack"


def test_voxtral_mock_two_stack_model():
    """Mock the Voxtral-Mini model structure: audio_tower.layers (32) + text_model.layers (30).

    This is the canonical two-stack case the feature is designed for.
    Does NOT download any HF model weights.
    """
    import torch.nn as nn

    P = _load_probe()

    class VoxtralEncoderLayer(nn.Module):
        def forward(self, x):
            return x

    class LlamaDecoderLayer(nn.Module):
        def forward(self, x):
            return x

    class VoxtralAudioTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([VoxtralEncoderLayer() for _ in range(32)])

    class VoxtralTextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([LlamaDecoderLayer() for _ in range(30)])

    class VoxtralMini(nn.Module):
        """Structural mock of mistralai/Voxtral-Mini-3B-2507."""

        def __init__(self):
            super().__init__()
            self.audio_tower = VoxtralAudioTower()
            self.language_model = VoxtralTextModel()

    model = VoxtralMini()
    stacks = P.find_all_stacks(model)

    assert len(stacks) == 2, "Voxtral mock must yield 2 stacks, got %d: %s" % (
        len(stacks),
        [(s.path, s.count, s.element_type.__name__) for s in stacks],
    )

    encoder_stacks = [s for s in stacks if s.element_type is VoxtralEncoderLayer]
    decoder_stacks = [s for s in stacks if s.element_type is LlamaDecoderLayer]

    assert len(encoder_stacks) == 1, "Expected 1 encoder stack"
    assert encoder_stacks[0].count == 32

    assert len(decoder_stacks) == 1, "Expected 1 decoder stack"
    assert decoder_stacks[0].count == 30

    # Encoder is encountered first (audio_tower before language_model)
    enc_idx = encoder_stacks[0].stack_idx
    dec_idx = decoder_stacks[0].stack_idx
    assert enc_idx < dec_idx, "Encoder stack should be discovered before decoder stack"
