# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for stack-prefixed signpost emission in _op_sig_probe.py (Task 2).

WHY THIS EXISTS
    Task 2 extends _install_block_signposts() to handle models with MULTIPLE block
    stacks (e.g. Voxtral-Mini: encoder + decoder).  The signpost format becomes:
        multi-stack:  PERF_BLOCK_SIGNPOST:stack{si}:{j}
        single-stack: PERF_BLOCK_SIGNPOST:{j}   (backward compat)

    The _STACK_TAG attribute (_perf_stack_idx) is also set on every tagged element
    so consumers can correlate signposts with the originating stack.

All tests use mock torch models — no HF model weights are downloaded.
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


def _signposts(seq, prefix):
    return [t for t in seq if isinstance(t, str) and t.startswith(prefix)]


# ---------------------------------------------------------------------------
# Test 1: Two-stack torch model emits stack-prefixed signposts
# ---------------------------------------------------------------------------


def test_two_stack_torch_model_emits_prefixed_signposts():
    """A torch model with two stacks must emit PERF_BLOCK_SIGNPOST:stack0:N and stack1:M.

    This is the Voxtral-Mini shape: audio_tower.layers (4 × EncoderLayer) and
    text_model.layers (3 × DecoderLayer).  We use short stacks to keep the test fast.
    The probe must emit both stack0 and stack1 prefixed signposts when the forward
    pass invokes all blocks from both stacks.
    """
    import torch
    import torch.nn as nn

    P = _load_probe()

    _saved_torch = torch.nn.Module.__call__
    try:

        class EncoderLayer(nn.Module):
            def forward(self, x):
                return x

        class DecoderLayer(nn.Module):
            def forward(self, x):
                return x

        class AudioTower(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([EncoderLayer() for _ in range(4)])

            def forward(self, x):
                for b in self.layers:
                    x = b(x)
                return x

        class TextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([DecoderLayer() for _ in range(3)])

            def forward(self, x):
                for b in self.layers:
                    x = b(x)
                return x

        class TwoStackModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.audio_tower = AudioTower()
                self.text_model = TextModel()

            def forward(self, x):
                x = self.audio_tower(x)
                return self.text_model(x)

        P._SEQ.clear()
        P._install_block_signposts()
        model = TwoStackModel()
        model(torch.zeros(1))

        signposts = _signposts(P._SEQ, P._SIGNPOST_PREFIX)

        # Must contain stack0-prefixed entries for the encoder stack
        stack0_posts = [s for s in signposts if ":stack0:" in s]
        assert len(stack0_posts) > 0, "No stack0 signposts found; got: %s" % signposts

        # Must contain stack1-prefixed entries for the decoder stack
        stack1_posts = [s for s in signposts if ":stack1:" in s]
        assert len(stack1_posts) > 0, "No stack1 signposts found; got: %s" % signposts

        # Verify exact signpost values for each stack
        expected_stack0 = ["%sstack0:%d" % (P._SIGNPOST_PREFIX, i) for i in range(4)]
        expected_stack1 = ["%sstack1:%d" % (P._SIGNPOST_PREFIX, i) for i in range(3)]

        for sp in expected_stack0:
            assert sp in signposts, "Missing signpost %r; got %s" % (sp, signposts)
        for sp in expected_stack1:
            assert sp in signposts, "Missing signpost %r; got %s" % (sp, signposts)

    finally:
        torch.nn.Module.__call__ = _saved_torch
        P._SEQ.clear()


# ---------------------------------------------------------------------------
# Test 2: Single-stack torch model preserves backward-compat format
# ---------------------------------------------------------------------------


def test_single_stack_torch_model_emits_unprefixed_signposts():
    """A torch model with exactly one block stack must emit PERF_BLOCK_SIGNPOST:N (no stack prefix).

    Backward compatibility: existing consumers that split on ':' and parse an integer
    must continue to work.
    """
    import torch
    import torch.nn as nn

    P = _load_probe()

    _saved_torch = torch.nn.Module.__call__
    try:

        class Block(nn.Module):
            def forward(self, x):
                return x

        class SingleStackModel(nn.Module):
            def __init__(self):
                super().__init__()
                # >8 total submodules so the tagger gate fires (same as test_block_signposts.py)
                self.layers = nn.ModuleList([Block() for _ in range(10)])

            def forward(self, x):
                for b in self.layers:
                    x = b(x)
                return x

        P._SEQ.clear()
        P._install_block_signposts()
        model = SingleStackModel()
        model(torch.zeros(1))

        signposts = _signposts(P._SEQ, P._SIGNPOST_PREFIX)

        # Must NOT contain "stack" prefix
        prefixed = [s for s in signposts if "stack" in s]
        assert prefixed == [], "Unexpected stack-prefixed signposts for single-stack model: %s" % prefixed

        # Must contain exactly the old-format entries
        expected = ["%s%d" % (P._SIGNPOST_PREFIX, i) for i in range(10)]
        for sp in expected:
            assert sp in signposts, "Missing signpost %r; got %s" % (sp, signposts)

    finally:
        torch.nn.Module.__call__ = _saved_torch
        P._SEQ.clear()


# ---------------------------------------------------------------------------
# Test 3: _perf_stack_idx attribute is set on every tagged element
# ---------------------------------------------------------------------------


def test_stack_tag_attribute_set_on_elements():
    """_perf_stack_idx must be set on every block in every stack.

    For a two-stack model:
      - encoder blocks must have _perf_stack_idx == 0 (or 1, whichever is first)
      - decoder blocks must have _perf_stack_idx == 1 (or 0 respectively)
      - both groups must have _perf_block_idx 0..N-1
    """
    import torch.nn as nn

    P = _load_probe()

    class EncoderLayer(nn.Module):
        def forward(self, x):
            return x

    class DecoderLayer(nn.Module):
        def forward(self, x):
            return x

    class AudioTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([EncoderLayer() for _ in range(4)])

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DecoderLayer() for _ in range(3)])

    class TwoStackModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.audio_tower = AudioTower()
            self.text_model = TextModel()

    model = TwoStackModel()

    # Tag by calling find_all_stacks + _tag_all_stacks directly (no hooks needed)
    stacks = P.find_all_stacks(model)
    assert len(stacks) == 2, "Expected 2 stacks, got %d" % len(stacks)

    P._tag_all_stacks(stacks)

    enc_stack = next(s for s in stacks if s.element_type is EncoderLayer)
    dec_stack = next(s for s in stacks if s.element_type is DecoderLayer)

    # Every encoder block must have correct _perf_block_idx and _perf_stack_idx
    for j, blk in enumerate(enc_stack.stack):
        assert getattr(blk, P._BLOCK_TAG) == j
        assert getattr(blk, P._STACK_TAG) == enc_stack.stack_idx

    # Every decoder block likewise
    for j, blk in enumerate(dec_stack.stack):
        assert getattr(blk, P._BLOCK_TAG) == j
        assert getattr(blk, P._STACK_TAG) == dec_stack.stack_idx

    # The two stacks must have different stack_idx values
    assert enc_stack.stack_idx != dec_stack.stack_idx


# ---------------------------------------------------------------------------
# Test 4: LightweightModule two-stack model emits prefixed signposts
# ---------------------------------------------------------------------------


def test_two_stack_lw_model_emits_prefixed_signposts():
    """The TTNN/LightweightModule path must also emit stack-prefixed signposts for 2-stack models."""
    P = _load_probe()
    from models.common.lightweightmodule import LightweightModule

    _saved_lw = LightweightModule.__call__
    try:

        class EncoderBlock(LightweightModule):
            def forward(self):
                return 1

        class DecoderBlock(LightweightModule):
            def forward(self):
                return 2

        class TwoStackLWModel(LightweightModule):
            def __init__(self):
                self.encoder_layers = [EncoderBlock() for _ in range(4)]
                self.decoder_layers = [DecoderBlock() for _ in range(3)]

            def forward(self):
                enc = [b() for b in self.encoder_layers]
                dec = [b() for b in self.decoder_layers]
                return enc, dec

        P._SEQ.clear()
        P._install_block_signposts()
        TwoStackLWModel()()

        signposts = _signposts(P._SEQ, P._SIGNPOST_PREFIX)

        stack0_posts = [s for s in signposts if ":stack0:" in s]
        stack1_posts = [s for s in signposts if ":stack1:" in s]

        assert len(stack0_posts) > 0, "No stack0 signposts; got: %s" % signposts
        assert len(stack1_posts) > 0, "No stack1 signposts; got: %s" % signposts

        # Combined, both stacks' indices 0..3 and 0..2 must appear
        all_stack_posts = stack0_posts + stack1_posts
        assert len(all_stack_posts) == 7, "Expected 4+3=7 stack signposts, got %d: %s" % (
            len(all_stack_posts),
            all_stack_posts,
        )

    finally:
        LightweightModule.__call__ = _saved_lw
        P._SEQ.clear()


# ---------------------------------------------------------------------------
# Test 5: Single-stack LightweightModule model preserves backward-compat format
# ---------------------------------------------------------------------------


def test_single_stack_lw_model_emits_unprefixed_signposts():
    """LightweightModule single-stack model must emit the old-style signposts (no stack prefix)."""
    P = _load_probe()
    from models.common.lightweightmodule import LightweightModule

    _saved_lw = LightweightModule.__call__
    try:

        class Block(LightweightModule):
            def forward(self):
                return 1

        class SingleStackLWModel(LightweightModule):
            def __init__(self):
                self.layers = [Block() for _ in range(5)]

            def forward(self):
                return [b() for b in self.layers]

        P._SEQ.clear()
        P._install_block_signposts()
        SingleStackLWModel()()

        signposts = _signposts(P._SEQ, P._SIGNPOST_PREFIX)

        # Must NOT contain any stack prefix
        prefixed = [s for s in signposts if "stack" in s]
        assert prefixed == [], "Unexpected stack-prefixed signposts: %s" % prefixed

        expected = ["%s%d" % (P._SIGNPOST_PREFIX, i) for i in range(5)]
        for sp in expected:
            assert sp in signposts, "Missing %r; got %s" % (sp, signposts)

    finally:
        LightweightModule.__call__ = _saved_lw
        P._SEQ.clear()


# ---------------------------------------------------------------------------
# Test 6: Signpost format strings are correct
# ---------------------------------------------------------------------------


def test_signpost_format_strings():
    """Verify the exact format of multi-stack and single-stack signpost strings."""
    P = _load_probe()

    # Multi-stack: stack{si}:{j}
    assert P._SIGNPOST_PREFIX == "PERF_BLOCK_SIGNPOST:"

    multi = "%sstack%d:%d" % (P._SIGNPOST_PREFIX, 0, 5)
    assert multi == "PERF_BLOCK_SIGNPOST:stack0:5"

    multi2 = "%sstack%d:%d" % (P._SIGNPOST_PREFIX, 1, 12)
    assert multi2 == "PERF_BLOCK_SIGNPOST:stack1:12"

    # Single-stack: old format
    single = "%s%d" % (P._SIGNPOST_PREFIX, 7)
    assert single == "PERF_BLOCK_SIGNPOST:7"

    # _STACK_TAG constant exists
    assert hasattr(P, "_STACK_TAG")
    assert P._STACK_TAG == "_perf_stack_idx"
