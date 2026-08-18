# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``generate_ttml`` must never sample an id outside ``config.vocab_size``.

The LM head's logits are wider than the vocabulary: padded to a multiple of 32
single-device, and to a multiple of ``32 * tp_size`` under TP (see
``tp_padded_vocab_size``). Those trailing columns are backed by zero weight rows,
so their logits sit at ~0 and win a greedy argmax whenever every real logit is
negative -- routine for a fine-tuned LM head. ``generate_ttml`` suppresses them by
passing a padding mask to ``ttml.ops.sample.sample_op``, which subtracts it before
the argmax.

The regression: that mask's width used to be computed as
``ceil(vocab_size / 32) * 32``, which is right single-device but undercounts under
TP. For a vocab that is already 32-aligned but not ``32 * tp_size``-aligned (Qwen3's
151936 at TP=8 is exactly this) it returns ``vocab_size`` itself, so
``_sample_logits_mask`` takes its ``orig_vocab >= padded_vocab`` early-out and
returns ``None`` -- no mask at all. ``generate_ttml`` now reads the width off the
actual post-all_gather logits.

Driving ``generate_ttml`` is what makes this a real test: the arithmetic that was
wrong lives in the caller, not in ``_sample_logits_mask``. A stub model supplies
logits wider than the vocab, standing in for a TP-padded LM head so the test needs
one device and no mesh.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import ttml
import ttnn

_QWEN3_EXAMPLE = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "sources", "examples", "qwen3")
)
# Force the qwen3 example dir to the front of sys.path so ``import utils`` resolves
# here even if a sibling example dir is already on the path.
if _QWEN3_EXAMPLE in sys.path:
    sys.path.remove(_QWEN3_EXAMPLE)
sys.path.insert(0, _QWEN3_EXAMPLE)

# The qwen3 example ships a top-level ``utils`` package, and so do sibling example
# dirs (e.g. examples/grpo, imported by test_grpo_trainer). In a single pytest
# session another test module may have already imported its own ``utils`` first,
# caching it in sys.modules and shadowing ours (sys.path.insert cannot override an
# already-imported module). Temporarily evict any cached ``utils*`` so the imports
# below resolve against _QWEN3_EXAMPLE, then restore the sibling's modules so we do
# not break whichever test imported them.
_saved_utils = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "utils" or k.startswith("utils.")}
try:
    from generate import _sample_logits_mask, generate_ttml  # noqa: E402
finally:
    for _k in [k for k in list(sys.modules) if k == "utils" or k.startswith("utils.")]:
        del sys.modules[_k]
    sys.modules.update(_saved_utils)

VOCAB = 128  # already 32-aligned, so ceil(V/32)*32 == V -> the old formula's blind spot
LOGITS_WIDTH = 256  # what a 32*tp_size-padded head actually emits (e.g. tp_size=8)
SEQ = 32
BEST_REAL_ID = 42
PROMPT = [1, 2, 3, 4]


@pytest.fixture(scope="module")
def device():
    """A device to run on, opening one only if this process has none.

    AutoContext holds at most one device and ``open_device`` raises if one already
    exists, so a sibling test module that left a device or mesh open would otherwise
    error every test here. Reuse whatever is open and close only what we opened.
    """
    ctx = ttml.autograd.AutoContext.get_instance()
    opened_here = True
    try:
        ctx.open_device()
    except RuntimeError:
        opened_here = False

    yield ctx.get_device()

    if opened_here:
        ctx.close_device()


class _StubTokenizer:
    def decode(self, ids):
        return "".join(f"<{i}>" for i in ids)


class _WideLogitsModel:
    """Emits [B, 1, SEQ, LOGITS_WIDTH] where the padding columns beat every real one.

    Real columns are negative (BEST_REAL_ID least so); the padding columns sit at
    0.0, exactly as zero-filled LM-head rows would. So an unmasked argmax lands on
    the first padding column and a correctly masked one lands on BEST_REAL_ID.
    """

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def eval(self):
        pass

    def __call__(self, input_tensor, attn_mask, past_key_values=None):
        data = np.full((self.batch_size, 1, SEQ, LOGITS_WIDTH), -1.0, dtype=np.float32)
        data[:, :, :, BEST_REAL_ID] = -0.5
        data[:, :, :, VOCAB:] = 0.0
        return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.bfloat16)


def _generate(device, max_tokens=3):
    config = SimpleNamespace(vocab_size=VOCAB, num_hidden_layers=1)
    generated, _ = generate_ttml(
        _WideLogitsModel(1),
        config,
        _StubTokenizer(),
        [list(PROMPT)],
        max_tokens,
        SEQ,
        device,
        temperature=0.0,  # skips the Gumbel path -> deterministic argmax
        collect_logits=False,
        distributed=False,
        dp_size=1,
        tp_size=1,
        kv_cache=False,
    )
    return generated[0]


def test_generate_never_samples_outside_the_vocabulary(device):
    """The regression. Fails if the mask width is guessed from ``vocab_size``."""
    ids = _generate(device)
    outside = [i for i in ids if i >= VOCAB]
    assert not outside, f"sampled ids outside vocab_size={VOCAB}: {outside} (all: {ids})"


def test_generate_picks_the_best_real_logit(device):
    """Stronger form: the mask must penalise only the padding, not shift the argmax."""
    ids = _generate(device)
    assert ids == [BEST_REAL_ID] * len(ids), f"expected all {BEST_REAL_ID}, got {ids}"


def test_collect_logits_path_slices_on_host(device):
    """``collect_logits=True`` samples on host from ``[:orig_vocab]`` and needs no mask."""
    config = SimpleNamespace(vocab_size=VOCAB, num_hidden_layers=1)
    generated, logits_lists = generate_ttml(
        _WideLogitsModel(1),
        config,
        _StubTokenizer(),
        [list(PROMPT)],
        2,
        SEQ,
        device,
        temperature=0.0,
        collect_logits=True,
        distributed=False,
        dp_size=1,
        tp_size=1,
        kv_cache=False,
    )
    assert all(i < VOCAB for i in generated[0]), generated[0]
    assert all(len(v) == VOCAB for v in logits_lists[0]), "collected logits must be sliced to vocab"


# --- _sample_logits_mask's own contract, relied on by the above -------------------


def test_mask_spans_the_width_it_is_given(device):
    mask = _sample_logits_mask(VOCAB, LOGITS_WIDTH, device)
    assert mask is not None
    assert int(mask.shape()[3]) == LOGITS_WIDTH

    values = mask.to_numpy(ttnn.DataType.FLOAT32)[0, 0, 0]
    assert np.all(values[:VOCAB] == 0.0), "real vocab columns must be untouched"
    assert np.all(values[VOCAB:] > 0.0), "padding columns must be penalised"


def test_no_mask_when_width_matches_vocab(device):
    """The early-out that turned the old width under-count into "no mask at all"."""
    assert _sample_logits_mask(VOCAB, VOCAB, device) is None
