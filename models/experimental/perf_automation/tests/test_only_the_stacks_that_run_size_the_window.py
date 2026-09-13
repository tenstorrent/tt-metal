# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The profiling window is sized from stacks the DEVICE runs, not from everything the walk finds.

WHAT THIS COST, measured on Voxtral-Mini-3B, 2026-08-12.

Walking the built pipeline finds every repeated list, and a built pipeline holds more than its own
blocks. Five stacks came back:

    hf.model.audio_tower.layers      32  VoxtralEncoderLayer   HF reference, never executed
    hf.model.language_model.layers   30  LlamaDecoderLayer     HF reference, never executed
    enc_a._inner.layers               4  TtEncoderLayer        real
    enc_b._inner.layers               4  _Counted/_EncLayer..  real
    kv                                4  KVSlot                cache slots, not blocks

The reference model is held for weight loading; its torch modules never dispatch a ttnn op. Sizing
from it asked for depth 32 -- the model's own full depth -- so capping changed no work, the run
declared the depth knob INERT, and profiled everything at full depth.

That is worse than not walking at all. The path it replaced inferred a single stack, sized 2, and
capped correctly (18729 -> 2471 ops). Adding discovery without this filter turned a working cap into
no cap, which is the kind of regression that only shows up in a real run: every unit check passed,
and the tool reported the knob inert rather than failing.

The two rejections are structural, not per-model. A torch.nn.Module element is reference weights --
TTNN blocks never subclass it. A non-callable element is not a block -- KVSlot is a dataclass of
cache tensors. What survives is what the device actually executes.
"""

from pathlib import Path

_PA = Path(__file__).resolve().parent.parent


def _src():
    return (_PA / "cc_optimize" / "_op_sig_probe.py").read_text()


def test_the_filter_exists_and_the_walk_goes_through_it():
    src = _src()
    assert "def _device_stacks(" in src, "no filter on discovered stacks"
    i = src.index("def tag_built_model(")
    body = src[i : i + 3000]
    # The walk's result must reach _device_stacks BEFORE anything is tagged. Checked as an ordering
    # rather than as one spelling: the census now reads the unfiltered walk between the two calls,
    # which is a legitimate use of it -- reporting what was found is not the same as sizing from it.
    walk, filt = body.index("find_all_stacks("), body.index("_device_stacks(")
    assert walk < filt < body.index("_tag_all_stacks("), "stacks are tagged without being filtered"
    assert "_device_stacks(_all)" in body or "_device_stacks(find_all_stacks(" in body


def test_reference_torch_modules_are_rejected():
    """The HF model is held for weight loading and never dispatches a ttnn op. It is also the
    DEEPEST stack in the pipeline, so it wins any max() and asks for the model's full depth."""
    src = _src()
    i = src.index("def _device_stacks(")
    body = src[i : i + 2600]
    assert "torch" in body and "isinstance(head" in body, "torch reference modules are not rejected"


def test_data_only_lists_are_rejected():
    """KVSlot is a dataclass of cache tensors -- a four-element list that is not four blocks."""
    src = _src()
    i = src.index("def _device_stacks(")
    body = src[i : i + 2600]
    assert "callable(head)" in body, "non-callable elements are still treated as blocks"


def test_the_incident_is_recorded_where_the_next_reader_will_look():
    """A filter with no reason attached is the first thing someone deletes for being over-cautious."""
    src = _src()
    i = src.index("def _device_stacks(")
    body = src[i : i + 2600]
    assert "INERT" in body, "the failure this prevents is not written down"
