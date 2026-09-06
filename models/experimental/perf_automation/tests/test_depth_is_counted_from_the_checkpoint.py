# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How deep a stack is comes from the checkpoint, not from guessing what a config calls the field.

_DEPTH_KEYS is nine guesses -- num_hidden_layers, n_layers, num_layers, n_layer, num_blocks,
num_decoder_layers, decoder_layers, gpt_layers, depth -- each added when a model used a spelling the
list did not have. A tenth spelling is one model away, and the list cannot tell two stacks apart
when they share a key name.

A checkpoint names its blocks by index:

    language_model.model.layers.29.mlp.gate_proj.weight

so a stack's depth is its highest index plus one. That is a property of the file, true of every
model with repeated blocks, in any architecture and any naming convention.

MEASURED ON VOXTRAL, and the config path was not merely less precise -- it produced NOTHING:

    before   full_depth_from_config -> None      declared_section_depths -> []
    after    full_depth_from_config -> 32        declared_section_depths -> [32, 30]

because transformers here does not recognise the `voxtral` model type (AutoConfig raises) and the
demo directory ships no config.json -- the weights are in the shared HF cache. Every caller was
falling back to establishing depth some other way.
"""

import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_MID = "mistralai/Voxtral-Mini-3B-2507"
_ROOT = _PA.parent.parent / "tt_transformers" / "demo" / "voxtral_mini_3b_2507"


def _skip_without_weights():
    from agent.checkpoint_sections import hf_cache_dir

    if not hf_cache_dir(_MID):
        pytest.skip("voxtral not in the local HF cache")


def test_depths_come_from_the_checkpoints_own_indices():
    _skip_without_weights()
    from agent.layer_depth import depths_from_checkpoint

    assert depths_from_checkpoint(_MID, _ROOT) == [32, 30]


def test_the_two_stacks_are_told_apart():
    """The config walk returns every depth it finds anywhere in the mapping and cannot say which
    stack each belongs to. The checkpoint names them, so 32 and 30 stay distinct."""
    _skip_without_weights()
    from agent.checkpoint_sections import declared_sections

    secs = declared_sections(str(_ROOT), _MID)
    assert secs == {"audio_tower.layers": 32, "language_model.model.layers": 30}


def test_the_checkpoint_is_consulted_before_the_config():
    for fn in ("full_depth_from_config", "declared_section_depths"):
        src = (_PA / "agent" / "layer_depth.py").read_text()
        i = src.index("def %s(" % fn)
        j = src.find("\ndef ", i + 1)
        body = src[i : j if j > 0 else len(src)]  # the last function has no next def
        code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
        assert "depths_from_checkpoint(" in code, "%s no longer counts from the checkpoint" % fn
        assert code.index("depths_from_checkpoint(") < code.index("AutoConfig"), (
            "%s asks the config before the checkpoint" % fn
        )


def test_an_unreadable_checkpoint_falls_through_rather_than_failing():
    from agent.layer_depth import depths_from_checkpoint

    assert depths_from_checkpoint("no-such-org/no-such-model", "/nonexistent") == []
