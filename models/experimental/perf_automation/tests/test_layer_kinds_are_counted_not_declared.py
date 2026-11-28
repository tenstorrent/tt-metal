# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How many KINDS of block a model has is counted from the checkpoint, not read from a config key.

The coverage window has to include one of every kind of block, or a profile of the first two layers
misses a kind that first appears deep in the stack. _config_layer_kinds answers that by reading a
per-layer pattern out of one of four attribute names:

    _LAYER_PATTERN_ATTRS = ("hybrid_override_pattern", "layer_types", "layers_block_type",
                            "block_types")

and it needs AutoConfig to load the model first. Voxtral fails at that step -- this transformers
does not know its model type -- so the config route returned (None, 0), and the caller ended the run
with "no_window: probe_failed".

Two blocks are the SAME KIND when they hold the same set of parameter names: an attention block has
q_proj/k_proj, a Mamba block has in_proj/conv1d, and telling them apart needs no vocabulary. The
indices are in the names too, so "the first k layers cover every kind" is a count.
"""

import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))

_MID = "mistralai/Voxtral-Mini-3B-2507"
_ROOT = _PA.parent.parent / "tt_transformers" / "demo" / "voxtral_mini_3b_2507"


def test_a_hybrid_stack_needs_enough_layers_to_see_every_kind():
    from agent.checkpoint_sections import layer_kinds_from_keys

    keys = []
    for i in range(8):
        fam = ["self_attn.q_proj.weight"] if i % 2 == 0 else ["mixer.in_proj.weight", "mixer.conv1d.weight"]
        keys += ["m.layers.%d.%s" % (i, f) for f in fam]

    assert layer_kinds_from_keys(keys) == (2, 2), "an alternating stack reads as one kind"


def test_a_kind_that_first_appears_deep_moves_the_window():
    """The case an observation-only climb misses: 15 identical layers, then a different one."""
    from agent.checkpoint_sections import layer_kinds_from_keys

    keys = ["m.layers.%d.self_attn.q_proj.weight" % i for i in range(15)]
    keys += ["m.layers.15.mixer.in_proj.weight"]

    k, n = layer_kinds_from_keys(keys)
    assert (k, n) == (16, 2), (k, n)


def test_a_homogeneous_stack_is_one_kind():
    from agent.checkpoint_sections import layer_kinds_from_keys

    assert layer_kinds_from_keys(["m.layers.%d.mlp.w" % i for i in range(30)]) == (1, 1)


def test_the_deepest_stack_wins_so_the_window_covers_every_tower():
    from agent.checkpoint_sections import layer_kinds_from_keys

    keys = ["a.layers.%d.mlp.w" % i for i in range(4)]  # homogeneous -> k=1
    keys += ["b.layers.%d.%s" % (i, "x.w" if i < 3 else "y.w") for i in range(5)]  # second kind at 3
    assert layer_kinds_from_keys(keys)[0] == 4


def test_a_model_with_no_repeated_blocks_declares_nothing():
    from agent.checkpoint_sections import layer_kinds_from_keys

    assert layer_kinds_from_keys(["lm_head.weight", "embed.weight"]) == (None, 0)


def test_voxtral_is_answerable_where_the_config_route_is_not():
    from agent.checkpoint_sections import hf_cache_dir, layer_kinds

    if not hf_cache_dir(_MID):
        pytest.skip("voxtral not in the local HF cache")
    assert layer_kinds(str(_ROOT), _MID) == (1, 1)


def test_stray_keys_in_the_model_dir_do_not_hide_the_real_checkpoint(tmp_path, monkeypatch):
    """A model directory can hold keys that are not the model. Voxtral's ships 24 -- 'data.pkl',
    '.format_version', 'byteorder' -- the pickle metadata of a captured reference tensor. Testing
    "the key list is non-empty" instead of "the answer came out" took those as the checkpoint and
    never looked in the cache where the weights are."""
    import agent.checkpoint_sections as CS

    stray = ["data.pkl", ".format_version", "byteorder"]
    real = ["m.layers.%d.%s" % (i, "a.w" if i < 2 else "b.w") for i in range(4)]

    monkeypatch.setattr(CS, "checkpoint_keys", lambda root: stray if str(root) == str(tmp_path) else [])
    monkeypatch.setattr(CS, "hf_cache_dir", lambda mid: tmp_path / "snap")
    monkeypatch.setattr(CS, "_index_keys", lambda snap: real)

    assert CS.layer_kinds(str(tmp_path), "org/model") == (3, 2), "the stray keys shadowed the cache"


def test_the_caller_falls_back_to_the_checkpoint_before_giving_up():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("k, n_kinds = _config_layer_kinds(")
    body = src[i : src.index('facts["no_window"]', i)]
    assert "layer_kinds" in body, "the run still ends at probe_failed when the config is silent"
