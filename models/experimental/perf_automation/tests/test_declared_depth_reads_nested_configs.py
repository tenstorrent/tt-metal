"""A multimodal HF config declares its depth in a nested CONFIG OBJECT, not a nested dict.

gemma-3-12b-it has 48 text layers. The run reported "96 layers" -- 48 decoder blocks each raising
two signposts -- because the clamp that should have bounded the coverage window at the model's
declared depth never fired:

    Gemma3Config.__dict__          -> {"text_config": Gemma3TextConfig(...), "vision_config": ...}
    _depth_from_mapping(that)      -> None      # no top-level num_hidden_layers
      -> obj.get("text_config")    -> Gemma3TextConfig OBJECT, not a dict
      -> isinstance(obj, dict)     -> False, return None

So full_depth_from_config() returned None, _cap_cov_depth() had nothing to clamp against, the window
stayed 96, and every downstream label -- the ledger depth stamp, the report header -- claimed a layer
count the model does not have. full_depth_from_config's own docstring already promises this case:
"also covers custom architectures via trust_remote_code, and nested text_config for multimodal
wrappers". The nested walk was written; it just could not see through an object.

This is not gemma-specific. Every HF multimodal wrapper nests the same way (llava, qwen-vl, voxtral,
paligemma), so the fix reads any nested value that exposes a __dict__, and none of them by name.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.layer_depth import _depth_from_mapping, full_depth_from_config  # noqa: E402


class _Cfg:
    """Stands in for a transformers PretrainedConfig: attributes, no Mapping interface."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ---------------------------------------------------------------- the reported bug


def test_the_gemma3_shape_reads_48_not_none():
    """The exact shape that produced '96 layers' for a 48-layer model."""
    cfg = _Cfg(
        model_type="gemma3",
        text_config=_Cfg(num_hidden_layers=48, hidden_size=3840),
        vision_config=_Cfg(num_hidden_layers=27),
    )
    assert _depth_from_mapping(cfg.__dict__) == 48


def test_the_vision_tower_does_not_win_by_being_first():
    """27 and 48 are both present; walk order must not decide. Max settles it without the walk
    needing to know which branch is the text stack."""
    for cfg in (
        _Cfg(vision_config=_Cfg(num_hidden_layers=27), text_config=_Cfg(num_hidden_layers=48)),
        _Cfg(text_config=_Cfg(num_hidden_layers=48), vision_config=_Cfg(num_hidden_layers=27)),
    ):
        assert _depth_from_mapping(cfg.__dict__) == 48


def test_the_deepest_declaration_wins_wherever_it_sits():
    """MAX, not first-found and not top-level-first. The value is used only as a CEILING (d = min(d,
    full)), so erring high just weakens the clamp while erring low silently hides layers. Position in
    the config carries no authority -- gemma3 offers 48 nested and nothing at the top."""
    assert _depth_from_mapping(_Cfg(num_hidden_layers=32, text_config=_Cfg(num_hidden_layers=99)).__dict__) == 99
    assert _depth_from_mapping(_Cfg(num_hidden_layers=99, text_config=_Cfg(num_hidden_layers=32)).__dict__) == 99


def test_a_list_of_sub_configs_is_walked():
    """transformers exposes sub_configs as a LIST; a dict/object-only walk misses it entirely."""
    assert _depth_from_mapping({"sub_configs": [_Cfg(num_hidden_layers=27), _Cfg(num_hidden_layers=48)]}) == 48


def test_an_unnamed_nesting_key_is_still_found():
    """The old walk only followed five hardcoded names. A model nesting under anything else -- and
    that is the residual hole in the whole scheme -- declared nothing at all."""
    assert _depth_from_mapping({"some_unlisted_wrapper": {"inner": _Cfg(num_hidden_layers=64)}}) == 64


# ---------------------------------------------------------------- shape coverage


@pytest.mark.parametrize("holder", ["text_config", "decoder", "model", "gpt", "llm_config"])
def test_every_nested_key_works_as_an_object_not_just_a_dict(holder):
    """The walk already listed these names; only dicts could be followed."""
    assert _depth_from_mapping({holder: _Cfg(num_hidden_layers=16)}) == 16
    assert _depth_from_mapping({holder: {"num_hidden_layers": 16}}) == 16


def test_a_dict_nested_inside_an_object_nested_inside_a_dict():
    """Real configs mix the two forms; the walk must not care which it is standing on."""
    assert _depth_from_mapping({"text_config": _Cfg(decoder={"num_hidden_layers": 7})}) == 7


def test_an_object_with_no_depth_anywhere_is_still_none():
    """None means 'nothing declares it' and lets the caller fall back to the builder. Returning a
    guess here would put a fabricated layer count on every report."""
    assert _depth_from_mapping(_Cfg(text_config=_Cfg(hidden_size=512)).__dict__) is None


def test_a_string_is_not_walked_for_attributes():
    """str has no __dict__, but bytes/other scalars must not be probed for one either -- a stray
    attribute lookup on a scalar is how a walker starts returning nonsense."""
    for junk in ("num_hidden_layers", b"48", 48, 4.8, None, True):
        assert _depth_from_mapping({"text_config": junk}) is None


def test_a_self_referential_config_terminates():
    """HF configs can hold back-references. An unbounded walk would hang the whole optimize run
    before it ever reached the device."""
    a = _Cfg(num_hidden_layers=None)
    b = _Cfg(text_config=a)
    a.text_config = b
    assert _depth_from_mapping({"text_config": a}) is None


def test_zero_and_negative_declared_depths_are_rejected():
    """A 0 would clamp the coverage window to nothing and profile an empty model."""
    for bad in (0, -1, -48):
        assert _depth_from_mapping({"text_config": _Cfg(num_hidden_layers=bad)}) is None


def test_a_bool_is_not_a_depth():
    """True is an int in Python. num_hidden_layers=True must not become a 1-layer window."""
    assert _depth_from_mapping({"text_config": _Cfg(num_hidden_layers=True)}) is None


# ---------------------------------------------------------------- through the public entry point


def test_full_depth_from_config_reads_a_nested_config_json(tmp_path):
    """The model_dir branch parses JSON, so text_config arrives as a dict -- this path always worked
    and must keep working."""
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 48}}))
    assert full_depth_from_config(model_dir=tmp_path) == 48


def test_full_depth_from_config_is_none_when_nothing_declares(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"hidden_size": 3840}))
    assert full_depth_from_config(model_dir=tmp_path) is None


# ---------------------------------------------------------------- the clamp that depends on it


def test_the_coverage_window_clamps_to_the_declared_depth(monkeypatch, tmp_path):
    """The whole point: 96 signpost blocks over a 48-layer model must yield a 48-layer window, or the
    ledger stamps a depth the model does not have and every label downstream repeats it."""
    from cc_optimize import run as R

    monkeypatch.setattr(R, "_declared_depth", lambda _root, mid="": 48 if mid else None)
    assert R._cap_cov_depth(96, "google/gemma-3-12b-it") == 48
    assert R._cap_cov_depth(12, "google/gemma-3-12b-it") == 12  # shallower windows are untouched
    assert R._cap_cov_depth(96, "") == 96  # no id, no claim, no clamp
