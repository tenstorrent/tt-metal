"""A config field that arrives as a list must cost that field at most, never the whole tower.

HF configs express per-layer geometry as a list on some models -- num_hidden_layers [32], head_dim
[128, 128]. tower_geometry read every field with a bare int(), and the guard around the first pair
answered TypeError with `continue`, which skips the ENTIRE sub-dict: layers, hidden size, the KV term
together. The stage then has no memory ceiling at all and the roofline falls back to a weaker
estimate with nothing said -- one structured field, and the model loses its ceiling.

Reproduced before the fix: a config identical but for list-wrapped values returned {} where the
scalar one returned full geometry.

perf_target._scalar already exists for this -- "a config value that may arrive as a list/dict ... so
a structured value degrades instead of crashing" -- and is used 25 times there. Imported rather than
repeated: it also handles dicts and non-finite floats, which a second copy written here would miss.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
for _p in (str(PERF), str(PERF.parent.parent.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_SCALAR = {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "intermediate_size": 16384,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_attention_heads": 32,
}


def _geo(tmp_path, cfg):
    from agent.checkpoint_sections import tower_geometry

    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return tower_geometry(str(tmp_path))


def test_list_valued_fields_give_the_same_geometry_as_scalars(tmp_path):
    """The whole defect in one comparison."""
    plain = _geo(tmp_path / "a", {**_SCALAR}) if (tmp_path / "a").mkdir() or True else None
    listed = _geo(tmp_path / "b", {k: [v, v] for k, v in _SCALAR.items()}) if (tmp_path / "b").mkdir() or True else None
    assert plain and plain == listed, "a structured field changed the answer"


def test_a_list_valued_depth_does_not_drop_the_tower(tmp_path):
    """The severe half: the guard around the FIRST pair used to skip the whole sub-dict."""
    (tmp_path / "c").mkdir()
    got = _geo(tmp_path / "c", {**_SCALAR, "num_hidden_layers": [32]})
    assert got, "the tower was dropped for one list-valued field"
    assert next(iter(got.values()))["layers"] == 32


def test_a_list_valued_attention_field_still_yields_the_kv_term(tmp_path):
    """The milder half degraded to 0, which silently understates the KV memory."""
    (tmp_path / "d").mkdir()
    got = _geo(tmp_path / "d", {**_SCALAR, "num_key_value_heads": [8], "head_dim": [128, 128]})
    geo = next(iter(got.values()))
    assert geo["kv_heads"] == 8 and geo["head_dim"] == 128


def test_a_field_the_config_does_not_state_is_zero_not_a_crash(tmp_path):
    """Absent stays absent -- the coercion must not invent a value."""
    from agent.checkpoint_sections import _cfg_int

    assert _cfg_int({}, "num_hidden_layers") == 0
    assert _cfg_int({"num_hidden_layers": None}, "num_hidden_layers") == 0
    assert _cfg_int({"num_hidden_layers": []}, "num_hidden_layers") == 0
    assert _cfg_int({"num_hidden_layers": ""}, "num_hidden_layers") == 0


def test_the_first_stated_name_wins(tmp_path):
    """These fields have aliases; the order they are tried is part of the contract."""
    from agent.checkpoint_sections import _cfg_int

    assert _cfg_int({"hidden_size": 4096, "d_model": 111}, "hidden_size", "d_model") == 4096
    assert _cfg_int({"d_model": 111}, "hidden_size", "d_model") == 111


def test_a_non_finite_value_is_unknown_rather_than_enormous(tmp_path):
    """json.loads accepts Infinity, and int(inf) raises OverflowError -- neither TypeError nor
    ValueError, so a hand-rolled guard would let it through."""
    from agent.checkpoint_sections import _cfg_int

    assert _cfg_int({"hidden_size": float("inf")}, "hidden_size") == 0
    assert _cfg_int({"hidden_size": float("nan")}, "hidden_size") == 0


def test_the_coercion_is_not_reimplemented_here():
    """perf_target._scalar owns this; a second copy would drift from the one with the edge cases."""
    src = (PERF / "agent" / "checkpoint_sections.py").read_text(encoding="utf-8")
    i = src.index("def _cfg_int")
    body = src[i : src.index("\ndef ", i + 10)]
    assert "from .perf_target import _scalar" in body
    assert "isinstance(" not in body, "list/dict handling is being written again instead of reused"


def test_every_geometry_field_uses_the_one_reader():
    """Each was its own try/int/except; the first of them is what dropped the tower."""
    src = (PERF / "agent" / "checkpoint_sections.py").read_text(encoding="utf-8")
    i = src.index("def tower_geometry")
    body = src[i : src.index("\ndef ", i + 10)]
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    for _f in ("num_hidden_layers", "hidden_size", "num_key_value_heads", "head_dim", "num_attention_heads"):
        assert "int(sub.get(%r)" % _f not in code.replace('"', "'"), _f
    assert code.count("_cfg_int(") >= 5
