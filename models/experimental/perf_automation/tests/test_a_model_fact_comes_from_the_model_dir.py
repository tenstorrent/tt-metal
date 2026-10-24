# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A fact about the model comes from the model's directory, or it is UNKNOWN.

_MODEL_ROOT defaulted to ".", so both ends of perf_target_inputs.json resolved against the WORKING
DIRECTORY: the writer dropped the file wherever the loop happened to be cd'd, and the reader then
adopted whatever file was lying there as this model's facts.

On gemma-3-12b that meant a stray file describing a 32-layer, hidden-1280, 30 MB model -- the vision
tower's geometry, not the text tower's -- was read as the facts for an 11B, 48-layer model. Three
sections of one report broke from that single file:

    PREFILL memory ceiling   0.061 ms   against a 100.46 ms measurement   (activations of hidden 1280)
    compute (both stages)    not measured                                (no param count -> no FLOPs)
    Fidelity ladder          absent entirely                             (no FLOPs -> no ladder)

None of it announced a problem; every cell rendered a confident number or a plausible blank.

The rule: unstated is UNKNOWN, and unknown renders as "not measured" -- which is true -- rather than
as a number, which was not. Pinned here because the previous fix was verified by hand in a throwaway
probe and left nothing behind that would fail if the default came back.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent

# The exact junk that was adopted, verbatim from the run that broke.
_JUNK = {
    "weight_bytes": 29869529,
    "dominant_dtype": "float16",
    "source": "checkpoint bytes + HF config",
    "layers": 32,
    "hidden_size": 1280,
}
_REAL = {
    "weight_bytes": 24374793024,
    "dominant_dtype": "bfloat16",
    "source": "checkpoint bytes + HF config",
    "total_params": 11180446320,
    "layers": 48,
}


def _mcp(monkeypatch, cwd, model_root=None):
    monkeypatch.chdir(cwd)
    if model_root is None:
        monkeypatch.delenv("PERF_MCP_MODEL_ROOT", raising=False)
    else:
        monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(model_root))
    _mf = cwd / "manifest.json"
    _mf.write_text(json.dumps({"config": {}, "perf_test_resolved": {"path": "t.py"}}))
    monkeypatch.setenv("PERF_MCP_MANIFEST", str(_mf))
    spec = _ilu.spec_from_file_location("pm_facts_ut", str(_PA / "cc_optimize" / "perf_mcp.py"))
    m = _ilu.module_from_spec(spec)
    sys.modules["pm_facts_ut"] = m
    spec.loader.exec_module(m)
    return m


def test_an_unstated_model_root_yields_no_facts(tmp_path, monkeypatch):
    """THE CASE. A junk perf_target_inputs.json sits in the working directory; nothing said where
    the model is. The answer must be "I don't know", not that file."""
    (tmp_path / "perf_target_inputs.json").write_text(json.dumps(_JUNK))
    m = _mcp(monkeypatch, tmp_path)
    assert m._MODEL_ROOT_STATED is False
    assert m._load_perf_target_inputs() is None


def test_the_rebuild_path_is_guarded_too(tmp_path, monkeypatch):
    """ONE RULE, TWO DOORS. Guarding only the direct read left the rebuild fallback resolving the
    same "." and re-adopting the same file -- which is what the first attempt at this fix did."""
    m = _mcp(monkeypatch, tmp_path)  # no file at all -> the rebuild path is the one that runs
    (tmp_path / "perf_target_inputs.json").write_text(json.dumps(_JUNK))
    assert m._load_perf_target_inputs() is None


def test_a_stated_model_root_reads_its_own_file(tmp_path, monkeypatch):
    """The rule withholds a guess; it must not withhold the answer."""
    mroot = tmp_path / "models" / "demos" / "gemma3"
    mroot.mkdir(parents=True)
    (mroot / "perf_target_inputs.json").write_text(json.dumps(_REAL))
    (tmp_path / "perf_target_inputs.json").write_text(json.dumps(_JUNK))  # decoy in the cwd
    m = _mcp(monkeypatch, tmp_path, model_root=mroot)
    facts = m._load_perf_target_inputs()
    assert facts and facts["total_params"] == 11180446320, facts
    assert facts["layers"] == 48, "read the cwd decoy instead of the model's own file"


def test_the_writer_refuses_an_unstated_root(tmp_path, monkeypatch, capsys):
    """The same defect at the other end: this is HOW the junk file came to exist. It carries the
    tool's own `source` marker, so the tool wrote it -- into the cwd, from a relative root."""
    spec = _ilu.spec_from_file_location("cc_run_facts_ut", str(_PA / "cc_optimize" / "run.py"))
    r = _ilu.module_from_spec(spec)
    sys.modules["cc_run_facts_ut"] = r
    spec.loader.exec_module(r)
    monkeypatch.chdir(tmp_path)
    r._emit_perf_target_inputs(".", tmp_path, None, {})
    assert not (tmp_path / "perf_target_inputs.json").exists(), "wrote a model fact into the cwd"
    assert "not a stated directory" in capsys.readouterr().out


def test_the_writer_refreshes_its_own_file_but_not_a_hand_tuned_one(tmp_path, monkeypatch):
    """`never overwrites` protected a hand-tuned file, which is right, and froze the tool's own first
    guess forever, which is not: the geometry keys prefill's byte model needs could never reach a
    model that already had a file, so its roof degraded silently for exactly the models run before."""
    spec = _ilu.spec_from_file_location("cc_run_facts_ut2", str(_PA / "cc_optimize" / "run.py"))
    r = _ilu.module_from_spec(spec)
    sys.modules["cc_run_facts_ut2"] = r
    spec.loader.exec_module(r)
    out = tmp_path / "perf_target_inputs.json"
    monkeypatch.setattr(r, "_perf_target_inputs", lambda *a, **k: dict(_REAL), raising=False)

    out.write_text(json.dumps(_JUNK))  # the tool's own stale output -> refreshed
    r._emit_perf_target_inputs(tmp_path, tmp_path, None, {})
    assert json.loads(out.read_text())["total_params"] == 11180446320

    hand = {"total_params": 1, "source": "hand-tuned per-tensor dtypes"}
    out.write_text(json.dumps(hand))  # a human's -> left alone
    r._emit_perf_target_inputs(tmp_path, tmp_path, None, {})
    assert json.loads(out.read_text()) == hand


def test_the_report_is_handed_the_model_dir_from_the_manifest(tmp_path, monkeypatch):
    """_latest_manifest returns a PATH, not the parsed document. _model_root_for_report called .get()
    on that Path, threw AttributeError, and its bare `except` turned that into None -- on every run,
    silently, from the day it was written.

    So the report never received the model directory this function exists to supply. It fell through
    to perf_mcp's "." fallback and read whatever perf_target_inputs.json sat in the working directory.
    That is the whole reason the junk file was reachable at all: this function failing open is what
    put the report in the cwd's hands.

    The `except` stays -- a missing manifest must not take the report down -- but a function that can
    only ever return None is not a fallback, it is a hole."""
    spec = _ilu.spec_from_file_location("cc_run_hint_ut", str(_PA / "cc_optimize" / "run.py"))
    r = _ilu.module_from_spec(spec)
    sys.modules["cc_run_hint_ut"] = r
    spec.loader.exec_module(r)

    runs = tmp_path / r.PERF_DIR / "runs" / "2026-01-01T00-00-00"
    runs.mkdir(parents=True)
    mroot = tmp_path / "models" / "demos" / "gemma3"
    mroot.mkdir(parents=True)
    (runs / "manifest.json").write_text(json.dumps({"config": {"model_root": str(mroot)}}))

    assert r._model_root_for_report(tmp_path) == mroot


def test_a_manifest_without_a_model_root_is_none_not_the_cwd(tmp_path):
    """None here means "unknown", which the reader now treats as unknown. It must not degrade into
    Path("") -- which is Path("."), the very cwd guess this whole fix removes."""
    spec = _ilu.spec_from_file_location("cc_run_hint_ut2", str(_PA / "cc_optimize" / "run.py"))
    r = _ilu.module_from_spec(spec)
    sys.modules["cc_run_hint_ut2"] = r
    spec.loader.exec_module(r)

    runs = tmp_path / r.PERF_DIR / "runs" / "2026-01-01T00-00-00"
    runs.mkdir(parents=True)
    (runs / "manifest.json").write_text(json.dumps({"config": {}}))
    got = r._model_root_for_report(tmp_path)
    assert got is None, got
    assert got != Path("."), "an empty model_root degraded back into the working directory"


# --- the geometry that makes prefill differ from decode ----------------------------------------

_GEMMA3_CFG = {
    "text_config": {
        "hidden_size": 3840,
        "intermediate_size": 15360,
        "num_key_value_heads": 8,
        "num_attention_heads": 16,
        "num_hidden_layers": 48,
    },
    "vision_config": {"hidden_size": 1152, "intermediate_size": 4304, "num_hidden_layers": 27},
}


def _cfg_probe(cfg):
    """Run the emitter's config reader over a config and return what it would record."""
    spec = _ilu.spec_from_file_location("cc_run_cfg_ut", str(_PA / "cc_optimize" / "run.py"))
    r = _ilu.module_from_spec(spec)
    sys.modules["cc_run_cfg_ut"] = r
    spec.loader.exec_module(r)
    return r


_TWO_TOWER_CFG = {
    "audio_config": {
        "num_hidden_layers": 32,
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_attention_heads": 20,
    },
    "text_config": {
        "num_hidden_layers": 30,
        "hidden_size": 3072,
        "intermediate_size": 8192,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
    },
}


def _write_cfg(doc):
    import json as _j
    import tempfile

    d = Path(tempfile.mkdtemp())
    (d / "config.json").write_text(_j.dumps(doc))
    return d


def test_the_geometry_is_read_per_block_not_as_loose_keys():
    """WHAT THIS REPLACES, and why the replacement is structural rather than a better lookup.

    Two rules picked geometry out of one config, independently:

        layers              = _depth_from_mapping(cfg)   "the DEEPEST depth anywhere"   -> 32
        hidden/intermediate = first sub-config named neither vision nor audio            -> 3072/8192

    On voxtral that is the AUDIO tower's depth welded to the LANGUAGE tower's widths: a 32-layer
    3072-wide model that does not exist. Every stage divided those numbers, so the audio encoder was
    priced at 0.041 ms against a 12.80 ms measurement.

    A better key lookup could not fix it. Geometry belongs to a BLOCK, and a model with two blocks
    has no single answer -- so blocks are read whole, keyed by depth."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _perf_target_inputs")
    body = src[i : src.index("\ndef ", i + 1)]
    # CODE ONLY. The comments name the towers to explain why keying on their names was the defect;
    # asserting over prose would forbid describing what was fixed.
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "_model_block_facts(" in code, "the facts no longer collect blocks"
    assert "_cfgv(" not in code, "the loose-key lookup is back"
    assert '"vision"' not in code and '"audio"' not in code, "a tower name blacklist is back"


def test_a_two_tower_model_publishes_no_flat_geometry():
    """The chimera cannot be constructed. With two blocks there is no single `hidden_size`, so none
    is emitted -- a caller that has not learned about blocks gets nothing rather than another
    tower's width. Missing degrades to a refused ceiling; wrong degrades to 312x."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _perf_target_inputs")
    body = src[i : src.index("\ndef ", i + 1)]
    assert "if len(_blocks or {}) == 1:" in body, "flat geometry is emitted for a multi-block model"


def test_each_tower_keeps_its_own_geometry():
    """gemma3's vision_config also carries hidden_size, and voxtral's audio tower its own kv_heads
    and head_dim. The old code excluded those towers BY NAME; blocks give each its own entry, so
    nothing has to be recognised and nothing has to be excluded."""
    from agent.checkpoint_sections import tower_geometry

    got = tower_geometry(_write_cfg(_TWO_TOWER_CFG))
    assert got[32]["hidden_size"] == 1280 and got[32]["kv_heads"] == 20 and got[32]["head_dim"] == 64
    assert got[30]["hidden_size"] == 3072 and got[30]["kv_heads"] == 8 and got[30]["head_dim"] == 128
