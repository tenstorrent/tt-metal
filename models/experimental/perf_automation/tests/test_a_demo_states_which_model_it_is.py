# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A model's IDENTITY is a thing the run was told; WHERE ITS FILES LIVE is not the same thing.

The cc engine had no tier that read a stated identity off the demo itself. `hint` is nulled for a
directory target (optimize.py: `model_id_hint=(None if model_dir else args.target)`) and HF_MODEL is
only set if the operator's shell exported it, so optimizing a brought-up demo normally arrived at the
two tiers that INFER: read every .py under the directory and take the first id that happens to be in
the HF cache. That answers "what does this folder mention", which is a different question, and on a
tree that mentions several models it answers it wrong -- gemma3's conftest pins the 12b while
test_ci_dispatch.py lists the 4b and the 27b.

Scaffold already wrote the answer down. bringup_status.json records the model the demo was generated
FROM, and find_demo_dir() already trusts that key enough to resolve an id back to a directory; the
resolver now reads the same fact in the other direction. Statements first (asked for, configured,
recorded), inference second.

The same confusion reached the end-of-run card by a second route: its model_targets lookup key fell
back to `model_name`, the demo DIRECTORY's name, so a model whose published targets sit in
model_targets.yaml printed "measured-only" because it was looked up by its folder.

  d1  the record decides, and outranks a tree full of decoys
  d2  what the caller/environment states still outranks the record
  d3  a checkpoint brought up from disk keeps its name -- the cache is not the arbiter of identity
  d4  no record, or an unusable one, falls through to exactly today's behaviour
  d5  resolving is read-only
  d6  the scorecard is keyed on the id, not on the folder
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))
sys.path.insert(0, str(_PA / "cc_optimize"))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_stated_id", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_M = _run()
_REAL = "google/gemma-3-12b-it"
_DECOY = "google/gemma-3-4b-it"


def _cached(*known):
    """Stand-in for _is_cached_model_id: only these ids exist locally."""
    s = set(known)
    return lambda v: bool(v) and str(v) in s


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A demo dir shaped like gemma3's: decoys in a CI matrix file, nothing stating the real model."""
    (tmp_path / "tests" / "e2e").mkdir(parents=True)
    (tmp_path / "tests" / "test_ci_dispatch.py").write_text(f'IDS = ["{_DECOY}", "google/gemma-3-27b-it"]\n')
    monkeypatch.delenv("HF_MODEL", raising=False)
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached(_REAL, _DECOY, "google/gemma-3-27b-it"))
    return tmp_path


def _record(demo_dir, **fields):
    (Path(demo_dir) / "bringup_status.json").write_text(json.dumps(fields))
    return demo_dir


# --------------------------------------------------------------------------- d1 THE RECORD DECIDES
def test_d1_the_recorded_model_beats_the_first_id_the_tree_mentions(tree):
    """THE FIX. Without the record this tree resolves to whichever decoy rglob reaches first."""
    _record(tree, new_model_id=_REAL)
    assert _M._resolve_model_id(tree, None) == _REAL


def test_d1_without_the_record_the_same_tree_answers_from_its_files(tree):
    """The other half of the claim above: the tree really does not know which model this is."""
    got = _M._resolve_model_id(tree, None)
    assert got in (_DECOY, "google/gemma-3-27b-it"), got


def test_d1_the_second_key_is_read_too(tree):
    """emit_e2e reads `new_model_id` then `model_id` off this record; so does the resolver."""
    _record(tree, model_id=_REAL)
    assert _M._resolve_model_id(tree, None) == _REAL


def test_d1_scaffolds_key_wins_when_both_are_present(tree):
    _record(tree, new_model_id=_REAL, model_id=_DECOY)
    assert _M._resolve_model_id(tree, None) == _REAL


# --------------------------------------------------------------------------- d2 STATEMENTS OUTRANK IT
def test_d2_an_explicit_target_still_wins(tree):
    """The record says what the demo was BUILT for; the operator is saying what to run now."""
    _record(tree, new_model_id=_REAL)
    assert _M._resolve_model_id(tree, "google/gemma-3-27b-it") == "google/gemma-3-27b-it"


def test_d2_hf_model_still_wins(tree, monkeypatch):
    """HF_MODEL is what the model itself will resolve its identity from when the test executes."""
    _record(tree, new_model_id=_REAL)
    monkeypatch.setenv("HF_MODEL", _DECOY)
    assert _M._resolve_model_id(tree, None) == _DECOY


def test_d2_but_an_unusable_env_does_not_displace_it(tree, monkeypatch):
    """A junk HF_MODEL is rejected by the tier above, and must fall to the record -- not past it."""
    _record(tree, new_model_id=_REAL)
    monkeypatch.setenv("HF_MODEL", "gemma3")
    assert _M._resolve_model_id(tree, None) == _REAL


# --------------------------------------------------------------------------- d3 BROUGHT UP FROM DISK
def test_d3_a_name_that_was_never_downloaded_is_still_this_models_name(tree):
    """The point of the clause. A checkpoint brought up from a local directory has no HF cache entry,
    so gating the record on _is_cached_model_id would drop it back to the scan -- which answers with a
    DIFFERENT model that does happen to be cached. Identity replaced by surroundings."""
    _record(tree, new_model_id="acme/Prototype-7B")
    assert _M._resolve_model_id(tree, None) == "acme/Prototype-7B"


def test_d3_a_checkpoint_directory_is_accepted_as_the_recorded_name(tree):
    """Bring-up accepts a local model directory as its target, so that is what the record may hold."""
    _record(tree, new_model_id=str(tree / "checkpoints" / "Llama-3.1-8B"))
    assert _M._resolve_model_id(tree, None) == str(tree / "checkpoints" / "Llama-3.1-8B")


@pytest.mark.parametrize("stated", ["acme/Prototype-7B", "/data/checkpoints/Llama-3.1-8B"])
def test_d3_an_uncached_name_still_carries_its_size(stated):
    """Not a cache miss dressed up as an answer: the size the roofline needs is IN the name, and the
    cache lookups it also feeds return empty rather than wrong."""
    total, _active = _M._params_from_model_id(stated)
    assert total in (7_000_000_000, 8_000_000_000)
    assert _M._hf_snapshots(stated) == []
    assert _M._hf_cache_weight_bytes(stated) == 0
    assert _M._hf_cache_dims(stated) == {}


# --------------------------------------------------------------------------- d4 FALL-THROUGH INTACT
@pytest.mark.parametrize(
    "body",
    [
        "",
        "   ",
        "{ not json",
        "[]",
        '"just-a-string"',
        "null",
        "{}",
        '{"new_model_id": ""}',
        '{"new_model_id": "   "}',
        '{"new_model_id": null, "model_id": null}',
        '{"new_model_id": {"id": "acme/x"}}',
        '{"new_model_id": 12}',
    ],
)
def test_d4_an_unusable_record_changes_nothing(tree, body):
    """Every one of these must resolve exactly as if the file were not there at all."""
    baseline = _M._resolve_model_id(tree, None)
    (tree / "bringup_status.json").write_text(body)
    assert _M._resolve_model_id(tree, None) == baseline


def test_d4_a_missing_directory_still_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert _M._resolve_model_id(tmp_path / "nope", None) is None


def test_d4_a_record_that_is_a_directory_is_not_a_record(tree):
    """An unreadable path must not raise out of a resolver every roofline figure depends on."""
    baseline = _M._resolve_model_id(tree, None)
    (tree / "bringup_status.json").mkdir()
    assert _M._resolve_model_id(tree, None) == baseline


def test_d4_the_reader_reports_absence_as_empty(tmp_path):
    assert _M._declared_model_id(tmp_path) == ""
    assert _M._declared_model_id(tmp_path / "nope") == ""


# --------------------------------------------------------------------------- d5 PURITY
def test_d5_resolving_does_not_touch_the_record(tree):
    _record(tree, new_model_id=_REAL)
    before = (tree / "bringup_status.json").read_bytes()
    stamp = (tree / "bringup_status.json").stat().st_mtime_ns
    for _ in range(3):
        assert _M._resolve_model_id(tree, None) == _REAL
    assert (tree / "bringup_status.json").read_bytes() == before
    assert (tree / "bringup_status.json").stat().st_mtime_ns == stamp


# --------------------------------------------------------------------------- d6 THE CARD'S KEY
class _FakeProfiles:
    """Stands in for scorecard_profiles so the assertion is on the KEY, not on the card's layout."""

    def __init__(self):
        self.seen = []

    def render(self, model_id, arch, chips, measured, repo_root=None):
        self.seen.append(model_id)
        return "card"


@pytest.fixture
def card(monkeypatch):
    fake = _FakeProfiles()
    monkeypatch.setitem(sys.modules, "scorecard_profiles", fake)
    monkeypatch.setattr(_M, "_LAST_SCORECARD", {"TTFT_ms": 12.5})
    return fake


def _show(model_name, model_id, manifest=None):
    _M._print_scorecard(
        "0",
        manifest or {"env": {"arch": "wormhole_b0", "device_count": 1}},
        {"task": "t"},
        {},
        None,
        None,
        model_name,
        model_id,
    )


def test_d6_the_lookup_uses_the_model_id(card, capsys):
    """model_targets.yaml is keyed by model name. Looking it up by the demo's folder finds nothing,
    and the card then says "not in model_targets.yaml" about a model that is in it."""
    _show("gemma3", _REAL)
    capsys.readouterr()
    assert card.seen == [_REAL]


def test_d6_the_folder_is_only_the_fallback(card, capsys):
    """Still better than nothing when no id resolved -- but only then."""
    _show("gemma3", "")
    capsys.readouterr()
    assert card.seen == ["gemma3"]


def test_d6_the_run_record_outranks_both(card, capsys):
    """If the manifest ever states the identity, that is the run's own word and stays first."""
    _show(
        "gemma3",
        _REAL,
        manifest={"env": {"arch": "wormhole_b0", "device_count": 1}, "model_id": "acme/from-manifest"},
    )
    capsys.readouterr()
    assert card.seen == ["acme/from-manifest"]


def test_d6_the_resolved_id_actually_reaches_the_card():
    """WIRING. The tests above prove _print_scorecard prefers the id; this proves it is GIVEN one.
    The value is resolved once in run_cc and handed down -- nothing re-derives it, and nothing
    silently reverts the card to naming the run after its directory."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert 'model_id=_model_id or ""' in src, "run_cc no longer hands the resolved id to the pipeline"
    call = src[src.index("_print_scorecard(devices") :]
    assert (
        call[: call.index("\n")].rstrip().endswith("model_name, model_id)")
    ), "optimize_pipeline stopped passing the resolved id to the scorecard"
