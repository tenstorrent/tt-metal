"""git_revert deletes the facts file, and two facts have no way back.

gitio.remove_new_untracked deletes untracked files created since the checkpoint -- deliberately, so
a lever that CREATED a kernel module cannot survive a revert. perf_target_inputs.json is untracked
and created after the baseline, so it looks exactly like such a file and every rejected attempt
removes it.

Most of it heals: the flat keys are re-derived from the checkpoint, the census is re-measured by the
next full-pipeline run. `blocks` needs a resolvable model id, and `stage_roots` is never written by
the emitter at all -- discovery merges it once and nothing re-runs that.

Run 16: with both gone, _stage_block("encode") returned None, the compute roof fell back to the flat
total_params (3.611B -- the LANGUAGE model's per-token read set) for an audio encoder holding
0.662B, and encode read 46% of a ceiling it was really at ~8% of. Decode looked correct only by
coincidence: total_params IS its read set, so the fallback was accidentally its right answer."""
import importlib.util as ilu
import json
import sys
import tempfile
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _run_mod():
    sys.path.insert(0, str(_PA))
    spec = ilu.spec_from_file_location("cc_run_arch", _PA / "cc_optimize" / "run.py")
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_FULL = {
    "weight_bytes": 9356474312,
    "dominant_dtype": "bfloat16",
    "source": "checkpoint bytes + HF config",
    "total_params": 3611483136,
    "blocks": {
        "audio_tower": {"layers": 32, "params": 636968960},
        "language_model": {"layers": 30, "params": 4014136320},
    },
    "stage_roots": {"encode": "audio_tower", "prefill": "language_model", "decode": "language_model"},
}
_REBUILD = {
    "weight_bytes": 9356474312,
    "dominant_dtype": "bfloat16",
    "source": "checkpoint bytes + HF config",
    "total_params": 3611483136,
    "layers": 32,
}


def _emit(mod, d, facts):
    mod._perf_target_inputs = lambda *a, **k: dict(facts)
    mod._emit_perf_target_inputs(d, d, None, {})


def test_the_map_survives_the_file_being_deleted(monkeypatch):
    """THE CASE THE CARRY-FORWARD CANNOT COVER. It reads the PREVIOUS file; a revert deletes it, so
    there is no previous file to read."""
    d, st = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(st))
    m = _run_mod()
    _emit(m, d, _FULL)
    (d / "perf_target_inputs.json").unlink()  # git_revert
    _emit(m, d, _REBUILD)  # rebuild: checkpoint facts only
    got = json.loads((d / "perf_target_inputs.json").read_text())
    assert got.get("blocks"), "blocks did not survive the delete"
    assert got.get("stage_roots", {}).get("encode") == "audio_tower"
    assert got["blocks"]["audio_tower"]["params"] == 636968960


def test_stage_roots_is_mirrored_by_discovery_too(monkeypatch):
    """The emitter never produces stage_roots -- only discovery does, once. If that is not mirrored
    at the point it is merged, it has no path back at all."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index('_merge_model_facts(model_root, {"stage_roots": _roots})')
    assert '_mirror_arch_facts({"stage_roots": _roots})' in src[i : i + 400]


def test_the_census_is_deliberately_not_mirrored():
    """Architecture is safe to cache forever; a MEASUREMENT is not. device_weight_bytes and
    bytes_per_param change the first time a precision knob lands, so a cached copy would be stale."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("ARCH_KEYS = (")
    stanza = src[i : i + 200]
    for k in ("device_weight_bytes", "bytes_per_param", "device_census_complete"):
        assert k not in stanza, "%s must not be mirrored -- it must be re-measured" % k


def test_it_writes_nothing_without_persist(monkeypatch):
    """Without --persist, state_dir() is the system temp dir -- the very place this exists to escape."""
    d, st = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    monkeypatch.delenv("PERF_MCP_STATE_DIR", raising=False)
    m = _run_mod()
    _emit(m, d, _FULL)
    assert not (st / "model_blocks.json").exists()


def test_a_mirror_that_cannot_be_written_never_costs_the_write(monkeypatch):
    """Best-effort: the facts file must still be written even if the mirror fails."""
    d, st = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(st))
    m = _run_mod()
    m.read_arch_mirror = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    _emit(m, d, _FULL)
    assert (d / "perf_target_inputs.json").is_file()
