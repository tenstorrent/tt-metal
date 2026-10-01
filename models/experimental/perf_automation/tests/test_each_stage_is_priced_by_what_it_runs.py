"""Each stage's compute roof is priced by the weights it runs, on the TP the pipeline runs at.

Qwen-Image-Edit on a WH Galaxy (2026-09-29): with no per-stage split every stage was charged the whole
28.85 B params (vae_encode read 64,000% of peak), the ceilings divided by TP=1 on 32 chips, the pinned
peak was looked up under "inferences" while stored under "inference", and DRAM compared the whole
model to one chip. These pin each fix. Stage and component names here are this fixture's own.
"""

import json
import struct
from pathlib import Path

import pytest

from agent import model_bytes as mb
from agent import stage_marks as sm
from agent import tracy_tool as tt

_PA = Path(__file__).resolve().parents[1]


# -- a pipeline, as its own code reads --------------------------------------------------------------


class _Mod:
    def __call__(self, *a, **k):
        return 0

    def __getattr__(self, n):
        return _Mod()


class _Encoder:
    def __init__(self):
        self.tower = _Mod()
        self.body = _Mod()

    def run_tower(self, x):
        return self.tower.forward(x)

    def run_body(self, x):
        return self._inner(x)

    def _inner(self, x):
        return self.body(x)


# Named once and bound below, like the other stage fixtures: a literal list on the class makes this
# test directory itself look like a model to the contract check that uses it as a non-model.
_STAGES = ["alpha", "beta", "gamma"]


class _Pipe:
    PIPELINE_STAGES = _STAGES

    def __init__(self):
        self.enc = _Encoder()
        self.core = _Mod()
        self.tp = 4

    def a(self, x):
        return self.enc.run_tower(x)

    def b(self, x):
        return self.enc.run_body(x)

    def c(self, x):
        return self._loop(x)

    def _loop(self, x):
        return self.core(x)

    def alpha_trace_step(self):
        return self.a(1)

    def beta_trace_step(self):
        return self.b(1)

    def gamma_trace_step(self):
        return self.c(1)


def test_each_stage_names_the_modules_its_own_code_runs():
    assert sm.stage_module_paths(_Pipe()) == {
        "alpha": ["enc.tower"],
        "beta": ["enc.body"],
        "gamma": ["core"],
    }


def test_the_pipeline_states_its_tp():
    assert sm.pipeline_tp(_Pipe()) == 4
    p = _Pipe()
    p.tp = "x"
    assert sm.pipeline_tp(p) == 0
    assert sm.pipeline_tp(object()) == 0


# -- the checkpoint side ----------------------------------------------------------------------------


def _st(path: Path, tensors: dict) -> None:
    hdr = {k: {"dtype": "BF16", "shape": list(s), "data_offsets": [0, 0]} for k, s in tensors.items()}
    raw = json.dumps(hdr).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)


def _snapshot(tmp_path):
    d = tmp_path / "snap"
    d.mkdir()
    (d / "model_index.json").write_text(json.dumps({"enc": ["lib", "E"], "core": ["lib", "C"]}))
    _st(
        d / "enc" / "m.safetensors",
        {"tower.w": (10, 10), "lm.layers.0.w": (20, 10), "lm.embed_tokens.weight": (50, 10)},
    )
    _st(d / "core" / "m.safetensors", {"blocks.0.w": (30, 30)})
    return d


def test_params_follow_the_modules_and_the_remainder_rule(tmp_path):
    got = mb.stage_params(_snapshot(tmp_path), {"alpha": ["enc.tower"], "beta": ["enc.body"], "gamma": ["core"]})
    # enc.body names no tensor: it takes what enc holds that alpha did not, minus the lookup table
    assert got == {"alpha": 100, "beta": 200, "gamma": 900}


def test_two_unresolved_stages_in_one_component_are_refused(tmp_path):
    got = mb.stage_params(_snapshot(tmp_path), {"x": ["enc.p"], "y": ["enc.q"], "gamma": ["core"]})
    assert got == {"gamma": 900}


# -- the pins and the TP the ceilings use -----------------------------------------------------------


class _FakeLedger:
    KIND_TP_DEGREE = "tp_degree"
    KIND_MATMUL_PARAMS = "matmul_params"

    def __init__(self):
        self.rows = []

    def anchor(self, kind, value, **kw):
        self.rows.append((kind, kw.get("depth"), value))
        return value

    def anchor_value(self, kind, depth="", **kw):
        return next((v for k, d, v in self.rows if k == kind and d == depth), None)


@pytest.fixture
def pm(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.delenv("TT_PERF_MESH_COLS", raising=False)
    import cc_optimize.perf_mcp as pm

    led = _FakeLedger()
    monkeypatch.setattr(pm, "_ledger", lambda: led)
    return pm, led


def test_the_marker_tp_and_each_stages_params_are_pinned(pm, tmp_path, monkeypatch):
    pm, led = pm
    snap = _snapshot(tmp_path)
    import agent.checkpoint_sections as cs

    monkeypatch.setattr(cs, "hf_cache_dir", lambda mid: snap)
    pm._pin_stage_params_and_tp({"alpha": ["enc.tower"], "beta": ["enc.body"], "gamma": ["core"]}, 8)
    assert ("tp_degree", "pipeline", 8.0) in led.rows
    assert {(d, v) for k, d, v in led.rows if k == "matmul_params"} == {
        ("alpha", 100.0),
        ("beta", 200.0),
        ("gamma", 900.0),
    }


def test_the_ceilings_divide_by_the_tp_the_run_reported_before_the_exported_mesh(pm, monkeypatch):
    pm, led = pm
    assert pm._tp_degree() == 1, "nothing observed, nothing exported"
    monkeypatch.setenv("TT_PERF_MESH_COLS", "32")
    assert pm._tp_degree() == 32, "before any marker, the exported topology"
    led.anchor("tp_degree", 8.0, depth="pipeline")
    assert pm._tp_degree() == 8, "the pipeline's own split beats a 1xN mesh planned without it"


def test_the_replay_reports_the_modules_and_the_pipelines_own_split():
    src = (_PA / "agent" / "trace_replay.py").read_text()
    assert 'print("TRACE_STAGE_MODULES[%s]=%s"' in src
    assert "_pipeline_tp(" in src and "(_dp * _tp) // _own_tp, _own_tp" in src


# -- the report --------------------------------------------------------------------------------------


@pytest.fixture
def summary(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("_summary_stage_price", _PA / "cc_optimize" / "summary.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("label,unit", [("inferences/s", "inference"), ("steps/s", "step"), ("tok/s/u", "token")])
def test_a_rate_label_finds_the_unit_its_anchor_is_keyed_by(summary, label, unit):
    assert summary._unit_key(label) == unit


def test_dram_is_held_against_one_chips_share(summary):
    src = (_PA / "cc_optimize" / "summary.py").read_text()
    assert "_resident = (active_bytes / max(1, int(tp_degree or 1)))" in src
    assert "_used = 100.0 * _resident / cap" in src and "_c = _resident / cap" in src


# -- the core count ----------------------------------------------------------------------------------


def test_the_core_count_is_the_detected_boards():
    from agent.environment import ARCH_FACTS

    assert tt.detected_worker_cores({"arch": "wormhole", "worker_cores": 64}) == 64
    assert tt.detected_worker_cores(dict(ARCH_FACTS["blackhole"])) == 130
    assert tt.detected_worker_cores({}) is None
    assert tt.DEFAULT_WORKER_CORES is None, "no board's grid is typed in as everyone's"


def test_an_unknown_board_never_reads_full():
    assert tt.normalize_grid(64, None) == "partial"
    assert tt.normalize_grid(5, None) == "tiny"
    assert tt.normalize_grid(64, 64) == "full"


# -- the data-parallel split -------------------------------------------------------------------------


class _Split:
    PIPELINE_STAGES = _STAGES

    def alpha_trace_split(self):
        return 4

    def beta_trace_split(self):
        raise RuntimeError("a split that cannot be read is not stated")


def test_a_stage_states_its_split_and_an_unstated_one_is_zero():
    from agent import perf_adapter as pa
    from agent import stage_seams

    assert stage_seams.SPLIT in stage_seams.OPTIONAL and stage_seams.SPLIT in stage_seams.ALL
    assert [pa._stated_count(_Split(), s, stage_seams.SPLIT) for s in _STAGES] == [4, 0, 0]
    assert pa._Stage("alpha", None, split=4).split == 4
    assert pa._Stage("alpha", None).split == 0, "unstated stays unstated"


def test_the_replay_prints_a_split_only_when_there_is_one():
    src = (_PA / "agent" / "trace_replay.py").read_text()
    assert 'print("TRACE_STAGE_SPLIT[%s]=%d" % (st.name, _sp)' in src and "if _sp > 1:" in src


def test_the_split_is_parsed_and_pinned_beside_the_item_count():
    src = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
    assert '("TRACE_STAGE_SPLIT[", stage_split)' in src
    assert "_ledger().KIND_STAGE_SPLIT, stage_split" in src


def _roofs(summary, monkeypatch, pinned):
    monkeypatch.setattr(summary, "_model_facts", lambda: {})
    monkeypatch.setattr(summary, "_pinned_peak_flops", lambda *a, **k: 64e12)
    monkeypatch.setattr(summary, "_peak_for_stage", lambda *a, **k: (64e12, "hifi4"))

    def _pin(kind, stage, model="", task=""):
        return pinned.get((kind, stage))

    monkeypatch.setattr(summary, "_pinned_ceiling_input", _pin)
    base = {("matmul_params", "alpha"): 1e9, ("stage_tokens", "alpha"): 1000}
    base.update(pinned)
    pinned.clear()
    pinned.update(base)
    return summary._stage_roofs(10e9, 288.0, 8, "inference", None, {"alpha": 1.0}, model="m", task="t")


def test_the_compute_roof_is_what_one_chip_does(summary, monkeypatch):
    one = _roofs(summary, monkeypatch, {})
    four = _roofs(summary, monkeypatch, {("stage_split", "alpha"): 4})
    assert one["alpha"]["flops"] == pytest.approx(2 * 1e9 * 1000 / 8), "unsplit: TP alone, as before"
    assert four["alpha"]["flops"] == pytest.approx(one["alpha"]["flops"] / 4), "split over 4 groups"
