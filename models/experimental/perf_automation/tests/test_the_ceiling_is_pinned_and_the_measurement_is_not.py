"""One rule, both directions: everything the THEORETICAL column rests on is pinned, and nothing the
MEASURED column rests on is.

A ceiling that moves cannot be compared across rounds -- the target retreats ahead of the work and a
real win reads as no progress. voxtral, measured: a fidelity lever took the pinned peak from 175.5
TFLOPS to 702.0 and prefill's ceiling from 203.82 ms to 50.95, a 4x change in the yardstick caused by
an optimization rather than by hardware.

A measurement that is pinned is the opposite failure: it would freeze at its first value and stop
reporting what the model does now.

The four inputs behind the two roofs:
    memory   = bytes / peak_bandwidth        bytes: KIND_ACTIVE_BYTES + KIND_STAGE_BYTES
    compute  = (2 x params x tokens) / peak  peak:  KIND_PEAK_FLOPS
                                             params: KIND_MATMUL_PARAMS
                                             tokens: KIND_STAGE_TOKENS
peak_bandwidth is an arch constant and cannot drift."""
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _src(rel):
    return (_PA / rel).read_text()


def _fn(body, name):
    i = body.index("def %s(" % name)
    return body[i : body.index("\ndef ", i + 10)]


def test_every_ceiling_input_has_an_anchor_kind():
    m = _src("cc_optimize/measurements.py")
    for kind in ("KIND_ACTIVE_BYTES", "KIND_STAGE_BYTES", "KIND_PEAK_FLOPS", "KIND_MATMUL_PARAMS", "KIND_STAGE_TOKENS"):
        assert "%s = " % kind in m, "no anchor kind for a ceiling input: %s" % kind


def test_every_ceiling_input_is_written_by_a_producer():
    """Anchored where the value is MADE. A renderer that pinned would make producing the report a
    side effect -- the first report written would fix a number every later one inherits."""
    mcp, run = _src("cc_optimize/perf_mcp.py"), _src("cc_optimize/run.py")
    both = mcp + run
    for kind in ("KIND_ACTIVE_BYTES", "KIND_STAGE_BYTES", "KIND_PEAK_FLOPS", "KIND_STAGE_TOKENS", "KIND_MATMUL_PARAMS"):
        assert "%s," % kind in both, "ceiling input never anchored by any producer: %s" % kind


def test_the_renderer_reads_anchors_and_never_writes_them():
    """summary.py may ask what is pinned; it may not pin."""
    body = _src("cc_optimize/summary.py")
    assert "anchor_value(" in body, "the renderer no longer consults the anchors"
    for writer in (".anchor(", "led.anchor(", "_ledger().anchor("):
        assert writer not in body, "the report writes an anchor (%s): rendering must not mutate" % writer


def test_the_compute_roof_prefers_pinned_params_and_tokens():
    body = _src("cc_optimize/summary.py")
    fn = _fn(body, "_stage_roofs")
    assert "_pinned_ceiling_input(_LED_PARAMS" in fn, "params still re-derived from the current model"
    assert "_pinned_ceiling_input(_LED_TOKENS" in fn, "tokens still re-observed each run"


def test_the_memory_roof_prefers_pinned_bytes():
    body = _src("cc_optimize/summary.py")
    fn = _fn(body, "stage_read_bytes")
    i, j = fn.index("_pinned_stage_bytes"), fn.index("measured or {}")
    assert i < j, "a live measurement outranks the pinned read set"


def test_the_measured_column_is_not_pinned():
    """The MEASURED side must keep moving: stage times come from the run's own file, never a ledger
    anchor. Pinning one would freeze it at its first reading."""
    body = _src("cc_optimize/summary.py")
    assert "read_stage_ms(" in body, "measured stage times no longer come from the run's own record"
    fn = _fn(body, "_measured_bw_gbps")
    for pinned in ("anchor_value", "_pinned_"):
        assert pinned not in fn, "the measured bandwidth reads a pinned value"


# --- and prove it by behaviour, not by reading the source ----------------------------------------


def _roofs(monkeypatch, tmp_path, params, toks, pin=None):
    """Run _stage_roofs against a one-stage model whose params/tokens we control."""
    import sys

    sys.path.insert(0, str(_PA))
    from cc_optimize import measurements as M
    from cc_optimize import summary as S

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(M, "ledger_path", lambda model="", task="": tmp_path / "led.jsonl", raising=False)
    if pin:
        M.anchor(M.KIND_MATMUL_PARAMS, float(pin[0]), depth="decode", mode="params", source="t", model="m", task="main")
        M.anchor(M.KIND_STAGE_TOKENS, float(pin[1]), depth="decode", mode="items", source="t", model="m", task="main")
    monkeypatch.setattr(
        S,
        "_stage_block",
        lambda *a, **k: {"layers": 30, "hidden_size": 3072, "matmul_params": params},
        raising=False,
    )
    monkeypatch.setattr(S, "_stage_items", lambda *a, **k: toks, raising=False)
    return S, M


def test_a_pinned_param_count_holds_the_compute_roof_still(monkeypatch, tmp_path):
    """The behaviour the anchor exists for: halve the model's params and the CEILING must not move."""
    from cc_optimize import measurements as M

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(M, "ledger_path", lambda model="", task="": tmp_path / "led.jsonl", raising=False)
    M.anchor(M.KIND_MATMUL_PARAMS, 3.6e9, depth="decode", mode="params", source="t", model="m", task="main")
    # a later, smaller reading must not move the pin
    M.anchor(M.KIND_MATMUL_PARAMS, 1.8e9, depth="decode", mode="params", source="t", model="m", task="main")
    assert M.anchor_value(M.KIND_MATMUL_PARAMS, depth="decode", model="m", task="main") == 3.6e9


def test_a_pinned_item_count_holds_too(monkeypatch, tmp_path):
    from cc_optimize import measurements as M

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(M, "ledger_path", lambda model="", task="": tmp_path / "led.jsonl", raising=False)
    M.anchor(M.KIND_STAGE_TOKENS, 4096.0, depth="prefill", mode="items", source="t", model="m", task="main")
    M.anchor(M.KIND_STAGE_TOKENS, 512.0, depth="prefill", mode="items", source="t", model="m", task="main")
    assert M.anchor_value(M.KIND_STAGE_TOKENS, depth="prefill", model="m", task="main") == 4096.0


def test_the_pins_are_keyed_per_stage(monkeypatch, tmp_path):
    """encode's params must not answer for decode -- the shared-key mistake the peak anchor made."""
    from cc_optimize import measurements as M

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(M, "ledger_path", lambda model="", task="": tmp_path / "led.jsonl", raising=False)
    M.anchor(M.KIND_MATMUL_PARAMS, 6.4e8, depth="encode", mode="params", source="t", model="m", task="main")
    assert M.anchor_value(M.KIND_MATMUL_PARAMS, depth="encode", model="m", task="main") == 6.4e8
    assert M.anchor_value(M.KIND_MATMUL_PARAMS, depth="decode", model="m", task="main") is None
