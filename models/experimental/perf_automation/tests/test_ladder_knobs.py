import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_mcp_ladderknobs",
    str(Path(__file__).resolve().parents[1] / "cc_optimize" / "perf_mcp.py"),
)
perf_mcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perf_mcp)
ladder = perf_mcp._op_ladder_status
MAX = perf_mcp._MAX_KNOB_RETRIES

MM = "MatmulDeviceOperation"


def _att(kind, n=1, wedged=False):
    return [{"op_signature": MM, "kernel_kind": kind, "wedged": wedged} for _ in range(n)]


def test_grid_knob_fires_when_grid_not_full():
    _, rung, _ = ladder({"grid": "partial", "bound_by": "memory"}, MM, [])
    assert rung == "knob:grid"


def test_grid_knob_bounded_by_retry_counter():
    _, rung, _ = ladder({"grid": "partial", "bound_by": "memory"}, MM, _att("grid", MAX))
    assert rung != "knob:grid"


def test_fidelity_knob_fires_on_compute_bound():
    _, rung, _ = ladder({"grid": "full", "bound_by": "compute"}, MM, [])
    assert rung == "knob:fidelity"


def test_fidelity_bounded_by_retry_counter():
    _, rung, _ = ladder({"grid": "full", "bound_by": "compute"}, MM, _att("fidelity", MAX))
    assert rung != "knob:fidelity"


def test_dtype_knob_fires_when_weight_dtype_unknown_failopen():
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": ""}, MM, [])
    assert rung == "knob:dtype"


def test_dtype_knob_skips_when_weight_already_low():
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, [])
    assert rung == "knob:shard"


def test_shard_knob_fires_on_memory_bound_after_dtype():
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, _att("shard", 0))
    assert rung == "knob:shard"


def test_shard_bounded_by_retry_counter(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    atts = _att("shard", MAX)
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, atts)
    assert rung != "knob:shard"


def test_shard_fires_for_memory_bound_nonmatmul():
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory"}, "LayerNormDeviceOperation", [])
    assert rung == "knob:shard"


def test_ladder_order_grid_before_fidelity_before_kernels(monkeypatch):
    monkeypatch.setattr(perf_mcp, "_ttl_available", lambda: True)
    _, rung, _ = ladder({"grid": "partial", "bound_by": "compute"}, MM, [])
    assert rung == "knob:grid"
    _, rung, _ = ladder({"grid": "full", "bound_by": "compute"}, MM, _att("fidelity", MAX))
    assert rung in ("knob:dtype", "knob:shard")
    _, rung, _ = ladder(
        {"grid": "full", "bound_by": "compute"},
        MM,
        _att("fidelity", MAX) + _att("dtype") + _att("shard"),
    )
    assert rung in ("tt-lang", "cpp", "tp-fracture", "structural")


def test_residual_report_propagates_grid_fidelity_weight_dtype():
    from agent import roofline

    prof = {
        "device_ms": 10.0,
        "buckets": [
            {
                "id": "matmul",
                "device_ms": 10.0,
                "top_ops": [
                    {
                        "op_code": MM,
                        "shape": "32x2048@2048x6144",
                        "device_ms": 10.0,
                        "count": 1,
                        "grid": "partial",
                        "fidelity": "hifi4",
                        "weight_dtype": "bf16",
                    }
                ],
            }
        ],
    }
    rep = roofline.residual_report(prof, {})
    row = rep["rows"][0]
    assert "grid" in row and "fidelity" in row and "weight_dtype" in row
    assert row["grid"] == "partial" and row["fidelity"] == "hifi4"


def test_host_gate_not_starved_by_blocking_device_ops():
    prof = {
        "buckets": [
            {"id": "host_overhead", "device_ms": 202.0, "tags": {"source": "op_gap"}},
        ]
    }
    block = perf_mcp._host_gate(prof, [{"op": "MatmulDeviceOperation"}], [])
    assert block is not None


def test_memory_bound_op_still_gets_a_fidelity_sweep_after_its_own_knobs():
    """Roofline sets PRIORITY, not MEMBERSHIP.

    llama3_1_8b_p150 recorded 0 fidelity attempts across 133, because every op resolved to
    memory/host/dispatch and the fidelity gate was `bound == "compute"`. Deleting the cheapest rung
    on a roofline ESTIMATE contradicts the rule the expensive rungs live by -- a measured attempt
    replaces "I reasoned it won't help". Memory-bound ops still do dtype/shard FIRST; fidelity comes
    after, but it comes.
    """
    tried = _att("dtype", MAX) + _att("shard", MAX)
    _, rung, reason = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, tried)
    assert rung == "knob:fidelity", rung
    assert "SWEEP" in reason


def test_compute_bound_op_still_gets_dtype_and_shard_sweeps():
    """The mirror case: a compute-bound op could never reach dtype/shard, which were memory-gated."""
    _, rung, _ = ladder({"grid": "full", "bound_by": "compute"}, MM, _att("fidelity", MAX))
    assert rung == "knob:dtype", rung
    _, rung, _ = ladder({"grid": "full", "bound_by": "compute"}, MM, _att("fidelity", MAX) + _att("dtype"))
    assert rung == "knob:shard", rung


def test_priority_order_is_unchanged_for_a_memory_bound_op():
    """The sweep must not reorder the ladder: the bound-appropriate rungs still come first."""
    op = {"grid": "partial", "bound_by": "memory"}
    seen, tried = [], []
    for _ in range(4):
        _, rung, _ = ladder(op, MM, list(tried))
        seen.append(rung)
        tried += _att(rung.split(":")[-1], MAX)
        if rung == "knob:grid":
            op = {**op, "grid": "full"}
    assert seen == ["knob:grid", "knob:dtype", "knob:shard", "knob:fidelity"], seen


def test_priority_order_is_unchanged_for_a_compute_bound_op():
    op = {"grid": "partial", "bound_by": "compute"}
    seen, tried = [], []
    for _ in range(4):
        _, rung, _ = ladder(op, MM, list(tried))
        seen.append(rung)
        tried += _att(rung.split(":")[-1], MAX)
        if rung == "knob:grid":
            op = {**op, "grid": "full"}
    assert seen == ["knob:grid", "knob:fidelity", "knob:dtype", "knob:shard"], seen


def test_one_measured_attempt_clears_a_swept_rung():
    """The sweep is a completeness floor, not a second search: one attempt is enough, so it cannot
    loop even when the knob does nothing for this op."""
    tried = _att("dtype", MAX) + _att("shard", MAX) + _att("fidelity")
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, tried)
    assert rung not in ("knob:fidelity", "knob:dtype", "knob:shard"), rung


def test_host_bucket_is_still_exempt_from_device_knobs():
    """A host_fallback entry is NOT a device op — grid/fidelity/dtype on it are meaningless. That
    exclusion is by KIND, not by a roofline estimate, so the sweep must not reach it."""
    _, rung, _ = ladder({"bound_by": "host", "bucket": "host_fallback"}, "host_overhead", [])
    assert rung == "trace-capture", rung
    # The rung retires on a measured win or at the attempt cap, not on one recorded row -- see
    # test_the_host_rung_clears_on_a_measurement.py. What this test guards is unchanged either way:
    # the device-knob sweep must never reach a host bucket.
    host_att = [
        {"op_signature": "host_overhead", "kernel_kind": k, "measured_ms": 400.0, "beat_baseline": False}
        for k in ("structural", "trace", "trace-capture")
    ]
    done, rung, _ = ladder({"bound_by": "host", "bucket": "host_fallback"}, "host_overhead", host_att)
    assert done and rung == "done"


def test_op_with_no_grid_tag_still_gets_a_grid_sweep():
    """The grid rung is gated on the profile TAG (`grid and grid != "full"`), so an op whose profile
    reports no grid never fired it -- omitted on a missing tag rather than a bad estimate, but
    omitted just the same. llama3_1_8b_p150 had one such bucket."""
    _, rung, _ = ladder({"bound_by": "memory", "weight_dtype": "bf8_b"}, MM, _att("shard", MAX))
    assert rung == "knob:grid", rung


def test_full_grid_is_not_swept():
    """An op already on the full grid has nothing to try -- the sweep is a floor, not busywork."""
    tried = _att("shard", MAX) + _att("fidelity") + _att("dtype")
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, tried)
    assert rung != "knob:grid", rung


def test_dtype_is_never_offered_to_a_non_matmul():
    """Applicability, not priority: a weight-dtype knob is meaningless on an op with no weights.
    The sweep must not turn 'deprioritised' into 'offered anyway' for a knob that cannot apply."""
    TOPK = "TopKDeviceOperation"

    def att(kind, n=1):
        return [{"op_signature": TOPK, "kernel_kind": kind} for _ in range(n)]

    tried, seen = [], []
    for _ in range(5):
        _, rung, _ = ladder({"grid": "full", "bound_by": "memory"}, TOPK, list(tried))
        if not rung.startswith("knob"):
            break
        seen.append(rung)
        tried += att(rung.split(":")[-1], MAX)
    assert seen, "no knob was offered at all"
    assert "knob:dtype" not in seen, seen
    assert "knob:shard" in seen and "knob:fidelity" in seen, seen


def test_grid_knob_is_skipped_once_the_grid_is_full():
    tried = _att("dtype", MAX) + _att("shard", MAX) + _att("fidelity", MAX)
    _, rung, _ = ladder({"grid": "full", "bound_by": "memory", "weight_dtype": "bf8_b"}, MM, tried)
    assert rung != "knob:grid", rung
