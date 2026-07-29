# SPDX-License-Identifier: Apache-2.0
"""Perf-target roofline: dense/MoE active_bytes, KV term, TP scaling, dtype bytes, status
mapping, and the per-module ms-floor band. Pure unit tests (plan §7)."""
import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "perf_target_ut",
    str(Path(__file__).resolve().parents[1] / "agent" / "perf_target.py"),
)
pt = importlib.util.module_from_spec(_SPEC)
sys.modules["perf_target_ut"] = pt  # dataclass annotation resolution needs the module registered
_SPEC.loader.exec_module(pt)

_BH = {"dram_bw_gbps": 512.0}
_WH = {"dram_bw_gbps": 288.0}


def test_bytes_per_elem_bf8_is_1_0625():
    assert pt._bytes_per_elem("bfloat8_b") == 1.0625
    assert pt._bytes_per_elem("bfloat16") == 2.0
    assert pt._bytes_per_elem("bfloat4_b") == 0.5625
    assert pt._bytes_per_elem("float32") == 4.0
    assert pt._bytes_per_elem("weird") == 2.0  # default


def test_dense_active_bytes_tensor_sum_dtype_aware():
    mf = {
        "weight_tensors": [
            {"numel": 1_000_000, "dtype": "bfloat16"},  # 2.0
            {"numel": 1_000_000, "dtype": "bfloat8_b"},  # 1.0625
        ]
    }
    assert pt.active_bytes(mf) == int(round(1_000_000 * 2.0 + 1_000_000 * 1.0625))


def test_dense_active_bytes_from_param_count():
    mf = {"total_params": 8_000_000_000, "dominant_dtype": "bfloat16"}
    assert pt.active_bytes(mf) == 16_000_000_000


def test_moe_uses_shared_plus_topk_not_all_experts():
    # 128 experts, top_k=8: only 8 experts' bytes count, not 128.
    mf = {
        "is_moe": True,
        "num_experts": 128,
        "top_k": 8,
        "shared_params": 1_000_000_000,
        "per_expert_params": 4_000_000,
        "dominant_dtype": "bfloat16",
    }
    got = pt.active_bytes(mf)
    expect = (1_000_000_000 + 8 * 4_000_000) * 2.0
    assert got == int(round(expect))
    # sanity: NOT all-experts (which would be far larger)
    all_experts = (1_000_000_000 + 128 * 4_000_000) * 2.0
    assert got < all_experts


def test_kv_term_off_by_default_on_when_seqlen():
    mf = {"total_params": 1_000_000, "dominant_dtype": "bfloat16", "layers": 32, "kv_heads": 8, "head_dim": 128}
    base = pt.active_bytes(mf)  # seq_len=0 -> weights only
    withkv = pt.active_bytes(mf, seq_len=2048)  # + KV
    assert base == 2_000_000
    assert withkv == base + int(round(2.0 * 32 * 8 * 128 * 2048 * 2.0))


def test_tp_divides_per_device_bytes():
    mf = {"total_params": 8_000_000_000, "dominant_dtype": "bfloat16"}
    t1 = pt.compute_target(mf, _BH, tp_degree=1)
    t4 = pt.compute_target(mf, _BH, tp_degree=4)
    # per-device bytes /4 -> theoretical tok/s x4
    assert abs(t4.theoretical_rate - 4 * t1.theoretical_rate) < 1e-6


def test_compute_target_ceiling_and_band():
    # 1B params -> 1 GB under xB -> xGB (NOT 2 GB from params x bf16: the stored dtype is not the
    # divisor any more), and the 0.80 sustained fraction is inside the ceiling: (512*0.8)/1 = 409.6.
    mf = {"total_params": 1_000_000_000, "dominant_dtype": "bfloat16"}
    t = pt.compute_target(mf, _BH)
    assert t.active_bytes == 1_000_000_000
    assert abs(t.theoretical_rate - 409.6) < 1e-3
    assert abs(t.band[0] - 0.60 * 409.6) < 1e-3 and abs(t.band[1] - 0.80 * 409.6) < 1e-3


def test_status_below_in_above():
    mf = {"total_params": 1_000_000_000, "dominant_dtype": "bfloat16"}
    t = pt.compute_target(mf, _BH)  # theo 409.6 tok/s ; band 245.8 - 327.7
    below = pt.score(t, forward_ms=1000.0 / 100.0)  # 100 tok/s < 245.8
    inb = pt.score(t, forward_ms=1000.0 / 300.0)  # 300 tok/s, >=245.8, <=409.6
    above = pt.score(t, forward_ms=1000.0 / 600.0)  # 600 tok/s > 409.6 ceiling
    assert below["status"] == "BELOW_BAND"
    assert inb["status"] == "IN_BAND"
    assert above["status"] == "ABOVE_BAND"


def test_score_unknown_on_bad_inputs():
    t = pt.compute_target({"total_params": 0}, _BH)  # active_bytes 0 -> theo 0
    assert pt.score(t, 5.0)["status"] == "UNKNOWN"
    good = pt.compute_target({"total_params": 1_000_000_000}, _BH)
    assert pt.score(good, 0.0)["status"] == "UNKNOWN"  # no measurement


def test_a_floor_target_carries_no_band():
    """60-80% is a statement about DRAM BANDWIDTH. The floor is a sum of per-op minimum times over one
    profiling window, so 1000/floor is an invocations-per-second figure with no hardware peak behind
    it -- banding it produced "achievable 671.54 - 895.38 ms" beside a 534 ms measurement, and the
    optimize stop gate consulted that same band, so a run could be declared done against a range
    never derived from the hardware.
    """
    t = pt.target_from_floor_ms(2.0)
    assert abs(t.theoretical_rate - 500.0) < 1e-6
    assert t.band == (0.0, 0.0)
    # no band to be in, at any measurement -- and never a silent IN_BAND from `measured >= 0`
    for ms in (3.5, 2.2, 2.05):
        assert pt.score(t, ms)["status"] == "NO_BAND", ms
    # beating the floor itself is still meaningful: one side of the pair is stale
    assert pt.score(t, 1.8)["status"] == "ABOVE_BAND"


def test_only_a_bandwidth_ceiling_produces_a_band():
    """The band must come from bandwidth-over-bytes, never from the ms floor: 8 GB on 512 GB/s -> a
    (512*0.8)/8 = 51.2 ceiling with the 30.7-41.0 achievable band."""
    t = pt.compute_target({"weight_bytes": int(8e9)}, {"dram_bw_gbps": 512.0})
    assert [round(b, 1) for b in t.band] == [30.7, 41.0]
    assert pt.score(t, 30.0)["status"] == "IN_BAND"  # 33.3 tok/s, inside 30.7-41.0
    assert pt.score(t, 50.0)["status"] == "BELOW_BAND"  # 20 tok/s
    assert pt.score(t, 10.0)["status"] == "ABOVE_BAND"  # 100 tok/s, past the 51.2 ceiling


def test_list_topk_degrades_not_crashes():
    mf = {
        "is_moe": True,
        "top_k": [8, 8, 8],
        "shared_params": 1_000_000,
        "per_expert_params": 1000,
        "dominant_dtype": "bfloat16",
    }
    got = pt.active_bytes(mf)  # top_k coerced to 8
    assert got == int(round((1_000_000 + 8 * 1000) * 2.0))


def test_prefill_stub_raises():
    """Explicit try/except: the repo prefers an error-context fixture that lives in the root
    conftest, which this suite's rootdir does not reach."""
    for call in (lambda: pt.active_bytes({"total_params": 1}, regime="prefill"), pt.prefill_ceiling):
        raised = None
        try:
            call()
        except NotImplementedError as exc:
            raised = exc
        assert raised is not None, call


# --- junk inputs must DEGRADE, never crash or invert (found by fuzzing the ceiling path) ---


def test_a_negative_bandwidth_is_unknown_not_a_negative_ceiling():
    """A junk dram_bw_gbps divided straight through to a NEGATIVE ceiling, which set a negative band
    and scored BELOW_BAND -- a verdict against a target that cannot exist. Unknown, not fast."""
    t = pt.compute_target({"total_params": int(8e9)}, {"dram_bw_gbps": -1, "dram_bw_per_chip_gbps": -1})
    assert t.theoretical_rate == 0.0
    assert t.band == (0.0, 0.0)
    assert pt.score(t, 19.4)["status"] == "UNKNOWN"


def test_a_non_finite_param_count_degrades_instead_of_raising():
    """json.loads accepts `Infinity`, so a corrupted or hand-edited perf_target_inputs.json reaches
    here; int(round(inf)) raises OverflowError -- neither TypeError nor ValueError, so it escaped the
    coercion guard and took the whole ceiling path down."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        assert pt._scalar(bad, 0) == 0
        t = pt.compute_target({"total_params": bad}, {"dram_bw_gbps": 512.0})
        assert t.theoretical_rate == 0.0
        assert pt.score(t, 19.4)["status"] == "UNKNOWN"
        assert pt.active_bytes({"total_params": bad, "dominant_dtype": "bfloat16"}) == 0


def test_rate_and_band_never_inverts_or_goes_negative():
    for byts, peak, frac, tp in (
        (0, 512e9, 0.8, 1),
        (8e9, 0, 0.8, 1),
        (8e9, 512e9, 0.0, 1),
        (8e9, -512e9, 0.8, 1),
        (8e9, 512e9, -0.8, 1),
        (8e9, 512e9, 0.8, 0),
        (8e9, 512e9, 0.8, -4),
        (-8e9, 512e9, 0.8, 1),
    ):
        theo, band = pt.rate_and_band(byts, peak, frac=frac, tp_degree=tp)
        assert theo >= 0.0, (byts, peak, frac, tp, theo)
        assert band[0] <= band[1], (byts, peak, frac, tp, band)
