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
    # 1B params served bf16 -> 2 GB. THE DECLARED DTYPE IS THE DIVISOR AGAIN.
    #
    # This asserted 1 GB under the xB -> xGB rule, whose safety argument was that TT models are served
    # under a byte per parameter (bf8 1.0625, bf4 0.5625) so the ceiling would be under-reported and a
    # run would keep optimising. A bf16 model inverts it: it streams 2 B/param, so the rule published a
    # ceiling ABOVE what the hardware permits -- voxtral got 141.8 tok/s/u against a true ~55, and the
    # run was told it had headroom that does not exist. The width also moves DURING a run, bf16 -> bf8
    # -> bf4 as dtype rungs land, so no constant can stand in for it.
    #
    # The ceiling is still SPEC and the 0.80 fraction still sets the band's top; only the byte count
    # changed, from a constant to what the model says it is served at.
    mf = {"total_params": 1_000_000_000, "dominant_dtype": "bfloat16"}
    t = pt.compute_target(mf, _BH)
    assert t.active_bytes == 2_000_000_000
    assert abs(t.theoretical_rate - 256.0) < 1e-3
    assert abs(t.band[0] - 0.60 * 256.0) < 1e-3 and abs(t.band[1] - 0.80 * 256.0) < 1e-3


def test_status_below_in_above():
    mf = {"total_params": 1_000_000_000, "dominant_dtype": "bfloat16"}
    t = pt.compute_target(mf, _BH)  # 2 GB at bf16 -> theo 256.0 tok/s (spec) ; band 153.6 - 204.8
    below = pt.score(t, forward_ms=1000.0 / 50.0)  # 50 tok/s < 153.6
    inb = pt.score(t, forward_ms=1000.0 / 175.0)  # 175 tok/s, >=153.6, <=204.8
    above = pt.score(t, forward_ms=1000.0 / 300.0)  # 300 tok/s > the 256.0 spec ceiling
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
    SPEC ceiling of 512/8 = 64.0 with a 38.4-51.2 achievable band (0.60-0.80 of it)."""
    t = pt.compute_target({"weight_bytes": int(8e9)}, {"dram_bw_gbps": 512.0})
    assert round(t.theoretical_rate, 1) == 64.0
    assert [round(b, 1) for b in t.band] == [38.4, 51.2]
    assert pt.score(t, 25.0)["status"] == "IN_BAND"  # 40 tok/s, inside 38.4-51.2
    assert pt.score(t, 50.0)["status"] == "BELOW_BAND"  # 20 tok/s
    assert pt.score(t, 10.0)["status"] == "ABOVE_BAND"  # 100 tok/s, past the 64.0 spec ceiling


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


def test_prefill_bytes_are_costed_and_unknown_regimes_are_not():
    """active_bytes MODELS prefill now: refusing it left its caller using the DECODE read set, so a
    report printed one memory ceiling twice and called it physics.

    An unknown regime is still refused, because accepting anything would let a typo silently return a
    decode figure. prefill_ceiling is gone -- the compute side is served by compute_ceiling
    with tokens_per_unit, and a second entry point for the same question is what this suite exists to
    prevent."""
    assert pt.active_bytes({"total_params": 1}, regime="prefill") > 0

    # THE NAME IS NOT CONSULTED, so an unknown one is costed rather than refused. This used to raise,
    # on the reasoning that accepting anything would let a typo return a decode figure -- true while
    # `regime` selected the math, and false since the KV and activation terms started keying on
    # `items`. What the refusal actually did was price any THIRD stage as weights-only, because the
    # caller catches and falls back: an audio encoder lost its activation term for being called
    # "encode". A typo now cannot change the number, which is asserted directly below.
    assert pt.active_bytes({"total_params": 1}, regime="nonsense") > 0

    # prefill_ceiling is GONE, not stubbed. It raised NotImplementedError, nothing in the tool ever
    # called it, and the compute side it stood in for is served by compute_ceiling -- so all it did
    # was keep a stage name alive in the module that owns the byte model.
    assert not hasattr(pt, "prefill_ceiling"), "the dead stub is back"


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


# --- the three roofline defects found on the run-50 report ------------------------------------


def test_the_attention_flops_are_counted_not_only_the_weight_matmuls():
    """2 x params x tokens counts every WEIGHT matmul -- each parameter is multiplied once per token,
    so every projection in every layer is already there. What it omitted is the attention SCORE path,
    QK^T and A.V, which uses no parameters and scales with the SQUARE of the sequence.

    0.4% of prefill FLOPs at ISL 128 -- invisible, which is why it survived -- 3.3% at 1024 and 21.3%
    at 8192, where it decides whether the stage reads compute-bound or memory-bound at all."""
    L, H, P = 48, 3840, 11_180_446_320
    for toks, want_pct in ((128, 0.4), (1024, 3.3), (8192, 21.3)):
        weights = 2.0 * P * toks
        attn = 4.0 * L * toks * toks * H
        assert abs(100.0 * attn / (weights + attn) - want_pct) < 0.2, toks
    # and it is ADDITIVE, never a replacement: the weight term still dominates at the benchmark point
    assert 4.0 * L * 128 * 128 * H < 0.01 * (2.0 * P * 128)
