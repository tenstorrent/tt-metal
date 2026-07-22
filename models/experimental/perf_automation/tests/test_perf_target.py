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
    mf = {"weight_tensors": [
        {"numel": 1_000_000, "dtype": "bfloat16"},   # 2.0
        {"numel": 1_000_000, "dtype": "bfloat8_b"},  # 1.0625
    ]}
    assert pt.active_bytes(mf) == int(round(1_000_000 * 2.0 + 1_000_000 * 1.0625))


def test_dense_active_bytes_from_param_count():
    mf = {"total_params": 8_000_000_000, "dominant_dtype": "bfloat16"}
    assert pt.active_bytes(mf) == 16_000_000_000


def test_moe_uses_shared_plus_topk_not_all_experts():
    # 128 experts, top_k=8: only 8 experts' bytes count, not 128.
    mf = {"is_moe": True, "num_experts": 128, "top_k": 8,
          "shared_params": 1_000_000_000, "per_expert_params": 4_000_000, "dominant_dtype": "bfloat16"}
    got = pt.active_bytes(mf)
    expect = (1_000_000_000 + 8 * 4_000_000) * 2.0
    assert got == int(round(expect))
    # sanity: NOT all-experts (which would be far larger)
    all_experts = (1_000_000_000 + 128 * 4_000_000) * 2.0
    assert got < all_experts


def test_kv_term_off_by_default_on_when_seqlen():
    mf = {"total_params": 1_000_000, "dominant_dtype": "bfloat16",
          "layers": 32, "kv_heads": 8, "head_dim": 128}
    base = pt.active_bytes(mf)                    # seq_len=0 -> weights only
    withkv = pt.active_bytes(mf, seq_len=2048)    # + KV
    assert base == 2_000_000
    assert withkv == base + int(round(2.0 * 32 * 8 * 128 * 2048 * 2.0))


def test_tp_divides_per_device_bytes():
    mf = {"total_params": 8_000_000_000, "dominant_dtype": "bfloat16"}
    t1 = pt.compute_target(mf, _BH, tp_degree=1)
    t4 = pt.compute_target(mf, _BH, tp_degree=4)
    # per-device bytes /4 -> theoretical tok/s x4
    assert abs(t4.theoretical_tok_s - 4 * t1.theoretical_tok_s) < 1e-6


def test_compute_target_ceiling_and_band():
    mf = {"total_params": 1_000_000_000, "dominant_dtype": "bfloat16"}  # 2 GB
    t = pt.compute_target(mf, _BH)  # 512e9 / 2e9 = 256 tok/s
    assert abs(t.theoretical_tok_s - 256.0) < 1e-3
    assert abs(t.band[0] - 0.60 * 256.0) < 1e-3 and abs(t.band[1] - 0.80 * 256.0) < 1e-3


def test_status_below_in_above():
    mf = {"total_params": 1_000_000_000, "dominant_dtype": "bfloat16"}
    t = pt.compute_target(mf, _BH)  # theo 256 tok/s ; band 153.6 - 204.8
    below = pt.score(t, forward_ms=1000.0 / 100.0)   # 100 tok/s < 153.6
    inb = pt.score(t, forward_ms=1000.0 / 200.0)     # 200 tok/s, >=153.6, <=256
    above = pt.score(t, forward_ms=1000.0 / 300.0)   # 300 tok/s > 256 ceiling
    assert below["status"] == "BELOW_BAND"
    assert inb["status"] == "IN_BAND"
    assert above["status"] == "ABOVE_BAND"


def test_score_unknown_on_bad_inputs():
    t = pt.compute_target({"total_params": 0}, _BH)  # active_bytes 0 -> theo 0
    assert pt.score(t, 5.0)["status"] == "UNKNOWN"
    good = pt.compute_target({"total_params": 1_000_000_000}, _BH)
    assert pt.score(good, 0.0)["status"] == "UNKNOWN"  # no measurement


def test_per_module_floor_band():
    # module floor 2.0 ms -> theo 500 inv/s ; band 300-400
    t = pt.target_from_floor_ms(2.0)
    assert abs(t.theoretical_tok_s - 500.0) < 1e-6
    # measured 3.5 ms -> 285.7 inv/s < 300 -> BELOW_BAND (headroom)
    assert pt.score(t, 3.5)["status"] == "BELOW_BAND"
    # measured 2.2 ms -> 454 inv/s, in band -> IN_BAND (near floor, done)
    assert pt.score(t, 2.2)["status"] == "IN_BAND"
    # measured 1.8 ms -> beat the 2.0ms floor -> ABOVE_BAND (floor suspect)
    assert pt.score(t, 1.8)["status"] == "ABOVE_BAND"


def test_list_topk_degrades_not_crashes():
    mf = {"is_moe": True, "top_k": [8, 8, 8], "shared_params": 1_000_000, "per_expert_params": 1000,
          "dominant_dtype": "bfloat16"}
    got = pt.active_bytes(mf)  # top_k coerced to 8
    assert got == int(round((1_000_000 + 8 * 1000) * 2.0))


def test_prefill_stub_raises():
    import pytest
    with pytest.raises(NotImplementedError):
        pt.active_bytes({"total_params": 1}, regime="prefill")
    with pytest.raises(NotImplementedError):
        pt.prefill_ceiling()
