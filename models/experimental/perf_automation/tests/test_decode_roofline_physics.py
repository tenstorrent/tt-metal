# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode roofline is peak DRAM bandwidth / bytes-streamed-per-token, and it must ACTIVATE.

The standard bound for LLM decode is:

    ceiling    = peak_BW / model_bytes            512 GB/s / 8 GB = 64 tok/s/u
    achievable = 60-80% of peak                   307-409 GB/s / 8 GB = 38-51 tok/s/u
    measured   = model_bytes / forward_time        8 GB / 19.4 ms = 412 GB/s -> 51.5 tok/s/u

perf_target implements exactly this, and read its inputs from perf_target_inputs.json -- which
NOTHING in the tool ever wrote. So active_bytes was always 0, every report fell back to the
Sigma-per-op ms floor, and the reports said "not an LLM decode pipeline" about Llama. The floor is a
weaker statement and it moves when the op mix changes; this bound does not.

These pin the arithmetic against the published Llama-3.1-8B figures above, and pin that the inputs
are produced from the checkpoint and HF config for any dense LLM without per-model wiring.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_BH_DRAM_GBPS = 512.0
_GB = 1e9


def _pt():
    from agent import perf_target

    return perf_target


def test_the_ceiling_is_peak_bandwidth_over_model_bytes():
    """8 GB of weights on a 512 GB/s part -> 64 tok/s/u."""
    pt = _pt()
    tgt = pt.compute_target({"weight_bytes": int(8 * _GB)}, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert tgt.active_bytes == int(8 * _GB)
    assert round(tgt.theoretical_rate, 1) == 64.0


def test_the_achievable_band_is_60_to_80_percent_of_peak():
    """307-409 GB/s of the 512 GB/s peak -> 38-51 tok/s/u."""
    pt = _pt()
    tgt = pt.compute_target({"weight_bytes": int(8 * _GB)}, {"dram_bw_gbps": _BH_DRAM_GBPS})
    lo, hi = tgt.band
    assert round(lo, 1) == 38.4 and round(hi, 1) == 51.2, (lo, hi)


def test_a_measured_forward_pass_scores_as_published():
    """19.4 ms/token on 8 GB = 412 GB/s = 80% utilisation = 51.5 tok/s/u, at the top of the band."""
    pt = _pt()
    tgt = pt.compute_target({"weight_bytes": int(8 * _GB)}, {"dram_bw_gbps": _BH_DRAM_GBPS})
    s = pt.score(tgt, 19.4)
    assert round(s["measured_tok_s"], 1) == 51.5
    assert round(s["effective_bw_bytes_s"] / 1e9) == 412
    # 8 GB / 19.4 ms = 412.4 GB/s, and 412.4/512 = 80.5%. The published "80%" rounds 412 first;
    # both agree to the nearest point, so pin the band rather than a single rounding convention.
    assert 80.0 <= s["bw_util"] * 100 <= 80.6
    assert s["status"] == "IN_BAND"


def test_a_slower_pass_reads_below_band_and_a_faster_one_above_the_ceiling():
    pt = _pt()
    tgt = pt.compute_target({"weight_bytes": int(8 * _GB)}, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert pt.score(tgt, 40.0)["status"] == "BELOW_BAND"
    assert pt.score(tgt, 10.0)["status"] == "ABOVE_BAND"


def test_measured_checkpoint_bytes_beat_params_times_dtype():
    """A quantised or mixed-dtype checkpoint cannot be reconstructed from params x one dtype, so the
    producer measures the bytes and active_bytes must prefer them."""
    pt = _pt()
    mf = {"weight_bytes": int(8 * _GB), "total_params": 8_000_000_000, "dominant_dtype": "bfloat16"}
    assert pt.active_bytes(mf) == int(8 * _GB)  # not 16 GB
    assert pt.active_bytes({"total_params": 8_000_000_000, "dominant_dtype": "bfloat16"}) == int(16 * _GB)


def test_the_kv_term_is_added_only_when_a_seq_len_is_given():
    pt = _pt()
    mf = {"weight_bytes": int(8 * _GB), "layers": 32, "kv_heads": 8, "head_dim": 128, "kv_dtype": "bfloat16"}
    assert pt.active_bytes(mf) == int(8 * _GB)
    with_kv = pt.active_bytes(mf, seq_len=4096)
    assert with_kv > int(8 * _GB)
    assert with_kv - int(8 * _GB) == 2 * 32 * 8 * 128 * 4096 * 2


# --- the producer: derived for any dense LLM, no per-model wiring -----------------------------------


def _run_mod():
    spec = importlib.util.spec_from_file_location("run_ptin_ut", _ROOT / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_ptin_ut"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cfg(**kw):
    base = {
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_size": 4096,
        "torch_dtype": "bfloat16",
    }
    base.update(kw)
    return base


def test_the_producer_derives_facts_from_the_checkpoint_and_config(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: int(8 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    facts = run._perf_target_inputs(tmp_path, None, {})
    assert facts["weight_bytes"] == int(8 * _GB)
    assert facts["dominant_dtype"] == "bfloat16"
    assert (facts["layers"], facts["kv_heads"], facts["head_dim"]) == (32, 8, 128)


def test_head_dim_is_derived_when_the_config_omits_it(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: 1000)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg(hidden_size=8192, num_attention_heads=64))
    assert run._perf_target_inputs(tmp_path, None, {})["head_dim"] == 128


def test_an_explicit_head_dim_wins_over_the_derived_one(tmp_path, monkeypatch):
    """Phi-3.5-mini has head_dim 96 with hidden/heads = 128; deriving it would be wrong."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: 1000)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg(head_dim=96))
    assert run._perf_target_inputs(tmp_path, None, {})["head_dim"] == 96


@pytest.mark.parametrize("moe_key", ["num_local_experts", "num_experts", "n_routed_experts"])
def test_moe_models_are_refused_rather_than_guessed(tmp_path, monkeypatch, moe_key):
    """The reachable read set is shared + top_k x per-expert, and the split cannot come from config
    alone without guessing FFN shapes. A guessed ceiling is worse than the floor fallback."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: int(60 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "x/moe")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg(**{moe_key: 128, "num_experts_per_tok": 8}))
    assert run._perf_target_inputs(tmp_path, None, {}) is None


def test_no_checkpoint_and_no_config_produce_nothing(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: None)
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: {})
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: 0)
    assert run._perf_target_inputs(tmp_path, None, {}) is None
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: int(8 * _GB))
    assert run._perf_target_inputs(tmp_path, None, {}) is None


def test_the_manifest_config_overrides_the_hf_cache(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: 1000)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg(num_hidden_layers=32))
    facts = run._perf_target_inputs(tmp_path, None, {"model_config": {"num_hidden_layers": 16}})
    assert facts["layers"] == 16


def test_emit_writes_the_file_once_and_never_clobbers_a_tuned_one(tmp_path, monkeypatch, capsys):
    """A file already present may carry real per-tensor dtypes, which beats anything derivable here."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: int(8 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())

    run._emit_perf_target_inputs(tmp_path, tmp_path, None, {})
    out = tmp_path / "perf_target_inputs.json"
    assert out.exists() and json.loads(out.read_text())["weight_bytes"] == int(8 * _GB)

    out.write_text(json.dumps({"weight_tensors": [{"numel": 10, "dtype": "bfloat16"}]}))
    run._emit_perf_target_inputs(tmp_path, tmp_path, None, {})
    assert "weight_tensors" in json.loads(out.read_text())


def test_emit_never_raises_on_a_broken_model_root(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: int(8 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    run._emit_perf_target_inputs(tmp_path / "does" / "not" / "exist", tmp_path, None, {})


def test_end_to_end_the_produced_facts_give_the_published_ceiling(tmp_path, monkeypatch):
    """Producer -> perf_target: the file this writes must yield 64 tok/s/u and the 38-51 band."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: int(8 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    run._emit_perf_target_inputs(tmp_path, tmp_path, None, {})
    facts = json.loads((tmp_path / "perf_target_inputs.json").read_text())

    pt = _pt()
    tgt = pt.compute_target(facts, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert round(tgt.theoretical_rate, 1) == 64.0
    assert [round(b, 1) for b in tgt.band] == [38.4, 51.2]
    assert round(pt.score(tgt, 19.4)["measured_tok_s"], 1) == 51.5


def test_on_device_weight_bytes_can_be_stated_when_they_differ_from_the_checkpoint(tmp_path, monkeypatch):
    """The bound is peak_BW / bytes the DEVICE reads. Llama-3.1-8B is 16.06 GB on disk but is served
    with bf8_b weights, so 8 GB stream and the ceiling is 512/8 = 64 tok/s/u, not 31.9. A run that
    quantises weights must be able to say so rather than be judged against its checkpoint's dtype."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: 16_060_556_376)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())

    monkeypatch.delenv("TT_PERF_WEIGHT_BYTES", raising=False)
    from_disk = run._perf_target_inputs(tmp_path, None, {})
    pt = _pt()
    assert round(pt.compute_target(from_disk, {"dram_bw_gbps": _BH_DRAM_GBPS}).theoretical_rate, 1) == 31.9
    assert "checkpoint" in from_disk["source"]

    monkeypatch.setenv("TT_PERF_WEIGHT_BYTES", str(int(8 * _GB)))
    on_device = run._perf_target_inputs(tmp_path, None, {})
    assert on_device["weight_bytes"] == int(8 * _GB)
    assert "on-device" in on_device["source"]
    tgt = pt.compute_target(on_device, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert round(tgt.theoretical_rate, 1) == 64.0
    assert [round(b, 1) for b in tgt.band] == [38.4, 51.2]


def test_a_junk_override_is_ignored_rather_than_trusted(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: int(16 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    for junk in ("", "  ", "abc", "0", "-8", "8e9x"):
        monkeypatch.setenv("TT_PERF_WEIGHT_BYTES", junk)
        assert run._perf_target_inputs(tmp_path, None, {})["weight_bytes"] == int(16 * _GB), junk


# --- the reported unit must match the ceiling's unit ------------------------------------------------


def _sm():
    spec = importlib.util.spec_from_file_location("sm_phys_ut", _ROOT / "cc_optimize" / "summary.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sm_phys_ut"] = mod
    spec.loader.exec_module(mod)
    return mod


def _led_mod():
    spec = importlib.util.spec_from_file_location("led_phys_ut", _ROOT / "cc_optimize" / "measurements.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["led_phys_ut"] = mod
    spec.loader.exec_module(mod)
    return mod


def _snap():
    pt = _pt()
    tgt = pt.compute_target({"weight_bytes": int(8 * _GB)}, {"dram_bw_gbps": _BH_DRAM_GBPS})
    return {
        "has_unit_ceiling": True,
        "theoretical_rate": tgt.theoretical_rate,
        "band": [tgt.band[0], tgt.band[1]],
        "active_bytes": tgt.active_bytes,
        "peak_bw_gbps": _BH_DRAM_GBPS,
        "tp_degree": 1,
        "perf_layers": "all",
    }


def test_the_per_token_reading_is_used_not_the_per_profile_sum(tmp_path, monkeypatch):
    """THE DEFECT: the ceiling is per TOKEN, and the renderer was handed the headline per-profile
    device_ms. 1000/534 ms reads 1.9 tok/s/u against a 64 tok/s/u ceiling -- 3% utilisation for a
    model actually running at 84%."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), 534.44, {"per_token_ms": 18.68}, "m", "main"))
    assert "53.5 tok/s/u" in out, out
    assert "1.9 tok/s/u" not in out
    assert "428 GB/s" in out
    assert "84%" in out


def test_published_figures_render_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), 534.44, {"per_token_ms": 19.4}, "m", "main"))
    assert "theoretical ceiling : 64.0 tok/s/u" in out
    assert "achievable (60-80%) : 38.4 - 51.2 tok/s/u" in out
    assert "51.5 tok/s/u" in out and "412 GB/s" in out


def test_with_no_per_token_reading_the_line_says_so(tmp_path, monkeypatch):
    """Better to report nothing than a number of the wrong kind."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), None, {}, "m", "main"))
    assert "n/a" in out
    assert "1.9" not in out and "tok/s/u   (1000" not in out


def test_the_report_never_hands_a_per_profile_sum_to_a_per_token_ceiling(tmp_path, monkeypatch):
    """THE GUARD: for a per-token ceiling the per-profile device_ms is not a fallback, it is a wrong
    answer -- it rendered 1.9 tok/s/u and 3% utilisation for a model running at 84%."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    kl = tmp_path / "kl.json"
    kl.write_text(
        json.dumps([{"op_signature": "Matmul", "kernel_kind": "grid", "measured_ms": 534.44, "beat_baseline": True}])
    )
    out = sm.render_summary(kl, model="m", task="main", finalized=True, throughput=_snap(), baseline_profile={})
    assert "3%" not in out and "1.9 tok/s/u" not in out, out
    assert "measured            : n/a" in out, out


def test_rates_carry_the_profiling_depth_when_the_window_is_truncated(tmp_path, monkeypatch):
    """tok/s/u is an ABSOLUTE throughput, so a 16-layer window on a 32-layer model reads ~2x the real
    figure. The ratios (GB/s, utilisation) are depth-invariant and stay unqualified; the rates say
    which depth they describe so nobody quotes them as the model's throughput."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    snap = dict(_snap(), perf_layers="16")
    out = "\n".join(sm._roofline_lines(snap, None, {"per_token_ms": 9.34}, "m", "main"))
    assert "[16-layer window, NOT the full model]" in out, out
    assert out.count("[16-layer window") == 2, out  # ceiling AND measured
    assert "GB/s" in out and "utilization" in out


def test_a_full_depth_profile_needs_no_qualifier(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), None, {"per_token_ms": 19.4}, "m", "main"))
    assert "NOT the full model" not in out


def test_a_truncated_measurement_is_refused_against_a_full_model_ceiling(tmp_path, monkeypatch):
    """THE DEFECT: a 16-layer per-token reading against a 32-layer ceiling reported 107.1 tok/s/u for a
    model that does 43.9 -- the window streams a fraction of the bytes the ceiling assumes, so the
    ratio is meaningless, not merely optimistic. Withheld with the reason, never annotated and shown.
    """
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    snap = dict(_snap(), perf_layers="all")
    out = "\n".join(sm._roofline_lines(snap, None, None, "m", "main", per_token_ms=9.34, measured_depth="16"))
    assert "107.1" not in out and "357 GB/s" not in out, out
    assert "measured            : n/a" in out and "16-layer window" in out and "full depth" in out


def test_matching_depths_are_reported_normally(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    snap = dict(_snap(), perf_layers="all")
    out = "\n".join(sm._roofline_lines(snap, None, None, "m", "main", per_token_ms=19.4, measured_depth="all"))
    assert "51.5 tok/s/u" in out and "412 GB/s" in out


def test_an_unknown_depth_on_either_side_does_not_block_the_report(tmp_path, monkeypatch):
    """Only a KNOWN disagreement is refused; missing depth information is not evidence of mismatch."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), None, None, "m", "main", per_token_ms=19.4, measured_depth=""))
    assert "51.5 tok/s/u" in out


# --- the anchored ceiling must work for EVERY unit, not just token ----------------------------------


def _unit_snap(unit, byts, theo):
    return {
        "scope": "model",
        "has_unit_ceiling": True,
        "theoretical_rate": theo,
        "band": [0.6 * theo, 0.8 * theo],
        "active_bytes": byts,
        "peak_bw_gbps": 512.0,
        "tp_degree": 1,
        "perf_layers": "all",
        "unit": unit,
    }


def test_the_anchor_is_read_under_the_models_own_unit(tmp_path, monkeypatch):
    """THE BUG: the producer anchors bytes under the model's unit (run.py: depth=facts["unit"]), and the
    reader hardcoded depth="token". For a diffusion or classifier model the lookup missed, the renderer
    silently fell back to the snapshot, and the anchor's whole purpose -- surviving the model directory
    being reverted mid-run -- applied to LLMs only. It passed unnoticed because the only model that has
    ever had an anchor is per-token."""
    sm, led = _sm(), _led_mod()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))

    # a diffusion model: 2.76 GB per STEP anchored, snapshot deliberately holding a WRONG 9 GB
    led.anchor(led.KIND_ACTIVE_BYTES, 2760.0, depth="step", mode="bytes_mb", source="test", model="m")
    txt = "\n".join(
        sm._roofline_lines(_unit_snap("step", 9_000_000_000, 56.9), None, {"per_token_ms": 30.0}, "m", "main")
    )
    # 512 / 2.76 = 185.5 steps/s -- from the ANCHOR, not the snapshot's 9 GB
    assert "185.5 steps/s" in txt, txt
    assert "56.9" not in txt, txt


def test_a_token_model_still_reads_its_anchor(tmp_path, monkeypatch):
    sm, led = _sm(), _led_mod()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    led.anchor(led.KIND_ACTIVE_BYTES, 6094.651392, depth="token", mode="bytes_mb", source="test", model="m")
    txt = "\n".join(
        sm._roofline_lines(_unit_snap("token", 3_330_000_000, 153.8), None, {"per_token_ms": 17.0}, "m", "main")
    )
    assert "84.0 tok/s/u" in txt, txt
    assert "153.8" not in txt, txt
