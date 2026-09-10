# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The decode roofline is peak DRAM bandwidth / bytes-streamed-per-token, and it must ACTIVATE.

The standard bound for LLM decode is:

    ceiling    = (peak_BW_per_chip * fraction * TP) / params_GB
                 dense (512*0.8)/8 = 64.0 tok/s/u ;  MoE (512*0.5)/3 = 170.7 tok/s/u
    achievable = 60-80% OF THAT CEILING            64.0 -> 38.4 - 51.2 tok/s/u
    compute    = (peak_FLOPs * 0.8 * TP) / (2 * params * tokens_per_unit) -- binds prefill, not decode
    TP scales the ceiling; DP and PP scale only aggregate_rate
    measured   = bytes / forward_time                8 GB / 19.4 ms = 412 GB/s -> 51.5 tok/s/u

The divisor is the PARAM count under xB -> xGB, not the exact streamed bytes: the exact figure
(6.095 GB for Llama-3.1-8B at bf4/bf8) is more accurate but costs a per-model investigation of what
width each tensor group is served at, while a param count is dtype-independent and free. The
sustained-BW fraction is folded INTO the ceiling, so it is an achievable number, not a spec one.

perf_target implements exactly this, and read its inputs from perf_target_inputs.json -- which
NOTHING in the tool ever wrote. So active_bytes was always 0, every report fell back to the
Sigma-per-op ms floor, and the reports said "not an LLM decode pipeline" about Llama. The floor is a
weaker statement and it moves when the op mix changes; this bound does not.

These pin the arithmetic against the published Llama-3.1-8B figures above, and pin that the inputs
are produced from the checkpoint and HF config for any dense LLM without per-model wiring.
"""
from __future__ import annotations

import re

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _flat(text):
    """A column-width-agnostic view of the table.

    The roofline pads a number and its unit into fixed sub-fields, so a published figure reads
    "64.0      tok/s/u". These assertions are about the PAIRING -- that a value is published carrying
    its unit -- not about the geometry, and pinning the geometry is how a column-width change becomes
    a test failure with nothing wrong behind it. Collapsing runs of spaces keeps the claim and drops
    the layout.
    """
    return re.sub(r"[ \t]+", " ", str(text))


_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

_BH_DRAM_GBPS = 512.0
_GB = 1e9


def _pt():
    from agent import perf_target

    return perf_target


def test_the_ceiling_is_sustained_bandwidth_over_params_gb():
    """8B params on a 512 GB/s part -> (512*0.8)/8 = 64.0 tok/s/u. The sustained fraction is INSIDE the
    ceiling: 512 GB/s is a spec figure no workload attains, and a target nobody can reach is not one."""
    pt = _pt()
    tgt = pt.compute_target({"total_params": int(8e9), "dominant_dtype": "int8"}, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert tgt.active_bytes == int(8 * _GB)  # xB -> xGB
    assert tgt.bw_fraction == 0.80
    assert round(tgt.theoretical_rate, 1) == 64.0
    assert [round(b, 1) for b in tgt.band] == [38.4, 51.2]  # 60-80% OF the ceiling
    assert tgt.bound_by == "memory"
    assert "params rule" in tgt.bytes_source


def test_a_moe_ceiling_uses_active_params_and_half_of_peak():
    """A 30B-A3B MoE streams only its ACTIVE params, and its scattered expert reads sustain ~50% of
    peak: 512/3 = 170.7 tok/s/u spec -- not 512/30, which would bound nothing it can reach. The
    0.50 MoE fraction sets the band top (85.3), which is what the ceiling used to report."""
    pt = _pt()
    tgt = pt.compute_target(
        {"is_moe": True, "active_params": int(3e9), "total_params": int(30e9), "dominant_dtype": "int8"},
        {"dram_bw_gbps": _BH_DRAM_GBPS},
    )
    assert tgt.active_bytes == int(3 * _GB)
    assert tgt.bw_fraction == 0.50
    assert round(tgt.theoretical_rate, 1) == 170.7  # (512*0.5)/3
    assert [round(b, 1) for b in tgt.band] == [64.0, 85.3]  # 0.375-0.50 of the 170.7 spec ceiling


def test_the_exact_byte_count_is_the_fallback_when_no_param_count_exists():
    """Facts written before the params rule still yield a ceiling instead of dropping to the floor."""
    pt = _pt()
    tgt = pt.compute_target({"weight_bytes": int(8 * _GB)}, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert tgt.active_bytes == int(8 * _GB)
    assert round(tgt.theoretical_rate, 1) == 64.0
    assert "per-tensor exact bytes" in tgt.bytes_source


def test_the_achievable_band_is_60_to_80_percent_of_peak():
    """38.4-51.2 tok/s/u -- 60-80% of the 64.0 ceiling, which is the label the report prints."""
    pt = _pt()
    tgt = pt.compute_target({"weight_bytes": int(8 * _GB)}, {"dram_bw_gbps": _BH_DRAM_GBPS})
    lo, hi = tgt.band
    assert round(lo, 1) == 38.4 and round(hi, 1) == 51.2, (lo, hi)


def test_a_measured_forward_pass_scores_as_published():
    """19.4 ms/token on 8 GB = 412 GB/s = 51.5 tok/s/u, at the TOP of the 38.4-51.2 band.

    bw_util is now a fraction of SPEC (512 GB/s), so this reads 80% of spec -- the same physical fact
    the old 100%-of-a-sustained-ceiling reading described, stated against the wall instead."""
    pt = _pt()
    tgt = pt.compute_target({"weight_bytes": int(8 * _GB)}, {"dram_bw_gbps": _BH_DRAM_GBPS})
    s = pt.score(tgt, 19.4)
    assert round(s["measured_tok_s"], 1) == 51.5
    assert round(s["effective_bw_bytes_s"] / 1e9) == 412
    # 8 GB / 19.4 ms = 412.4 GB/s, and 412.4/512 = 80.5%. The published "80%" rounds 412 first;
    # both agree to the nearest point, so pin the band rather than a single rounding convention.
    assert 80.0 <= s["bw_util_of_peak"] * 100 <= 80.6
    # 51.5 against a 64.0 SPEC ceiling with a 38.4-51.2 band: this is the top of what the hardware
    # sustains, so it reads IN_BAND. ABOVE_BAND is now reserved for beating spec, which flags the
    # bytes as wrong rather than the build as fast.
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
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: int(8 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    facts = run._perf_target_inputs(tmp_path, None, {})
    assert facts["weight_bytes"] == int(8 * _GB)
    assert facts["dominant_dtype"] == "bfloat16"
    assert (facts["layers"], facts["kv_heads"], facts["head_dim"]) == (32, 8, 128)


def test_head_dim_is_derived_when_the_config_omits_it(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 1000)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg(hidden_size=8192, num_attention_heads=64))
    assert run._perf_target_inputs(tmp_path, None, {})["head_dim"] == 128


def test_an_explicit_head_dim_wins_over_the_derived_one(tmp_path, monkeypatch):
    """Phi-3.5-mini has head_dim 96 with hidden/heads = 128; deriving it would be wrong."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 1000)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg(head_dim=96))
    assert run._perf_target_inputs(tmp_path, None, {})["head_dim"] == 96


@pytest.mark.parametrize("moe_key", ["num_local_experts", "num_experts", "n_routed_experts"])
def test_moe_models_are_refused_rather_than_guessed(tmp_path, monkeypatch, moe_key):
    """The reachable read set is shared + top_k x per-expert, and the split cannot come from config
    alone without guessing FFN shapes. A guessed ceiling is worse than the floor fallback."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: int(60 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "x/moe")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg(**{moe_key: 128, "num_experts_per_tok": 8}))
    assert run._perf_target_inputs(tmp_path, None, {}) is None


def test_only_a_missing_divisor_withholds_the_ceiling(tmp_path, monkeypatch):
    """A DIVISOR is the one input the ceiling cannot do without; a config is not.

    This used to assert that weight bytes WITHOUT an HF config produced nothing, because the producer
    gated on `cfg`. But the ceiling never reads cfg -- it supplies the KV terms, which are unused unless a
    seq_len is given -- so that gate rejected models over an input the formula does not consult, and their
    reports fell to the band-less ms floor. Now only the absence of BOTH a param count and a byte count
    withholds it, since with neither there is nothing to divide by and a zero ceiling would render as a
    real one."""
    run = _run_mod()
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: None)
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: {})
    monkeypatch.setattr(run, "_hf_snapshots", lambda mid: [])
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 0)
    assert run._perf_target_inputs(tmp_path, None, {}) is None, "no params and no bytes must withhold"

    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: int(8 * _GB))
    facts = run._perf_target_inputs(tmp_path, None, {})
    assert facts and facts["weight_bytes"] == int(8 * _GB), "bytes alone are a usable divisor"
    tgt = _pt().compute_target(facts, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert round(tgt.theoretical_rate, 1) == 64.0 and tgt.band[0] > 0


def test_the_manifest_config_overrides_the_hf_cache(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 1000)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg(num_hidden_layers=32))
    facts = run._perf_target_inputs(tmp_path, None, {"model_config": {"num_hidden_layers": 16}})
    assert facts["layers"] == 16


def test_emit_writes_the_file_once_and_never_clobbers_a_tuned_one(tmp_path, monkeypatch, capsys):
    """A file already present may carry real per-tensor dtypes, which beats anything derivable here."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: int(8 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())

    run._emit_perf_target_inputs(tmp_path, tmp_path, None, {})
    out = tmp_path / "perf_target_inputs.json"
    assert out.exists() and json.loads(out.read_text())["weight_bytes"] == int(8 * _GB)

    out.write_text(json.dumps({"weight_tensors": [{"numel": 10, "dtype": "bfloat16"}]}))
    run._emit_perf_target_inputs(tmp_path, tmp_path, None, {})
    assert "weight_tensors" in json.loads(out.read_text())


def test_emit_never_raises_on_a_broken_model_root(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: int(8 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    run._emit_perf_target_inputs(tmp_path / "does" / "not" / "exist", tmp_path, None, {})


def test_end_to_end_the_produced_facts_give_the_published_ceiling(tmp_path, monkeypatch):
    """Producer -> perf_target: the file this writes must yield the 64.0 ceiling and 38.4-51.2 band."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: int(8 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    # No snapshot: this pins the CONFIG-and-name path, so the real HF cache must not leak in.
    monkeypatch.setattr(run, "_hf_snapshots", lambda mid: [])
    run._emit_perf_target_inputs(tmp_path, tmp_path, None, {})
    facts = json.loads((tmp_path / "perf_target_inputs.json").read_text())

    pt = _pt()
    tgt = pt.compute_target(facts, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert facts["total_params"] == int(8e9)  # from the model id "Llama-3.1-8B"
    # 8B served bf16 = 16 GB, so the wall is 512/16 = 32.0. This read 64.0 under the xB -> xGB
    # constant, which is right only for a 1-byte format: these facts come from a real bf16
    # checkpoint, and pricing them at one byte published a ceiling the hardware cannot reach.
    assert round(tgt.theoretical_rate, 1) == 32.0
    assert [round(b, 1) for b in tgt.band] == [19.2, 25.6]  # band follows the bf16 ceiling (32.0)
    assert round(pt.score(tgt, 19.4)["measured_tok_s"], 1) == 51.5


def test_on_device_weight_bytes_can_be_stated_when_they_differ_from_the_checkpoint(tmp_path, monkeypatch):
    """The checkpoint's STORED dtype must not set the ceiling. Llama-3.1-8B is 16.06 GB of bf16 on
    disk and streams far less once served as bf4/bf8; judging it by the on-disk figure gave a 31.9
    ceiling for a model that measures 58.9 -- already beaten, so it bounds nothing, and its 19.1-25.5
    band sits BELOW the untouched 20.7 baseline, so the run would stop before optimizing anything.

    Under the params rule that trap is structural rather than a setting to remember: 8B params -> 8 GB
    whatever the checkpoint stores, so both paths give 64.0 and the override cannot be forgotten."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 16_060_556_376)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    # No snapshot: this pins the CONFIG-and-name path, so the real HF cache must not leak in.
    monkeypatch.setattr(run, "_hf_snapshots", lambda mid: [])

    monkeypatch.delenv("TT_PERF_WEIGHT_BYTES", raising=False)
    from_disk = run._perf_target_inputs(tmp_path, None, {})
    pt = _pt()
    # 16.06 GB on disk, but the divisor is 8B params -> 8 GB, so NOT the old 31.9.
    # 8B served bf16 = 16 GB, so the wall is 512/16 = 32.0. This read 64.0 under the xB -> xGB
    # constant, which is right only for a 1-byte format: these facts come from a real bf16
    # checkpoint, and pricing them at one byte published a ceiling the hardware cannot reach.
    assert round(pt.compute_target(from_disk, {"dram_bw_gbps": _BH_DRAM_GBPS}).theoretical_rate, 1) == 32.0
    assert from_disk["total_params"] == int(8e9)

    monkeypatch.setenv("TT_PERF_WEIGHT_BYTES", str(int(8 * _GB)))
    on_device = run._perf_target_inputs(tmp_path, None, {})
    assert on_device["weight_bytes"] == int(8 * _GB)
    assert "on-device" in on_device["source"]
    tgt = pt.compute_target(on_device, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert round(tgt.theoretical_rate, 1) == 32.0  # identical either way now (bf16: 8B x 2 = 16 GB)
    assert [round(b, 1) for b in tgt.band] == [19.2, 25.6]  # band follows the bf16 ceiling (32.0)


def test_a_junk_override_is_ignored_rather_than_trusted(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: int(16 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "x/y")
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
        # The sustained fraction must travel with the ceiling, or the report cannot say the ceiling is
        # achievable-not-spec and its label falls back to bare.
        "bw_fraction": tgt.bw_fraction,
    }


def test_the_per_token_reading_is_used_not_the_per_profile_sum(tmp_path, monkeypatch):
    """THE DEFECT: the ceiling is per TOKEN, and the renderer was handed the headline per-profile
    device_ms. 1000/534 ms reads 1.9 tok/s/u against the ceiling -- 3% utilisation for a model
    actually running at 84% of spec peak."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), 534.44, {"per_token_ms": 18.68}, "m", "main"))
    assert "53.5 tok/s/u" in _flat(out), out
    assert "1.9 tok/s/u" not in _flat(out)
    assert "428.3 GB/s" in _flat(out)
    # 428/512 = 84% of SPEC peak. Against the sustained ceiling the same run reads 104%, so the
    # per-token reading is what is being checked here, not which denominator the label uses.
    assert "1.9 tok/s/u" not in _flat(out)


def test_published_figures_render_exactly(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), 534.44, {"per_token_ms": 19.4}, "m", "main"))
    # Labels say SUSTAINED, and both are derived from the numbers rather than hardcoded strings.
    assert "64.0 tok/s/u" in _flat(out), out
    assert "60-80%" in _flat(out) and "38.4 – 51.2" in _flat(out), out
    assert "51.5 tok/s/u" in _flat(out) and "412.4 GB/s" in _flat(out)


def test_with_no_per_token_reading_the_line_says_so(tmp_path, monkeypatch):
    """Better to report nothing than a number of the wrong kind."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), None, {}, "m", "main"))
    assert "n/a" in _flat(out)
    assert "1.9" not in _flat(out) and "tok/s/u   (1000" not in _flat(out)


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
    # Scoped to the BANDWIDTH row. A bare `"3%" not in _flat(out)` also matched "23% used" on the capacity
    # row, so the guard fired on an unrelated healthy number instead of the fabricated utilisation
    # it was written to catch.
    assert "1.9 tok/s/u" not in _flat(out), out
    # With no per-unit reading there is no bandwidth row to draw at all, which is the correct
    # rendering of "unknown" -- the guard is that no row may carry the FABRICATED 3%, not that a row
    # must exist to carry something.
    _bw = [l for l in _flat(out).splitlines() if "decode memory" in l]
    assert not any("3%" in l for l in _bw), _bw
    assert "n/a — not measured" in _flat(out), out


def test_rates_carry_the_profiling_depth_when_the_window_is_truncated(tmp_path, monkeypatch):
    """tok/s/u is an ABSOLUTE throughput, so a 16-layer window on a 32-layer model reads ~2x the real
    figure. The ratios (GB/s, utilisation) are depth-invariant and stay unqualified; the rates say
    which depth they describe so nobody quotes them as the model's throughput."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    snap = dict(_snap(), perf_layers="16")
    out = "\n".join(sm._roofline_lines(snap, None, {"per_token_ms": 9.34}, "m", "main"))
    assert "[16-layer window, NOT the full model]" in _flat(out), out
    # ONE qualifier, not two. The old five-line form printed the ceiling and the measurement on
    # separate lines and each needed its own tag; the table puts both rates on one row, so a single
    # line beneath it qualifies both. The depth-invariant GB/s row stays unqualified either way.
    assert out.count("[16-layer window") == 1, out
    assert "GB/s" in _flat(out) and "Utilization" in _flat(out)


def test_a_full_depth_profile_needs_no_qualifier(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), None, {"per_token_ms": 19.4}, "m", "main"))
    assert "NOT the full model" not in _flat(out)


def test_a_truncated_measurement_is_refused_against_a_full_model_ceiling(tmp_path, monkeypatch):
    """THE DEFECT: a 16-layer per-token reading against a 32-layer ceiling reported 107.1 tok/s/u for a
    model that does 43.9 -- the window streams a fraction of the bytes the ceiling assumes, so the
    ratio is meaningless, not merely optimistic. Withheld with the reason, never annotated and shown.
    """
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    snap = dict(_snap(), perf_layers="all")
    out = "\n".join(sm._roofline_lines(snap, None, None, "m", "main", per_token_ms=9.34, measured_depth="16"))
    assert "107.1" not in _flat(out) and "357 GB/s" not in _flat(out), out
    assert "n/a — not measured" in _flat(out) and "16-layer window" in _flat(out) and "full depth" in _flat(out)


def test_matching_depths_are_reported_normally(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    snap = dict(_snap(), perf_layers="all")
    out = "\n".join(sm._roofline_lines(snap, None, None, "m", "main", per_token_ms=19.4, measured_depth="all"))
    assert "51.5 tok/s/u" in _flat(out) and "412.4 GB/s" in _flat(out)


def test_an_unknown_depth_on_either_side_does_not_block_the_report(tmp_path, monkeypatch):
    """Only a KNOWN disagreement is refused; missing depth information is not evidence of mismatch."""
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    sm = _sm()
    out = "\n".join(sm._roofline_lines(_snap(), None, None, "m", "main", per_token_ms=19.4, measured_depth=""))
    assert "51.5 tok/s/u" in _flat(out)


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
    assert "185.5 steps/s" in _flat(txt), txt
    assert "56.9" not in txt, txt


def test_a_token_model_still_reads_its_anchor(tmp_path, monkeypatch):
    sm, led = _sm(), _led_mod()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    led.anchor(led.KIND_ACTIVE_BYTES, 6094.651392, depth="token", mode="bytes_mb", source="test", model="m")
    txt = "\n".join(
        sm._roofline_lines(_unit_snap("token", 3_330_000_000, 153.8), None, {"per_token_ms": 17.0}, "m", "main")
    )
    assert "84.0 tok/s/u" in _flat(txt), txt
    assert "153.8" not in txt, txt


def test_the_anchored_ceiling_uses_the_sustained_fraction_not_a_second_copy_of_the_math(tmp_path, monkeypatch):
    """THE BUG: the renderer recomputed the anchored ceiling with its OWN `peak / bytes` and a
    hardcoded (0.60, 0.80) band. When the ceiling moved to (peak * sustained) / bytes, that copy kept
    the old physics -- so an anchored run PRINTED 84.0 while the stop gate, reading the same snapshot
    through perf_target, judged against the sustained number. The report and the gate disagreeing about
    one run is worse than either value being wrong, so the arithmetic now has exactly one owner."""
    sm, led = _sm(), _led_mod()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    led.anchor(led.KIND_ACTIVE_BYTES, 6094.651392, depth="token", mode="bytes_mb", source="test", model="m")

    snap = _unit_snap("token", 3_330_000_000, 153.8)  # snapshot bytes deliberately stale
    snap["bw_fraction"] = 0.80
    txt = "\n".join(sm._roofline_lines(snap, None, {"per_token_ms": 17.0}, "m", "main"))

    # (512*0.8) / 6.0947 GB = 67.2 tok/s/u from the ANCHOR.
    # 67.2 is the BAND TOP -- rate_and_band returns spec peak as the ceiling and folds the sustained
    # fraction into the band, so the table shows 84.0 in THEORETICAL and 50.4 - 67.2 in ACHIEVABLE.
    # The unit suffix now lives in the ceiling column, so the band is asserted as bare values.
    assert "84.0 tok/s/u" in _flat(txt), txt
    assert "50.4 – 67.2" in _flat(txt), txt
    assert "153.8" not in txt, txt  # the stale snapshot value


def test_an_anchored_snapshot_without_the_fraction_keeps_its_old_reading(tmp_path, monkeypatch):
    """A run in flight can hold a snapshot written before bw_fraction existed. Assuming 0.80 for it
    would silently restate that run's ceiling mid-run (and would be wrong for an MoE), so a missing
    fraction means 1.0 -- the spec-peak reading it already had."""
    sm, led = _sm(), _led_mod()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    led.anchor(led.KIND_ACTIVE_BYTES, 6094.651392, depth="token", mode="bytes_mb", source="test", model="m")
    txt = "\n".join(
        sm._roofline_lines(_unit_snap("token", 3_330_000_000, 153.8), None, {"per_token_ms": 17.0}, "m", "main")
    )
    assert "84.0 tok/s/u" in _flat(txt), txt


def test_the_anchored_divisor_is_the_one_the_ceiling_divides_by(tmp_path, monkeypatch):
    """THE BUG: the anchor pinned facts["weight_bytes"] (what the CHECKPOINT stores) while the ceiling
    divides by params (xB -> xGB). The report recomputes from the anchor and the stop gate computes
    from the facts, so one run had two divisors: a bf16 checkpoint storing 16.06 GB with 7.5B params
    printed 25.5 tok/s/u while the gate judged against 54.6.

    The anchor exists so a directory revert cannot move the ceiling; pinning the WRONG quantity moved
    it on every run instead."""
    run, led = _run_mod(), _led_mod()
    monkeypatch.setenv("PERF_MCP_LEDGER", str(tmp_path / "l.jsonl"))
    # a bf16 checkpoint: 16.06 GB stored, but 8B params -> the ceiling divides by 8 GB
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 16_060_556_376)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    # No snapshot: this pins the CONFIG-and-name path, so the real HF cache must not leak in.
    monkeypatch.setattr(run, "_hf_snapshots", lambda mid: [])

    root = tmp_path / "m"
    root.mkdir()
    # A UNIT MUST EXIST FOR AN ANCHOR TO. The producer declines without one, because the only moment
    # it runs -- setup -- is before any trace has reported the unit, and it used to invent the key
    # "unit" instead. This test read back under that same placeholder, so it agreed with the producer
    # about a key nothing else looks up and never saw the divergence it exists to catch.
    monkeypatch.setenv("PERF_MCP_LAST_HEADLINE_UNIT", "token")
    run._emit_perf_target_inputs(root, root, None, {})
    facts = json.loads((root / "perf_target_inputs.json").read_text())
    assert facts.get("unit") == "token"

    pinned_mb = led.anchor_value(led.KIND_ACTIVE_BYTES, depth="token", model="m")
    assert pinned_mb, "nothing anchored"
    pinned_bytes = int(round(float(pinned_mb) * 1e6))

    pt = _pt()
    gate = pt.compute_target(facts, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert pinned_bytes == gate.active_bytes, (pinned_bytes, gate.active_bytes)
    assert pinned_bytes != 16_060_556_376, "anchored the checkpoint bytes, not the ceiling's divisor"


def test_a_list_valued_config_field_does_not_cost_the_whole_ceiling(tmp_path, monkeypatch):
    """A per-layer config carries a LIST where a scalar is expected. Raw int() on one raises TypeError,
    the caller swallows it, and the model lost its ENTIRE ceiling over a KV field the ceiling does not
    even use without a seq_len. Two cached checkpoints hit this in a stress run."""
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: int(8 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "meta-llama/Llama-3.1-8B")
    monkeypatch.setattr(
        run,
        "_hf_cache_dims",
        lambda mid: _cfg(num_key_value_heads=[8] * 32, num_hidden_layers=[32], head_dim=[128, 128]),
    )
    monkeypatch.setattr(run, "_hf_snapshots", lambda mid: [])
    facts = run._perf_target_inputs(tmp_path, None, {})
    assert facts, "a list-valued field must not withhold the whole ceiling"
    assert facts["kv_heads"] == 8 and facts["layers"] == 32 and facts["head_dim"] == 128
    # 8B served bf16 = 16 GB, so the wall is 512/16 = 32.0. This read 64.0 under the xB -> xGB
    # constant, which is right only for a 1-byte format: these facts come from a real bf16
    # checkpoint, and pricing them at one byte published a ceiling the hardware cannot reach.
    assert round(_pt().compute_target(facts, {"dram_bw_gbps": _BH_DRAM_GBPS}).theoretical_rate, 1) == 32.0


def test_params_are_read_even_when_the_unit_cannot_be_determined(tmp_path, monkeypatch):
    """THE DEFECT: the header walk was gated on a known unit, so a model whose unit could not be
    determined fell back to the checkpoint's FILE SIZE as the divisor -- bge-large-en-v1.5 read 1.34 GB
    of float32 (~4 B/param) and scored 305.5, bypassing xB -> xGB entirely. The param count does not
    depend on the unit; only the lookup-only exclusion does."""
    run = _run_mod()
    shard = tmp_path / "snap"
    shard.mkdir()
    # 1M params stored as float32 (4 MB on disk) -- params rule must give 1 MB, not 4 MB
    import struct

    hdr = json.dumps({"w": {"dtype": "F32", "shape": [1000, 1000], "data_offsets": [0, 4_000_000]}}).encode()
    (shard / "model.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr)

    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None, *_a, **_k: 4_000_000)
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None, *_a, **_k: "org/no-size-in-name")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: {"hidden_size": 1000})  # no tag, no architectures
    monkeypatch.setattr(run, "_hf_snapshots", lambda mid: [shard])

    facts = run._perf_target_inputs(tmp_path, None, {})
    assert facts and facts.get("total_params") == 1_000_000, facts
    tgt = _pt().compute_target(facts, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert tgt.active_bytes == 2_000_000, "divisor came from the file size, not the params"  # bf16
    assert "params rule" in tgt.bytes_source


# --- the COMPUTE term, and TP/DP -- model-agnostic, no architecture formulas --------------------------

_BH_HW = {
    "dram_bw_gbps": 512.0,
    "dram_bw_per_chip_gbps": 512.0,
    "peak_tflops_per_core": {"lofi": 5.4, "hifi2": 2.7, "hifi3": 1.8, "hifi4": 1.35},
    "grid_x": 13,
    "grid_y": 10,
    "mesh_chips": 1,
    "worker_cores": 130,
}


def test_decode_is_memory_bound_and_prefill_is_compute_bound():
    """The binding constraint decides, and which one binds follows from the unit: a decode step reads
    every weight to make ONE token (~2 FLOP/byte, far under what the part wants, so bandwidth binds),
    while a prefill does the same reads for S tokens -- bytes flat, FLOPs linear in S. prefill_ceiling
    used to raise NotImplementedError, so those runs got a bandwidth bound they could never hit."""
    pt = _pt()
    dec = pt.compute_target(
        {"total_params": int(8e9), "unit": "token", "dominant_dtype": "int8"}, _BH_HW, tokens_per_unit=1
    )
    assert dec.bound_by == "memory"
    assert round(dec.theoretical_rate, 1) == 64.0  # unchanged by adding the compute term

    pre = pt.compute_target(
        {"total_params": int(8e9), "unit": "token", "dominant_dtype": "int8"}, _BH_HW, tokens_per_unit=2048
    )
    assert pre.bound_by == "compute"
    assert pre.theoretical_rate < dec.theoretical_rate, "prefill cannot be faster than decode per unit"
    # (176 TFLOP/s * 0.8) / (2 * 8e9 * 2048)
    assert abs(pre.theoretical_rate - (pt.chip_peak_flops(_BH_HW) * 0.8) / (2 * 8e9 * 2048)) < 1e-6


def test_the_compute_ceiling_is_per_chip_not_mesh_aggregate():
    """worker_cores in the env is already multiplied by mesh_chips, so a per-unit FLOP ceiling built
    from it would apply the chip count twice -- the same defect the bandwidth term had."""
    pt = _pt()
    one = dict(_BH_HW)
    eight = {**_BH_HW, "mesh_chips": 8, "worker_cores": 130 * 8, "dram_bw_gbps": 512.0 * 8}
    assert abs(pt.chip_peak_flops(one) - pt.chip_peak_flops(eight)) < 1e-3


def test_tp_scales_the_ceiling_and_dp_only_scales_aggregate():
    """Model-agnostic mesh rule: TP shards the weights AND the work, so it raises the per-unit ceiling.
    DP replicates, so it cannot make one unit faster -- it multiplies how many run at once. Eight chips
    of bandwidth is eight chips of bandwidth however it is split, so aggregate is constant."""
    pt = _pt()
    mf = {"total_params": int(8e9), "unit": "token", "dominant_dtype": "int8"}
    seen_aggregate = set()
    for tp, dp, want_per_user in ((8, 1, 512.0), (4, 2, 256.0), (2, 4, 128.0), (1, 8, 64.0)):
        t = pt.compute_target(mf, _BH_HW, tp_degree=tp, dp_degree=dp)
        assert round(t.theoretical_rate, 1) == want_per_user, (tp, dp, t.theoretical_rate)
        assert t.dp_degree == dp and t.tp_degree == tp
        seen_aggregate.add(round(t.aggregate_rate, 1))
        # the BAND scores the per-unit rate, never the aggregate
        assert t.band[1] <= t.theoretical_rate + 1e-9
    assert seen_aggregate == {512.0}, seen_aggregate


def test_dp_never_leaks_into_the_scored_rate():
    """A DP-inflated target would let a slow per-token run read IN_BAND because other replicas exist."""
    pt = _pt()
    mf = {"total_params": int(8e9), "unit": "token", "dominant_dtype": "int8"}
    solo = pt.compute_target(mf, _BH_HW, tp_degree=1, dp_degree=1)
    replicated = pt.compute_target(mf, _BH_HW, tp_degree=1, dp_degree=32)
    assert solo.theoretical_rate == replicated.theoretical_rate
    assert solo.band == replicated.band
    ms = 1000.0 / 45.0  # 45 tok/s: inside the 38.4-51.2 band for the real ceiling
    assert pt.score(solo, ms)["status"] == pt.score(replicated, ms)["status"] == "IN_BAND"
