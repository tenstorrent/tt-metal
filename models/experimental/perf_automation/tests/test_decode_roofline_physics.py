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
    assert round(tgt.theoretical_tok_s, 1) == 64.0


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
    assert round(tgt.theoretical_tok_s, 1) == 64.0
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
    assert round(pt.compute_target(from_disk, {"dram_bw_gbps": _BH_DRAM_GBPS}).theoretical_tok_s, 1) == 31.9
    assert "checkpoint" in from_disk["source"]

    monkeypatch.setenv("TT_PERF_WEIGHT_BYTES", str(int(8 * _GB)))
    on_device = run._perf_target_inputs(tmp_path, None, {})
    assert on_device["weight_bytes"] == int(8 * _GB)
    assert "on-device" in on_device["source"]
    tgt = pt.compute_target(on_device, {"dram_bw_gbps": _BH_DRAM_GBPS})
    assert round(tgt.theoretical_tok_s, 1) == 64.0
    assert [round(b, 1) for b in tgt.band] == [38.4, 51.2]


def test_a_junk_override_is_ignored_rather_than_trusted(tmp_path, monkeypatch):
    run = _run_mod()
    monkeypatch.setattr(run, "_model_weight_bytes", lambda d, h=None: int(16 * _GB))
    monkeypatch.setattr(run, "_resolve_model_id", lambda d, h=None: "x/y")
    monkeypatch.setattr(run, "_hf_cache_dims", lambda mid: _cfg())
    for junk in ("", "  ", "abc", "0", "-8", "8e9x"):
        monkeypatch.setenv("TT_PERF_WEIGHT_BYTES", junk)
        assert run._perf_target_inputs(tmp_path, None, {})["weight_bytes"] == int(16 * _GB), junk
