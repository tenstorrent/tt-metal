# SPDX-License-Identifier: Apache-2.0
"""Two-layer routing fix for the auto-sync slug!=model_type miss (e.g. Qwen2.5-VL):

  1. Deterministic matcher normalizes separators so HF ``qwen2_5_vl`` exact-matches
     the synced path-slug key ``qwen25_vl`` (free, 100% reliable) -- and never
     collides distinct models.
  2. ``resolve_backend_with_quality`` short-circuits on that exact match (no LLM),
     and otherwise defers the family choice to the LLM-first ranker, degrading to
     deterministic when the LLM is gated off / unavailable / unconfident.
  3. The sync derivation classifies a ``*_vl`` / llava / paligemma demo as VLM with
     the image-text-to-text tag (not an impoverished LLM entry with no tags).
"""
import json

import scripts.tt_hw_planner.family_backends as fb
import scripts.tt_hw_planner.sibling_ranker as sr
from scripts.tt_hw_planner.registry_sync import _derive_demo_families


def _fake_backend(name, cat="VLM", keys=None):
    return fb.FamilyBackend(
        category=cat,
        name=name,
        demo_path="x",
        routing_mode="template",
        canonical_hf_id=None,
        model_type_keys=keys or [],
    )


# ---- layer 1: normalization ------------------------------------------------
def test_norm_collapses_separator_and_grouping_variants():
    assert fb._norm_mt("qwen2_5_vl") == fb._norm_mt("qwen25_vl") == "qwen25vl"
    assert fb._norm_mt("qwen2-vl") == fb._norm_mt("qwen2_vl") == "qwen2vl"
    # distinct models must NOT collapse together
    assert fb._norm_mt("qwen2_5_vl") != fb._norm_mt("qwen2_vl")


def test_hf_model_type_exact_matches_synced_slug(monkeypatch):
    fake = _fake_backend("qwen25_vl (test)", keys=["qwen25_vl"])
    monkeypatch.setattr(fb, "backends_for_category", lambda category: [fake] if category == "VLM" else [])
    monkeypatch.setattr(fb, "all_backends", lambda: [fake])
    b, q = fb.pick_backend_with_quality(category="VLM", model_type="qwen2_5_vl")
    assert q == "exact" and b is fake


def test_no_normalization_collisions_in_real_registry():
    from collections import defaultdict

    seen = defaultdict(set)
    for b in fb.all_backends():
        for k in b.model_type_keys:
            seen[fb._norm_mt(k)].add(k.lower())
    collisions = {n: ks for n, ks in seen.items() if len(ks) > 1}
    assert not collisions, collisions


# ---- layer 2: LLM-first resolver ------------------------------------------
def test_exact_short_circuits_without_llm(monkeypatch):
    sr._RESOLVE_CACHE.clear()
    exact_b = _fake_backend("exact-be")
    monkeypatch.setattr(fb, "pick_backend_with_quality", lambda **k: (exact_b, "exact"))

    def boom(**k):
        raise AssertionError("LLM must not run on an exact match")

    monkeypatch.setattr(sr, "rank_siblings", boom)
    b, q = sr.resolve_backend_with_quality(model_id="m", category="VLM", model_type="x")
    assert q == "exact" and b is exact_b


def test_non_exact_defers_to_llm(monkeypatch):
    sr._RESOLVE_CACHE.clear()
    det_b, llm_b = _fake_backend("mistral"), _fake_backend("qwen25")
    monkeypatch.setattr(fb, "pick_backend_with_quality", lambda **k: (det_b, "pipeline"))
    monkeypatch.setattr(sr, "rank_siblings", lambda **k: [(llm_b, 100, "LLM: exact same model")])
    b, q = sr.resolve_backend_with_quality(model_id="m", category="VLM", model_type="x")
    assert q == "llm" and b is llm_b


def test_gate_off_forces_deterministic(monkeypatch):
    sr._RESOLVE_CACHE.clear()
    det_b = _fake_backend("mistral")
    monkeypatch.setenv("TT_HW_PLANNER_LLM_ROUTE", "0")
    monkeypatch.setattr(fb, "pick_backend_with_quality", lambda **k: (det_b, "pipeline"))

    def boom(**k):
        raise AssertionError("LLM must not run when gated off")

    monkeypatch.setattr(sr, "rank_siblings", boom)
    b, q = sr.resolve_backend_with_quality(model_id="m", category="VLM", model_type="x")
    assert q == "pipeline" and b is det_b


def test_llm_unavailable_or_unconfident_degrades(monkeypatch):
    sr._RESOLVE_CACHE.clear()
    det_b, weak = _fake_backend("mistral"), _fake_backend("weak")
    monkeypatch.setattr(fb, "pick_backend_with_quality", lambda **k: (det_b, "pipeline"))
    # empty (unavailable)
    monkeypatch.setattr(sr, "rank_siblings", lambda **k: [])
    assert sr.resolve_backend_with_quality(model_id="m1", category="VLM", model_type="x") == (det_b, "pipeline")
    # below strong-score threshold (unconfident)
    sr._RESOLVE_CACHE.clear()
    monkeypatch.setattr(sr, "rank_siblings", lambda **k: [(weak, 30, "LLM: not sure")])
    assert sr.resolve_backend_with_quality(model_id="m2", category="VLM", model_type="x") == (det_b, "pipeline")


# ---- layer 3: sync derivation metadata ------------------------------------
def test_vl_demos_derive_as_vlm_with_tag():
    fams = {f["name"]: f for f in _derive_demo_families(["qwen25_vl", "qwen3_vl", "llava_next", "resnet50", "whisper"])}
    for n in ("qwen25_vl (auto-upstream)", "qwen3_vl (auto-upstream)", "llava_next (auto-upstream)"):
        assert fams[n]["category"] == "VLM", fams[n]
        assert fams[n]["pipeline_tags"] == ["image-text-to-text"], fams[n]
    # no false positives for non-VLM families
    assert fams["resnet50 (auto-upstream)"]["category"] == "CNN"
    assert fams["whisper (auto-upstream)"]["category"] == "STT"


def test_sibling_voting_runs_n_asks_and_caches(monkeypatch):
    import threading

    sr._SIBLING_CACHE.clear()
    monkeypatch.setenv("TT_HW_PLANNER_SIBLING_VOTES", "3")
    A, B = _fake_backend("A"), _fake_backend("B")
    calls = []
    lock = threading.Lock()

    def fake(**kw):
        with lock:
            calls.append(1)
        return [(A, 90, "a"), (B, 40, "b")]

    monkeypatch.setattr(sr, "rank_backends_llm", fake)
    monkeypatch.setattr(sr, "rank_backends", lambda **k: [])
    r = sr.rank_siblings(model_id="m", category="X", model_type="mt", top_n=2)
    assert r[0][0].name == "A", r
    assert len(calls) == 3, "should vote 3x"
    # cached: a second call must not re-vote
    sr.rank_siblings(model_id="m", category="X", model_type="mt", top_n=2)
    assert len(calls) == 3, "second call should hit the cache"


def test_sibling_votes_one_is_single_ask(monkeypatch):
    sr._SIBLING_CACHE.clear()
    monkeypatch.setenv("TT_HW_PLANNER_SIBLING_VOTES", "1")
    A = _fake_backend("A")
    calls = []
    monkeypatch.setattr(sr, "rank_backends_llm", lambda **k: (calls.append(1), [(A, 90, "a")])[1])
    monkeypatch.setattr(sr, "rank_backends", lambda **k: [])
    sr.rank_siblings(model_id="m2", category="X", model_type="mt")
    assert len(calls) == 1


def test_arch_fingerprint_backbone_families():
    from scripts.tt_hw_planner.fingerprint import arch_descriptor

    # structural backbone, derived the same way for target + backends
    assert arch_descriptor(model_type="hunyuan", architectures=["HunyuanImage3ForCausalMM"]).startswith(
        "decoder-only causal LM"
    )
    assert arch_descriptor(model_type="stable_diffusion") == "diffusion UNet+VAE"
    assert arch_descriptor(model_type="acestep").startswith("DiT")
    assert arch_descriptor(model_type="speecht5").startswith("encoder-decoder transformer")
    assert arch_descriptor(model_type="whisper").startswith("encoder-decoder transformer")
    assert arch_descriptor(model_type="qwen2_5_vl").startswith("VLM")
    # a causal-LM that emits images must NOT read as a diffusion backbone
    d = arch_descriptor(model_type="hunyuan", architectures=["HunyuanImage3ForCausalMM"])
    assert "diffusion" not in d and "decoder-only" in d
    # unknown model_type falls back to architectures-class inference
    assert arch_descriptor(model_type="totallynovel", architectures=["FooForCausalLM"]).startswith("decoder-only")


def test_derivation_reads_real_model_type_from_config(tmp_path):
    # a fetched tree that ships the real HF config under a synced model_params dir
    cfg_dir = tmp_path / "models" / "tt_transformers" / "model_params" / "Qwen2.5-VL-7B-Instruct"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen2_5_vl", "architectures": ["Qwen2_5_VLForConditionalGeneration"]})
    )
    fams = {f["name"]: f for f in _derive_demo_families(["qwen25_vl"], tmp_path)}
    got = fams["qwen25_vl (auto-upstream)"]
    # key is the TRUE HF model_type read from config, not the folder slug
    assert got["model_type_keys"] == ["qwen2_5_vl"], got
    assert got["category"] == "VLM" and got["pipeline_tags"] == ["image-text-to-text"], got
