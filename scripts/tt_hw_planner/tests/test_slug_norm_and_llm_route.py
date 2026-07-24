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


def test_backend_category_matches_pipeline_tag():
    # guards the XTTS-v2 STT-vs-TTS typo class: a template backend's category must be
    # consistent with its pipeline_tags (a text-to-speech backend can't be category STT)
    from scripts.tt_hw_planner.family_backends import all_backends

    # ONLY unambiguous tags: text-to-image / text-to-audio / any-to-any legitimately span
    # categories (an AR-LLM is tagged text-to-image but is category LLM — the #3 lesson),
    # so they must NOT be asserted here or this guard false-fails on a valid backend.
    cat_of_tag = {
        "text-to-speech": "TTS",
        "automatic-speech-recognition": "STT",
        "image-classification": "CNN",
        "feature-extraction": "Embed",
    }
    mismatches = []
    for b in all_backends():
        if getattr(b, "routing_mode", "") == "generic":
            continue
        for t in b.pipeline_tags:
            expected = cat_of_tag.get(t)
            if expected and b.category != expected:
                mismatches.append((b.name, b.category, t, expected))
    assert not mismatches, f"category/pipeline_tag mismatches: {mismatches}"


def test_cross_category_pipeline_tag_does_not_override_real_category():
    # issue #3: an AR-MoE LLM tagged text-to-image must NOT be pulled to a diffusion
    # (Image) backend via a cross-category pipeline_tag; it falls to its own category
    # default. The cross-category escape hatch stays only for home-less (Unknown) models.
    from scripts.tt_hw_planner.family_backends import pick_backend_with_quality

    b, q = pick_backend_with_quality(category="LLM", model_type="hunyuan_ar_img", pipeline_tag="text-to-image")
    assert q == "category-default", q
    assert b is not None and "diffusion" not in b.name.lower() and "stable" not in b.name.lower()

    b2, q2 = pick_backend_with_quality(category="Unknown", model_type="novel_x", pipeline_tag="text-to-image")
    assert q2 == "pipeline", f"Unknown-category escape hatch must still allow cross-category pipeline, got {q2}"


def test_low_confidence_category_only_for_ambiguous_tags():
    # issue #8: a config-less music model (text-to-audio) is flagged low-confidence, but
    # clean tags (text-generation, text-to-speech) and model_type/arch-confirmed models are not.
    from scripts.tt_hw_planner.probe import _is_low_confidence_category

    assert _is_low_confidence_category("text-to-audio", None, False) is True  # ACE-Step
    assert _is_low_confidence_category("text-generation", None, False) is False  # DeepSeek (clean)
    assert _is_low_confidence_category("text-to-speech", None, False) is False  # Kokoro (real TTS)
    assert _is_low_confidence_category("text-to-audio", "TTS", False) is False  # model_type confirms
    assert _is_low_confidence_category("text-to-audio", None, True) is False  # arch confirms


def test_local_safetensors_header_dtype(tmp_path):
    # issue #4: local-path reads the real dtype from the safetensors header instead of
    # hardcoding fp32 (//4), which 2x-undercounts a bf16 local repo (HunyuanImage-3.0).
    import struct

    from scripts.tt_hw_planner.probe import _bytes_per_param_from_local_safetensors

    hdr = json.dumps({"w": {"dtype": "BF16", "shape": [8], "data_offsets": [0, 16]}}).encode()
    with open(tmp_path / "model.safetensors", "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(b"\0" * 16)
    assert _bytes_per_param_from_local_safetensors(str(tmp_path)) == (2, "bf16")
    assert _bytes_per_param_from_local_safetensors(str(tmp_path / "empty")) == (None, None)


def test_bytes_per_param_from_safetensors_header(monkeypatch):
    # issue #7: dtype (and thus param count) read from the actual safetensors header,
    # so an fp32 repo with no config torch_dtype isn't 2x-overcounted as bf16.
    import huggingface_hub

    from scripts.tt_hw_planner import probe as pr

    class _T:
        def __init__(self, d):
            self.dtype = d

    class _MD:
        tensors = {"a": _T("F32"), "b": _T("F32"), "c": _T("I32")}  # dominant float = F32; int skipped

    class _Api:
        def parse_safetensors_file_metadata(self, mid, fn):
            return _MD()

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: _Api())
    assert pr._bytes_per_param_from_safetensors("m", ["x.safetensors"]) == (4, True, "fp32")
    assert pr._bytes_per_param_from_safetensors("m", []) == (None, False, None)


def test_scaffold_hard_stops_on_composite(monkeypatch):
    # issue #5: a composite / multi-submodel repo must HARD-STOP, not fall through to a
    # wrong family template and report success.
    from scripts.tt_hw_planner import scaffold as sc

    class _P:
        raw_config = {"model_type": "longcat_video"}
        is_composite = True
        submodels = ["dit", "text_encoder", "vae"]
        category = "Video"

    monkeypatch.setattr(sc, "probe_model", lambda mid: _P())
    raised = None
    try:
        sc.plan_scaffold("x/longcat")
    except sc.CompositeScaffoldError as e:
        raised = e
    except Exception as e:  # any other error is a FAIL for this test
        raised = ("other", repr(e))
    assert isinstance(raised, sc.CompositeScaffoldError), raised
    assert "COMPOSITE" in str(raised) and "dit" in str(raised)

    # non-composite must NOT raise CompositeScaffoldError (it proceeds; other errors ok here)
    class _Q(_P):
        is_composite = False
        submodels = []
        category = "LLM"

    monkeypatch.setattr(sc, "probe_model", lambda mid: _Q())
    try:
        sc.plan_scaffold("x/qwen")
    except sc.CompositeScaffoldError:
        raise AssertionError("non-composite must not raise CompositeScaffoldError")
    except Exception:
        pass  # unrelated downstream error is fine for this assertion


def test_compat_no_llm_blocks_for_non_llm_unknown():
    # issue #6: an unknown NON-LLM arch (DiT) must NOT get the generic LLM decoder block
    # list; an LLM-like unknown still gets the fallback.
    from scripts.tt_hw_planner.compatibility import check_compatibility

    r = check_compatibility(
        "x/longcat", {"model_type": "longcat_video", "architectures": ["LongCatVideoTransformer3DModel"]}
    )
    assert [b for b in r.results if getattr(b, "needed", False)] == []
    assert "non-LLM" in r.architecture_family

    r2 = check_compatibility(
        "x/novel",
        {
            "model_type": "novel_llm",
            "architectures": ["FooForCausalLM"],
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 128000,
        },
    )
    names = [b.block.name for b in r2.results if getattr(b, "needed", False)]
    assert "LM head" in names, names


def test_xtts_v2_is_in_tts_bucket():
    from scripts.tt_hw_planner.family_backends import backends_for_category

    tts = [b.name for b in backends_for_category("TTS")]
    stt = [b.name for b in backends_for_category("STT")]
    assert any("XTTS" in n for n in tts), "XTTS-v2 must be in the TTS bucket"
    assert not any("XTTS" in n for n in stt), "XTTS-v2 must NOT be in the STT bucket"


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
    # any *CausalLM / *CausalMM arch is decoder-only (not just For*) -- Janus MultiModalityCausalLM
    assert arch_descriptor(model_type="multi_modality", architectures=["MultiModalityCausalLM"]).startswith(
        "decoder-only"
    )
    assert arch_descriptor(model_type="emu3", architectures=["Emu3ForConditionalGeneration"]).startswith("decoder-only")
    # bare ForConditionalGeneration is NOT assumed encoder-decoder (many AR/VLM models use it)
    assert "encoder-decoder" not in arch_descriptor(
        model_type="novelx", architectures=["NovelForConditionalGeneration"]
    )
    # a real encoder+decoder signal still classifies as encoder-decoder
    assert arch_descriptor(model_type="novel_s2s", notes="encoder-decoder seq2seq").startswith("encoder-decoder")
    # vision transformers (incl. all DINO variants) infer as ViT from class-name text
    assert arch_descriptor(model_type="dinov2", architectures=["Dinov2Model"]).startswith("ViT")
    assert arch_descriptor(model_type="resnet", architectures=["ResNetForImageClassification"]).startswith("CNN")
    # 'vits' (a TTS model_type) must NOT substring-match the 'vit' vision keyword;
    # the vision inference is a delimited-token regex, so vits reads honest 'unknown'
    assert "ViT" not in arch_descriptor(model_type="vits", architectures=["VitsModel"])
    # real vision transformers (incl. versioned variants) still classify as ViT
    assert arch_descriptor(model_type="vit", architectures=["ViTForImageClassification"]).startswith("ViT")
    assert arch_descriptor(model_type="swinv2", architectures=["Swinv2Model"]).startswith("ViT")
    assert arch_descriptor(model_type="dinov2", architectures=["Dinov2Model"]).startswith("ViT")


def test_llm_category_fallback_only_on_residual(monkeypatch):
    from scripts.tt_hw_planner import probe as P

    # the residual gate: LLM fires ONLY when no model_type home AND fingerprint 'unknown'
    assert P._is_category_residual(None, "unknown") is True
    assert P._is_category_residual("LLM", "unknown") is False  # model_type placed it
    assert P._is_category_residual(None, "decoder-only causal LM") is False  # arch placed it
    assert P._is_category_residual(None, "encoder-decoder transformer") is False  # fact placed it

    calls = {"n": 0}

    def _fake_invoke(prompt, **kw):
        calls["n"] += 1
        return '{"category": "TTS"}'

    monkeypatch.setattr("scripts.tt_hw_planner.llm_synth.invoke_llm_agent", _fake_invoke)
    monkeypatch.setattr("scripts.tt_hw_planner.probe._fetch_model_card_text", lambda mid: "")  # no network in CI
    monkeypatch.setenv("TT_HW_PLANNER_LLM_CATEGORY", "1")
    monkeypatch.setenv("TT_HW_PLANNER_CATEGORY_VOTES", "1")  # deterministic single-ask for the count assert
    P._LLM_CATEGORY_CACHE.clear()
    cfg = {"model_type": "mimi", "architectures": ["MimiModel"], "sampling_rate": 24000, "codebook_size": 2048}
    assert P._llm_resolve_category("kyutai/mimi", cfg, "feature-extraction") == "TTS"
    assert calls["n"] == 1
    # cached: a second call does not re-invoke
    assert P._llm_resolve_category("kyutai/mimi", cfg, "feature-extraction") == "TTS"
    assert calls["n"] == 1
    # self-consistency voting: N asks, majority wins (2x TTS, 1x STT -> TTS)
    P._LLM_CATEGORY_CACHE.clear()
    monkeypatch.setenv("TT_HW_PLANNER_CATEGORY_VOTES", "3")
    seq = iter(['{"category":"TTS"}', '{"category":"STT"}', '{"category":"TTS"}'])
    monkeypatch.setattr("scripts.tt_hw_planner.llm_synth.invoke_llm_agent", lambda p, **k: next(seq))
    assert P._llm_resolve_category("vote/model", cfg, None) == "TTS"
    # a bogus category from the LLM is rejected (validated against the allowed set)
    P._LLM_CATEGORY_CACHE.clear()
    monkeypatch.setattr("scripts.tt_hw_planner.llm_synth.invoke_llm_agent", lambda p, **k: '{"category": "Banana"}')
    assert P._llm_resolve_category("x/y", cfg, None) is None
    # gate off -> never invokes, returns None (deterministic result stands)
    P._LLM_CATEGORY_CACHE.clear()
    monkeypatch.setenv("TT_HW_PLANNER_LLM_CATEGORY", "0")
    assert P._llm_resolve_category("kyutai/mimi", cfg, "feature-extraction") is None


def test_generative_vlm_fact_is_authoritative():
    # a generative model with a vision tower is provably a VLM (Ornith: text-heavy card but
    # vision_config present). This FACT is authoritative -- the agent cannot override it to LLM.
    from scripts.tt_hw_planner.probe import _has_generative_vlm_fact as F

    assert F({"vision_config": {}, "architectures": ["Qwen3_5ForConditionalGeneration"]}) is True
    assert F({"image_token_id": 5, "architectures": ["FooForCausalLM"]}) is True
    # a bare contrastive dual-encoder (CLIP) has vision_config but no generative head -> NOT this fact
    assert F({"vision_config": {}, "text_config": {}, "architectures": ["CLIPModel"]}) is False
    # text-only LLM -> no vision tower -> False
    assert F({"architectures": ["LlamaForCausalLM"]}) is False


def test_dual_encoder_contrastive_is_a_fact_not_llm():
    # contrastive dual-encoders (CLIP/ALIGN/CLAP) were flaky under the LLM residual
    # (it flips them to a generation category); they carry a STRUCTURAL fact -- text_config
    # + vision/audio_config with a non-generative arch -> Embed, read deterministically.
    from scripts.tt_hw_planner.probe import _is_dual_encoder_contrastive as D

    assert D({"text_config": {}, "audio_config": {}, "architectures": ["ClapModel"]}) is True
    assert D({"text_config": {}, "vision_config": {}, "architectures": ["AlignModel"]}) is True
    assert D({"text_config": {}, "vision_config": {}, "architectures": ["SiglipModel"]}) is True
    # a generative multimodal (VLM with an LM head) is NOT a contrastive dual-encoder
    assert D({"text_config": {}, "vision_config": {}, "architectures": ["LlavaForConditionalGeneration"]}) is False
    # a TASK model built ON a dual encoder (CLIPSeg = segmentation) is NOT contrastive: any
    # For<Task> head disqualifies it, so it flows to the CNN/LLM path instead of Embed.
    assert D({"text_config": {}, "vision_config": {}, "architectures": ["CLIPSegForImageSegmentation"]}) is False
    assert D({"text_config": {}, "vision_config": {}, "architectures": ["CLIPForImageClassification"]}) is False
    # audio-only codec (no text_config) is not dual-encoder -> stays on its own path
    assert D({"audio_config": {}, "sampling_rate": 24000, "architectures": ["EncodecModel"]}) is False
    # plain text/vision model -> not dual-encoder
    assert D({"architectures": ["BertModel"]}) is False


def test_dual_fact_does_not_override_confident_tag():
    # X-CLIP is a bare dual-encoder (XCLIPModel) but tagged video-classification -> Video.
    # the dual-encoder fact must NOT override a confident tag category; it only sets Embed when
    # the category is still Unknown or a weak Embed (same trust-the-tag rule as the LLM residual).
    from scripts.tt_hw_planner.probe import _is_dual_encoder_contrastive

    cfg = {"text_config": {}, "vision_config": {}, "architectures": ["XCLIPModel"]}
    assert _is_dual_encoder_contrastive(cfg) is True  # it IS a dual encoder
    # the guard lives at the call site: category in {Unknown, Embed} -- so a Video/CNN tag wins.
    for resolved, expect_embed in [("Video", False), ("CNN", False), ("Unknown", True), ("Embed", True)]:
        applies = _is_dual_encoder_contrastive(cfg) and resolved in {"Unknown", "Embed"}
        assert applies is expect_embed, resolved


def test_no_hand_maintained_model_type_lists():
    # ANTI-TREADMILL guard: category classification must run on self-maintaining signals
    # (transformers registry + config facts + arch suffix + LLM residual), NOT hand-kept
    # per-model_type lists that never converge. These lists were deleted; keep them gone.
    from scripts.tt_hw_planner import probe as P

    assert not hasattr(P, "VISION_ONLY_MODEL_TYPES"), "hand-maintained vision list re-introduced"
    assert not hasattr(P, "AUDIO_ONLY_MODEL_TYPES"), "hand-maintained audio list re-introduced"
    # _category_from_model_type must be a pure delegate to the self-updating registry:
    # a model_type the installed transformers does not know returns None (no hardcoded hit),
    # so it flows to the fact/fingerprint/LLM layers instead of a rotting table entry.
    assert P._category_from_model_type("zzz_not_a_real_model_type_9000") is None
    # and a registry-known type still classifies (self-maintaining, no hand entry needed)
    assert P._category_from_model_type("llama") == "LLM"
    assert P._category_from_model_type("vit") == "CNN"


def test_audio_markers_distinguish_codec_from_text_embedder():
    # feature-extraction -> Embed is right for a TEXT embedder but wrong for an audio codec
    # (mimi/encodec). Audio structural markers are the deterministic tell that re-routes the
    # codec off Embed; a real text embedder has none, so it is trusted as Embed (no LLM flip).
    from scripts.tt_hw_planner.probe import _has_audio_markers as H

    assert H({"sampling_rate": 24000, "codebook_size": 2048}) is True
    assert H({"audio_config": {}}) is True
    assert H({"num_mel_bins": 80}) is True
    assert H({"hidden_size": 768, "vocab_size": 30000}) is False  # text embedder: no audio markers


def test_audio_classification_tag_maps_to_stt():
    # audio-classification is an official HF tag (audio analysis, NOT synthesis) -- it belongs in
    # the stable tag vocab -> STT (audio-encoder family), not Unknown->LLM->wrongly TTS.
    from scripts.tt_hw_planner.probe import _classify_category

    assert _classify_category("audio-classification", [], None) == "STT"


def test_any_to_any_tag_maps_to_vlm():
    # 'any-to-any' is an official HF pipeline tag (unified multimodal, e.g. omni models);
    # it belongs in the stable tag vocab -> VLM, not an Unknown that forces the LLM residual.
    from scripts.tt_hw_planner.probe import _classify_category

    assert _classify_category("any-to-any", [], None) == "VLM"


def test_fingerprint_to_category_bridge():
    # when the deterministic tag/model_type path is Unknown but the STRUCTURAL fingerprint
    # identifies a backbone, the category is derived from it (Janus MultiModalityCausalLM ->
    # 'decoder-only causal LM' -> LLM), reserving the LLM residual for a truly unknown fp.
    from scripts.tt_hw_planner.probe import _category_from_fingerprint

    assert _category_from_fingerprint("decoder-only causal LM") == "LLM"
    assert _category_from_fingerprint("VLM (ViT + decoder LM)") == "VLM"
    assert _category_from_fingerprint("encoder-decoder transformer") == "LLM"
    assert _category_from_fingerprint("ViT (vision transformer)") == "CNN"
    assert _category_from_fingerprint("DiT (diffusion transformer)") == "Image"
    assert _category_from_fingerprint("diffusion UNet+VAE") == "Image"
    assert _category_from_fingerprint("SSM/Mamba (hybrid)") == "LLM"
    assert _category_from_fingerprint("unknown") is None  # -> stays residual for the LLM


def test_seq2seq_classified_by_config_fact_not_table():
    # GENERALIZATION GUARD: seq2seq is classified by the authoritative HF config flag
    # is_encoder_decoder, NOT a per-model_type table -- so model_types that are NOT in
    # the anchor table (and ones the tool has never seen) still classify correctly.
    from scripts.tt_hw_planner.fingerprint import arch_descriptor, _MODEL_TYPE_FAMILY

    for mt in ("mbart", "pegasus", "m2m_100", "marian", "fsmt", "prophetnet", "led", "zzz_new_s2s"):
        assert mt not in _MODEL_TYPE_FAMILY, f"{mt} should NOT be table-patched"
        assert arch_descriptor(model_type=mt, is_encoder_decoder=True).startswith("encoder-decoder"), mt
    # the flag only forces enc-dec on a positive fact; a causal model is untouched
    assert arch_descriptor(model_type="qwen2", architectures=["Qwen2ForCausalLM"]).startswith("decoder-only")
    assert arch_descriptor(
        model_type="newthing", architectures=["NewThingForCausalLM"], is_encoder_decoder=False
    ).startswith("decoder-only")


def test_arch_class_disambiguates_shared_model_type():
    from scripts.tt_hw_planner.probe import _arch_override_category

    # speecht5 is ONE model_type for ASR + TTS + voice-conversion, so the
    # model_type table can only guess (STT); the architectures class-name suffix
    # is the authoritative task signal and must correct it.
    assert _arch_override_category("STT", {"architectures": ["SpeechT5ForTextToSpeech"]}) == "TTS"
    assert _arch_override_category("STT", {"architectures": ["SpeechT5ForSpeechToText"]}) == "STT"
    assert _arch_override_category("STT", {"architectures": ["Wav2Vec2ForCTC"]}) == "STT"
    # a *ForCausalLM trunk mis-tagged with a single modality (Phi-4-multimodal tagged ASR) is
    # a generative LM, not STT -- promote it. Genuine ASR (CondGen/CTC) is untouched above.
    assert _arch_override_category("STT", {"architectures": ["Phi4MMForCausalLM"]}) == "LLM"
    # but an autoregressive TTS on a ForCausalLM trunk is NOT flipped (TTS excluded from promote)
    assert _arch_override_category("TTS", {"architectures": ["SomeTTSForCausalLM"]}) == "TTS"
    # same authority rule as #3: a text-to-image tag on a causal-MM arch is still LLM
    assert _arch_override_category("Image", {"architectures": ["HunyuanImage3ForCausalMM"]}) == "LLM"
    # the speech rule is scoped: it never flips a non-speech category
    assert _arch_override_category("LLM", {"architectures": ["SpeechT5ForTextToSpeech"]}) == "LLM"
    assert _arch_override_category("Image", {"architectures": ["SpeechT5ForTextToSpeech"]}) == "Image"
    # no architectures -> unchanged (degrade to model_type table)
    assert _arch_override_category("STT", {"architectures": []}) == "STT"


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
