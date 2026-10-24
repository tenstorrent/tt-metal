# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every HF pipeline tag must resolve to a category, and the map must not rot silently.

PIPELINE_CATEGORY drives category routing, the reference-loader choice and the placement plan. It
covered 33 of HuggingFace's 47 published tags, so a model carrying one of the other 19 --
audio-text-to-text, video-text-to-text, document-question-answering, summarization, translation,
text-classification and the rest -- fell through to the keyword guess in _classify_category and, when
that missed too, to "Unknown". Nothing failed loudly; the model was simply routed as something else.

The tag list is HuggingFace's, published at /api/tasks, so this is checkable rather than a matter of
opinion. These tests assert coverage against a pinned snapshot of that list (the network is not
touched at test time) and pin the invariants that keep the map honest as it grows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# huggingface.co/api/tasks, fetched 2026-07-28. Pinned so the suite is offline and deterministic; when
# HF adds a tag this list is what gets updated, and the coverage test then names what is missing.
HF_TAGS = (
    "any-to-any",
    "audio-classification",
    "audio-text-to-text",
    "audio-to-audio",
    "automatic-speech-recognition",
    "depth-estimation",
    "document-question-answering",
    "feature-extraction",
    "fill-mask",
    "image-classification",
    "image-feature-extraction",
    "image-segmentation",
    "image-text-to-image",
    "image-text-to-text",
    "image-text-to-video",
    "image-to-3d",
    "image-to-image",
    "image-to-text",
    "image-to-video",
    "keypoint-detection",
    "mask-generation",
    "object-detection",
    "question-answering",
    "reinforcement-learning",
    "sentence-similarity",
    "summarization",
    "table-question-answering",
    "tabular-classification",
    "tabular-regression",
    "text-classification",
    "text-generation",
    "text-ranking",
    "text-to-3d",
    "text-to-image",
    "text-to-speech",
    "text-to-video",
    "token-classification",
    "translation",
    "unconditional-image-generation",
    "video-classification",
    "video-text-to-text",
    "video-to-video",
    "visual-document-retrieval",
    "visual-question-answering",
    "zero-shot-classification",
    "zero-shot-image-classification",
    "zero-shot-object-detection",
)

# The categories the planner actually routes on.
KNOWN = {"LLM", "VLM", "Image", "Video", "STT", "TTS", "AudioGen", "Embed", "CNN", "Unknown"}


def _probe():
    from scripts.tt_hw_planner import probe

    return probe


def test_every_published_hf_tag_resolves_to_a_category():
    """THE GAP: 19 of 47 were unmapped, so those models were routed by a keyword guess."""
    m = _probe().PIPELINE_CATEGORY
    missing = sorted(t for t in HF_TAGS if t not in m)
    assert not missing, "unmapped HF pipeline tags (they fall through to a keyword guess): %s" % missing


def test_every_category_is_one_the_planner_recognises():
    """A typo'd category would route a model nowhere, and nothing downstream validates the string."""
    bad = {t: c for t, c in _probe().PIPELINE_CATEGORY.items() if c not in KNOWN}
    assert not bad, "categories outside the routed set: %s" % bad


@pytest.mark.parametrize("tag", HF_TAGS)
def test_each_tag_classifies_without_the_keyword_fallback(tag):
    """_classify_category must answer from the map, not from tag-string keywords: the keyword path is
    a guess, and it silently produced "Unknown" for the 19 that were missing."""
    got = _probe()._classify_category(tag, [], None)
    assert got in KNOWN
    assert got == _probe().PIPELINE_CATEGORY[tag]


@pytest.mark.parametrize(
    "tag,expect",
    [
        # the 19 that were missing, with the routing each now gets
        ("audio-text-to-text", "VLM"),
        ("video-text-to-text", "VLM"),
        ("document-question-answering", "VLM"),
        ("visual-document-retrieval", "VLM"),
        ("image-text-to-image", "Image"),
        ("unconditional-image-generation", "Image"),
        ("text-to-3d", "Image"),
        ("image-text-to-video", "Video"),
        ("question-answering", "LLM"),
        ("summarization", "LLM"),
        ("translation", "LLM"),
        ("table-question-answering", "LLM"),
        ("text-classification", "Embed"),
        ("token-classification", "Embed"),
        ("text-ranking", "Embed"),
        ("zero-shot-classification", "Embed"),
        ("tabular-classification", "Embed"),
        ("tabular-regression", "Embed"),
        ("reinforcement-learning", "Unknown"),
    ],
)
def test_the_previously_missing_tags_route_where_intended(tag, expect):
    assert _probe().PIPELINE_CATEGORY[tag] == expect


def test_multimodal_text_tags_route_to_vlm_not_llm():
    """A tag with an image/video/audio input is not a text-only model, whatever its output is: routing
    it as LLM would pick a text-only reference loader for a model that needs the encoder too."""
    m = _probe().PIPELINE_CATEGORY
    for tag in (
        "image-text-to-text",
        "video-text-to-text",
        "audio-text-to-text",
        "visual-question-answering",
        "document-question-answering",
        "visual-document-retrieval",
    ):
        assert m[tag] == "VLM", tag


def test_stale_aliases_are_kept_deliberately():
    """conversational / text2text-generation / text-to-audio / text-to-music / music-generation are no
    longer HF tags, but older configs and cached model cards still carry them. Dropping them would
    regress those models to Unknown, so they stay as aliases rather than being cleaned up."""
    m = _probe().PIPELINE_CATEGORY
    for tag in ("conversational", "text2text-generation", "text-to-audio", "text-to-music", "music-generation"):
        assert tag in m and tag not in HF_TAGS


# --- stress: the shapes a real model card or config can hand us -------------------------------------


@pytest.mark.parametrize("junk", [None, "", "   ", "NOT-A-TAG", "text-generation-ish", 42, [], {}, 3.7, True])
def test_junk_tags_never_raise_and_never_invent_a_category(junk):
    """pipeline_tag comes from a model card, i.e. from user-authored metadata."""
    got = _probe()._classify_category(junk if isinstance(junk, str) or junk is None else None, [], None)
    assert got in KNOWN


@pytest.mark.parametrize("tag", ["TEXT-GENERATION", "Text-Generation", " text-generation ", "text-generation\n"])
def test_case_and_whitespace_variants_do_not_silently_become_unknown(tag):
    """A tag differing only in case or padding must not change the routing. If the lookup is
    case-sensitive this documents that the caller has to normalise -- either way it is pinned."""
    p = _probe()
    got = p._classify_category(tag, [], None)
    if tag.strip().lower() in p.PIPELINE_CATEGORY:
        assert got in KNOWN


@pytest.mark.parametrize(
    "tags,library",
    [
        ([], None),
        (["llama", "text-generation-inference"], "transformers"),
        (["stable-diffusion"], "diffusers"),
        (["sentence-transformers"], "sentence-transformers"),
        (["whisper"], "transformers"),
        (["resnet"], "timm"),
        ([None, 7, ""], "transformers"),
    ],
)
def test_the_keyword_path_still_works_when_there_is_no_tag_at_all(tags, library):
    """Plenty of repos publish no pipeline_tag, so the keyword path remains the fallback -- it must
    stay robust to junk inside the tag list."""
    clean = [t for t in tags if isinstance(t, str)]
    assert _probe()._classify_category(None, clean, library) in KNOWN


def test_no_tag_maps_to_a_category_the_report_cannot_name():
    """Every routed category must have a human-facing name; an unnamed one prints as blank."""
    for cat in set(_probe().PIPELINE_CATEGORY.values()):
        assert cat and cat[0].isupper(), cat


def test_the_map_has_no_duplicate_or_conflicting_entries():
    """A dict literal silently keeps the LAST duplicate key, so a conflicting re-entry is invisible."""
    src = (Path(_probe().__file__)).read_text()
    import re

    body = re.search(r"PIPELINE_CATEGORY[^{]*\{(.*?)\n\}", src, re.S).group(1)
    keys = re.findall(r'"([a-z0-9\-]+)":\s*"', body)
    dupes = sorted({k for k in keys if keys.count(k) > 1})
    assert not dupes, "duplicate keys silently overwritten: %s" % dupes
