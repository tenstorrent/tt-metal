# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A package's card carries the licence and pipeline tag of the model it was built from.

Voxtral-4B-TTS, 2026-11: the base model is CC BY-NC 4.0 and text-to-speech, but every package was written as
apache-2.0 (and every vLLM package as text-generation), so the Hub listed a non-commercial speech model as a
permissively licensed text generator. The terms now come from the base model's HF card.

No network: the Hub is a fake.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_hw_planner.commands import publish_hf as ph  # noqa: E402


def _hub(monkeypatch, models):
    """A fake huggingface_hub whose HfApi().model_info(id) answers from ``models`` and raises for anything else."""

    class _Api:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id):
            if repo_id not in models:
                raise OSError(f"{repo_id} not found")
            card, tag = models[repo_id]
            return types.SimpleNamespace(card_data=card, pipeline_tag=tag)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=_Api))


def _yaml(tmp_path, terms):
    yaml = pytest.importorskip("yaml")
    demo = tmp_path / "models" / "demos" / "tts"
    demo.mkdir(parents=True)
    out = tmp_path / "tt-model.yaml"
    ph._write_http_server_yaml(
        out,
        state={"env": {"arch": "blackhole"}},
        slug="tts",
        repo_id="org/repo",
        checkout=str(tmp_path),
        demo_rel="models/demos/tts",
        weights="org/weights",
        box="QB2",
        arch=None,
        hardware=None,
        http={"module": "models.demos.tts.serve", "command": ["python"], "ready_line": "ready", "mesh": "(1, 1)"},
        packages=[],
        commit=None,
        card_terms=terms,
    )
    return yaml.safe_load(out.read_text())["card"]


def test_a_non_commercial_speech_model_is_packaged_as_one(monkeypatch, tmp_path):
    _hub(monkeypatch, {"org/weights": ({"license": "cc-by-nc-4.0"}, "text-to-speech")})
    card = _yaml(tmp_path, ph._upstream_card_terms("org/weights", None))
    assert card["license"] == {"id": "cc-by-nc-4.0"}
    assert card["pipeline_tag"] == "text-to-speech"


def test_a_custom_licence_keeps_its_name_and_link(monkeypatch, tmp_path):
    lic = {"license": "other", "license_name": "model-license", "license_link": "https://example.com/LICENSE"}
    _hub(monkeypatch, {"org/weights": (lic, None)})
    card = _yaml(tmp_path, ph._upstream_card_terms("org/weights", None))
    assert card["license"] == {"id": "other", "name": "model-license", "link": "https://example.com/LICENSE"}
    assert "pipeline_tag" not in card


def test_an_unreachable_base_model_falls_back_to_apache_and_no_pipeline_for_its_own_server(monkeypatch, tmp_path):
    _hub(monkeypatch, {})
    assert ph._upstream_card_terms("org/weights", None) == {}
    assert ph._upstream_card_terms(None, None) == {}
    card = _yaml(tmp_path, {})
    assert card["license"] == {"id": "apache-2.0"}
    assert "pipeline_tag" not in card


def test_a_vllm_package_still_defaults_to_text_generation():
    assert ph._card_terms_lines({}, "text-generation") == [
        "  license:",
        "    id: apache-2.0",
        "  pipeline_tag: text-generation",
    ]
    assert ph._card_terms_lines({"license": "mit", "pipeline_tag": "image-text-to-text"}, "text-generation") == [
        "  license:",
        "    id: mit",
        "  pipeline_tag: image-text-to-text",
    ]
