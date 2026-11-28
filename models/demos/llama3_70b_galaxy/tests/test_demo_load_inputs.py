# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for the Galaxy demo prompt loaders.

``load_inputs`` / ``load_inputs_advanced`` replicate the prompt file once per
batch slot so that ``all_prompts`` covers the full rotation the demo later
walks. The per-user target length, however, is indexed by batch slot, so the
replicated index has to wrap at ``batch``. These tests pin that contract.
"""

import importlib.abc
import importlib.machinery
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from models.demos.llama3_70b_galaxy.demo.demo_common import load_inputs_advanced

REPO_ROOT = Path(__file__).resolve().parents[4]
SAMPLE_PROMPTS = REPO_ROOT / "models/demos/llama3_70b_galaxy/demo/sample_prompts"


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Stand-in for the ttnn runtime so the demo modules import on a host box."""

    def find_spec(self, name, path=None, target=None):
        if name == "ttnn" or name.startswith("ttnn."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = lambda attr: (
            (_ for _ in ()).throw(AttributeError(attr)) if attr.startswith("__") else MagicMock(name=attr)
        )
        return module

    def exec_module(self, module):
        pass


@pytest.fixture
def text_demo_load_inputs(monkeypatch):
    """``text_demo.load_inputs`` with the ttnn runtime stubbed out."""
    monkeypatch.setattr(sys, "meta_path", [_StubFinder()] + sys.meta_path)
    for name in [n for n in sys.modules if n == "ttnn" or n.startswith("ttnn.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    import models.demos.llama3_70b_galaxy.demo.text_demo as text_demo

    return text_demo


class _Response:
    status_code = 200

    def __init__(self, text):
        self.text = text


@pytest.fixture
def context_server(monkeypatch):
    """Answer context fetches locally with a long fixed body."""
    body = "x" * 300000
    calls = []

    def fake_get(url):
        calls.append(url)
        return _Response(body)

    monkeypatch.setattr("models.demos.llama3_70b_galaxy.demo.demo_common.requests.get", fake_get)
    return body, calls


def _write_prompts(tmp_path, entries):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps(entries))
    return str(path)


@pytest.mark.parametrize(
    "entries",
    [
        [{"prompt": "p", "context": "https://example.com/a.txt", "max_length": 16000}],
        [
            {"prompt": "p", "context": "https://example.com/a.txt", "max_length": 3500},
            {"prompt": "q", "context": "https://example.com/b.txt", "max_length": 7000},
            {"prompt": "r", "context": "https://example.com/c.txt", "max_length": 16000},
        ],
    ],
)
def test_advanced_loader_indexes_per_user_length_by_batch_slot(tmp_path, context_server, entries):
    body, _ = context_server
    len_per_batch = [534, 1008, 1111 * 4, 3333 * 4] * 8
    in_prompt, all_prompts = load_inputs_advanced(
        _write_prompts(tmp_path, entries), len_per_batch, True, tmp_path / "cache"
    )

    assert len(in_prompt) == len(len_per_batch)
    assert len(all_prompts) == len(entries) * len(len_per_batch)
    for slot in range(len(len_per_batch)):
        expected = body[: len_per_batch[slot]] + "```\n\n" + entries[slot % len(entries)]["prompt"]
        assert in_prompt[slot] == "```" + expected
        assert len(all_prompts[slot]) == len(in_prompt[slot])


def test_advanced_loader_keeps_file_max_length_for_single_user(tmp_path, context_server):
    context_server
    in_prompt, all_prompts = load_inputs_advanced(
        _write_prompts(tmp_path, [{"prompt": "p", "context": "https://example.com/a.txt", "max_length": 42}]),
        [15384 * 8],
        True,
        tmp_path / "cache",
    )

    assert len(in_prompt) == 1
    assert in_prompt[0].startswith("```" + "x" * 42 + "```")


def test_text_demo_loader_indexes_per_user_length_by_batch_slot(tmp_path, monkeypatch, text_demo_load_inputs):
    text_demo = text_demo_load_inputs
    body = "x" * 300000
    monkeypatch.setattr(text_demo.requests, "get", lambda url: _Response(body))
    len_per_batch = [534, 1008, 1111 * 4, 3333 * 4] * 8
    entries = [
        {"prompt": "p", "context": "https://example.com/a.txt", "max_length": 16000},
        {"prompt": "q", "context": "https://example.com/b.txt", "max_length": 32000},
    ]
    in_prompt, all_prompts = text_demo.load_inputs(_write_prompts(tmp_path, entries), len_per_batch, True)

    assert len(in_prompt) == len(len_per_batch)
    assert len(all_prompts) == len(entries) * len(len_per_batch)
    for slot in range(len(len_per_batch)):
        assert in_prompt[slot] == "```" + body[: len_per_batch[slot]] + "```\n\n" + entries[slot % 2]["prompt"]
