"""The vendored package is the recorded upstream revision, and its vLLM metadata names the adapter. CPU only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.model import entry

PHASE = Path(__file__).resolve().parents[2]
UPSTREAM = json.loads((PHASE / 'provenance/upstream.json').read_text())
#: Files edited after vendoring, by their own hash.
MODIFIED = UPSTREAM.get('modified_sha256', {})
#: Files vendored from other sources than the bring-up model directory.
ADDITIONAL = UPSTREAM.get('additional_sources', {})


@pytest.mark.parametrize('rel', sorted(set(UPSTREAM['source_sha256']) | set(MODIFIED) | set(ADDITIONAL)))
def test_vendored_file_matches_upstream(rel):
    data = (PHASE / rel).read_bytes()
    if rel in MODIFIED:
        assert hashlib.sha256(data).hexdigest() == MODIFIED[rel]
        return
    if rel in ADDITIONAL:
        assert hashlib.sha256(data).hexdigest() == ADDITIONAL[rel]['upstream_sha256']
        return
    rewrite = UPSTREAM['import_rewrite']
    if rel.endswith(('.py', '.json')):
        data = data.decode().replace(rewrite['to'], rewrite['from']).encode()
    assert hashlib.sha256(data).hexdigest() == UPSTREAM['source_sha256'][rel]


def test_no_bring_up_paths_remain():
    """Only the provenance record names the bring-up directory; code, data and strings use the package path."""
    token = UPSTREAM['path'].split('/')[1].encode()  # the bring-up models/ subdirectory
    hits = [
        str(p.relative_to(PHASE))
        for p in PHASE.rglob('*')
        if p.is_file() and '__pycache__' not in p.parts and p != PHASE / 'provenance/upstream.json'
        and token in p.read_bytes()
    ]
    assert hits == []


def test_metadata_names_the_vendored_adapter():
    module, _, cls = entry.METADATA['main_class'].partition(':')
    assert module.startswith('agentic_models.models.nemotron_3_5_lightning_30b_a3b.vllm.')
    tree = ast.parse(Path(importlib.util.find_spec(module).origin).read_text())
    assert cls in [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert entry.ARCH == 'TT' + entry.METADATA['arch'] == 'TTNemotronHForCausalLM'


def test_register_overwrites_an_existing_entry(monkeypatch):
    """register() must replace an earlier class for the arch (an if-missing form would keep a stale one)."""
    calls = []

    class Registry:
        @staticmethod
        def get_supported_archs():
            return [entry.ARCH]

        @staticmethod
        def register_model(arch, target):
            calls.append((arch, target))

    registry = pytest.importorskip('vllm.model_executor.models.registry')
    monkeypatch.setattr(registry, 'ModelRegistry', Registry)
    monkeypatch.setattr(entry.importlib, 'import_module', lambda name: type('M', (), {'NemotronForCausalLM': 1}))
    assert entry.register() == entry.ARCH
    assert calls == [(entry.ARCH, entry.METADATA['main_class'])]
