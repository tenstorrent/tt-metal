"""The vendored package is the recorded upstream revision, and its vLLM metadata names the adapter. CPU only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from agentic_models.models.qwen3_6_35b_a3b.vllm.model import entry

PHASE = Path(__file__).resolve().parents[2]
UPSTREAM = json.loads((PHASE / 'provenance/upstream.json').read_text())
#: Files edited after vendoring, by their own hash.
MODIFIED = UPSTREAM.get('modified_sha256', {})
#: Package files written here rather than vendored.
OWN = {'tt/__init__.py'}


@pytest.mark.parametrize('rel', sorted(set(UPSTREAM['source_sha256']) | set(MODIFIED)))
def test_vendored_file_matches_upstream(rel):
    data = (PHASE / rel).read_bytes()
    if rel in MODIFIED:
        assert hashlib.sha256(data).hexdigest() == MODIFIED[rel]
        return
    rewrite = UPSTREAM['import_rewrite']
    if rel.endswith(('.py', '.json')):
        data = data.decode().replace(rewrite['to'], rewrite['from']).encode()
    assert hashlib.sha256(data).hexdigest() == UPSTREAM['source_sha256'][rel]


def test_every_runtime_file_is_recorded():
    recorded = set(UPSTREAM['source_sha256']) | set(MODIFIED) | OWN
    present = {
        str(p.relative_to(PHASE))
        for top in ('tt', 'doc')
        for p in (PHASE / top).rglob('*')
        if p.is_file() and '__pycache__' not in p.parts
    }
    assert present == recorded
    assert all(rel in UPSTREAM['edits'] for rel in MODIFIED)


def test_no_bring_up_package_path_left():
    # The import rewrite covers string literals too (fast_mtp's __import__); only provenance names the source path.
    source = UPSTREAM['import_rewrite']['from']
    bring_up_dir = source.split('.')[1]
    for p in PHASE.rglob('*'):
        if p.is_file() and 'provenance' not in p.parts and '__pycache__' not in p.parts:
            text = p.read_text(errors='ignore')
            assert bring_up_dir not in text and source not in text, p


def test_metadata_names_the_vendored_adapter():
    module, _, cls = entry.METADATA['main_class'].partition(':')
    assert module.startswith('agentic_models.models.qwen3_6_35b_a3b.vllm.')
    tree = ast.parse(Path(importlib.util.find_spec(module).origin).read_text())
    assert cls in [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    # The checkpoint's own architecture, so vLLM keeps its hybrid (mamba align mode) handling.
    assert entry.METADATA['arch'] == 'Qwen3_5MoeForConditionalGeneration'
    assert entry.ARCH == 'TT' + entry.METADATA['arch']
    assert entry.TT_MODEL_CLASS_OVERRIDES == f"{entry.ARCH}={entry.METADATA['main_class']}"


def test_manifest_names_the_validated_checkpoint():
    manifest = json.loads((PHASE / 'manifest.json').read_text())
    assert manifest['hf_model'] == 'Qwen/Qwen3.6-35B-A3B'
    assert manifest['entrypoint'] == entry.__name__
    source = (PHASE / 'tt/model.py').read_text()
    assert f'HF_REVISION = "{manifest["hf_revision"]}"' in source


def test_register_overwrites_an_earlier_registration():
    registry = pytest.importorskip('vllm.model_executor.models.registry').ModelRegistry
    previous = registry.models.get(entry.ARCH)
    registry.register_model(entry.ARCH, 'some.other.package:Model')
    try:
        assert entry.register() == entry.ARCH
        model = registry.models[entry.ARCH]
        assert f'{model.module_name}:{model.class_name}' == entry.METADATA['main_class']
    finally:
        if previous is None:
            registry.models.pop(entry.ARCH, None)
        else:
            registry.models[entry.ARCH] = previous
