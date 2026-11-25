"""The vendored package is the recorded upstream revision, and its vLLM metadata names the adapter. CPU only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from agentic_models.models.ornith_1_5_35b_a3b.vllm.model import entry

PHASE = Path(__file__).resolve().parents[2]
UPSTREAM = json.loads((PHASE / 'provenance/upstream.json').read_text())
#: Files edited after vendoring, by their own hash.
MODIFIED = UPSTREAM.get('modified_sha256', {})
#: Files the KV-capacity gate (tt/model.py measured_kv_capacity) hashes at startup.
CAPACITY_GATED = ('tt/model.py', 'tt/generator.py', 'tt/multichip_decoder.py', 'tt/optimized_decoder.py', 'tt/precision.py')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize('rel', sorted(set(UPSTREAM['source_sha256']) | set(MODIFIED)))
def test_vendored_file_matches_upstream(rel):
    data = (PHASE / rel).read_bytes()
    if rel in MODIFIED:
        assert sha256(data) == MODIFIED[rel]
        return
    if rel.endswith(('.py', '.json')):
        text = data.decode()
        for rule in (UPSTREAM['import_rewrite'], UPSTREAM['path_rewrite']):
            text = text.replace(rule['to'], rule['from'])
        data = text.encode()
    assert sha256(data) == UPSTREAM['source_sha256'][rel]


@pytest.mark.parametrize('rel', sorted(UPSTREAM['vendored_modules']))
def test_vendored_module_matches_record(rel):
    assert sha256((PHASE / rel).read_bytes()) == UPSTREAM['vendored_modules'][rel]['modified_sha256']


def test_capacity_gate_names_the_vendored_sources():
    # tt/model.py refuses to start when these hashes differ, so they must track any edit to the gated files.
    record = json.loads((PHASE / 'doc/full_model/kv_capacity.json').read_text())
    for rel in CAPACITY_GATED:
        assert record['source_sha256'][rel] == sha256((PHASE / rel).read_bytes()), rel
    assert record['evidence'] and all((PHASE / rel).is_file() for rel in record['evidence'])


def test_no_bring_up_paths_remain():
    # Only provenance/upstream.json may name the bring-up directory (split so this file does not match itself).
    needle = b'auto' + b'ports'
    hits = [
        str(p.relative_to(PHASE))
        for p in PHASE.rglob('*')
        if p.is_file() and not {'provenance', '__pycache__'} & set(p.parts) and needle in p.read_bytes()
    ]
    assert hits == []


def test_metadata_names_the_vendored_adapter():
    module, _, cls = entry.METADATA['main_class'].partition(':')
    assert module.startswith('agentic_models.models.ornith_1_5_35b_a3b.vllm.')
    tree = ast.parse(Path(importlib.util.find_spec(module).origin).read_text())
    assert cls in [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert entry.ARCH == 'TT' + entry.METADATA['arch'] == 'TTQwen3_5MoeForConditionalGeneration'
