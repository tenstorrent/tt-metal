"""The vendored package is the recorded upstream revision, and its vLLM metadata names the adapter. CPU only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from agentic_models.models.gemma_4_12b_it.vllm.model import entry

PHASE = Path(__file__).resolve().parents[2]
UPSTREAM = json.loads((PHASE / 'provenance/upstream.json').read_text())
#: Files edited after vendoring, by their own hash.
MODIFIED = UPSTREAM.get('modified_sha256', {})


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


def test_metadata_names_the_vendored_adapter():
    module, _, cls = entry.METADATA['main_class'].partition(':')
    assert module.startswith('agentic_models.models.gemma_4_12b_it.vllm.')
    tree = ast.parse(Path(importlib.util.find_spec(module).origin).read_text())
    assert cls in [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert entry.ARCH == 'TT' + entry.METADATA['arch']
