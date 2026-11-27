"""The vendored package is the recorded upstream revision, and its vLLM metadata names the adapter. CPU only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from agentic_models.models.nanbeige_4_3_33b.vllm.model import entry

PHASE = Path(__file__).resolve().parents[2]
UPSTREAM = json.loads((PHASE / 'provenance/upstream.json').read_text())
FILES = sorted(set(UPSTREAM['source_sha256']) | set(UPSTREAM['modified_sha256']))


@pytest.mark.parametrize('rel', FILES)
def test_vendored_file_matches_upstream(rel):
    data = (PHASE / rel).read_bytes()
    if rel in UPSTREAM['modified_sha256']:
        # Edited after vendoring: pinned by its own hash.
        assert hashlib.sha256(data).hexdigest() == UPSTREAM['modified_sha256'][rel]
        return
    rewrite = UPSTREAM['import_rewrite']
    if rel.endswith(('.py', '.json')):
        data = data.decode().replace(rewrite['to'], rewrite['from']).encode()
    assert hashlib.sha256(data).hexdigest() == UPSTREAM['source_sha256'][rel]


def test_every_runtime_file_is_recorded():
    ours = {'README.md', 'manifest.json', 'vllm_metadata.json'}
    for path in PHASE.rglob('*'):
        rel = path.relative_to(PHASE).as_posix()
        if not path.is_file() or '__pycache__' in rel or rel.startswith(('model/', 'provenance/')):
            continue
        if path.name == '__init__.py' and not path.read_bytes() or rel in ours:
            continue
        assert rel in FILES, rel


def test_no_bring_up_import_paths_remain():
    # The bring-up tree ('models/<tree>/<dir>') appears only in provenance: not in imports, string literals or paths.
    tree = UPSTREAM['path'].split('/')[1]
    for path in PHASE.rglob('*'):
        rel = path.relative_to(PHASE).as_posix()
        if path.is_file() and path.suffix in ('.py', '.json', '.cpp', '.hpp') and not rel.startswith('provenance/'):
            assert tree not in path.read_text(), rel


def test_metadata_names_the_vendored_adapter():
    module, _, cls = entry.METADATA['main_class'].partition(':')
    assert module.startswith('agentic_models.models.nanbeige_4_3_33b.vllm.')
    tree = ast.parse(Path(importlib.util.find_spec(module).origin).read_text())
    assert cls in [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    # The checkpoint's own Llama architecture; the plugin also registers tt-transformers Llama under this name.
    assert entry.ARCH == 'TT' + entry.METADATA['arch'] == 'TTLlamaForCausalLM'
