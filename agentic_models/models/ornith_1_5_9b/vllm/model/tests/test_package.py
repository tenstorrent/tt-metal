"""The vendored package is the recorded upstream revisions, and its vLLM metadata names the adapter. CPU only."""
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

from agentic_models.models.ornith_1_5_9b.vllm.model import entry

PHASE = Path(__file__).resolve().parents[2]
UPSTREAM = json.loads((PHASE / 'provenance/upstream.json').read_text())
#: The bring-up model directory, then the shared modules vendored next to it (tt/sampling, tt/_tt_transformers).
SOURCES = [UPSTREAM, *UPSTREAM['vendored']]
FILES = [
    (source, rel) for source in SOURCES for rel in sorted(set(source['source_sha256']) | set(source['modified_sha256']))
]


@pytest.mark.parametrize('source, rel', FILES, ids=[rel for _, rel in FILES])
def test_vendored_file_matches_upstream(source, rel):
    data = (PHASE / rel).read_bytes()
    if rel in source['modified_sha256']:
        # Edited after vendoring: pinned by its own hash.
        assert hashlib.sha256(data).hexdigest() == source['modified_sha256'][rel]
        return
    rewrite = source.get('import_rewrite')
    if rewrite and rel.endswith(('.py', '.json')):
        data = data.decode().replace(rewrite['to'], rewrite['from']).encode()
    assert hashlib.sha256(data).hexdigest() == source['source_sha256'][rel]


def test_every_runtime_file_is_recorded():
    recorded = {rel for source, rel in FILES}
    ours = {'README.md', 'manifest.json', 'vllm_metadata.json'}
    for path in PHASE.rglob('*'):
        rel = path.relative_to(PHASE).as_posix()
        if not path.is_file() or '__pycache__' in rel or rel.startswith(('model/', 'provenance/')):
            continue
        if path.name == '__init__.py' and not path.read_bytes() or rel in ours:
            continue
        assert rel in recorded, rel


def test_no_bring_up_import_paths_remain():
    # The bring-up tree ('models/<tree>/<dir>') appears only in provenance: not in imports, string literals or paths.
    tree = UPSTREAM['path'].split('/')[1]
    for path in PHASE.rglob('*'):
        rel = path.relative_to(PHASE).as_posix()
        if path.is_file() and path.suffix in ('.py', '.json', '.cpp', '.hpp') and not rel.startswith('provenance/'):
            text = path.read_text()
            assert tree not in text, rel
            # tt-transformers is not installed with agentic-models: its modules are vendored under tt/_tt_transformers.
            assert not re.search(r'^\s*(from|import) tt_transformers\b', text, re.M), rel


def test_metadata_names_the_vendored_adapter():
    module, _, cls = entry.METADATA['main_class'].partition(':')
    assert module.startswith('agentic_models.models.ornith_1_5_9b.vllm.')
    tree = ast.parse(Path(importlib.util.find_spec(module).origin).read_text())
    assert cls in [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    # The checkpoint's own hybrid architecture; the plugin also registers its in-tree Qwen3.5 port under this name.
    assert entry.ARCH == 'TT' + entry.METADATA['arch'] == 'TTQwen3_5ForConditionalGeneration'
