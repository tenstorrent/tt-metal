"""The vendored package is the recorded bring-up revision, and its vLLM metadata names the adapter. CPU only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from agentic_models.models.qwen3_6_27b.vllm.model import entry

PHASE = Path(__file__).resolve().parents[2]
UPSTREAM = json.loads((PHASE / 'provenance/upstream.json').read_text())
#: Files edited (or added) after vendoring, by their own hash.
MODIFIED = UPSTREAM.get('modified_sha256', {})
REWRITE = UPSTREAM['import_rewrite']


@pytest.mark.parametrize('rel', sorted(set(UPSTREAM['source_sha256']) | set(MODIFIED)))
def test_vendored_file_matches_upstream(rel):
    data = (PHASE / rel).read_bytes()
    if rel in MODIFIED:
        assert hashlib.sha256(data).hexdigest() == MODIFIED[rel]
        return
    if rel.endswith(('.py', '.json')):
        data = data.decode().replace(REWRITE['to'], REWRITE['from']).encode()
    assert hashlib.sha256(data).hexdigest() == UPSTREAM['source_sha256'][rel]


def test_every_shipped_file_is_recorded():
    """Nothing in the package escapes provenance, apart from the agentic-models files around the vendored code."""
    own = {'__init__.py', 'manifest.json', 'README.md', 'vllm_metadata.json'}
    recorded = set(UPSTREAM['source_sha256']) | set(MODIFIED)
    for path in PHASE.rglob('*'):
        rel = path.relative_to(PHASE).as_posix()
        if not path.is_file() or '__pycache__' in rel or rel in own or rel.startswith(('model/', 'provenance/')):
            continue
        assert rel in recorded, rel


def test_no_bringup_module_path_left():
    """The import rewrite covered imports and string literals: no code refers to the bring-up module path."""
    bringup_tree = '.'.join(REWRITE['from'].split('.')[:2])  # any bring-up model, not only this one
    for path in PHASE.rglob('*'):
        rel = path.relative_to(PHASE).as_posix()
        if path.suffix in ('.py', '.json') and not rel.startswith('provenance/') and '__pycache__' not in rel:
            assert bringup_tree not in path.read_text(), rel


def test_metadata_names_the_vendored_adapter():
    module, _, cls = entry.METADATA['main_class'].partition(':')
    assert module.startswith('agentic_models.models.qwen3_6_27b.vllm.')
    tree = ast.parse(Path(importlib.util.find_spec(module).origin).read_text())
    assert cls in [n.name for n in tree.body if isinstance(n, ast.ClassDef)]
    assert entry.ARCH == 'TTQwen3_5ForConditionalGeneration'


def test_register_overwrites_an_earlier_registration():
    """The plugin builtin TTQwen3_5ForConditionalGeneration (the tt-metal demo) must not survive register()."""
    registry = pytest.importorskip('vllm.model_executor.models.registry').ModelRegistry
    registry.register_model(entry.ARCH, 'models.demos.blackhole.qwen36.tt.generator_vllm:Builtin')
    assert entry.register() == entry.ARCH
    module, _, cls = entry.METADATA['main_class'].partition(':')
    registered = registry.models[entry.ARCH]
    assert (registered.module_name, registered.class_name) == (module, cls)


def test_selected_precision_is_config_p():
    """Only the validated serving precision ships, as the selected config precision.py loads by default."""
    selected = json.loads((PHASE / 'doc/datatype_sweep/selected_precision_config.json').read_text())
    assert selected['config_id'] == 'bfp8_baseline'
