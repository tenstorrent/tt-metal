# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Two failures seen on a model larger than host RAM, offline and device-free:

1. A manifest an earlier run left with NO components (the model could not be read then) made
   Step 2 report "ALREADY SCAFFOLDED" forever, so discovery never ran again and every later step
   had nothing to bring up. Such a manifest must be re-planned; one WITH components still
   short-circuits exactly as before.
2. Step 6's input capture loaded the whole model and was OOM-killed. A model that fits loads whole
   through the identical call; one that does not loads only the blocks the components need.
"""
import json
import types

import pytest
import torch
import transformers

from models.common import hf_reference_loading as h
from scripts.tt_hw_planner import bringup_loop
from scripts.tt_hw_planner import scaffold as scaffold_mod
from scripts.tt_hw_planner.agentic import probe as probe_mod

MODEL = "org/some-model"


class _ReachedPlanning(Exception):
    def __init__(self):
        super().__init__("re-planned past the short-circuit")


@pytest.fixture
def scaffolded(tmp_path, monkeypatch):
    demo = tmp_path / "models" / "demos" / "some_model"
    demo.mkdir(parents=True)
    monkeypatch.setattr(scaffold_mod, "BRINGUP_ROOT", lambda: tmp_path)
    monkeypatch.setattr(bringup_loop, "find_demo_dir", lambda m, repo_root=None: demo)
    monkeypatch.setattr(
        scaffold_mod,
        "probe_model",
        lambda m: types.SimpleNamespace(is_composite=False, raw_config={"x": 1}, category="LLM"),
    )

    def _compat(*a, **k):
        raise _ReachedPlanning

    monkeypatch.setattr(scaffold_mod, "check_compatibility", _compat)

    def _write(components):
        (demo / scaffold_mod.BRINGUP_STATUS_FILENAME).write_text(
            json.dumps({"new_model_id": MODEL, "components": components})
        )

    return _write


def test_a_manifest_with_components_still_short_circuits(scaffolded):
    scaffolded([{"name": "attn"}])
    assert scaffold_mod.plan_scaffold(MODEL).compat_overall == "ALREADY SCAFFOLDED"


def test_a_manifest_with_no_components_is_replanned(scaffolded, expect_error):
    scaffolded([])
    with expect_error(_ReachedPlanning, "re-planned"):
        scaffold_mod.plan_scaffold(MODEL)


def test_an_unparsable_manifest_keeps_the_previous_behavior(scaffolded, tmp_path):
    scaffolded([])
    next(tmp_path.rglob(scaffold_mod.BRINGUP_STATUS_FILENAME)).write_text("{not json")
    assert scaffold_mod.plan_scaffold(MODEL).compat_overall == "ALREADY SCAFFOLDED"


# ----------------------------------------------------------------------------- low-RAM capture

LAYERS = 4


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    cfg = transformers.LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("tiny_llama")
    transformers.LlamaForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d)
    return str(d)


@pytest.fixture(autouse=True)
def _fresh_cache():
    h._skeleton_cached.cache_clear()
    yield
    h._skeleton_cached.cache_clear()


def test_a_model_that_fits_is_loaded_by_the_identical_call(monkeypatch):
    seen = []

    class _Cls:
        @staticmethod
        def from_pretrained(model_id, **kw):
            seen.append((model_id, kw))
            return object()

    monkeypatch.setattr(transformers, "AutoModelForCausalLM", _Cls, raising=False)
    monkeypatch.setattr(h, "fits_in_host_ram", lambda *a, **k: True)
    model, loader = probe_mod.load_hf_model_cascade(MODEL, needed_paths=["model.layers.1"])
    assert loader == "AutoModelForCausalLM"
    assert seen == [(MODEL, {"trust_remote_code": True, "torch_dtype": torch.bfloat16, "low_cpu_mem_usage": True})]


def test_a_model_that_does_not_fit_loads_only_the_needed_blocks(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(h, "host_ram_available_bytes", lambda: 1)
    model, loader = probe_mod.load_hf_model_cascade(tiny_model_dir, needed_paths=["model.layers.1"])
    assert loader == "AutoModelForCausalLM"
    assert len(model.model.layers) == 2
    variants = list(probe_mod.iter_hf_model_variants(tiny_model_dir, needed_paths=["model.layers.0"]))
    assert variants and all(len(m.model.layers) == 1 for m, _ in variants if hasattr(m, "model"))


def test_without_paths_the_whole_model_loads_as_before(tiny_model_dir, monkeypatch):
    monkeypatch.setattr(h, "host_ram_available_bytes", lambda: 1)
    model, _ = probe_mod.load_hf_model_cascade(tiny_model_dir)
    assert len(model.model.layers) == LAYERS
