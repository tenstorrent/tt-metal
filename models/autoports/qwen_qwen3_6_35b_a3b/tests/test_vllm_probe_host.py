# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host validation of the hardware probe's adapter construction boundary."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_vllm_adapter_host import adapter_class


def probe_factory():
    path = Path(__file__).with_name("run_serving_commands_probe.py")
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_adapter_probe"]
    scope = dict(os=os, SimpleNamespace=SimpleNamespace)
    exec(compile(tree, str(path), "exec"), scope)
    return scope["build_adapter_probe"]


@pytest.mark.parametrize("all_layers", [False, True])
def test_probe_uses_real_adapter_factory_and_external_cache(monkeypatch, all_layers):
    cls, scope = adapter_class()
    model = scope["QwenModel"].return_value
    model.layer_indices = tuple(range(40)) if all_layers else (0, 3)
    descriptors = [dict(default_shape=[32, 1, 32, 256], dtype="bfloat16", tensor_idx=i) for i in range(10)]
    model.cache_spec.return_value = dict(layers=descriptors)
    cache = model.allocate_kv_cache_per_layer.return_value
    scope["QwenGenerator"].return_value.cache = cache
    monkeypatch.setenv("QWEN_VLLM_REDUCED", "prior-value")
    mesh = object()
    args = SimpleNamespace(all_layers=all_layers, model_dir=None, context=1024)

    adapter = probe_factory()(cls, mesh, args)

    assert adapter.generator.cache is cache
    assert adapter.uses_mrope
    assert adapter.decode_input_update_contract == 1
    scope["QwenModel"].assert_called_once_with(
        mesh, checkpoint=None, layer_indices=None if all_layers else [0, 3], max_seq_len=1024
    )
    expected = [((32, 1, 32, 256), "bfloat16", i) for i in range(10 if all_layers else 1)]
    model.allocate_kv_cache_per_layer.assert_called_once_with(expected, batch=3)
    scope["QwenGenerator"].assert_called_once_with(model, max_batch_size=3, kv_cache=cache, decode_buckets=[2, 3])
    assert os.environ["QWEN_VLLM_REDUCED"] == "prior-value"


def test_probe_restores_reduced_selection_after_constructor_failure(monkeypatch):
    cls, scope = adapter_class()
    scope["QwenModel"].side_effect = RuntimeError("construction failed")
    monkeypatch.delenv("QWEN_VLLM_REDUCED", raising=False)
    with pytest.raises(RuntimeError, match="construction failed"):  # allow-pytest.raises: CPU-only validation
        probe_factory()(cls, object(), SimpleNamespace(all_layers=False, model_dir=None, context=1024))
    assert "QWEN_VLLM_REDUCED" not in os.environ
