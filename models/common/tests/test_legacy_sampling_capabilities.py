# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""vLLM generators that sample on device must advertise the device sampler's top-k limit,
and the TTTv1 generators with device grammar must keep their sampling capabilities.

Host-only: reads the generator sources with ``ast``; nothing is imported or opened.
These checks used to run from the in-tree TTTv2 sampling tests. tt-transformers skips
them because the generators live here, so they stay in tt-metal.
"""

import ast
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEVICE_TOP_K = 32


def _advertised_device_sampling_capabilities(relative_path):
    source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)
    advertised = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for statement in node.body:
            if not isinstance(statement, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "model_capabilities" for target in statement.targets
            ):
                continue
            capabilities = ast.literal_eval(statement.value)
            if capabilities.get("supports_sample_on_device"):
                advertised.append((node.name, capabilities))
    return advertised


@pytest.mark.parametrize(
    "relative_path",
    [
        "models/tt_transformers/tt/generator_vllm.py",
        "models/demos/llama3_70b_galaxy/tt/generator_vllm.py",
    ],
)
def test_device_sampling_capabilities_declare_the_exact_limit(relative_path):
    advertised = _advertised_device_sampling_capabilities(relative_path)

    assert advertised
    assert all(capabilities.get("max_device_top_k") == DEVICE_TOP_K for _, capabilities in advertised)


@pytest.mark.parametrize("class_name", ["LlamaForCausalLM", "QwenForCausalLM", "MistralForCausalLM"])
def test_device_grammar_generators_retain_sampling_capabilities(class_name):
    relative_path = "models/tt_transformers/tt/generator_vllm.py"
    source = (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    assignment = next(
        node
        for node in class_node.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "model_capabilities" for target in node.targets)
    )
    capabilities = ast.literal_eval(assignment.value)

    assert capabilities["supports_sample_on_device"] is True
    assert capabilities["supports_device_grammar"] is True
    assert capabilities["max_device_top_k"] == DEVICE_TOP_K
