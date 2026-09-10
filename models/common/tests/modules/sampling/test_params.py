# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from models.common.modules.sampling.params import (
    PreparedSamplingParams,
    place_prepared_sampling_params,
    prepare_sampling_params,
    slice_prepared_sampling_params,
    slice_sampling_params,
)
from models.common.sampling.sampling_params import SamplingParams


def _prepare(params, *, batch_size=4, max_device_top_k=32, allow_force_argmax=True):
    return prepare_sampling_params(
        params,
        batch_size,
        max_device_top_k=max_device_top_k,
        allow_force_argmax=allow_force_argmax,
    )


@pytest.mark.parametrize("top_k", [1, 32])
def test_stochastic_top_k_device_boundaries_are_preserved_exactly(top_k):
    prepared = _prepare(SamplingParams(temperature=0.7, top_k=top_k, top_p=0.9))

    assert isinstance(prepared, PreparedSamplingParams)
    assert prepared.top_k[0] == top_k
    assert prepared.sampling_path == "topk"


@pytest.mark.parametrize("top_k", [0, -1, 33, 50, 128256])
def test_unsupported_stochastic_top_k_raises_instead_of_clamping(top_k, expect_error):
    with expect_error(ValueError, "route this request to host sampling"):
        _prepare(SamplingParams(temperature=0.7, top_k=top_k, top_p=0.9))


@pytest.mark.parametrize("top_k", [-1, 0, 33, 50, 128256])
def test_greedy_normalization_precedes_top_k_validation(top_k):
    prepared = _prepare(SamplingParams(temperature=0.0, top_k=top_k, top_p=0.2))

    assert prepared.top_k[0] == 1
    assert prepared.top_p[0] == 0.0
    assert prepared.temperature[0] == 1.0
    assert prepared.greedy_mask[0] is True
    assert prepared.sampling_path == "argmax"


def test_mixed_greedy_and_stochastic_rows_use_batch_wide_topk_path():
    prepared = _prepare(
        SamplingParams(
            temperature=[0.0, 0.5],
            top_k=[128256, 32],
            top_p=[0.4, 0.8],
        )
    )

    assert prepared.row_paths[:2] == ("argmax", "topk")
    assert prepared.sampling_path == "topk"
    assert prepared.top_k[:2] == (1, 32)


def test_one_unsupported_stochastic_row_rejects_the_complete_batch(expect_error):
    with expect_error(ValueError, r"top_k\[1\]=33"):
        _prepare(
            SamplingParams(
                temperature=[0.0, 0.5],
                top_k=[128256, 33],
                top_p=[0.4, 0.8],
            )
        )


def test_prepared_structure_preserves_all_request_owned_sampling_fields():
    params = SamplingParams(
        temperature=torch.tensor([0.0, 0.5]),
        top_k=torch.tensor([128256, 7]),
        top_p=torch.tensor([0.6, 0.75]),
        presence_penalty=[0.1, 0.2],
        frequency_penalty=[0.3, 0.4],
        repetition_penalty=[1.1, 1.2],
        seed=[None, 19],
        enable_log_probs=[False, True],
        num_logprobs=[0, 0],
    )

    prepared = _prepare(params)

    assert prepared.presence_penalty[:2] == pytest.approx((0.1, 0.2))
    assert prepared.frequency_penalty[:2] == pytest.approx((0.3, 0.4))
    assert prepared.repetition_penalty[:2] == pytest.approx((1.1, 1.2))
    assert prepared.seeds[:2] == (None, 19)
    assert prepared.logprob_modes[:2] == ("none", "sampled_token")
    assert prepared.penalties_enabled is True
    assert prepared.log_probs_enabled is True
    assert torch.equal(params.top_k, torch.tensor([128256, 7]))


def test_greedy_logprob_request_uses_logprob_capable_topk_path():
    prepared = _prepare(
        SamplingParams(
            temperature=0.0,
            top_k=128256,
            top_p=1.0,
            enable_log_probs=True,
            num_logprobs=0,
        )
    )

    assert prepared.logprob_modes[0] == "sampled_token"
    assert prepared.row_paths[0] == "topk"
    assert prepared.sampling_path == "topk"


def test_vllm_disabled_logprob_sentinel_normalizes_without_rejecting_device_sampling():
    prepared = _prepare(
        SamplingParams(
            temperature=[0.0, 0.8],
            top_k=[128256, 7],
            top_p=[1.0, 0.9],
            enable_log_probs=[False, True],
            num_logprobs=[-2, 0],
        )
    )

    assert prepared.num_logprobs[:2] == (0, 0)
    assert prepared.logprob_modes[:2] == ("none", "sampled_token")


def test_force_argmax_capability_is_explicit_policy():
    params = SamplingParams(temperature=0.0, top_k=128256, top_p=1.0)

    assert _prepare(params, allow_force_argmax=True).sampling_path == "argmax"
    assert _prepare(params, allow_force_argmax=False).sampling_path == "topk"


def test_scalar_seed_stays_request_scoped_and_inactive_rows_use_safe_defaults():
    prepared = _prepare(SamplingParams(temperature=0.8, top_k=4, top_p=0.9, seed=123))

    assert prepared.seeds == (123, None, None, None)
    assert prepared.top_k[1:] == (1, 1, 1)
    assert prepared.top_p[1:] == (0.0, 0.0, 0.0)
    assert prepared.temperature[1:] == (1.0, 1.0, 1.0)
    assert prepared.row_paths[1:] == ("inactive", "inactive", "inactive")


def test_vllm_negative_one_seed_sentinel_is_native_unseeded_state():
    prepared = _prepare(
        SamplingParams(
            temperature=[0.8, 0.8],
            top_k=[4, 4],
            top_p=[0.9, 0.9],
            seed=[-1, 123],
        )
    )

    assert prepared.seeds[:2] == (None, 123)


def test_slice_sampling_params_preserves_field_alignment_without_mutating_input():
    params = SamplingParams(
        temperature=[0.1, 0.2, 0.3],
        top_k=[1, 2, 3],
        top_p=[0.4, 0.5, 0.6],
        presence_penalty=[0.7, 0.8, 0.9],
        seed=[11, None, 33],
        enable_log_probs=[False, True, False],
    )

    sliced = slice_sampling_params(params, [2, 0])

    assert sliced.temperature == [0.3, 0.1]
    assert sliced.top_k == [3, 1]
    assert sliced.top_p == [0.6, 0.4]
    assert sliced.presence_penalty == [0.9, 0.7]
    assert sliced.seed == [33, 11]
    assert sliced.enable_log_probs == [False, False]
    assert params.temperature == [0.1, 0.2, 0.3]


def test_prepared_slice_preserves_prompt_output_and_slot_remap_alignment():
    prompt_tokens = torch.tensor([[10, 11], [20, 21], [30, 31]])
    output_tokens = [[100], [200, 201], [300]]
    slot_remap = torch.tensor([2, 0, 1], dtype=torch.int32)
    prepared = prepare_sampling_params(
        SamplingParams(
            temperature=[0.0, 0.5, 0.8],
            top_k=[128256, 5, 7],
            top_p=[1.0, 0.9, 0.8],
            seed=[11, 22, 33],
        ),
        4,
        max_device_top_k=32,
        allow_force_argmax=True,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        slot_remap=slot_remap,
    )

    lane = slice_prepared_sampling_params(prepared, [2, 0])

    assert lane.top_k == (7, 1)
    assert lane.seeds == (33, 11)
    assert lane.row_paths == ("topk", "argmax")
    assert lane.sampling_path == "topk"
    assert torch.equal(lane.prompt_tokens, torch.tensor([[30, 31], [10, 11]]))
    assert lane.output_tokens == [[300], [100]]
    assert torch.equal(lane.slot_remap, torch.tensor([1, 2], dtype=torch.int32))
    assert torch.equal(prepared.prompt_tokens, prompt_tokens)
    assert prepared.output_tokens is output_tokens
    assert prepared.slot_remap is slot_remap


def test_prefill_request_rows_are_placed_into_lane_local_slots_with_history():
    prepared = prepare_sampling_params(
        SamplingParams(
            temperature=[0.0, 0.5],
            top_k=[128256, 7],
            top_p=[1.0, 0.8],
            seed=[11, 22],
            repetition_penalty=[1.1, 1.2],
        ),
        4,
        max_device_top_k=32,
        allow_force_argmax=True,
        prompt_tokens=torch.tensor([[10, 11], [20, 21]]),
        output_tokens=torch.tensor([[100, -1], [200, 201]]),
    )

    placed = place_prepared_sampling_params(prepared, [3, 1])

    assert placed.active_mask == (False, True, False, True)
    assert placed.top_k == (1, 7, 1, 1)
    assert placed.seeds == (None, 22, None, 11)
    assert placed.row_paths == ("inactive", "topk", "inactive", "argmax")
    assert torch.equal(
        placed.prompt_tokens,
        torch.tensor([[-1, -1], [20, 21], [-1, -1], [10, 11]]),
    )
    assert torch.equal(
        placed.output_tokens,
        torch.tensor([[-1, -1], [200, 201], [-1, -1], [100, -1]]),
    )


@pytest.mark.parametrize(
    ("relative_path", "class_name"),
    [
        ("models/common/models/llama3_8b/generator.py", "Llama3Generator"),
        ("models/common/models/llama33_70b/generator.py", "Llama33_70BGenerator"),
        ("models/common/models/qwen3_32b/generator.py", "Qwen3_32BGenerator"),
    ],
)
def test_target_generator_capabilities_advertise_exact_device_top_k(relative_path, class_name):
    repository_root = Path(__file__).parents[5]
    source = (repository_root / relative_path).read_text(encoding="utf-8")
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
    assert capabilities["max_device_top_k"] == 32


@pytest.mark.parametrize(
    "relative_path",
    [
        "models/tt_transformers/tt/generator_vllm.py",
        "models/demos/llama3_70b_galaxy/tt/generator_vllm.py",
    ],
)
def test_legacy_device_sampling_capabilities_declare_the_exact_limit(relative_path):
    tree = ast.parse(Path(relative_path).read_text(encoding="utf-8"), filename=relative_path)
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

    assert advertised
    assert all(capabilities.get("max_device_top_k") == 32 for _, capabilities in advertised)


def test_tttv2_runtime_import_boundary_excludes_legacy_sampling_state():
    roots = [Path("models/common/llm_runtime")]
    files = [path for root in roots for path in root.rglob("*.py")]
    files.extend(
        Path(path)
        for path in (
            "models/common/models/llama3_8b/executor.py",
            "models/common/models/llama33_70b/executor.py",
            "models/common/models/qwen3_32b/executor.py",
        )
    )
    forbidden_modules = {
        "models.common.sampling",
        "models.common.sampling.generator",
        "models.common.sampling.tt_penalties",
    }
    violations = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in forbidden_modules:
                violations.append((str(path), node.lineno, node.module))
    assert violations == []


def test_loading_tttv2_runtime_does_not_load_legacy_sampling_modules():
    script = """
import importlib
import sys
importlib.import_module('models.common.llm_runtime.decode')
forbidden = {
    'models.common.sampling.generator',
    'models.common.sampling.tt_penalties',
    'models.common.sampling.tt_sampling',
}
loaded = sorted(forbidden.intersection(sys.modules))
if loaded:
    raise SystemExit('loaded legacy sampling modules: ' + ', '.join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
