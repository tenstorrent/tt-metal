# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Focused contracts for the family-neutral model composition root."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import ttnn
from models.common.llm_runtime.config import PagedKVCacheConfig, TraceConfig, WarmupConfig
from models.common.models.executor import ModelExecutor, ModelExecutorConfig

_EXECUTOR_PATH = Path(__file__).parents[2] / "models" / "executor.py"
_MODELS_ROOT = _EXECUTOR_PATH.parent


def _config(*, device_sampling_enabled: bool = False) -> ModelExecutorConfig:
    return ModelExecutorConfig(
        trace=TraceConfig(mode="none"),
        warmup=WarmupConfig(),
        paged_kv_cache=PagedKVCacheConfig(
            block_size=32,
            max_num_blocks=128,
            num_blocks=128,
            dtype=ttnn.bfloat8_b,
        ),
        device_sampling_enabled=device_sampling_enabled,
    )


def test_common_executor_has_no_concrete_model_dependencies_or_dispatch() -> None:
    tree = ast.parse(_EXECUTOR_PATH.read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None}
    imported_names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not any(module.startswith("models.common.models.") for module in imports)
    assert not any(name.startswith("models.common.models.") for name in imported_names)

    control_flow = [
        node.test for node in ast.walk(tree) if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.Assert))
    ]
    dispatch_names = {"model_name", "model_id", "provider_id", "checkpoint_path", "model_version"}
    assert not any(
        isinstance(node, ast.Name) and node.id in dispatch_names
        for expression in control_flow
        for node in ast.walk(expression)
    )

    torch_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
    ]
    assert torch_calls == []


def test_model_layer_has_only_the_approved_family_modules_and_readmes() -> None:
    assert (_MODELS_ROOT / "llama3_executor.py").is_file()
    assert (_MODELS_ROOT / "qwen2_executor.py").is_file()
    assert not (_MODELS_ROOT / "qwen3_executor.py").exists()

    model_directories = sorted(
        path for path in _MODELS_ROOT.iterdir() if path.is_dir() and (path / "model.py").is_file()
    )
    assert len(model_directories) == 12
    assert all((path / "README.md").is_file() for path in model_directories)


@pytest.mark.parametrize(
    "relative_path",
    (
        "deepseek_r1_distill_qwen_14b/executor.py",
        "mistral_7b/executor.py",
        "phi4/executor.py",
    ),
)
def test_direct_composition_examples_do_not_depend_on_shared_family_executors(relative_path: str) -> None:
    tree = ast.parse((_MODELS_ROOT / relative_path).read_text())
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "models.common.models.executor" not in imports
    assert "models.common.models.llama3_executor" not in imports
    assert "models.common.models.qwen2_executor" not in imports


def test_common_config_is_frozen_and_rejects_non_exact_nested_config_types() -> None:
    config = _config()
    with pytest.raises(AttributeError):
        config.device_sampling_enabled = True

    class TraceConfigSubclass(TraceConfig):
        pass

    with pytest.raises(TypeError, match="trace must be exactly TraceConfig"):
        ModelExecutorConfig(
            trace=TraceConfigSubclass(mode="none"),
            warmup=config.warmup,
            paged_kv_cache=config.paged_kv_cache,
            device_sampling_enabled=False,
        )


def test_sampling_state_inputs_are_an_optional_owned_pair() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        ModelExecutor(
            None,
            None,
            _config(device_sampling_enabled=True),
            sampling_state_controller=object(),
        )
    with pytest.raises(ValueError, match="requires device sampling"):
        ModelExecutor(
            None,
            None,
            _config(),
            sampling_state_controller=object(),
            sampling_state=object(),
        )


@pytest.mark.parametrize(
    ("enable_trace", "expected"),
    [(True, ["prime", "coordinator"]), (False, ["coordinator", "prime"])],
)
def test_prefill_warmup_policy_controls_order_through_one_continuation(enable_trace, expected) -> None:
    events = []

    def policy(executor, default_warmup, *, kv_cache, can_sample_on_device, enable_trace):
        assert executor is target
        assert kv_cache is cache
        assert can_sample_on_device
        if enable_trace:
            events.append("prime")
        default_warmup()
        if not enable_trace:
            events.append("prime")

    cache = object()
    target = object.__new__(ModelExecutor)
    target._terminal = False
    target.prefill_runtime = SimpleNamespace(transient_orphan_count=0)
    target.decode_runtime = SimpleNamespace(transient_orphan_count=0)
    target._prefill_warmup = policy
    target.warmup = SimpleNamespace(
        warmup_prefill=lambda **kwargs: events.append("coordinator"),
    )

    target.warmup_model_prefill(
        kv_cache=cache,
        can_sample_on_device=True,
        enable_trace=enable_trace,
    )

    assert events == expected


def test_request_state_and_execution_target_are_forwarded_by_identity() -> None:
    target = object.__new__(ModelExecutor)
    target._terminal = False
    target.prefill_runtime = SimpleNamespace(transient_orphan_count=0)
    target.decode_runtime = SimpleNamespace(transient_orphan_count=0)
    target._validate_bound_cache = MagicMock()
    target._ensure_sampling_for = MagicMock()
    target._prefill_execution = MagicMock()
    target._request_state_fields = ("prompt_tokens", "output_tokens", "slot_remap")

    values = {name: object() for name in ("tokens", "page_table", "prompt_tokens", "output_tokens", "slot_remap")}
    target.compile_prefill(
        tokens=values["tokens"],
        page_table=values["page_table"],
        prompt_tokens=values["prompt_tokens"],
        output_tokens=values["output_tokens"],
        slot_remap=values["slot_remap"],
    )

    forwarded = target._prefill_execution.compile_prefill.call_args.kwargs
    for name, value in values.items():
        assert forwarded[name] is value


def test_layout_refresh_preserves_owner_and_sampling_state_identity() -> None:
    class _Config:
        def __init__(self, name, state):
            self.name = name
            self.sampling_state = state

        def with_page_table_layout(self, layout):
            replacement = _Config(self.name, self.sampling_state)
            replacement.page_table_layout = layout
            return replacement

    state = object()
    layout = object()
    target = object.__new__(ModelExecutor)
    target.prefill_runtime = SimpleNamespace(config=_Config("prefill", state))
    target.decode_runtime = SimpleNamespace(config=_Config("decode", state))
    target.warmup = SimpleNamespace(config=_Config("warmup", state))
    target._resolve_page_table_layout = lambda: layout
    owners = (target.prefill_runtime, target.decode_runtime, target.warmup)

    target._refresh_page_table_layout()

    assert (target.prefill_runtime, target.decode_runtime, target.warmup) == owners
    assert target.page_table_layout is layout
    assert all(owner.config.page_table_layout is layout for owner in owners)
    assert target.prefill_runtime.config.sampling_state is state
    assert target.decode_runtime.config.sampling_state is state


def test_cleanup_is_ordered_retryable_idempotent_and_terminal() -> None:
    events = []
    failing = {"reader", "trace"}

    class _Owner:
        def __init__(self, name):
            self.name = name

        def action(self, *args):
            events.append(self.name)
            if self.name in failing:
                raise RuntimeError(self.name)

        cleanup = action
        drain = action
        drain_external_outputs = action
        cleanup_transients = action
        release = action

    target = object.__new__(ModelExecutor)
    target._terminal = False
    target._cleaned_up = False
    target._owner_name = "TestExecutor"
    target.decode_runtime = _Owner("decode")
    target.output_reader = _Owner("reader")
    target.prefill_runtime = _Owner("prefill")
    target.trace_compiler = _Owner("trace")
    target.program_compiler = _Owner("program")
    target.config = SimpleNamespace(device_sampling_enabled=True)
    target.sampling_state_controller = _Owner("sampling-state")
    target.sampling_state = object()
    target.model = SimpleNamespace(sampling=_Owner("sampling"))
    target.kv_cache_manager = _Owner("kv")

    expected = ["decode", "reader", "prefill", "decode", "trace", "program", "sampling-state", "sampling", "kv"]
    with pytest.raises(RuntimeError, match="reader") as raised:
        target.cleanup()
    assert events == expected
    assert [str(error) for error in raised.value.cleanup_failures] == ["trace"]
    assert target.terminal
    assert not target._cleaned_up

    failing.clear()
    target.cleanup()
    target.cleanup()
    assert events == expected * 2
    assert target._cleaned_up
