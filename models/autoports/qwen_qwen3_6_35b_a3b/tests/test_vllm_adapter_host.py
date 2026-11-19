# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only boundary tests, loading the actual adapter class without TTNN."""

import ast
import hashlib
import json
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import torch


def adapter_class():
    path = Path(__file__).resolve().parents[1] / "tt/generator_vllm.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    scope = dict(
        hashlib=hashlib,
        json=json,
        os=os,
        time=time,
        Path=Path,
        torch=torch,
        replace=replace,
        QwenServingCapabilities=SimpleNamespace(serving_capabilities={}),
        QwenModel=Mock(),
        QwenGenerator=Mock(),
    )
    exec(compile(tree, str(path), "exec"), scope)
    return scope["QwenForCausalLM"], scope


def test_external_cache_is_bound_before_generator_construction():
    cls, scope = adapter_class()
    model = scope["QwenModel"].return_value
    model.layer_indices = (0, 3)
    instance = cls.initialize_vllm_model(SimpleNamespace(model_type="qwen3_5_moe"), object(), 3, 262144)
    scope["QwenGenerator"].assert_not_called()
    cache = instance.allocate_kv_cache_per_layer([((8195, 1, 32, 256), torch.bfloat16, 0)])
    scope["QwenGenerator"].assert_called_once_with(model, max_batch_size=3, kv_cache=cache, decode_buckets=[2, 3])
    assert cache is model.allocate_kv_cache_per_layer.return_value
    assert instance.tt_supported_decode_batch_sizes == [2, 3]


def test_all_accepted_capacities_dispatch_supported_hidden_masks():
    """Execute actual model dispatch for every prepared bucket at capacities 1–32."""
    path = Path(__file__).resolve().parents[1] / "tt/model.py"
    tree = ast.parse(path.read_text())
    model_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "QwenModel")
    decode = next(node for node in model_class.body if isinstance(node, ast.FunctionDef) and node.name == "decode")
    scope = dict(ttnn=MagicMock(), advance_position=Mock())
    exec(compile(ast.Module(body=[decode], type_ignores=[]), str(path), "exec"), scope)
    cls, adapter_scope = adapter_class()
    adapter_scope["QwenModel"].return_value.layer_indices = (0, 3)
    for capacity in range(1, 33):
        instance = cls.initialize_vllm_model(SimpleNamespace(model_type="qwen3_5_moe"), object(), capacity, 262144)
        assert instance.max_batch_size == capacity
        for rows in instance.tt_supported_decode_batch_sizes:
            layer = Mock()
            layer.decode_batch.return_value = [object() for _ in range(rows)]
            model = Mock(
                suppress_inactive_moe_routes=True,
                fuse_hidden_row_mask=True,
                layers=[layer],
            )
            model._mask_hidden_rows.side_effect = lambda values, slots, activity: values
            mask = object()
            scope["decode"](
                model,
                MagicMock(),
                cache=[[object() for _ in range(capacity)]],
                positions=[object() for _ in range(capacity)],
                slots=tuple(range(rows)),
                activity=[object() for _ in range(capacity)],
                active_mask=mask,
            )
            fused = rows in (2, 4, 32)
            assert layer.decode_batch.call_args.kwargs["hidden_row_mask"] is (mask if fused else None)
            assert model._mask_hidden_rows.call_count == (1 if fused else 2)


def test_decode_forwards_stale_inputs_and_independent_commands_without_reading():
    cls, _ = adapter_class()
    instance = cls()
    instance.generator = Mock()
    instance._event = Mock()
    tokens, positions, pages, cache = object(), object(), object(), object()
    commands = dict(
        reload_inputs=False,
        reload_page_table=True,
        reload_sampling_params=False,
        reset_sampling_state=False,
        page_table=pages,
        kv_cache=cache,
        read_from_device=False,
        enable_trace=True,
    )
    instance.decode_forward(tokens, positions, rope_deltas_all_users=None, **commands)
    instance.generator.decode_forward_v1.assert_called_once_with(tokens, positions, **commands)
    assert cls.decode_input_update_contract == 1


def test_text_prefill_host_mode_uses_serving_output_and_zero_rope_delta():
    cls, _ = adapter_class()
    instance = cls()
    instance.generator = Mock(active_slots=(0,))
    instance.uses_mrope = True
    instance._event = Mock()
    tokens = torch.ones((1, 33), dtype=torch.int32)
    kwargs = dict(prompt_lens=[33], start_pos=[0], kv_cache=object(), enable_trace=False)
    output, deltas = instance.prefill_forward(tokens, pixel_values=[None], image_grid_thw=[None], **kwargs)
    instance.generator.prefill_forward.assert_called_once_with(tokens, read_from_device=True, **kwargs)
    assert output is instance.generator.prefill_forward.return_value
    assert deltas.tolist() == [0]


def test_disabled_logprob_sentinel_translation_preserves_real_feature_requests():
    from math import isfinite
    from numbers import Integral

    from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_serving_commands_cpu import functions

    cls, _ = adapter_class()
    scope = dict(torch=torch, isfinite=isfinite, Integral=Integral)
    functions(Path(__file__).resolve().parents[1] / "tt/serving.py", scope)
    validate = scope["_neutral_device_features"]

    @dataclass(frozen=True)
    class Params:
        num_logprobs: list
        enable_log_probs: list
        seed: list
        presence_penalty: float = 0.0

    original = Params([-2, -2, -2], [False] * 3, [42, None, 13])
    fields = {"sampling_params": original}
    cls._translate_sampling(fields)
    normalized = fields["sampling_params"]
    validate(normalized, 3)
    assert normalized.num_logprobs == [0, 0, 0]
    assert original.num_logprobs == [-2, -2, -2]
    assert normalized.seed == original.seed
    for bad in (
        Params([0], [True], [42]),
        Params([3], [True], [42]),
        Params([-2], [True], [42]),
        Params([-2], [False], [42], presence_penalty=0.5),
    ):
        fields = {"sampling_params": bad}
        cls._translate_sampling(fields)
        try:
            validate(fields["sampling_params"], 1)
        except ValueError:
            pass
        else:
            raise AssertionError("Translation discarded an unsupported feature")
    fields = {"sampling_params": None}
    cls._translate_sampling(fields)
    assert fields["sampling_params"] is None


def test_page_audit_records_logical_growth_from_cpu_shadows():
    cls, _ = adapter_class()
    instance = cls()
    instance.max_batch_size = 2
    instance.generator = SimpleNamespace(
        cache=[
            [
                {"page_table": object(), "_host_pages": torch.tensor([[5, 6, 6, 6]])},
                {"page_table": object(), "_host_pages": torch.tensor([[8, 8, 8, 8]])},
            ]
        ]
    )
    kwargs = dict(
        reload_page_table=True, reload_inputs=False, page_table=torch.tensor([[5, 6, 7, -1], [8, -1, -1, -1]])
    )
    assert instance._page_change_audit(kwargs) == [dict(slot=0, logical_page=2, old=6, new=7)]
    kwargs["reload_inputs"] = True
    assert instance._page_change_audit(kwargs) == []


def test_optional_route_audit_pairs_submissions_without_using_stale_positions():
    cls, _ = adapter_class()
    instance = cls()
    instance.audit_enabled = True
    instance.max_batch_size = 2
    instance.generator = SimpleNamespace(
        counters={"decode": 0},
        active_slots=(1,),
        serving_rows=(0, 1),
        output_pool=SimpleNamespace(leases=[None, None]),
        decode_forward_v1=Mock(return_value=object()),
    )
    instance._event = Mock()
    stale_tokens = object()  # Not inspectable, even when metadata auditing is on.
    stale_positions = torch.tensor([-1, 17])
    kwargs = dict(reload_inputs=False, reload_page_table=False, reset_sampling_state=False)
    instance.decode_forward(stale_tokens, stale_positions, **kwargs)
    instance.decode_forward(stale_tokens, stale_positions, **kwargs)
    events = instance._event.call_args_list
    assert [event.kwargs["submission_id"] for event in events] == [1, 1, 2, 2]
    assert events[0].kwargs["advisory_host_positions"] == [-1, 17]
    assert events[0].kwargs["remembered_rows"] == [0, 1]
    assert events[1].kwargs["active_slots"] == [1]
    instance.generator.decode_forward_v1.assert_called_with(stale_tokens, stale_positions, **kwargs)


def test_optional_prefill_digest_identifies_logical_chunk_and_physical_slot():
    cls, _ = adapter_class()
    instance = cls()
    instance.audit_enabled = True
    instance.uses_mrope = False
    instance.generator = Mock(active_slots=(0,))
    instance._event = Mock()
    tokens = torch.tensor([[999, 10, 11, 777]])
    instance.prefill_forward(tokens, start_pos=[1], prompt_lens=[3], empty_slots=[7])
    assert instance._event.call_args.kwargs["prompt_chunks"] == [
        dict(slot=7, start=1, end=3, token_digest=hashlib.sha256(b"[10,11]").hexdigest())
    ]
