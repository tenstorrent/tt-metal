# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Fast contract checks for the full-model stage (no device required)."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from transformers import AutoConfig

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.generator import Gemma4Generator
from models.autoports.google_gemma_4_26b_a4b_it.tt.model import (
    DEFAULT_MAX_CONTEXT,
    FULL_KIND,
    SLIDING_CACHE_TOKENS,
    Gemma4FullModel,
    PagedCacheSpec,
)
from models.common.readiness_check.contract import Generator


def _config():
    return AutoConfig.from_pretrained(
        "models/demos/gemma4/configs/gemma-4-26B-A4B-it", local_files_only=True
    ).text_config


def test_full_model_architecture_and_cache_geometry_contract():
    config = _config()
    model = Gemma4FullModel.__new__(Gemma4FullModel)
    model.hf_config = config
    model.num_layers = config.num_hidden_layers
    model.layer_indices = list(range(config.num_hidden_layers))
    model.max_seq_len = DEFAULT_MAX_CONTEXT
    specs = model._make_cache_specs()

    assert len(specs) == 30
    assert sum(spec.layer_type == "sliding_attention" for spec in specs) == 25
    assert sum(spec.layer_type == "full_attention" for spec in specs) == 5
    for spec in specs:
        if spec.layer_type == "sliding_attention":
            assert (spec.block_size, spec.local_kv_heads, spec.head_dim) == (64, 2, 256)
            assert spec.capacity_tokens_per_slot == SLIDING_CACHE_TOKENS
        else:
            assert (spec.block_size, spec.local_kv_heads, spec.head_dim) == (128, 1, 512)
            assert spec.capacity_tokens_per_slot == DEFAULT_MAX_CONTEXT


def test_generator_implements_readiness_contract_with_explicit_trace_keyword():
    assert issubclass(Gemma4Generator, Generator)
    assert not Gemma4Generator.__abstractmethods__
    generate = inspect.signature(Gemma4Generator.generate)
    assert "enable_trace" in generate.parameters
    assert generate.parameters["enable_trace"].kind is inspect.Parameter.KEYWORD_ONLY
    for method in (Gemma4Generator.prefill_forward, Gemma4Generator.decode_forward):
        signature = inspect.signature(method)
        assert "page_table" in signature.parameters
        assert "kv_cache" in signature.parameters


def test_optimized_decode_source_has_no_host_logits_or_argmax_boundary():
    model_source = inspect.getsource(Gemma4FullModel.decode_forward)
    trace_source = inspect.getsource(Gemma4Generator._get_or_capture_decode_trace)
    forbidden = ("to_torch", "torch.argmax", ".argmax(", ".cpu(", ".numpy(")
    assert not any(token in model_source for token in forbidden)
    assert not any(token in trace_source for token in forbidden)
    assert "tt_out_tok=token_input" in trace_source
    decode_source = inspect.getsource(Gemma4Generator.decode_forward)
    assert "ttnn.plus_one(trace.current_pos" in decode_source
    assert "ttnn.plus_one(trace.position_ids" in decode_source


def test_sampling_specs_are_validated_and_trace_distinct(expect_error):
    generator = object.__new__(Gemma4Generator)
    greedy = generator._sampling_spec(1, temperature=0.0)
    sampled = generator._sampling_spec(1, top_k=8, top_p=0.95, temperature=0.8, seeds=42)
    assert greedy.greedy
    assert not sampled.greedy
    assert sampled.top_k == (8,)
    assert sampled.top_p == (0.95,)
    assert sampled.temperature == (0.8,)
    assert sampled.seeds == (42,)
    assert greedy.key != sampled.key
    with expect_error(ValueError, "top_k"):
        generator._sampling_spec(1, top_k=33, top_p=0.9, temperature=1.0)
    with expect_error(ValueError, "top_p"):
        generator._sampling_spec(1, top_k=8, top_p=1.1, temperature=1.0)


def test_sampler_choice_is_semantically_greedy_split_topk():
    source = inspect.getsource(Gemma4Generator._sampling_params)
    assert "return None, None, None" in source
    assert "exactly greedy" in source
    assert "max_top_k=32" in inspect.getsource(Gemma4Generator.__init__)
    assert "Sampling1D" in inspect.getsource(Gemma4Generator.__init__)


def test_cache_spec_rounding():
    spec = PagedCacheSpec(0, "full_attention", 128, 1, FULL_KIND.head_dim, 129)
    assert spec.blocks_per_slot == 2


def _load_probe_state(*, full_stack: bool = False):
    roots = Path(os.environ.get("HF_HOME", "/huggingface")) / "hub/models--google--gemma-4-26B-A4B-it/snapshots"
    snapshots = [p for p in roots.iterdir() if (p / "model.safetensors.index.json").is_file()]
    if not snapshots:
        pytest.skip("local Gemma4 snapshot unavailable")
    snapshot = snapshots[0]
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefixes = (
        ("model.language_model.layers.",)
        if full_stack
        else (
            "model.language_model.layers.0.",
            "model.language_model.layers.5.",
        )
    )
    terminal = {"model.language_model.embed_tokens.weight", "model.language_model.norm.weight"}
    wanted = {name for name in weight_map if name.startswith(prefixes) or name in terminal}
    state = {}
    for shard_name in sorted({weight_map[name] for name in wanted}):
        with safe_open(snapshot / shard_name, framework="pt", device="cpu") as shard:
            for name in wanted:
                if weight_map[name] == shard_name:
                    state[name] = shard.get_tensor(name)
    return state


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_reduced_real_weight_full_model_probe(mesh_device):
    if os.environ.get("GEMMA4_FULL_MODEL_PROBE") != "1":
        pytest.skip("set GEMMA4_FULL_MODEL_PROBE=1 for the serialized TP4 hardware probe")
    full_stack = os.environ.get("GEMMA4_FULL_STACK_PROBE") == "1"
    batch32 = os.environ.get("GEMMA4_BATCH32_PROBE") == "1"
    long_context = os.environ.get("GEMMA4_LONG_CONTEXT_PROBE") == "1"
    prompt_len = 262_111 if long_context else 32
    state = _load_probe_state(full_stack=full_stack)
    model = Gemma4FullModel(
        mesh_device=mesh_device,
        hf_config=_config(),
        state_dict=state,
        max_seq_len=262_144 if (batch32 or long_context) else 128,
        max_batch_size=32 if batch32 else 1,
        layer_indices=None if full_stack else [0, 5],
        tensor_cache_path=("/tmp/gemma4_full_model_cache" if full_stack else "/tmp/gemma4_full_model_probe_cache"),
    )
    probe_batch = 32 if batch32 else 1
    # Prefill is intentionally scheduled one slot at a time; the public
    # generator owns mixed-prompt iteration.  Decode then exercises all 32
    # fixed slots together against the same fully allocated state.
    tokens = ttnn.from_torch(
        torch.ones((1, prompt_len), dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    positions = ttnn.from_torch(
        torch.arange(prompt_len, dtype=torch.int32).reshape(1, prompt_len),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    if os.environ.get("GEMMA4_EMBED_TRACE_ONLY") == "1":
        decode_token = ttnn.from_torch(
            torch.ones((1, 1), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        model.embed_tokens(decode_token)
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        model.embed_tokens(decode_token)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.release_trace(mesh_device, trace_id)
        return
    generator = Gemma4Generator(model, tokenizer=None, sampling_mode="device")
    if long_context:
        logits = generator.prefill_forward(
            torch.ones((1, prompt_len), dtype=torch.long),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            prompt_lens=[prompt_len],
        )
    else:
        logits = model.prefill_forward(tokens, state=model.state, prompt_lens=[prompt_len], position_ids=positions)
    assert tuple(logits.shape) == (1, 1, 1, 65_536)

    if os.environ.get("GEMMA4_MODEL_TRACE_ONLY") == "1":
        decode_token = generator._host_tokens_to_device(torch.ones((1, 1), dtype=torch.long), rank4=True)
        current_pos = generator._positions_to_device(torch.tensor([32], dtype=torch.int32))
        position_ids = generator._positions_to_device(torch.tensor([32], dtype=torch.int32), dtype=ttnn.uint32)
        model.decode_forward(decode_token, state=model.state, current_pos=current_pos, position_ids=position_ids)
        ttnn.synchronize_device(mesh_device)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        model.decode_forward(decode_token, state=model.state, current_pos=current_pos, position_ids=position_ids)
        if os.environ.get("GEMMA4_TRACE_PLUS_ONE") == "1":
            ttnn.plus_one(current_pos, skip_negative_entries=True)
            ttnn.plus_one(position_ids, skip_negative_entries=True)
        elif os.environ.get("GEMMA4_TRACE_INPLACE_ADD") == "1":
            ttnn.add(current_pos, 1, output_tensor=current_pos)
            ttnn.add(position_ids, 1, output_tensor=position_ids)
        ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
        ttnn.release_trace(mesh_device, trace_id)
        return
    sampled = generator.decode_forward(
        torch.ones((probe_batch, 1), dtype=torch.long),
        torch.full((probe_batch,), prompt_len, dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        enable_trace=True,
    )
    assert tuple(sampled.shape) == (1, 1, 1, 32)
    assert generator.trace_counters.replays == 1
    trace = next(iter(generator._trace_cache.values()))
    assert int(ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)[0]) == prompt_len + 1

    # A second replay consumes the prior sampled token without a host token
    # refresh and advances both position buffers exactly once.
    if os.environ.get("GEMMA4_FULL_MODEL_DEVICE_PROFILE") == "1":
        from tracy import signpost

        signpost("FULL_MODEL_REDUCED_DECODE")
    generator.decode_forward(
        torch.zeros((probe_batch, 1), dtype=torch.long),
        torch.full((probe_batch,), 999, dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        enable_trace=True,
    )
    if os.environ.get("GEMMA4_FULL_MODEL_DEVICE_PROFILE") == "1":
        ttnn.synchronize_device(mesh_device)
        signpost("FULL_MODEL_REDUCED_DECODE_END")
    assert generator.trace_counters.token_refreshes == 0
    assert int(ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)[0]) == prompt_len + 2

    if os.environ.get("GEMMA4_SAMPLED_TRACE_PROBE") == "1":
        sampled = generator.decode_forward(
            torch.zeros((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), 34, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
            top_k=8,
            top_p=0.95,
            temperature=0.8,
            seeds=42,
        )
        assert tuple(sampled.shape) == (1, 1, 1, 32)
        sampled_traces = [entry for entry in generator._trace_cache.values() if not entry.sampling_spec.greedy]
        assert len(sampled_traces) == 1
        sampled_trace = sampled_traces[0]
        assert sampled_trace.sampling_trace_id is not None
        assert all(tensor is not None for tensor in sampled_trace.sampling_params)
        assert sampled_trace.sampled_tokens.buffer_address() == sampled_trace.token_input.buffer_address()
        greedy_trace_id = trace.model_trace_id
        generator.decode_forward(
            torch.zeros((probe_batch, 1), dtype=torch.long),
            torch.full((probe_batch,), 34, dtype=torch.int32),
            page_table=model.state.page_tables,
            kv_cache=model.state,
            enable_trace=True,
        )
        assert len(generator._trace_cache) == 1
        recaptured_greedy = next(entry for entry in generator._trace_cache.values() if entry.sampling_spec.greedy)
        assert recaptured_greedy.model_trace_id != greedy_trace_id


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_reduced_mixed_prompt_and_inactive_slot_probe(mesh_device):
    if os.environ.get("GEMMA4_MIXED_PROBE") != "1":
        pytest.skip("set GEMMA4_MIXED_PROBE=1 for the serialized TP4 mixed-prompt probe")
    model = Gemma4FullModel(
        mesh_device=mesh_device,
        hf_config=_config(),
        state_dict=_load_probe_state(full_stack=False),
        max_seq_len=128,
        max_batch_size=2,
        layer_indices=[0, 5],
        tensor_cache_path="/tmp/gemma4_full_model_probe_cache",
    )
    generator = Gemma4Generator(model, tokenizer=None, sampling_mode="device")
    tokens = torch.ones((2, 64), dtype=torch.long)
    outputs = generator.prefill_forward(
        tokens,
        page_table=model.state.page_tables,
        kv_cache=model.state,
        prompt_lens=[33, 47],
    )
    assert len(outputs) == 2
    assert all(tuple(output.shape) == (1, 1, 1, 65_536) for output in outputs)
    table_addresses = [table.buffer_address() for table in model.state.page_tables]
    alternate_tables = []
    for table in model.state.page_tables:
        host_table = ttnn.to_torch(ttnn.get_device_tensors(table)[0]).clone()
        host_table[0, 0], host_table[1, 0] = host_table[1, 0].clone(), host_table[0, 0].clone()
        alternate_tables.append(
            ttnn.from_torch(
                host_table,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        )
    sampled = generator.decode_forward(
        torch.ones((2, 1), dtype=torch.long),
        torch.tensor([33, 47], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        active_mask=torch.tensor([True, False]),
        enable_trace=True,
    )
    assert tuple(sampled.shape) == (1, 1, 1, 32)
    trace = next(iter(generator._trace_cache.values()))
    current = ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)
    assert current[:3].tolist() == [34, -1, -1]
    generator.decode_forward(
        torch.zeros((2, 1), dtype=torch.long),
        torch.tensor([999, 999], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        active_mask=torch.tensor([True, False]),
        enable_trace=True,
    )
    current = ttnn.to_torch(ttnn.get_device_tensors(trace.current_pos)[0]).reshape(-1)
    assert current[:3].tolist() == [35, -1, -1]
    assert [table.buffer_address() for table in model.state.page_tables] == table_addresses
    assert generator.trace_counters.token_refreshes == 0
    assert generator.trace_counters.page_table_refreshes == 0

    # A scheduler-boundary mapping change invalidates and recaptures once;
    # subsequent tokens reuse the new mapping without copying tables per token.
    model.state.page_tables = alternate_tables
    generator.decode_forward(
        torch.zeros((2, 1), dtype=torch.long),
        torch.tensor([35, 47], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        active_mask=torch.tensor([True, False]),
        enable_trace=True,
    )
    changed_trace = next(iter(generator._trace_cache.values()))
    assert changed_trace.page_table_ids == tuple(id(table) for table in alternate_tables)
    assert generator.trace_counters.page_table_refreshes == 1
    generator.decode_forward(
        torch.zeros((2, 1), dtype=torch.long),
        torch.tensor([999, 999], dtype=torch.int32),
        page_table=model.state.page_tables,
        kv_cache=model.state,
        active_mask=torch.tensor([True, False]),
        enable_trace=True,
    )
    assert generator.trace_counters.page_table_refreshes == 1
