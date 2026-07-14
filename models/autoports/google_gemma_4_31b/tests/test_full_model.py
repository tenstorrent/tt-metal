# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch
from tracy import signpost

import ttnn
from models.autoports.google_gemma_4_31b.tt.generator import Gemma4GreedyTP4Sampler, build_generator
from models.autoports.google_gemma_4_31b.tt.model import (
    ROPE_POSITION_INACTIVE_SENTINEL,
    Gemma4FullModel,
    Gemma4FullModelConfig,
)
from models.common.modules.sampling.sampling_1d import Sampling1D, Sampling1DConfig
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.readiness_check.schema import load_reference

MODEL_DIR = Path("models/autoports/google_gemma_4_31b")
REFERENCE_PATH = MODEL_DIR / "doc/full_model/readiness_aime24_plain.refpt"


def _replicated_device_tensor(mesh, source, dtype):
    return ttnn.from_torch(
        source,
        device=mesh,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


@pytest.mark.timeout(600)
def test_tp_vocab_row_materialization_and_sampler_boundaries():
    if os.environ.get("GEMMA4_31B_FULL_MODEL_RUN_SAMPLER_BOUNDARY") != "1":
        pytest.skip("set GEMMA4_31B_FULL_MODEL_RUN_SAMPLER_BOUNDARY=1 to run the isolated sampler probe")

    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    try:
        winners = {0: 0, 1: 32_767, 2: 32_768, 3: 65_535, 17: 65_536, 31: 262_143}
        host = torch.full((1, 1, 32, 262_144), -100.0, dtype=torch.bfloat16)
        for row, token in winners.items():
            host[0, 0, row, token] = 100.0
        # BF16 softcap plateaus occur in real Gemma logits. Semantic greedy
        # must match torch.argmax and choose the lower token ID on a tie.
        host[0, 0, 7, 177] = 100.0
        host[0, 0, 7, 192] = 100.0
        source = ttnn.from_torch(
            host,
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )
        outputs = []
        source_addresses = [tensor.buffer_address() for tensor in ttnn.get_device_tensors(source)]
        for row in winners:
            view = ttnn.slice(
                source,
                [0, 0, row, 0],
                [1, 1, row + 1, 65_536],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            output = ttnn.clone(view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            assert all(
                tensor.buffer_address() != source_address
                for tensor, source_address in zip(ttnn.get_device_tensors(output), source_addresses)
            )
            outputs.append(output)
        tie_view = ttnn.slice(
            source,
            [0, 0, 7, 0],
            [1, 1, 8, 65_536],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tie_output = ttnn.clone(tie_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        batch2_view = ttnn.slice(
            source,
            [0, 0, 0, 0],
            [1, 1, 2, 65_536],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        batch2_output = ttnn.clone(batch2_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        source.deallocate(True)
        ttnn.synchronize_device(mesh)
        assert all(tensor.is_allocated() for output in outputs for tensor in ttnn.get_device_tensors(output))

        common = dict(
            vocab_size=262_144,
            mesh_device=mesh,
            max_batch_size=1,
            max_top_k=32,
            pad_to_power_of_2=False,
            ag_topology=ttnn.Topology.Linear,
            num_gather_links=2,
            sampling_cluster_axis=1,
            use_broadcast_all_gather=True,
            gather_values_dtype=ttnn.float32,
        )
        sampler = Sampling1D.from_config(Sampling1DConfig(**common, allow_force_argmax=False))
        sampler.load_device_buffers()
        run_rejected_force_argmax = os.environ.get("GEMMA4_31B_FULL_MODEL_RUN_REJECTED_FORCE_ARGMAX") == "1"
        argmax_sampler = None
        if run_rejected_force_argmax:
            argmax_sampler = Sampling1D.from_config(Sampling1DConfig(**common, allow_force_argmax=True))
            argmax_sampler.load_device_buffers()
        assert sampler._local_indices.dtype == ttnn.uint32
        local_values, local_indices = ttnn.topk(outputs[0], k=32, dim=-1, indices_tensor=sampler._local_indices)
        assert local_indices.dtype == ttnn.uint32
        first_values = ttnn.to_torch(ttnn.get_device_tensors(local_values)[0]).reshape(-1)
        first_indices = ttnn.to_torch(ttnn.get_device_tensors(local_indices)[0]).reshape(-1)
        assert int(first_indices[int(torch.argmax(first_values))]) == 0
        local_values.deallocate(True)
        local_indices.deallocate(True)
        gathered_values, gathered_indices = sampler._topk(outputs[0])
        assert gathered_values.layout == ttnn.TILE_LAYOUT
        assert gathered_indices.layout == ttnn.TILE_LAYOUT
        assert tuple(gathered_values.shape) == (1, 1, 1, 128)
        assert tuple(gathered_indices.shape) == (1, 1, 1, 128)
        assert gathered_values.dtype == ttnn.bfloat16
        assert gathered_indices.dtype == ttnn.uint32
        candidate_values = ttnn.to_torch(ttnn.get_device_tensors(gathered_values)[0]).reshape(-1)
        candidate_indices = ttnn.to_torch(ttnn.get_device_tensors(gathered_indices)[0]).reshape(-1)
        winner_slot = int(torch.argmax(candidate_values))
        assert int(candidate_indices[winner_slot]) == 0
        candidate_indices_i32 = ttnn.typecast(gathered_indices, dtype=ttnn.int32)
        global_indices = ttnn.add(sampler._index_offsets, candidate_indices_i32, dtype=ttnn.int32)
        global_host = ttnn.to_torch(ttnn.get_device_tensors(global_indices)[0]).reshape(-1)
        assert int(global_host[winner_slot]) == 0
        candidate_indices_i32.deallocate(True)
        global_indices.deallocate(True)
        gathered_values.deallocate(True)
        gathered_indices.deallocate(True)
        k = _replicated_device_tensor(mesh, torch.tensor([1], dtype=torch.int32), ttnn.uint32)
        p = _replicated_device_tensor(mesh, torch.tensor([0.0], dtype=torch.bfloat16), ttnn.bfloat16)
        temp = _replicated_device_tensor(mesh, torch.tensor([1.0], dtype=torch.bfloat16), ttnn.bfloat16)

        topk_results = []
        force_argmax_results = []
        for output, expected in zip(outputs, winners.values()):
            topk_out = _replicated_device_tensor(mesh, torch.zeros((1, 1, 1, 1), dtype=torch.int32), ttnn.uint32)
            sampler.decode_forward(output, k=k, p=p, temp=temp, tt_out_tok=topk_out)
            ttnn.synchronize_device(mesh)
            topk_value = int(ttnn.to_torch(ttnn.get_device_tensors(topk_out)[0]).reshape(-1)[0])
            assert topk_value == expected
            topk_results.append(topk_value)
            topk_out.deallocate(True)
            if run_rejected_force_argmax:
                argmax_out = _replicated_device_tensor(mesh, torch.zeros((1, 1, 1, 1), dtype=torch.int32), ttnn.uint32)
                assert argmax_sampler is not None
                argmax_sampler.decode_forward(output, tt_out_tok=argmax_out)
                ttnn.synchronize_device(mesh)
                force_argmax_results.append(int(ttnn.to_torch(ttnn.get_device_tensors(argmax_out)[0]).reshape(-1)[0]))
                argmax_out.deallocate(True)

        # Keep the common candidate-reduction path as comparison evidence.
        assert topk_results == list(winners.values())
        # Full-logits force-argmax is a measured rejected alternative and may poison watcher
        # state. Re-run it only via its dedicated opt-in when collecting rejection evidence.
        if run_rejected_force_argmax:
            assert force_argmax_results != list(winners.values())
            assert force_argmax_results[1] != 65_535

        greedy = Gemma4GreedyTP4Sampler(
            mesh_device=mesh,
            vocab_per_device=65_536,
            max_batch_size=1,
        )
        greedy_results = []
        for output, expected in zip(outputs, winners.values()):
            greedy_out = _replicated_device_tensor(mesh, torch.zeros((1, 1, 1, 1), dtype=torch.int32), ttnn.uint32)
            greedy.decode_forward(output, tt_out_tok=greedy_out)
            ttnn.synchronize_device(mesh)
            value = int(ttnn.to_torch(ttnn.get_device_tensors(greedy_out)[0]).reshape(-1)[0])
            assert value == expected
            greedy_results.append(value)
            greedy_out.deallocate(True)
        assert greedy_results == list(winners.values())

        tie_out = _replicated_device_tensor(mesh, torch.zeros((1, 1, 1, 1), dtype=torch.int32), ttnn.uint32)
        greedy.decode_forward(tie_output, tt_out_tok=tie_out)
        ttnn.synchronize_device(mesh)
        assert int(ttnn.to_torch(ttnn.get_device_tensors(tie_out)[0]).reshape(-1)[0]) == 177
        tie_out.deallocate(True)

        greedy_batch2 = Gemma4GreedyTP4Sampler(
            mesh_device=mesh,
            vocab_per_device=65_536,
            max_batch_size=2,
        )
        batch2_out = _replicated_device_tensor(mesh, torch.zeros((1, 1, 1, 2), dtype=torch.int32), ttnn.uint32)
        greedy_batch2.decode_forward(batch2_output, tt_out_tok=batch2_out)
        ttnn.synchronize_device(mesh)
        assert ttnn.to_torch(ttnn.get_device_tensors(batch2_out)[0]).reshape(-1)[:2].tolist() == [0, 32_767]
        batch2_out.deallocate(True)

        trace_output = _replicated_device_tensor(mesh, torch.zeros((1, 1, 1, 1), dtype=torch.int32), ttnn.uint32)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        greedy.decode_forward(outputs[-1], tt_out_tok=trace_output)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        for _ in range(3):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        traced_value = int(ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0]).reshape(-1)[0])
        ttnn.release_trace(mesh, trace_id)
        assert traced_value == 262_143
    finally:
        close_readiness_mesh_device(mesh, "FABRIC_1D")


@pytest.mark.timeout(900)
def test_reduced_embedding_layout_boundary():
    if os.environ.get("GEMMA4_31B_FULL_MODEL_RUN_EMBED_BOUNDARY") != "1":
        pytest.skip("set GEMMA4_31B_FULL_MODEL_RUN_EMBED_BOUNDARY=1 to run the isolated boundary probe")

    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    model = None
    try:
        model = Gemma4FullModel.from_pretrained(
            mesh_device=mesh,
            config=Gemma4FullModelConfig(max_seq_len=128, layer_indices=(0,)),
            tensor_cache_path=Path("/tmp/gemma4_31b_full_model_tensor_cache"),
        )
        for tokens, mode, expected_m in (
            (torch.arange(33, dtype=torch.long).reshape(1, 33), "prefill", 33),
            (torch.tensor([[17]], dtype=torch.long), "decode", 1),
        ):
            hidden = model.embed_tokens(tokens, mode=mode)
            assert hidden.layout == ttnn.TILE_LAYOUT
            assert tuple(hidden.shape) == (1, 1, expected_m, model.hidden_size)
            normalized = model.layers[0].layer.input_layernorm.forward(hidden)
            ttnn.synchronize_device(mesh)
            normalized.deallocate(True)
            hidden.deallocate(True)
    finally:
        if model is not None:
            model.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")


@pytest.mark.timeout(900)
def test_reduced_short_prompt_repeated_generate():
    if os.environ.get("GEMMA4_31B_FULL_MODEL_RUN_SHORT_PREFILL") != "1":
        pytest.skip("set GEMMA4_31B_FULL_MODEL_RUN_SHORT_PREFILL=1 to run the short-prompt ownership probe")

    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    try:
        generator = build_generator(
            MODEL_DIR,
            mesh,
            model_config=Gemma4FullModelConfig(
                max_seq_len=128,
                layer_indices=(0, 5),
                max_batch_size=1,
            ),
            cache_context=128,
            tensor_cache_path=Path("/tmp/gemma4_31b_full_model_tensor_cache"),
        )
        prompts = (
            "Write a haiku about machine learning.",
            "What are the three laws of thermodynamics?",
        )
        for prompt_text in prompts:
            prompt = generator.tokenizer.encode(prompt_text, add_special_tokens=True)
            assert len(prompt) <= 32
            generated = generator.generate(prompt, 4, enable_trace=True)
            assert len(generated) == 4
            assert all(0 <= token < generator.model.vocab_size for token in generated)
        assert generator.model.trace_state.counters["model_trace_replays"] == 3
        assert generator.model.trace_state.counters["full_logits_readbacks"] == 0
        assert generator.model.trace_state.counters["token_host_refreshes"] == 0
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")


@pytest.mark.timeout(1800)
def test_reduced_full_model_prefill_split_greedy_and_trace():
    if os.environ.get("GEMMA4_31B_FULL_MODEL_RUN_REDUCED") != "1":
        pytest.skip("set GEMMA4_31B_FULL_MODEL_RUN_REDUCED=1 to run the two-kind full-model probe")

    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    try:
        generator = build_generator(
            MODEL_DIR,
            mesh,
            model_config=Gemma4FullModelConfig(
                max_seq_len=128,
                layer_indices=(0, 5),
                max_batch_size=2,
            ),
            cache_context=128,
            tensor_cache_path=Path("/tmp/gemma4_31b_full_model_tensor_cache"),
        )
        mixed_prompts = [
            generator.tokenizer.encode("A short proof that 1 + 1 = 2 is", add_special_tokens=True),
            generator.tokenizer.encode("The next prime number is", add_special_tokens=True),
        ]
        mixed_lengths = [33, 17]
        for row, length in enumerate(mixed_lengths):
            repeats = (length + len(mixed_prompts[row]) - 1) // len(mixed_prompts[row])
            mixed_prompts[row] = (mixed_prompts[row] * repeats)[:length]
        mixed_input = torch.zeros((2, max(mixed_lengths)), dtype=torch.long)
        for row, prompt_tokens in enumerate(mixed_prompts):
            mixed_input[row, : len(prompt_tokens)] = torch.tensor(prompt_tokens)
        mixed_logits = generator.prefill_forward(
            mixed_input,
            page_table=generator.page_tables,
            kv_cache=generator.kv_cache,
            prompt_lens=mixed_lengths,
            return_device_logits=True,
        )
        assert mixed_logits.shape[-2] == 2
        mixed_output_buffer = generator._new_token_buffer(2)
        mixed_tokens_out, _ = generator._sample_eager(mixed_logits, tt_out_tok=mixed_output_buffer)
        sampled_rows = ttnn.to_torch(ttnn.get_device_tensors(mixed_tokens_out)[0]).reshape(-1)[:2]
        assert all(0 <= int(token) < generator.model.vocab_size for token in sampled_rows)
        mixed_logits.deallocate(True)
        mixed_output_buffer.deallocate(True)
        generator.reset()

        prompt = generator.tokenizer.encode("A short proof that 1 + 1 = 2 is", add_special_tokens=True)
        prompt = (prompt * ((33 + len(prompt) - 1) // len(prompt)))[:33]
        tokens = torch.tensor([prompt], dtype=torch.long)

        device_logits = generator.prefill_forward(
            tokens,
            page_table=generator.page_tables,
            kv_cache=generator.kv_cache,
            prompt_lens=[33],
            return_device_logits=True,
        )
        topk_buffer = generator._new_token_buffer(1)
        topk_token, _ = generator._sample_eager(device_logits, tt_out_tok=topk_buffer)
        ttnn.synchronize_device(mesh)
        topk_value = generator.read_sampled_token(topk_token)
        host_argmax_value = int(torch.argmax(generator.model.logits_to_torch(device_logits).reshape(-1)).item())
        device_logits.deallocate(True)
        topk_buffer.deallocate(True)
        assert topk_value == host_argmax_value

        alternate_page_tables = [ttnn.clone(table) for table in generator.page_tables]
        generator.reset()
        generated = generator.generate(prompt, 4, enable_trace=True)
        assert len(generated) == 4
        assert all(0 <= token < generator.model.vocab_size for token in generated)
        counters = generator.model.trace_state.counters
        assert counters["model_trace_replays"] == 3
        assert counters["full_logits_readbacks"] == 0
        assert counters["token_host_refreshes"] == 0
        assert counters["page_table_refreshes"] == 0
        assert generator._sampling_trace_output[0] is generator.model.trace_state.token_input
        assert int(generator.model.trace_state.logits.shape[-2]) == 2
        assert tuple(generator.model.trace_state.token_input.shape) == (1, 1, 1, 2)
        assert generator._sampling_params is None
        assert generator._sampling_trace_key == (2, 1, 0.0, 1.0, False)
        cache_position = ttnn.to_torch(ttnn.get_device_tensors(generator.model.trace_state.cache_position)[0]).reshape(
            -1
        )
        rope_position = ttnn.to_torch(ttnn.get_device_tensors(generator.model.trace_state.rope_position)[0]).reshape(-1)
        assert int(cache_position[0]) == len(prompt) + counters["model_trace_replays"]
        assert int(rope_position[0]) == len(prompt) + counters["model_trace_replays"]
        assert int(cache_position[1]) == -1
        assert int(rope_position[1]) & 0xFFFF_FFFF == ROPE_POSITION_INACTIVE_SENTINEL
        assert generator.model.trace_state.active_batch_size == 1
        assert generator.model.trace_state.prompt_lengths == (len(prompt), 0)

        before = dict(counters)
        generator.decode_next_token_traced()
        ttnn.synchronize_device(mesh)
        after = dict(generator.model.trace_state.counters)
        assert after["page_table_refreshes"] == before["page_table_refreshes"]
        assert after["token_host_refreshes"] == before["token_host_refreshes"]
        assert after["position_host_refreshes"] == before["position_host_refreshes"]

        generator.decode_next_token_traced(
            page_table=alternate_page_tables,
            kv_cache=generator.kv_cache,
            page_table_generations=[1] * len(alternate_page_tables),
        )
        changed = dict(generator.model.trace_state.counters)
        expected_refreshes = len(
            {
                (id(source), id(target))
                for source, target in zip(alternate_page_tables, generator.model.trace_state.page_tables)
            }
        )
        assert changed["page_table_refreshes"] == after["page_table_refreshes"] + expected_refreshes
        generator.decode_next_token_traced(
            page_table=alternate_page_tables,
            kv_cache=generator.kv_cache,
            page_table_generations=[1] * len(alternate_page_tables),
        )
        assert generator.model.trace_state.counters["page_table_refreshes"] == changed["page_table_refreshes"]

        compatibility = generator.generate(
            prompt,
            2,
            enable_trace=False,
            host_sampling_compat=True,
        )
        assert len(compatibility) == 2
        assert generator.model.trace_state.trace_id is None
        assert generator.model.trace_state.counters["full_logits_readbacks"] == 2
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")


@pytest.mark.timeout(1800)
def test_reduced_full_model_token_out_perf_signposts():
    if os.environ.get("GEMMA4_31B_FULL_MODEL_RUN_REDUCED_PERF") != "1":
        pytest.skip("set GEMMA4_31B_FULL_MODEL_RUN_REDUCED_PERF=1 to profile the reduced token-out path")

    max_new_tokens = int(os.environ.get("GEMMA4_31B_FULL_MODEL_REDUCED_PERF_TOKENS", "6"))
    if max_new_tokens < 3:
        raise ValueError("reduced token-out profiler requires at least three generated tokens")
    output_path = Path(
        os.environ.get(
            "GEMMA4_31B_FULL_MODEL_REDUCED_PERF_OUT",
            MODEL_DIR / "doc/full_model/reduced_token_out_perf.json",
        )
    )
    reference = load_reference(REFERENCE_PATH)
    prompt = reference.entries[0].prompt_tokens.reshape(-1).tolist()

    mesh = open_readiness_mesh_device("P150_X4", "FABRIC_1D")
    generator = None
    try:
        generator = build_generator(
            MODEL_DIR,
            mesh,
            model_config=Gemma4FullModelConfig(layer_indices=(0, 5), max_batch_size=1),
            tensor_cache_path=Path("/tmp/gemma4_31b_full_model_tensor_cache"),
        )
        generator.reset()
        start = time.perf_counter()
        logits = generator.prefill_forward(
            torch.tensor([prompt], dtype=torch.long),
            page_table=generator.page_tables,
            kv_cache=generator.kv_cache,
            prompt_lens=[len(prompt)],
            return_device_logits=True,
        )
        first_buffer = generator._new_token_buffer(1)
        first_output, _ = generator._sample_eager(logits, tt_out_tok=first_buffer)
        first_token = generator.read_sampled_token(first_output)
        logits.deallocate(True)
        first_buffer.deallocate(True)
        ttft = time.perf_counter()

        generator.prepare_token_out_decode(first_input_tokens=[first_token], start_positions=[len(prompt)])
        # The profiler records model construction, prefill, warmup, and both
        # trace captures before the signposted steady window. Flush that setup
        # traffic so the finite device buffers retain every selected replay.
        ttnn.ReadDeviceProfiler(mesh)
        steady_start = time.perf_counter()
        signpost("GEMMA4_FULL_MODEL_TOKEN_OUT_STEADY")
        for _ in range(max_new_tokens - 2):
            generator.decode_next_token_traced()
        ttnn.synchronize_device(mesh)
        signpost("GEMMA4_FULL_MODEL_TOKEN_OUT_STEADY_END")
        end = time.perf_counter()

        result = {
            "workload": {
                "layers": [0, 5],
                "prompt_len": len(prompt),
                "gen_len": max_new_tokens,
                "batch": 1,
                "context": generator.cache_context,
            },
            "ttft_ms": (ttft - start) * 1000.0,
            "prepare_decode_ms": (steady_start - ttft) * 1000.0,
            "decode_t/s/u": (max_new_tokens - 1) / max(end - ttft, 1.0e-9),
            "steady_decode_t/s/u": (max_new_tokens - 2) / max(end - steady_start, 1.0e-9),
            "trace_counters": dict(generator.model.trace_state.counters),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        assert result["trace_counters"]["model_trace_replays"] == max_new_tokens - 1
        assert result["trace_counters"]["token_host_refreshes"] == 0
        assert result["trace_counters"]["page_table_refreshes"] == 0
        assert result["trace_counters"]["full_logits_readbacks"] == 0
        assert result["trace_counters"]["sampled_token_readbacks"] == 1
        assert generator._sampling_trace_output[0] is generator.model.trace_state.token_input
    finally:
        if generator is not None:
            generator.teardown()
        close_readiness_mesh_device(mesh, "FABRIC_1D")
