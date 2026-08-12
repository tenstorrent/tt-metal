# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.generator import (
    MistralSmall24BGenerator,
    SafetensorStateDict,
    _first_device_to_torch,
)
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.model import (
    HF_CONTEXT_LENGTH,
    HF_VOCAB_SIZE,
    FullModelConfig,
    MistralSmall24BFullModel,
)
from models.autoports.mistralai_mistral_small_24b_instruct_2501.tt.multichip_decoder import MultichipDecoder
from models.common.readiness_check.contract import Generator
from models.common.sampling import SamplingGenerator, SamplingParams, format_sampling_params

REDUCED_REAL_ENV = "MISTRAL_SMALL_24B_FULL_MODEL_REDUCED_REAL"
MESH_PARAMS = [(1, 4)]
DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200000000}]
MODEL_DIR = Path(__file__).resolve().parents[1]


def _hf_config():
    return SimpleNamespace(
        max_position_embeddings=HF_CONTEXT_LENGTH,
        hidden_size=5120,
        num_attention_heads=32,
        num_key_value_heads=8,
        vocab_size=HF_VOCAB_SIZE,
        num_hidden_layers=40,
    )


def test_full_model_policy_preserves_accepted_decoder_defaults():
    config = FullModelConfig()
    config.validate(_hf_config())
    assert config.max_context_len == 32768
    assert config.kv_cache_dtype == ttnn.bfloat8_b
    assert config.lm_head_weight_dtype == ttnn.bfloat16
    assert config.num_blocks == 1024

    signature = inspect.signature(MultichipDecoder.from_state_dict)
    assert signature.parameters["mlp_weight_dtype"].default == ttnn.bfloat4_b
    assert signature.parameters["attention_weight_dtype"].default == ttnn.bfloat4_b
    assert signature.parameters["mlp_math_fidelity"].default == ttnn.MathFidelity.LoFi
    assert signature.parameters["attention_math_fidelity"].default == ttnn.MathFidelity.LoFi
    assert signature.parameters["collective_family"].default == "persistent"
    assert signature.parameters["collective_dtype"].default == "bfp8"
    assert signature.parameters["prefill_collective_dtype"].default == "bf16"


def test_full_context_capacity_accounting_includes_split_trace_region():
    contract = json.loads((MODEL_DIR / "doc/context_contract.json").read_text())
    plan = contract["multichip_decoder_plan"]
    steady_state = (
        plan["per_device_weights_plus_kv_bytes_at_advertised_context"]
        + plan["per_device_shared_rope_bytes_at_advertised_context"]
        + plan["per_device_shared_position_index_bytes_at_advertised_context"]
        + plan["per_device_page_table_bytes_at_advertised_context"]
        + plan["per_device_persistent_decode_position_bytes"]
        + plan["per_device_sampling_static_bytes"]
        + plan["per_device_minimum_trace_terminal_bytes"]
        + plan["per_device_reserved_trace_region_bytes"]
    )
    assert contract["current_supported_context"] == contract["hf_advertised_context"] == 32768
    assert plan["supported_active_batch_range"] == {"minimum": 1, "maximum": 32}
    assert plan["tested_active_batches"] == [1, 3, 32]
    assert plan["active_batch_contract"]["tested_batches"] == [1, 3, 32]
    position_contract = plan["trace_position_contract"]
    assert position_contract["cache_sdpa_position"]["persistent_tensor_dtype"] == "INT32"
    assert position_contract["cache_sdpa_position"]["inactive_row_value"] == -1
    assert position_contract["rope_position"]["persistent_tensor_dtype"] == "UINT32"
    assert position_contract["rope_position"]["inactive_row_value"] == 0
    assert plan["per_device_persistent_decode_position_bytes"] == 2 * 32 * 4
    assert plan["per_device_prefill_matrix_stack_weight_bytes"] == 3_099_852_800
    assert plan["per_device_reserved_trace_region_bytes"] == 8 * plan["trace_region_bytes_per_dram_bank"]
    assert plan["capacity_probe_result"]["prefill_weights_released_before_decode"] is False
    assert plan["capacity_probe_result"]["resident_prefill_matrix_layers"] == 40
    assert "batches 2 through 31" in plan["active_batch_contract"]["tile_padding_policy"]
    assert steady_state == plan["per_device_full_stack_steady_state_bytes_at_advertised_context"]
    headroom = contract["capacity_evidence"]["total_device_dram_bytes"] - steady_state
    assert headroom == plan["per_device_dram_headroom_bytes_before_runtime_reserve"]
    assert (
        headroom - plan["runtime_reserve_bytes"]
        == plan["per_device_margin_after_runtime_reserve_at_advertised_context"]
    )
    fixed_bytes = (
        plan["per_device_estimated_full_model_weight_bytes"]
        + plan["per_device_persistent_decode_position_bytes"]
        + plan["per_device_sampling_static_bytes"]
        + plan["per_device_minimum_trace_terminal_bytes"]
        + plan["per_device_reserved_trace_region_bytes"]
        + plan["runtime_reserve_bytes"]
    )
    physical_ceiling = (
        (contract["capacity_evidence"]["total_device_dram_bytes"] - fixed_bytes)
        // plan["per_device_variable_bytes_per_logical_token"]
        // 32
        * 32
    )
    assert physical_ceiling == plan["calculated_context_ceiling_with_reserve"] == 34464
    claim = plan["claim"]
    assert f'{plan["per_device_margin_after_runtime_reserve_at_advertised_context"]:,}' in claim
    assert "1.5 GiB runtime reserve" in claim


def test_readiness_and_explicit_state_signatures():
    assert issubclass(MistralSmall24BGenerator, Generator)
    generate = inspect.signature(MistralSmall24BGenerator.generate)
    assert generate.parameters["enable_trace"].default is True
    assert generate.parameters["sampling_mode"].default == "device"

    prefill = inspect.signature(MistralSmall24BGenerator.prefill_forward)
    assert "page_table" in prefill.parameters
    assert "kv_cache" in prefill.parameters
    assert "prompt_lens" in prefill.parameters
    assert "return_all_logits" in prefill.parameters

    decode = inspect.signature(MistralSmall24BGenerator.decode_forward)
    assert "start_pos" in decode.parameters
    assert "page_table" in decode.parameters
    assert "kv_cache" in decode.parameters


def test_fixed_slot_page_table_preserves_active_mapping_and_fills_inactive_rows():
    generator = object.__new__(MistralSmall24BGenerator)
    generator.batch = 3
    generator.blocks_per_slot = 4
    generator.model = SimpleNamespace(config=SimpleNamespace(num_blocks=12))

    caller = torch.arange(4, dtype=torch.int32).reshape(1, 4)
    actual = generator._normalise_page_table(caller, active_batch=1)
    expected = torch.tensor(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(actual, expected)

    caller = torch.tensor([[8, 9, 10, 11]], dtype=torch.int32)
    actual = generator._normalise_page_table(caller, active_batch=1)
    assert torch.equal(actual[0], caller[0])
    assert actual.unique().numel() == 12


def test_decode_separates_signed_cache_positions_from_rope_positions():
    signature = inspect.signature(MultichipDecoder.decode_forward)
    assert "current_pos_tensor" in signature.parameters
    assert "rotary_pos_tensor" in signature.parameters

    source = inspect.getsource(MistralSmall24BFullModel.decode_forward)
    assert "skip_negative_entries=True" in source
    assert "ttnn.plus_one(rotary_position)" in source


def test_selected_split_sampler_and_trace_feedback_are_canonical():
    sampler_source = inspect.getsource(MistralSmall24BFullModel.sample_split)
    assert "self.sampler.decode_forward" in sampler_source
    assert "k=k" in sampler_source
    assert "p=p" in sampler_source
    assert "temp=temp" in sampler_source
    assert "tt_out_tok=tt_out_tok" in sampler_source

    replay_source = inspect.getsource(MistralSmall24BGenerator._replay_split_sampling)
    model_replay = replay_source.index("self._trace_model_id")
    sampling_replay = replay_source.index("self._trace_sampling_id")
    assert model_replay < sampling_replay


def test_optimized_generate_has_no_host_argmax_or_full_logit_feedback():
    source = inspect.getsource(MistralSmall24BGenerator.generate)
    assert ".argmax(" not in source
    assert "_local_logits_to_torch" not in source
    assert "_replay_split_sampling" in source
    assert "_copy_forced_tokens" in source
    assert '"token_feedback": "host_forced" if next_input is not None else "device"' in source
    assert '"final_device_positions"' in source
    assert '"sampled_token_is_feedback_buffer"' in source

    reset_source = inspect.getsource(MistralSmall24BGenerator.reset)
    assert "reset_kv_cache" in reset_source
    assert "_release_decode_traces" not in reset_source


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_reduced_real_shape_full_model_and_split_trace(mesh_device):
    snapshot_text = os.environ.get(REDUCED_REAL_ENV)
    if not snapshot_text:
        pytest.skip(f"Set {REDUCED_REAL_ENV} to the complete local HF snapshot")
    ttnn.CONFIG.throw_exception_on_fallback = True
    print("FULL_MODEL_RUNTIME_FALLBACK_POLICY throw_exception_on_fallback=true")
    snapshot = os.fspath(snapshot_text)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    model = MistralSmall24BFullModel.from_state_dict(
        SafetensorStateDict(snapshot),
        hf_config=config,
        mesh_device=mesh_device,
        full_model_config=FullModelConfig(
            max_batch_size=1,
            max_context_len=128,
            num_blocks=4,
            prefill_chunk_size=32,
            override_num_layers=1,
        ),
    )
    tokenizer = SimpleNamespace(eos_token_id=2)
    generator = MistralSmall24BGenerator(model, tokenizer)
    prompt = [1, 101, 202, 303, 404, 505, 606]

    # Exercise non-aligned prefill and the explicit host-logit compatibility
    # boundary before entering the optimized token-out path.
    logits = generator.prefill_forward(
        torch.tensor([prompt], dtype=torch.long),
        page_table=torch.arange(4, dtype=torch.int32).reshape(1, 4),
        kv_cache=None,
        prompt_lens=[len(prompt)],
    )
    assert tuple(logits.shape) == (1, 1, config.vocab_size)
    generator.reset()
    repeated_logits = generator.prefill_forward(
        torch.tensor([prompt], dtype=torch.long),
        page_table=torch.arange(4, dtype=torch.int32).reshape(1, 4),
        kv_cache=None,
        prompt_lens=[len(prompt)],
    )
    assert torch.equal(logits, repeated_logits)
    print("FULL_MODEL_LOGIT_DETERMINISM repeated_prompt=true exact_logits=true")

    generator.reset()
    host_outputs = generator.generate(prompt, 2, enable_trace=False, sampling_mode="host")
    assert len(host_outputs) == 2
    generator.reset()
    full_logit_readbacks_before_optimized = generator.trace_stats["full_logit_readbacks"]
    page_device = generator._page_table_device(generator._make_page_table())
    token_device = generator._tokens_device(generator._decode_token_host(torch.tensor([707])))
    current_pos, rotary_pos = generator._positions_device(torch.tensor([len(prompt)]))
    device_logits = model.decode_forward(
        token_device,
        current_pos,
        rotary_pos,
        page_table=page_device,
        kv_cache=generator._ensure_kv_cache(),
        advance_positions=False,
    )
    split_token = generator._device_tt(
        torch.zeros((1, 1, 1, 32), dtype=torch.int32),
        dtype=ttnn.uint32,
    )
    alternative_token = generator._device_tt(
        torch.zeros((1, 1, 1, 32), dtype=torch.int32),
        dtype=ttnn.uint32,
    )
    alternative_args = SimpleNamespace(
        vocab_size=config.vocab_size,
        padded_vocab_size=config.vocab_size,
        max_batch_size=32,
        max_top_k=32,
        sampling_dp=1,
        cluster_shape=(1, 4),
        sampling_all_gather_axis=1,
        pad_logits_to_power_of_2=True,
        sub_core_grids=None,
        model_config={
            "SAMPLING_AG_CONFIG": {
                "allow_force_argmax": True,
                "num_links": 2,
                "topology": ttnn.Topology.Linear,
            }
        },
    )
    alternative = SamplingGenerator(
        args=alternative_args,
        mesh_device=mesh_device,
        tt_ccl=model.tt_ccl,
    )
    alternative_params = format_sampling_params(
        SamplingParams(
            temperature=0.0,
            top_k=1,
            top_p=1.0,
            seed=None,
            enable_log_probs=False,
        ),
        max_batch_size=32,
    )
    alternative.apply_prefill_state(
        sampling_params=alternative_params,
        prompt_tokens=torch.tensor([prompt], dtype=torch.long),
        empty_slots=[0],
    )
    assert alternative.tt_sampling.force_argmax_sampling
    assert not alternative._penalties_active
    assert not alternative._log_probs_active
    assert not alternative.seed_manager.has_active_request_seed()

    model.sampler.load_device_buffers()
    model.sample_split(
        device_logits,
        k=generator._sampling_k,
        p=generator._sampling_p,
        temp=generator._sampling_temp,
        tt_out_tok=split_token,
    )
    alternative_output = alternative.sample(
        device_logits,
        enable_trace=True,
        tt_out_tok=alternative_token,
    )
    alternative.sample(
        device_logits,
        enable_trace=True,
        tt_out_tok=alternative_token,
    )
    ttnn.synchronize_device(mesh_device)
    split_host = _first_device_to_torch(split_token).reshape(-1)
    alternative_host = _first_device_to_torch(alternative_token).reshape(-1)
    assert alternative_output[0] is alternative_token
    assert len(alternative._trace_states) == 1
    assert int(split_host[0]) == int(alternative_host[0])
    alternative.reset_trace()

    iterations = 5
    start = time.perf_counter()
    for _ in range(iterations):
        model.sample_split(
            device_logits,
            k=generator._sampling_k,
            p=generator._sampling_p,
            temp=generator._sampling_temp,
            tt_out_tok=split_token,
        )
    ttnn.synchronize_device(mesh_device)
    split_ms = 1000.0 * (time.perf_counter() - start) / iterations
    start = time.perf_counter()
    for _ in range(iterations):
        alternative.sample(
            device_logits,
            enable_trace=False,
            tt_out_tok=alternative_token,
        )
    ttnn.synchronize_device(mesh_device)
    alternative_ms = 1000.0 * (time.perf_counter() - start) / iterations
    print(
        f"FULL_MODEL_COMMON_SAMPLER_COMPARISON sampling1d_split_greedy_ms={split_ms:.6f} "
        f"sampling_generator_force_argmax_ms={alternative_ms:.6f} "
        f"semantic_token={int(split_host[0])} alternative_trace_slots=1 "
        "alternative_seed_mode=unseeded alternative_penalties=off alternative_log_probs=off"
    )

    generator.reset()
    outputs = generator.generate(prompt, 3, enable_trace=True, sampling_mode="device")
    assert len(outputs) == 3
    assert generator.trace_stats["captures"] == 1
    assert generator.trace_stats["model_replays"] == 2
    assert generator.trace_stats["sampling_replays"] == 2
    assert generator.trace_stats["full_logit_readbacks"] == full_logit_readbacks_before_optimized

    current_position = _first_device_to_torch(generator._trace_current_pos).reshape(-1)
    assert int(current_position[0]) == len(prompt) + 2
    sampled_address = ttnn.get_device_tensors(generator._trace_sampled)[0].buffer_address()
    token_address = ttnn.get_device_tensors(generator._trace_token)[0].buffer_address()
    assert sampled_address == token_address

    unchanged_copies = generator.trace_stats["page_table_host_copies"]
    page_table = generator._make_page_table()
    generator._copy_trace_state(tokens=None, positions=None, page_host=page_table)
    assert generator.trace_stats["page_table_host_copies"] == unchanged_copies
    unchanged_position = int(_first_device_to_torch(generator._trace_current_pos).reshape(-1)[0])
    generator._replay_split_sampling()
    ttnn.synchronize_device(mesh_device)
    assert int(_first_device_to_torch(generator._trace_current_pos).reshape(-1)[0]) == unchanged_position + 1

    changed_page_table = torch.flip(page_table, dims=[1]).contiguous()
    generator._copy_trace_state(tokens=None, positions=None, page_host=changed_page_table)
    assert generator.trace_stats["page_table_host_copies"] == unchanged_copies + 1
    logical_page = unchanged_position // 32
    old_physical_page = int(page_table[0, logical_page])
    changed_physical_page = int(changed_page_table[0, logical_page])
    assert old_physical_page != changed_physical_page
    key_cache = generator._ensure_kv_cache()[0][0]
    before_changed_replay = _first_device_to_torch(key_cache).clone()
    changed_position = int(_first_device_to_torch(generator._trace_current_pos).reshape(-1)[0])
    generator._replay_split_sampling()
    ttnn.synchronize_device(mesh_device)
    after_changed_replay = _first_device_to_torch(key_cache)
    assert int(_first_device_to_torch(generator._trace_current_pos).reshape(-1)[0]) == changed_position + 1
    assert torch.equal(
        before_changed_replay[old_physical_page],
        after_changed_replay[old_physical_page],
    )
    assert not torch.equal(
        before_changed_replay[changed_physical_page],
        after_changed_replay[changed_physical_page],
    )
    print(
        "FULL_MODEL_CHANGED_PAGE_REPLAY "
        f"logical_page={logical_page} old_physical_page={old_physical_page} "
        f"changed_physical_page={changed_physical_page} consumed=true"
    )

    model_trace_id = generator._trace_model_id
    sampling_trace_id = generator._trace_sampling_id
    generator.reset()
    assert generator._trace_model_id == model_trace_id
    assert generator._trace_sampling_id == sampling_trace_id
    generator.teardown()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_reduced_real_mixed_fixed_slots_and_inactive_rows(mesh_device):
    snapshot_text = os.environ.get(REDUCED_REAL_ENV)
    if not snapshot_text:
        pytest.skip(f"Set {REDUCED_REAL_ENV} to the complete local HF snapshot")
    ttnn.CONFIG.throw_exception_on_fallback = True
    print("FULL_MODEL_RUNTIME_FALLBACK_POLICY throw_exception_on_fallback=true")
    snapshot = os.fspath(snapshot_text)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    model = MistralSmall24BFullModel.from_state_dict(
        SafetensorStateDict(snapshot),
        hf_config=config,
        mesh_device=mesh_device,
        full_model_config=FullModelConfig(
            max_batch_size=3,
            max_context_len=64,
            num_blocks=6,
            prefill_chunk_size=32,
            override_num_layers=1,
        ),
    )
    generator = MistralSmall24BGenerator(model, SimpleNamespace(eos_token_id=2))
    tokens = torch.tensor(
        [
            [1, 11, 12, 13, 14, 15, 16, 0, 0, 0, 0],
            [1, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
        ],
        dtype=torch.long,
    )
    caller_page_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    logits = generator.prefill_forward(
        tokens,
        page_table=caller_page_table,
        kv_cache=None,
        prompt_lens=[7, 11],
    )
    assert tuple(logits.shape) == (2, 1, config.vocab_size)
    assert torch.equal(
        generator._page_table_host,
        torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32),
    )
    generator.reset()
    swapped_logits = generator.prefill_forward(
        torch.flip(tokens, dims=[0]),
        page_table=caller_page_table,
        kv_cache=None,
        prompt_lens=[11, 7],
    )
    assert torch.equal(logits[0], swapped_logits[1])
    assert torch.equal(logits[1], swapped_logits[0])
    print("FULL_MODEL_BATCH_POSITION_DETERMINISM swapped_slots=true exact_logits=true")
    generator.reset()
    generator.prefill_forward(
        tokens,
        page_table=caller_page_table,
        kv_cache=None,
        prompt_lens=[7, 11],
    )

    sampled = generator.decode_forward(
        torch.tensor([[101], [202]], dtype=torch.long),
        torch.tensor([7, 11], dtype=torch.long),
        page_table=caller_page_table,
        kv_cache=generator._ensure_kv_cache(),
        sampling_mode="device",
        enable_trace=True,
    )
    sampled_host = _first_device_to_torch(sampled).reshape(-1).to(torch.int64)
    assert sampled_host[:2].ge(0).all()
    current_positions = _first_device_to_torch(generator._trace_current_pos).reshape(-1)
    assert current_positions[:3].tolist() == [8, 12, -1]

    page_copies = generator.trace_stats["page_table_host_copies"]
    normalized = generator._normalise_page_table(caller_page_table, active_batch=2)
    generator._copy_trace_state(tokens=None, positions=None, page_host=normalized)
    assert generator.trace_stats["page_table_host_copies"] == page_copies
    changed = torch.tensor([[1, 0], [3, 2], [5, 4]], dtype=torch.int32)
    generator._copy_trace_state(tokens=None, positions=None, page_host=changed)
    assert generator.trace_stats["page_table_host_copies"] == page_copies + 1
    assert torch.equal(generator._trace_page_table_snapshot, changed)
    generator._replay_split_sampling()
    ttnn.synchronize_device(mesh_device)
    assert _first_device_to_torch(generator._trace_current_pos).reshape(-1)[:3].tolist() == [9, 13, -1]
    generator.teardown()


@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", MESH_PARAMS, indirect=True)
def test_reduced_real_full_terminal_trace_profile(mesh_device):
    """One real layer plus final norm, full LM head, split sampler, and feedback trace."""

    snapshot_text = os.environ.get(REDUCED_REAL_ENV)
    if not snapshot_text:
        pytest.skip(f"Set {REDUCED_REAL_ENV} to the complete local HF snapshot")
    import tracy

    ttnn.CONFIG.throw_exception_on_fallback = True
    print("FULL_MODEL_RUNTIME_FALLBACK_POLICY throw_exception_on_fallback=true")
    snapshot = os.fspath(snapshot_text)
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    model = MistralSmall24BFullModel.from_state_dict(
        SafetensorStateDict(snapshot),
        hf_config=config,
        mesh_device=mesh_device,
        full_model_config=FullModelConfig(
            max_batch_size=1,
            max_context_len=320,
            num_blocks=10,
            prefill_chunk_size=128,
            override_num_layers=1,
        ),
    )
    generator = MistralSmall24BGenerator(model, SimpleNamespace(eos_token_id=2))
    prompt = [1] + [1000 + (index % 100) for index in range(127)]
    outputs = generator.generate(prompt, 2, enable_trace=True, sampling_mode="device")
    assert len(outputs) == 2
    tracy.signpost("FULL_MODEL_REDUCED_TRACE")
    for _ in range(10):
        generator._replay_split_sampling()
    ttnn.synchronize_device(mesh_device)
    tracy.signpost("FULL_MODEL_REDUCED_TRACE_END")
    assert generator.trace_stats["model_replays"] == 11
    assert generator.trace_stats["sampling_replays"] == 11
    assert generator.trace_stats["full_logit_readbacks"] == 0
    generator.teardown()
