# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

import ttnn
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.generator import (
    Llama31Generator,
    _first_device_to_torch,
    _round_up,
    build_generator,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.model import (
    DEFAULT_NUM_BLOCKS,
    HF_CONTEXT_LENGTH,
    PADDED_VOCAB_SIZE,
    FullModelConfig,
)
from models.autoports.meta_llama_llama_3_1_8b_instruct.tt.multichip_decoder import TARGET_TP_DEGREE

MODEL_DIR = Path("models/autoports/meta_llama_llama_3_1_8b_instruct")
MODEL_CONFIG_DIR = Path("models/tt_transformers/model_params/Llama-3.1-8B-Instruct")
P300_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 100_000_000,
    "require_exact_physical_num_devices": True,
}


def _p300_test(function):
    parametrized = pytest.mark.parametrize(
        "mesh_device, device_params",
        [(4, P300_DEVICE_PARAMS)],
        indirect=True,
        ids=["p300-1x4-ring"],
    )(function)
    return pytest.mark.timeout(1800)(parametrized)


def test_full_model_config_preserves_hf_context_and_tp4_policy():
    hf_config = AutoConfig.from_pretrained(MODEL_CONFIG_DIR, local_files_only=True)
    config = FullModelConfig()

    config.validate(hf_config)

    assert config.max_context_len == HF_CONTEXT_LENGTH == hf_config.max_position_embeddings
    assert config.num_blocks == DEFAULT_NUM_BLOCKS
    assert config.num_blocks * 64 == HF_CONTEXT_LENGTH
    assert TARGET_TP_DEGREE == 4
    assert config.multichip.num_links == 2
    assert config.multichip.topology == ttnn.Topology.Ring
    assert config.kv_cache_dtype == ttnn.bfloat8_b


def test_page_allocator_handles_non_aligned_and_mixed_prompt_lengths():
    generator = object.__new__(Llama31Generator)
    generator.batch = 4
    generator.model = SimpleNamespace(config=SimpleNamespace(num_blocks=16))

    page_table = generator._make_page_table([1, 65, 127])

    assert page_table.shape == (4, 16)
    assert page_table[0, :2].tolist() == [0, -1]
    assert page_table[1, :3].tolist() == [1, 2, -1]
    assert page_table[2, :3].tolist() == [3, 4, -1]
    assert page_table[3].eq(-1).all()
    assert _round_up(65, 128) == 128


def test_sampling_params_use_inverse_temperature_and_zero_greedy_sentinel():
    public_k = [2, 2]
    public_p = [0.9, 0.9]
    public_temperature = [0.5, 0.0]

    device_k, device_p, device_temperature = Llama31Generator._format_device_sampling_params(
        public_k,
        public_p,
        public_temperature,
    )

    assert device_k == [2, 1]
    assert device_p == [0.9, 0.0]
    assert device_temperature == [2.0, 1.0]
    assert public_k == [2, 2] and public_p == [0.9, 0.9]


@_p300_test
def test_exact_weight_reduced_full_model_trace_and_sampler_contract(mesh_device):
    """Cheap exact-weight proof for fixed slots, reset, trace feedback, and greedy split sampling."""

    generator = build_generator(
        MODEL_DIR,
        mesh_device,
        max_batch_size=2,
        max_context_len=256,
        num_blocks=4,
        prefill_chunk_size=128,
        override_num_layers=1,
    )
    prompt = generator.tokenizer.encode("Explain why 7 is prime.", add_special_tokens=True)
    # Allocate the independent eager-control cache before trace reservations
    # pin allocator addresses.
    control_cache = generator.model.allocate_kv_cache()

    page_table = generator._make_page_table([_round_up(len(prompt) + 1, 64)])
    logits = generator._prefill_single_device(prompt, page_table, generator._ensure_kv_cache())
    k, p, temp = generator._ensure_sampling_params()
    split_token = generator.model.sample_greedy_split(logits, k=k, p=p, temp=temp)
    argmax_token = generator.model.sample_force_argmax(logits)
    ttnn.synchronize_device(mesh_device)
    assert int(generator._sampled_tokens_to_torch(split_token)[0]) == int(
        generator._sampled_tokens_to_torch(argmax_token)[0]
    )
    generator.reset()

    device_tokens = generator.generate(prompt, 5, enable_trace=True, sampling_mode="device")
    trace_stats = dict(generator.trace_stats)

    assert len(device_tokens) == 5
    assert all(0 <= token < generator.model.vocab_size for token in device_tokens)
    assert trace_stats["captures"] == 1
    assert trace_stats["replays"] == 4
    assert trace_stats["releases"] == 0
    assert trace_stats["prefill_captures"] == 1
    assert trace_stats["prefill_replays"] == 1
    assert trace_stats["prefill_releases"] == 0
    assert trace_stats["decode_warmups"] == 1
    assert trace_stats["sampling_param_host_copies"] == 0
    assert trace_stats["sampling_seed_host_copies"] == 0
    assert trace_stats["caller_token_readbacks"] == 7

    trace_ids = (generator._trace_model_id, generator._trace_sampling_id)
    captures_before_reset = generator.trace_stats["captures"]
    replays_before_reset = generator.trace_stats["replays"]
    token_copies_before_reset = generator.trace_stats["token_host_copies"]
    position_copies_before_reset = generator.trace_stats["position_host_copies"]
    page_copies_before_reset = generator.trace_stats["page_table_host_copies"]
    syncs_before_reset = generator.trace_stats["explicit_synchronizations"]
    cache_probe = generator._ensure_kv_cache()[0][0]
    assert torch.count_nonzero(_first_device_to_torch(cache_probe)).item() > 0

    generator.reset()
    assert (generator._trace_model_id, generator._trace_sampling_id) == trace_ids
    assert torch.count_nonzero(_first_device_to_torch(cache_probe)).item() == 0
    repeated_tokens = generator.generate(prompt, 5, enable_trace=True, sampling_mode="device")
    assert repeated_tokens == device_tokens
    assert generator.trace_stats["captures"] == captures_before_reset
    assert generator.trace_stats["replays"] - replays_before_reset == 4
    assert generator.trace_stats["token_host_copies"] - token_copies_before_reset == 1
    assert generator.trace_stats["position_host_copies"] - position_copies_before_reset == 2
    assert generator.trace_stats["page_table_host_copies"] == page_copies_before_reset
    # Reset is the only request-boundary synchronization. Compatible prefill
    # replay and every steady decode replay add no explicit per-token sync.
    assert generator.trace_stats["explicit_synchronizations"] == syncs_before_reset + 1
    assert generator.trace_stats["prefill_captures"] == 1
    assert generator.trace_stats["prefill_replays"] == 2

    generator.reset()
    forced_feedback_tokens = generator.generate(
        prompt,
        5,
        next_input=lambda _step, predicted: predicted,
        enable_trace=True,
        sampling_mode="device",
    )
    assert device_tokens == forced_feedback_tokens
    assert generator.trace_stats["captures"] == captures_before_reset
    assert generator.trace_stats["prefill_captures"] == 1
    assert generator.trace_stats["prefill_replays"] == 3

    # Public replay keeps token/position state on device. An unchanged page
    # table performs no H2D update; a changed table updates the exact persistent
    # trace input and is consumed by the next replay. Compare both mappings to
    # independently reset eager controls.
    base_page_table = generator._make_page_table([128])
    changed_page_table = base_page_table.clone()
    changed_page_table[0, 0], changed_page_table[0, 1] = (
        base_page_table[0, 1].clone(),
        base_page_table[0, 0].clone(),
    )

    def traced_decode_token(page_table):
        generator.reset()
        prefill_sampled = generator.prefill_forward(
            torch.tensor([prompt]),
            page_table=base_page_table,
            kv_cache=generator._ensure_kv_cache(),
            prompt_lens=[len(prompt)],
            sampling_mode="device",
        )
        input_token = int(generator._sampled_tokens_to_torch(prefill_sampled)[0])
        sampled = generator.decode_forward(
            torch.tensor([[input_token]]),
            torch.tensor([len(prompt)]),
            page_table=page_table,
            kv_cache=generator._ensure_kv_cache(),
            sampling_mode="device",
            enable_trace=True,
            active_batch=1,
        )
        output_token = int(generator._sampled_tokens_to_torch(sampled)[0])
        return input_token, output_token

    page_copies_before = generator.trace_stats["page_table_host_copies"]
    base_input_token, base_trace_token = traced_decode_token(base_page_table)
    assert generator.trace_stats["page_table_host_copies"] == page_copies_before
    assert torch.equal(
        _first_device_to_torch(generator._trace_inputs[3]).to(torch.int32),
        base_page_table,
    )
    assert _first_device_to_torch(generator._trace_inputs[1]).reshape(-1)[:2].tolist() == [len(prompt) + 1, -1]

    changed_input_token, changed_trace_token = traced_decode_token(changed_page_table)
    assert changed_input_token == base_input_token
    assert generator.trace_stats["page_table_host_copies"] == page_copies_before + 1
    assert torch.equal(
        _first_device_to_torch(generator._trace_inputs[3]).to(torch.int32),
        changed_page_table,
    )
    assert generator.trace_stats["captures"] == captures_before_reset

    def eager_control_token(page_table):
        generator.model.reset_kv_cache(control_cache)
        control_logits = generator.prefill_forward(
            torch.tensor([prompt]),
            page_table=base_page_table,
            kv_cache=control_cache,
            prompt_lens=[len(prompt)],
        )
        input_token = int(control_logits[0, 0].argmax())
        decoded = generator.decode_forward(
            torch.tensor([[input_token]]),
            torch.tensor([len(prompt)]),
            page_table=page_table,
            kv_cache=control_cache,
            sampling_mode="host",
            enable_trace=False,
            active_batch=1,
        )
        return input_token, int(decoded[0].argmax())

    base_control_input, base_control_token = eager_control_token(base_page_table)
    changed_control_input, changed_control_token = eager_control_token(changed_page_table)
    assert base_control_input == changed_control_input == base_input_token
    assert base_trace_token == base_control_token
    assert changed_trace_token == changed_control_token

    generator.reset()
    host_tokens = generator.generate(prompt, 5, enable_trace=False, sampling_mode="host")
    assert len(host_tokens) == 5
    assert host_tokens[0] == device_tokens[0]
    assert all(0 <= token < generator.model.vocab_size for token in host_tokens)

    generator.reset()
    sampled_tokens = generator.generate(
        prompt,
        2,
        enable_trace=True,
        sampling_mode="device",
        top_k=2,
        top_p=1.0,
        temperature=1.0,
    )
    assert len(sampled_tokens) == 2
    assert all(0 <= token < generator.model.vocab_size for token in sampled_tokens)
    assert generator.trace_stats["sampling_param_host_copies"] == 3
    assert generator.trace_stats["sampling_seed_host_copies"] == 2
    generator.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=1)

    generator.reset()
    mixed_tokens = torch.tensor(
        [
            prompt[:7] + [0] * 6,
            prompt + [generator.tokenizer.eos_token_id] * (13 - len(prompt)),
        ]
    )
    mixed_lengths = [7, len(prompt)]
    mixed_page_table = generator._make_page_table([128, 128])
    mixed_logits = generator.prefill_forward(
        mixed_tokens,
        page_table=mixed_page_table,
        kv_cache=generator._ensure_kv_cache(),
        prompt_lens=mixed_lengths,
    )
    assert mixed_logits.shape == (2, 1, generator.model.vocab_size)
    mixed_greedy = mixed_logits[:, 0].argmax(dim=-1).tolist()
    mixed_decode = generator.decode_forward(
        mixed_logits.argmax(dim=-1),
        torch.tensor(mixed_lengths),
        page_table=mixed_page_table,
        kv_cache=generator._ensure_kv_cache(),
        sampling_mode="host",
        enable_trace=False,
    )
    assert mixed_decode.shape == (2, generator.model.vocab_size)

    generator.reset()
    generator.set_sampling_params(top_k=1, top_p=0.0, temperature=1.0, active_batch=2)
    mixed_sampled = generator.prefill_forward(
        mixed_tokens,
        page_table=mixed_page_table,
        kv_cache=generator._ensure_kv_cache(),
        prompt_lens=mixed_lengths,
        sampling_mode="device",
    )
    assert generator._sampled_tokens_to_torch(mixed_sampled)[:2].tolist() == mixed_greedy

    generator.reset()
    chunked_prompt = [generator.tokenizer.bos_token_id] + [42] * 128
    chunked_page_table = generator._make_page_table([256])
    chunked_logits = generator.prefill_forward(
        torch.tensor([chunked_prompt]),
        page_table=chunked_page_table[:1],
        kv_cache=generator._ensure_kv_cache(),
        prompt_lens=[len(chunked_prompt)],
    )
    assert chunked_logits.shape == (1, 1, generator.model.vocab_size)

    # Fixed equal-probability logits isolate the trace-stable seed lifecycle:
    # one real request seed, then UINT32_MAX so device RNG advances per replay.
    generator.reset()
    generator.set_sampling_params(top_k=2, top_p=1.0, temperature=1.0, active_batch=1)
    host_logits = torch.full((1, 1, 32, PADDED_VOCAB_SIZE), -30.0, dtype=torch.bfloat16)
    host_logits[..., 7] = 1.0
    host_logits[..., 8] = 1.0
    fixed_logits = ttnn.from_torch(
        host_logits,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=3),
    )
    k, p, temp = generator._ensure_sampling_params()
    trace_output = generator.model.sample_stochastic_split(fixed_logits, k=k, p=p, temp=temp)
    ttnn.synchronize_device(mesh_device)
    sampling_trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    trace_output = generator.model.sample_stochastic_split(
        fixed_logits,
        k=k,
        p=p,
        temp=temp,
        tt_out_tok=trace_output,
    )
    ttnn.end_trace_capture(mesh_device, sampling_trace_id, cq_id=0)

    def seeded_sequence() -> list[int]:
        generator.begin_sampling_request(seed=17, active_batch=1)
        ttnn.execute_trace(mesh_device, sampling_trace_id, cq_id=0, blocking=False)
        sequence = [int(_first_device_to_torch(trace_output).reshape(-1)[0])]
        generator._transition_sampling_seed_to_device_advance()
        for _ in range(16):
            ttnn.execute_trace(mesh_device, sampling_trace_id, cq_id=0, blocking=False)
            sequence.append(int(_first_device_to_torch(trace_output).reshape(-1)[0]))
        return sequence

    seed_copies_before = generator.trace_stats["sampling_seed_host_copies"]
    sequence_one = seeded_sequence()
    sequence_two = seeded_sequence()
    ttnn.release_trace(mesh_device, sampling_trace_id)
    assert sequence_one == sequence_two
    assert len(set(sequence_one)) > 1
    assert generator.trace_stats["sampling_seed_host_copies"] - seed_copies_before == 4
    assert generator._trace_model_id is None and generator._trace_sampling_id is None


@_p300_test
def test_exact_weight_reduced_full_model_watcher_path(mesh_device):
    """Canonical split-sampling path only; run this target under TT_METAL_WATCHER."""

    generator = build_generator(
        MODEL_DIR,
        mesh_device,
        max_batch_size=2,
        max_context_len=256,
        num_blocks=4,
        prefill_chunk_size=128,
        override_num_layers=1,
    )
    prompt = generator.tokenizer.encode("Explain why 7 is prime.", add_special_tokens=True)
    tokens = generator.generate(prompt, 5, enable_trace=True, sampling_mode="device")
    assert len(tokens) == 5
    assert generator.trace_stats["captures"] == 1
    assert generator.trace_stats["replays"] == 4

    generator.reset()
    sampled = generator.generate(
        prompt,
        2,
        enable_trace=True,
        sampling_mode="device",
        top_k=2,
        top_p=1.0,
        temperature=1.0,
    )
    assert len(sampled) == 2
    assert generator.trace_stats["sampling_param_host_copies"] == 3
    assert generator.trace_stats["sampling_seed_host_copies"] == 2
