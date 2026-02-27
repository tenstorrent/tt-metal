# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pure TT-metal sampling tests inspired by vLLM sampling tests.

It should keep the high-level organization from vLLM request-level tests,
but validate behavior directly at token-ID level using synthetic logits and
on-device sampling primitives (`TTSampling` + `SamplingGenerator`).
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.sampling.tt_sampling import TTSampling


# --- Constants & helpers ---

BATCH_SIZE = 32
MAX_TOP_K = 32
VOCAB_SIZE = 32000
FAST_NUM_TRIES = 6
FAST_NUM_STEPS = 5


# Lightweight args container expected by TTSampling/SamplingGenerator in tests.
@dataclass
class _SamplingArgs:
    vocab_size: int
    padded_vocab_size: int
    max_batch_size: int
    max_top_k: int
    cluster_shape: tuple[int, int]
    sampling_all_gather_axis: int
    sampling_dp: int
    sub_core_grids: ttnn.CoreRangeSet | None
    model_config: dict


def _compute_per_device_vocab(vocab_size: int, num_tp: int) -> int:
    per_device = (((vocab_size + num_tp - 1) // num_tp + 31) // 32) * 32
    return 1 << (per_device - 1).bit_length()


def _broadcast(value, *, size: int = BATCH_SIZE):
    if isinstance(value, list):
        assert len(value) == size, f"Expected list of length {size}, got {len(value)}"
        return list(value)
    return [value] * size


def make_sampling_args(mesh_device, sampling_dp: int = 1) -> _SamplingArgs:
    """Build sampling args for synthetic tests on the current mesh shape."""
    cluster_shape = tuple(mesh_device.shape)
    # For 1D meshes, gather across the non-singleton axis.
    if cluster_shape[0] == 1 and cluster_shape[1] > 1:
        sampling_all_gather_axis = 1
    elif cluster_shape[1] == 1 and cluster_shape[0] > 1:
        sampling_all_gather_axis = 0
    else:
        sampling_all_gather_axis = 1 if cluster_shape[0] > 1 and cluster_shape[1] > 1 else 0
    num_tp = cluster_shape[sampling_all_gather_axis] if cluster_shape[sampling_all_gather_axis] > 0 else 1
    per_device_vocab = _compute_per_device_vocab(VOCAB_SIZE, num_tp)
    padded_vocab_size = per_device_vocab * num_tp
    return _SamplingArgs(
        vocab_size=VOCAB_SIZE,
        padded_vocab_size=padded_vocab_size,
        max_batch_size=BATCH_SIZE,
        max_top_k=MAX_TOP_K,
        cluster_shape=cluster_shape,
        sampling_all_gather_axis=sampling_all_gather_axis,
        sampling_dp=sampling_dp,
        sub_core_grids=None,
        model_config={},
    )


def make_sharded_logits(torch_logits: torch.Tensor, mesh_device, args: _SamplingArgs):
    """Create device logits with vocab sharded along the sampling all-gather axis."""
    if mesh_device.get_num_devices() == 1:
        mesh_mapper = None
    elif args.cluster_shape[0] > 1 and args.cluster_shape[1] > 1:
        dims = (None, 3) if args.sampling_all_gather_axis == 1 else (3, None)
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=args.cluster_shape)
    else:
        mesh_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=3)
    return ttnn.from_torch(
        torch_logits,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_mapper,
    )


def extract_tokens(tt_out_tok, batch_size: int = BATCH_SIZE) -> list[int]:
    first_device_out = ttnn.get_device_tensors(tt_out_tok)[0]
    out_torch = ttnn.to_torch(first_device_out).reshape(-1).to(torch.int64)
    return out_torch[:batch_size].tolist()


def build_hot_logits(
    args: _SamplingArgs,
    *,
    batch_size: int = BATCH_SIZE,
    hot_tokens: list[int] | None = None,
    per_user_hot_tokens: list[list[int]] | None = None,
    base_logit: float = -10.0,
    top_logit: float = 10.0,
    step: float = 0.25,
) -> torch.Tensor:
    """Create logits with a small hot token set that dominates sampling."""
    if hot_tokens is None and per_user_hot_tokens is None:
        raise ValueError("Either hot_tokens or per_user_hot_tokens must be provided")
    if per_user_hot_tokens is None:
        per_user_hot_tokens = [list(hot_tokens)] * batch_size
    assert len(per_user_hot_tokens) == batch_size

    logits = torch.full((1, 1, batch_size, args.padded_vocab_size), base_logit, dtype=torch.float32)
    for user_idx, user_hot in enumerate(per_user_hot_tokens):
        for rank, tok in enumerate(user_hot):
            logits[0, 0, user_idx, tok] = top_logit - rank * step
    logits[:, :, :, args.vocab_size :] = -float("inf")
    return logits


def build_penalty_logits(
    args: _SamplingArgs,
    *,
    target_token: int,
    batch_size: int = BATCH_SIZE,
    base_logit: float = 4.0,
    target_logit: float = 5.0,
) -> torch.Tensor:
    """Create logits where one token is the greedy target before penalties."""
    logits = torch.full((1, 1, batch_size, args.padded_vocab_size), base_logit, dtype=torch.float32)
    logits[:, :, :, target_token] = target_logit
    logits[:, :, :, args.vocab_size :] = -float("inf")
    return logits


def run_ttsampling_once(
    mesh_device,
    args: _SamplingArgs,
    torch_logits: torch.Tensor,
    *,
    top_k,
    top_p,
    temperature,
) -> list[int]:
    """Run one direct TTSampling forward and return host token IDs."""
    tt_sampling = TTSampling(
        mesh_device=mesh_device,
        tt_ccl=None,
        args=args,
        k=torch.tensor(_broadcast(top_k), dtype=torch.int64),
        p=torch.tensor(_broadcast(top_p), dtype=torch.float32),
        temp=torch.tensor(_broadcast(temperature), dtype=torch.float32),
    )
    tt_input = make_sharded_logits(torch_logits, mesh_device, args)
    tt_tokens, _ = tt_sampling(tt_input)
    return extract_tokens(tt_tokens, BATCH_SIZE)


def run_sampling_generator(
    mesh_device,
    args: _SamplingArgs,
    torch_logits: torch.Tensor,
    sampling_params: SamplingParams,
    *,
    num_steps: int = 1,
    advance_seeds: bool = True,
    seed_values: list[int] | None = None,
    write_seed_values_to_device: bool = False,
    state_setup=None,
) -> list[list[int]]:
    """Run SamplingGenerator for num_steps and return per-step token lists."""
    sg = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None, enable_internal_trace=False)
    formatted = format_sampling_params(sampling_params, BATCH_SIZE)
    sg.reset_sampling_params(formatted)
    if seed_values is not None:
        user_ids = list(range(len(seed_values)))
        sg.seed_manager.reset_seed(seed_values, user_ids)
    if state_setup is not None:
        state_setup(sg)

    if write_seed_values_to_device:
        if seed_values is None:
            raise ValueError("write_seed_values_to_device=True requires seed_values")
        if len(seed_values) != BATCH_SIZE:
            raise ValueError(f"Expected {BATCH_SIZE} seed values, got {len(seed_values)}")
        wrapped = [(seed & 0xFFFFFFFF) for seed in seed_values]
        seed_tt = ttnn.from_torch(
            torch.tensor(wrapped, dtype=torch.uint32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=sg.seed_manager._seed_mapper,
        )
        ttnn.copy_host_to_device_tensor(seed_tt, sg.tt_sampling.seeds_tt_tensor)

    tt_input = make_sharded_logits(torch_logits, mesh_device, args)
    outputs = []
    for _ in range(num_steps):
        # SamplingGenerator keeps per-user RNG state in SeedManager.
        if advance_seeds:
            sg.seed_manager.get_new_values()
        tt_tokens, _ = sg.sample(tt_input, enable_trace=False)
        outputs.append(extract_tokens(tt_tokens, BATCH_SIZE))
    return outputs


def _assert_tokens_in_vocab(tokens: list[int]):
    assert all(0 <= tok < VOCAB_SIZE for tok in tokens), f"Found out-of-range token(s): {tokens}"


# --- Test: prefill parameter behavior ---

class TestPrefillWithDifferentParams:
    @pytest.mark.parametrize("mesh_device", [1, (4, 8)], indirect=True)
    @pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
        indirect=True,
    )
    def test_prefill_temperature_varied_in_batch(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[100, 101, 102, 103, 104, 105, 106, 107])
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        outputs = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)
        tokens = outputs[0]
        _assert_tokens_in_vocab(tokens)
        assert len(set(tokens)) >= 2, f"Expected stochastic diversity in batch, got tokens={tokens}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_prefill_temperature_varied_between_batches(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[120, 121, 122, 123, 124, 125, 126, 127])
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        outputs = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=FAST_NUM_TRIES, advance_seeds=True
        )
        user0 = [step[0] for step in outputs]
        assert len(set(user0)) >= 2, f"Expected variation across runs for stochastic sampling, got {user0}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_prefill_topk(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[140, 141, 142, 143, 144, 145, 146, 147])

        half = BATCH_SIZE // 2
        temperature = [0.0] * half + [1.5] * (BATCH_SIZE - half)
        top_k = [32] * half + [8] * (BATCH_SIZE - half)
        top_p = [1.0] * BATCH_SIZE
        params = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)

        seed_values = [1000 + i for i in range(BATCH_SIZE)]
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seed_values
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seed_values
        )[0]

        assert out1 == out2, "Same per-user seeds must replay exactly across batches"
        greedy = out1[:half]
        stochastic = out1[half:]
        assert len(set(greedy)) == 1, f"Greedy half should be deterministic and identical, got {greedy}"
        assert len(set(stochastic)) >= 2, f"Stochastic half should vary across slots, got {stochastic}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_prefill_seeding(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[160, 161, 162, 163, 164, 165, 166, 167])

        thirds = BATCH_SIZE // 3
        greedy_count = BATCH_SIZE - 2 * thirds
        temperature = [0.0] * greedy_count + [1.5] * (BATCH_SIZE - greedy_count)
        top_k = [32] * greedy_count + [8] * (BATCH_SIZE - greedy_count)
        top_p = [1.0] * BATCH_SIZE
        params = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p)

        seed_values = [2000 + i for i in range(BATCH_SIZE)]
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seed_values
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seed_values
        )[0]
        assert out1 == out2, "Replay with same seeds must be deterministic for every slot"
        assert len(set(out1[greedy_count:])) >= 2, "Different stochastic seeds should yield diverse tokens"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_prefill_topk_1_is_greedy(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[180, 181, 182, 183])
        greedy_tokens = run_ttsampling_once(mesh_device, args, logits, top_k=1, top_p=0.0, temperature=1.0)
        topk1_tokens = run_ttsampling_once(mesh_device, args, logits, top_k=1, top_p=1.0, temperature=5.0)
        assert greedy_tokens == topk1_tokens, "top_k=1 should match greedy behavior"


# --- Test: per-request penalties ---

@pytest.mark.parametrize("mesh_device", [1], indirect=True)
class TestRepetitionPenaltyPerRequest:
    def test_different_repetition_penalties(self, mesh_device):
        args = make_sampling_args(mesh_device)
        target_token = 500
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = ([1.0, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0] * 4)[:BATCH_SIZE]

        def _state_setup(sg):
            seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)

        params = SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            repetition_penalty=penalties,
        )
        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=_state_setup)[0]
        low_penalty = [tok for i, tok in enumerate(tokens) if penalties[i] <= 1.2]
        high_penalty = [tok for i, tok in enumerate(tokens) if penalties[i] >= 1.5]
        assert all(tok == target_token for tok in low_penalty), "Low repetition penalty should keep target token"
        assert any(tok != target_token for tok in high_penalty), "High repetition penalties should alter output"

    def test_repetition_penalty_vs_no_penalty(self, mesh_device):
        args = make_sampling_args(mesh_device)
        target_token = 520
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [1.0 if i % 2 == 0 else 2.5 for i in range(BATCH_SIZE)]

        def _state_setup(sg):
            seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)

        params = SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            repetition_penalty=penalties,
        )
        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=_state_setup)[0]
        no_penalty = [tokens[i] for i in range(0, BATCH_SIZE, 2)]
        with_penalty = [tokens[i] for i in range(1, BATCH_SIZE, 2)]
        assert all(tok == target_token for tok in no_penalty), "No-penalty lanes should keep target"
        assert all(tok != target_token for tok in with_penalty), "Penalty lanes should change token"
        assert no_penalty[0] != with_penalty[0], "Penalty and no-penalty outputs should differ"


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
class TestPresencePenaltyPerRequest:
    def test_different_presence_penalties(self, mesh_device):
        args = make_sampling_args(mesh_device)
        target_token = 700
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = ([0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, 4.0] * 4)[:BATCH_SIZE]

        def _state_setup(sg):
            seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)

        params = SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            presence_penalty=penalties,
        )
        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=_state_setup)[0]
        assert any(tok == target_token for tok in tokens), "Some lanes should retain target token"
        assert any(tok != target_token for tok in tokens), "Some lanes should shift off target with higher penalties"

    def test_presence_penalty_mixed_batch(self, mesh_device):
        args = make_sampling_args(mesh_device)
        target_token = 720
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.0 if i % 2 == 0 else 2.0 for i in range(BATCH_SIZE)]

        def _state_setup(sg):
            seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)

        params = SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            presence_penalty=penalties,
        )
        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=_state_setup)[0]
        no_penalty = [tokens[i] for i in range(0, BATCH_SIZE, 2)]
        with_penalty = [tokens[i] for i in range(1, BATCH_SIZE, 2)]
        assert all(tok == target_token for tok in no_penalty), "No presence-penalty lanes should keep target"
        assert all(tok != target_token for tok in with_penalty), "Presence-penalty lanes should move off target"


@pytest.mark.parametrize("mesh_device", [1], indirect=True)
class TestFrequencyPenaltyPerRequest:
    def test_different_frequency_penalties(self, mesh_device):
        args = make_sampling_args(mesh_device)
        target_token = 900
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = ([0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, 4.0] * 4)[:BATCH_SIZE]

        def _state_setup(sg):
            seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)

        params = SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            frequency_penalty=penalties,
        )
        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=_state_setup)[0]
        assert any(tok == target_token for tok in tokens), "Some lanes should retain target token"
        assert any(tok != target_token for tok in tokens), "Some lanes should shift off target with higher penalties"

    def test_frequency_penalty_mixed_batch(self, mesh_device):
        args = make_sampling_args(mesh_device)
        target_token = 920
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.0 if i % 2 == 0 else 2.0 for i in range(BATCH_SIZE)]

        def _state_setup(sg):
            seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)

        params = SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            frequency_penalty=penalties,
        )
        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=_state_setup)[0]
        no_penalty = [tokens[i] for i in range(0, BATCH_SIZE, 2)]
        with_penalty = [tokens[i] for i in range(1, BATCH_SIZE, 2)]
        assert all(tok == target_token for tok in no_penalty), "No frequency-penalty lanes should keep target"
        assert all(tok != target_token for tok in with_penalty), "Frequency-penalty lanes should move off target"


# --- Test: seed behavior ---
class TestSeededSamplingPerRequest:
    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_different_seeds_produce_different_outputs(self, mesh_device):
        args = make_sampling_args(mesh_device)
        hot_tokens = [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007]
        hot_token_set = set(hot_tokens)
        logits = build_hot_logits(args, hot_tokens=hot_tokens)
        params = SamplingParams(temperature=1.5, top_k=8, top_p=1.0)

        # Same seed vector should replay exactly across fresh generator instances.
        seeds_a = list(range(BATCH_SIZE))
        out_a1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_a
        )[0]
        out_a2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_a
        )[0]

        assert out_a1 == out_a2, "Same seed vector should replay exactly across independent runs"

        # Shift every seed; at least one slot must change for stochastic config.
        seeds_b = [s + 12345 for s in seeds_a]
        out_b = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_b
        )[0]

        changed_indices = [i for i, (a, b) in enumerate(zip(out_a1, out_b)) if a != b]
        assert len(changed_indices) > 0, (
            "Different seed vector should change at least one slot. "
            f"changed={len(changed_indices)}, first_diffs={changed_indices[:8]}, "
            f"out_a1={out_a1}, out_b={out_b}"
        )

        # Sanity checks: valid token range and no sampling outside the hot set.
        for outputs in (out_a1, out_a2, out_b):
            _assert_tokens_in_vocab(outputs)
            unexpected = [tok for tok in outputs if tok not in hot_token_set]
            assert not unexpected, (
                f"Sampled tokens outside expected hot set {sorted(hot_token_set)}: {unexpected}. "
                f"outputs={outputs}"
            )

    @pytest.mark.parametrize("mesh_device", [1, (4, 8)], indirect=True)
    @pytest.mark.parametrize(
        "device_params",
        [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
        indirect=True,
    )
    def test_same_seeds_reproduce_across_batches(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1100, 1101, 1102, 1103, 1104, 1105, 1106, 1107])
        params = SamplingParams(temperature=1.25, top_k=8, top_p=1.0)
        seeds = [500 + i for i in range(BATCH_SIZE)]
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=FAST_NUM_STEPS, advance_seeds=True, seed_values=seeds
        )
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=FAST_NUM_STEPS, advance_seeds=True, seed_values=seeds
        )
        assert out1 == out2, "Same seed replay should match exactly for all users and steps"

    @pytest.mark.parametrize("seed", [42, 123, 999, 0])
    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_specific_seed_reproducible(self, mesh_device, seed):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1200, 1201, 1202, 1203, 1204, 1205, 1206, 1207])
        params = SamplingParams(temperature=1.25, top_k=8, top_p=1.0)
        seeds = [10000 + i for i in range(BATCH_SIZE)]
        seeds[0] = seed
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        assert out1[0] == out2[0], f"Seed {seed} should be reproducible for user-0"

    @pytest.mark.parametrize("seed", [1, 0])
    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_batch1_seed_reproducible(self, mesh_device, seed):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307])
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        results = []
        for _ in range(FAST_NUM_TRIES):
            seeds = [seed] + [100000 + i for i in range(1, BATCH_SIZE)]
            out = run_sampling_generator(
                mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
            )[0]
            results.append(out[0])
        assert len(set(results)) == 1, f"Single seeded slot should reproduce, got {results}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_batch1_no_seed_varied(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407])
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        results = []
        for _ in range(FAST_NUM_TRIES):
            out = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)[0]
            results.append(out[0])
        assert len(set(results)) >= 2, f"Unseeded single slot should vary across requests, got {results}"

    @pytest.mark.parametrize("seed", [1, 0])
    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_uniform_seed_deterministic(self, mesh_device, seed):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1500, 1501, 1502, 1503, 1504, 1505, 1506, 1507])
        params = SamplingParams(temperature=1.0, top_k=8, top_p=1.0)
        seeds = [seed] * BATCH_SIZE
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        assert out1 == out2, f"Uniform seed {seed} should be deterministic across runs"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_uniform_noseed_varied(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607])
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        outputs = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=FAST_NUM_STEPS, advance_seeds=True
        )
        user0 = [step[0] for step in outputs]
        assert len(set(user0)) >= 2, f"Expected unseeded variation across steps, got {user0}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_seed_0_produces_deterministic_outputs(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1700, 1701, 1702, 1703, 1704, 1705, 1706, 1707])
        params = SamplingParams(temperature=1.0, top_k=8, top_p=1.0)
        seeds = [0] * BATCH_SIZE
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        assert out1 == out2, "Seed 0 must be deterministic across requests"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_negative_seed_does_not_crash(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[1800, 1801, 1802, 1803, 1804, 1805, 1806, 1807])
        params = SamplingParams(temperature=1.0, top_k=8, top_p=1.0)
        seeds = [-1] + [2000 + i for i in range(1, BATCH_SIZE)]
        out = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=1,
            advance_seeds=False,
            seed_values=seeds,
            write_seed_values_to_device=True,
        )[0]
        _assert_tokens_in_vocab(out)


# --- Test: batch isolation ---

@pytest.mark.parametrize("mesh_device", [1], indirect=True)
class TestBatchIsolation:
    def test_mixed_params_batch(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007])

        temperature_template = [0.0, 1.5, 0.5, 0.5, 1.0, 0.01, 2.0, 0.7]
        top_k_template = [32, 8, 8, 8, 1, 32, 5, 8]
        top_p_template = [1.0, 1.0, 1.0, 0.7, 1.0, 1.0, 1.0, 0.9]
        repetition_template = [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.5]
        presence_template = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]
        frequency_template = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]

        temperature = (temperature_template * 4)[:BATCH_SIZE]
        top_k = (top_k_template * 4)[:BATCH_SIZE]
        top_p = (top_p_template * 4)[:BATCH_SIZE]
        repetition = (repetition_template * 4)[:BATCH_SIZE]
        presence = (presence_template * 4)[:BATCH_SIZE]
        frequency = (frequency_template * 4)[:BATCH_SIZE]
        seeds = [9000 + i for i in range(BATCH_SIZE)]

        def _state_setup(sg):
            seen = torch.full((BATCH_SIZE, 1), 2000, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)

        params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition,
            presence_penalty=presence,
            frequency_penalty=frequency,
        )

        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds, state_setup=_state_setup
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds, state_setup=_state_setup
        )[0]
        assert out1 == out2, "Deterministic lanes (and seeded lanes) should replay identically across runs"
        assert len(set(out1)) >= 2, "Mixed batch should not collapse to a single token"

    def test_outputs_not_mixed_different_prompts(self, mesh_device):
        args = make_sampling_args(mesh_device)
        tokens_per_user = 4
        per_user_hot = []
        expected_sets = []
        token_cursor = 2200
        for _ in range(BATCH_SIZE):
            hot = [token_cursor + i for i in range(tokens_per_user)]
            per_user_hot.append(hot)
            expected_sets.append(set(hot))
            token_cursor += tokens_per_user

        logits = build_hot_logits(args, per_user_hot_tokens=per_user_hot)
        params = SamplingParams(temperature=1.0, top_k=tokens_per_user, top_p=1.0)
        seeds = [10000 + i for i in range(BATCH_SIZE)]
        tokens = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        for i, tok in enumerate(tokens):
            assert tok in expected_sets[i], f"User {i} token leaked across users: tok={tok}, expected={expected_sets[i]}"


# --- Test: batch size variations ---

@pytest.mark.parametrize("mesh_device", [1], indirect=True)
class TestBatchSizeVariations:
    def _make_active_user_logits(self, args: _SamplingArgs, active_users: int):
        per_user_hot = []
        cursor = 2800
        for i in range(BATCH_SIZE):
            if i < active_users:
                hot = [cursor, cursor + 1, cursor + 2]
                cursor += 3
            else:
                hot = [42]
            per_user_hot.append(hot)
        return build_hot_logits(args, per_user_hot_tokens=per_user_hot), per_user_hot

    def test_small_batch_different_params(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits, per_user_hot = self._make_active_user_logits(args, active_users=2)
        params = SamplingParams(
            temperature=[0.0, 1.0],
            top_k=[1, 3],
            top_p=[1.0, 1.0],
        )
        tokens = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)[0]
        assert tokens[0] in set(per_user_hot[0])
        assert tokens[1] in set(per_user_hot[1])
        _assert_tokens_in_vocab(tokens)

    def test_full_batch_different_params(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits, per_user_hot = self._make_active_user_logits(args, active_users=BATCH_SIZE)
        temperature = [0.5 + (i * 0.02) for i in range(BATCH_SIZE)]
        params = SamplingParams(temperature=temperature, top_k=[3] * BATCH_SIZE, top_p=[1.0] * BATCH_SIZE)
        tokens = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)[0]
        for i, tok in enumerate(tokens):
            assert tok in set(per_user_hot[i]), f"User {i} expected token from {per_user_hot[i]}, got {tok}"

    def test_partial_batch_different_params(self, mesh_device):
        args = make_sampling_args(mesh_device)
        active_users = BATCH_SIZE // 2
        logits, per_user_hot = self._make_active_user_logits(args, active_users=active_users)
        temperature = [0.0 if i % 2 == 0 else 1.0 for i in range(active_users)]
        top_k = [1 if i % 2 == 0 else 3 for i in range(active_users)]
        params = SamplingParams(temperature=temperature, top_k=top_k, top_p=[1.0] * active_users)
        tokens = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)[0]
        for i in range(active_users):
            assert tokens[i] in set(per_user_hot[i]), f"Active user {i} expected token from {per_user_hot[i]}"
        _assert_tokens_in_vocab(tokens)


# --- Test: mixed-parameter batches ---

@pytest.mark.parametrize("mesh_device", [1], indirect=True)
class TestMixedParameterBatches:
    def test_all_parameter_types_in_batch(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007])

        temperature_template = [0.0, 1.5, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0]
        top_k_template = [32, 8, 10, 8, 8, 8, 8, 8]
        top_p_template = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5]
        repetition_template = [1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.5, 1.0]
        presence_template = [0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.0]
        frequency_template = [0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 0.0]

        temperature = (temperature_template * 4)[:BATCH_SIZE]
        top_k = (top_k_template * 4)[:BATCH_SIZE]
        top_p = (top_p_template * 4)[:BATCH_SIZE]
        repetition = (repetition_template * 4)[:BATCH_SIZE]
        presence = (presence_template * 4)[:BATCH_SIZE]
        frequency = (frequency_template * 4)[:BATCH_SIZE]
        seeds = [12000 + i for i in range(BATCH_SIZE)]

        def _state_setup(sg):
            seen = torch.full((BATCH_SIZE, 1), 3000, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)

        params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition,
            presence_penalty=presence,
            frequency_penalty=frequency,
        )

        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds, state_setup=_state_setup
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds, state_setup=_state_setup
        )[0]
        assert out1 == out2, "Mixed parameter batch should replay deterministically with fixed seeds"
        _assert_tokens_in_vocab(out1)
        assert len(set(out1)) >= 2, "Mixed parameter batch should produce non-trivial diversity"
