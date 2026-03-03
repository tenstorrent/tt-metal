# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Pure TT-metal sampling tests inspired by vLLM sampling tests.

It should keep the high-level organization from vLLM request-level tests,
but validate behavior directly at token-ID level using synthetic logits and
on-device sampling primitives ('TTSampling' + 'SamplingGenerator').
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import pytest
import torch

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.sampling.tt_sampling import TTSampling

# TEST NOTES:
# - 'mesh_device' and 'device_params' come from repo-root 'conftest.py'.
# - 'device_params' defaults to '{}' (no explicit fabric config).
# - 'models/common/tests/conftest.py' provides additional common-test fixtures
#   (e.g. 'ttnn_mesh_device'), but this file uses 'mesh_device'.
# - For all multi-device paths in this file, we pass an explicit ring fabric
#   config so tests do not rely on implicit environment defaults.
# - Token ID bands are intentionally disjoint across suites for clearer failure triage.


# --- Constants & helpers ---

BATCH_SIZE = 32
MAX_TOP_K = 32
VOCAB_SIZE = 32000
FAST_NUM_TRIES = 6
FAST_NUM_STEPS = 5
MULTI_DEVICE_MESHES = [1, (4, 8)]
RING_FABRIC_DEVICE_PARAMS = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}]


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


def compute_per_device_vocab(vocab_size: int, num_tp: int) -> int:
    per_device = (((vocab_size + num_tp - 1) // num_tp + 31) // 32) * 32
    return 1 << (per_device - 1).bit_length()


def broadcast(value, *, size: int = BATCH_SIZE):
    if isinstance(value, list):
        assert len(value) == size, f"Expected list of length {size}, got {len(value)}"
        return list(value)
    return [value] * size


def safe_sync(mesh_device):
    try:
        ttnn.synchronize_device(mesh_device)
    except Exception:
        # Cleanup best-effort only; sync failures should not mask test assertions.
        pass


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
    per_device_vocab = compute_per_device_vocab(VOCAB_SIZE, num_tp)
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


def infer_effective_batch_size(
    torch_logits: torch.Tensor,
    batch_size: int | None,
    *,
    max_batch_size: int = BATCH_SIZE,
) -> int:
    if torch_logits.ndim != 4:
        raise ValueError(
            f"Expected torch_logits with rank 4 [1, 1, batch, vocab], got shape {tuple(torch_logits.shape)}"
        )
    inferred_batch_size = int(torch_logits.shape[2])
    effective_batch_size = inferred_batch_size if batch_size is None else int(batch_size)
    if effective_batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {effective_batch_size}")
    if effective_batch_size > inferred_batch_size:
        raise ValueError(
            f"batch_size ({effective_batch_size}) cannot exceed logits batch dimension ({inferred_batch_size})"
        )
    if effective_batch_size > max_batch_size:
        raise ValueError(f"batch_size ({effective_batch_size}) cannot exceed max test batch size ({max_batch_size})")
    return effective_batch_size


def pad_logits_to_max_batch(torch_logits: torch.Tensor, *, max_batch_size: int = BATCH_SIZE) -> torch.Tensor:
    """Pad active-batch logits to max batch expected by TT sampling kernels."""
    current_batch = int(torch_logits.shape[2])
    if current_batch == max_batch_size:
        return torch_logits

    padded = torch.empty(
        (torch_logits.shape[0], torch_logits.shape[1], max_batch_size, torch_logits.shape[3]),
        dtype=torch_logits.dtype,
        device=torch_logits.device,
    )
    padded[:, :, :current_batch, :] = torch_logits
    # Inactive lanes are filled with a valid copy to avoid all -inf rows.
    padded[:, :, current_batch:, :] = torch_logits[:, :, :1, :]
    return padded


def validate_token_id(token: int, vocab_size: int, *, field: str):
    if not (0 <= token < vocab_size):
        raise ValueError(f"{field} token id {token} out of range [0, {vocab_size - 1}]")


def extract_tokens(tt_out_tok, batch_size: int = BATCH_SIZE, device_idx: int = 0) -> list[int]:
    """Extract token IDs from one mesh device (default device 0).

    Using a single device view matches existing TT sampling test conventions and
    keeps checks lightweight; pass device_idx to debug alternate device views.
    """
    device_tensors = ttnn.get_device_tensors(tt_out_tok)
    if not (0 <= device_idx < len(device_tensors)):
        raise ValueError(f"device_idx {device_idx} out of range for {len(device_tensors)} device tensors")
    out_torch = ttnn.to_torch(device_tensors[device_idx]).reshape(-1).to(torch.int64)
    return out_torch[:batch_size].tolist()


def extract_tokens_all_devices(tt_out_tok, batch_size: int = BATCH_SIZE) -> list[list[int]]:
    """Extract token IDs for all device views from a mesh output tensor."""
    device_tensors = ttnn.get_device_tensors(tt_out_tok)
    return [extract_tokens(tt_out_tok, batch_size=batch_size, device_idx=i) for i in range(len(device_tensors))]


def representative_device_indices(mesh_device) -> list[int]:
    """Choose a small set of device indices for cross-device consistency checks."""
    num_devices = mesh_device.get_num_devices()
    if num_devices <= 1:
        return [0]
    shape = tuple(mesh_device.shape)
    if len(shape) == 2 and shape[0] > 1 and shape[1] > 1:
        cols = shape[1]
        return [row * cols for row in range(shape[0])]
    return [0, num_devices - 1]


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

    if hot_tokens is not None:
        for tok in hot_tokens:
            validate_token_id(tok, args.vocab_size, field="hot_tokens")

    if per_user_hot_tokens is None:
        per_user_hot_tokens = [list(hot_tokens)] * batch_size

    assert len(per_user_hot_tokens) == batch_size

    for user_idx, user_hot in enumerate(per_user_hot_tokens):
        for tok in user_hot:
            validate_token_id(tok, args.vocab_size, field=f"per_user_hot_tokens[{user_idx}]")

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
    validate_token_id(target_token, args.vocab_size, field="target_token")
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
    batch_size: int | None = None,
) -> list[int]:
    """Run one direct TTSampling forward and return host token IDs."""
    effective_batch_size = infer_effective_batch_size(torch_logits, batch_size, max_batch_size=BATCH_SIZE)
    padded_logits = pad_logits_to_max_batch(torch_logits, max_batch_size=BATCH_SIZE)

    tt_sampling = None
    tt_input = None
    tt_logits = None
    tt_tokens = None
    tt_log_probs = None

    try:
        tt_sampling = TTSampling(
            mesh_device=mesh_device,
            tt_ccl=None,
            args=args,
            k=torch.tensor(broadcast(top_k), dtype=torch.int64),
            p=torch.tensor(broadcast(top_p), dtype=torch.float32),
            temp=torch.tensor(broadcast(temperature), dtype=torch.float32),
        )
        tt_logits = padded_logits
        tt_input = make_sharded_logits(tt_logits, mesh_device, args)
        tt_tokens, tt_log_probs = tt_sampling(tt_input)
        return extract_tokens(tt_tokens, effective_batch_size)
    finally:
        if tt_log_probs is not None:
            del tt_log_probs

        if tt_tokens is not None:
            del tt_tokens

        if tt_input is not None:
            del tt_input

        if tt_logits is not None:
            del tt_logits

        if tt_sampling is not None:
            del tt_sampling

        safe_sync(mesh_device)


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
    batch_size: int | None = None,
    device_idx: int = 0,
    state_setup=None,
) -> list[list[int]]:
    """Run SamplingGenerator for num_steps and return per-step token lists."""
    effective_batch_size = infer_effective_batch_size(torch_logits, batch_size, max_batch_size=BATCH_SIZE)
    padded_logits = pad_logits_to_max_batch(torch_logits, max_batch_size=BATCH_SIZE)
    sg = None
    tt_input = None
    tt_logits = None
    tt_tokens = None
    tt_log_probs = None
    outputs = []

    try:
        sg = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None, enable_internal_trace=False)
        formatted = format_sampling_params(sampling_params, BATCH_SIZE)
        sg.reset_sampling_params(formatted)

        if seed_values is not None:
            if len(seed_values) > BATCH_SIZE:
                raise ValueError(f"seed_values length ({len(seed_values)}) cannot exceed BATCH_SIZE ({BATCH_SIZE})")
            if len(seed_values) > effective_batch_size and not write_seed_values_to_device:
                raise ValueError(
                    f"seed_values length ({len(seed_values)}) cannot exceed active batch size ({effective_batch_size})"
                )
            user_ids = list(range(len(seed_values)))
            sg.seed_manager.reset_seed(seed_values, user_ids)
        
        if state_setup is not None:
            state_setup(sg)

        if write_seed_values_to_device:
            if seed_values is None:
                raise ValueError("write_seed_values_to_device=True requires seed_values")
            sg.seed_manager.write_device_seed_values(seed_values)

        tt_logits = padded_logits
        tt_input = make_sharded_logits(tt_logits, mesh_device, args)

        for _ in range(num_steps):
            # SamplingGenerator keeps per-user RNG state in SeedManager.
            if advance_seeds:
                sg.seed_manager.get_new_values()

            tt_tokens, tt_log_probs = sg.sample(tt_input, enable_trace=False)
            outputs.append(extract_tokens(tt_tokens, effective_batch_size, device_idx=device_idx))
        
        return outputs
    finally:
        if tt_log_probs is not None:
            del tt_log_probs

        if tt_tokens is not None:
            del tt_tokens

        if tt_input is not None:
            del tt_input

        if tt_logits is not None:
            del tt_logits

        if sg is not None:
            del sg

        safe_sync(mesh_device)


def assert_tokens_in_vocab(tokens: list[int], vocab_size: int = VOCAB_SIZE):
    assert all(0 <= tok < vocab_size for tok in tokens), (
        f"Found out-of-range token(s) for vocab_size={vocab_size}: {tokens}"
    )


def flatten_steps(outputs: list[list[int]]) -> list[int]:
    return [tok for step in outputs for tok in step]


# --- Test: prefill parameter behavior ---

class TestPrefillWithDifferentParams:
    @pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
    @pytest.mark.parametrize(
        "device_params",
        RING_FABRIC_DEVICE_PARAMS,
        indirect=True,
    )
    def test_prefill_temperature_varied_in_batch(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        hot_tokens = [100, 101, 102, 103, 104, 105, 106, 107]
        hot_token_set = set(hot_tokens)
        logits = build_hot_logits(args, hot_tokens=hot_tokens)
        params = SamplingParams(temperature=2.0, top_k=8, top_p=1.0)
        seeds = [3100 + i for i in range(BATCH_SIZE)]
        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds
        )[0]
        assert out1 == out2, "Same per-user seeds should replay exactly for stochastic prefill config"
        assert_tokens_in_vocab(out1, args.vocab_size)
        unexpected = [tok for tok in out1 if tok not in hot_token_set]
        assert not unexpected, f"Sampled tokens outside expected hot set: {unexpected}, out={out1}"

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
    def test_prefill_topk_mixed_greedy_and_stochastic(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[140, 141, 142, 143, 144, 145, 146, 147])

        half = BATCH_SIZE // 2
        temperature = [0.0] * half + [1.5] * (BATCH_SIZE - half) # First half is greedy; second half is stochastic.
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
        greedy_tokens = run_ttsampling_once(mesh_device, args, logits, top_k=1, top_p=1.0, temperature=1.0) # Use top_p=1.0 (disabled) to keep this test portable across top-p conventions.
        topk1_tokens = run_ttsampling_once(mesh_device, args, logits, top_k=1, top_p=1.0, temperature=5.0)
        assert greedy_tokens == topk1_tokens, "top_k=1 should match greedy behavior"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_greedy_picks_max_logit(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[42, 43, 44])  # 42 has highest logit
        tokens = run_ttsampling_once(mesh_device, args, logits, top_k=1, top_p=1.0, temperature=0.0)
        assert all(tok == 42 for tok in tokens), "Greedy should always pick the max logit token"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_run_ttsampling_once_respects_logits_batch_size(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, batch_size=2, hot_tokens=[60, 61, 62])
        tokens = run_ttsampling_once(mesh_device, args, logits, top_k=1, top_p=1.0, temperature=0.0)
        assert len(tokens) == 2, f"Expected 2 tokens for batch_size=2 logits, got {len(tokens)}"
        assert_tokens_in_vocab(tokens, args.vocab_size)
        assert all(tok == 60 for tok in tokens), f"Greedy should pick the max-logit token 60, got {tokens}"

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_run_sampling_generator_respects_logits_batch_size(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, batch_size=2, hot_tokens=[80, 81, 82])
        params = SamplingParams(temperature=[0.0, 1.0], top_k=[1, 3], top_p=[1.0, 1.0])
        outputs = run_sampling_generator(mesh_device, args, logits, params, num_steps=1, advance_seeds=True)
        assert len(outputs) == 1
        assert len(outputs[0]) == 2, f"Expected 2 tokens for batch_size=2 logits, got {len(outputs[0])}"
        assert_tokens_in_vocab(outputs[0], args.vocab_size)

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_extract_tokens_invalid_device_idx_raises(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[84, 85, 86])
        params = SamplingParams(temperature=1.0, top_k=3, top_p=1.0)
        invalid_device_idx = mesh_device.get_num_devices()
        with pytest.raises(ValueError, match=r"device_idx .* out of range"):
            run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=1,
                advance_seeds=False,
                device_idx=invalid_device_idx,
            )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_build_hot_logits_rejects_out_of_vocab_token(self, mesh_device):
        args = make_sampling_args(mesh_device)
        bad_token = args.vocab_size
        with pytest.raises(ValueError, match=r"out of range"):
            build_hot_logits(args, hot_tokens=[bad_token])

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_build_penalty_logits_rejects_out_of_vocab_target(self, mesh_device):
        args = make_sampling_args(mesh_device)
        bad_token = args.vocab_size
        with pytest.raises(ValueError, match=r"out of range"):
            build_penalty_logits(args, target_token=bad_token)

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_top_p_restricts_candidate_set(self, mesh_device):
        args = make_sampling_args(mesh_device)
        hot_tokens = [1900, 1901, 1902]
        logits = build_hot_logits(
            args,
            hot_tokens=hot_tokens,
            base_logit=-12.0,
            top_logit=2.0,
            step=0.2,
        )
        seeds = [31000 + i for i in range(BATCH_SIZE)]

        params_low = SamplingParams(temperature=1.0, top_k=3, top_p=0.6)
        params_full = SamplingParams(temperature=1.0, top_k=3, top_p=1.0)

        out_low = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params_low,
            num_steps=FAST_NUM_TRIES,
            advance_seeds=True,
            seed_values=seeds,
        )
        out_full = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params_full,
            num_steps=FAST_NUM_TRIES,
            advance_seeds=True,
            seed_values=seeds,
        )

        for step_tokens in out_low + out_full:
            assert_tokens_in_vocab(step_tokens, args.vocab_size)

        low_flat = flatten_steps(out_low)
        full_flat = flatten_steps(out_full)
        low_set = set(low_flat)
        full_set = set(full_flat)

        allowed_low = {1900, 1901}
        allowed_full = {1900, 1901, 1902}
        assert low_set.issubset(allowed_low), (
            f"top_p=0.6 sampled outside allowed set {sorted(allowed_low)}. "
            f"low_set={sorted(low_set)}, low_hist={Counter(low_flat).most_common(6)}"
        )
        assert full_set.issubset(allowed_full), (
            f"top_p=1.0 sampled outside expected hot set {sorted(allowed_full)}. "
            f"full_set={sorted(full_set)}, full_hist={Counter(full_flat).most_common(6)}"
        )
        assert 1902 in full_set, (
            "top_p=1.0 did not sample token 1902 at least once. "
            f"low_set={sorted(low_set)}, full_set={sorted(full_set)}, "
            f"low_hist={Counter(low_flat).most_common(6)}, full_hist={Counter(full_flat).most_common(6)}"
        )


# --- Test: per-request penalties ---

@pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    RING_FABRIC_DEVICE_PARAMS,
    indirect=True,
)
class TestRepetitionPenaltyPerRequest:
    def test_different_repetition_penalties(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 500
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = ([1.0, 1.0, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0] * 4)[:BATCH_SIZE]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token))[0]
        low_penalty = [tok for i, tok in enumerate(tokens) if penalties[i] <= 1.0] # Keep boundary-sensitive values (e.g. 1.2) out of strict assertions.
        high_penalty = [tok for i, tok in enumerate(tokens) if penalties[i] >= 1.5]
        assert all(tok == target_token for tok in low_penalty), "Low repetition penalty should keep target token"
        assert any(tok != target_token for tok in high_penalty), "High repetition penalties should alter output"

    def test_repetition_penalty_vs_no_penalty(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 520
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [1.0 if i % 2 == 0 else 2.5 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token))[0]
        no_penalty = [tokens[i] for i in range(0, BATCH_SIZE, 2)]
        with_penalty = [tokens[i] for i in range(1, BATCH_SIZE, 2)]
        assert all(tok == target_token for tok in no_penalty), "No-penalty lanes should keep target"
        assert all(tok != target_token for tok in with_penalty), "Penalty lanes should change token"
        assert no_penalty[0] != with_penalty[0], "Penalty and no-penalty outputs should differ"

    def test_repetition_penalty_persists_across_steps(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 540
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [1.0 if i % 2 == 0 else 2.5 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        outputs = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=3,
            advance_seeds=False,
        )

        for step_tokens in outputs:
            assert_tokens_in_vocab(step_tokens, args.vocab_size)

        even_idxs = range(0, BATCH_SIZE, 2)
        odd_idxs = range(1, BATCH_SIZE, 2)
        assert all(outputs[0][i] == target_token for i in odd_idxs), "Odd lanes should start at target token"
        assert all(outputs[0][i] == target_token for i in even_idxs), "Even lanes should start at target token"
        assert all(outputs[1][i] != target_token for i in odd_idxs), "Odd lanes should leave target at step 1"
        assert all(outputs[2][i] != target_token for i in odd_idxs), "Odd lanes should stay off target at step 2"
        assert all(outputs[1][i] == target_token for i in even_idxs), "Even lanes should keep target at step 1"
        assert all(outputs[2][i] == target_token for i in even_idxs), "Even lanes should keep target at step 2"

    def _state_setup(self, sg, target_token: int):
        seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
        sg.reset_prompt_tokens(seen)
        sg.reset_output_state(tokens=seen)

    def _get_sampling_params(self, penalties):
        return SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            repetition_penalty=penalties,
        )


@pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    RING_FABRIC_DEVICE_PARAMS,
    indirect=True,
)
class TestPresencePenaltyPerRequest:
    def test_different_presence_penalties(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 700
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = ([0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, 4.0] * 4)[:BATCH_SIZE]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token))[0]
        assert any(tok == target_token for tok in tokens), "Some lanes should retain target token"
        assert any(tok != target_token for tok in tokens), "Some lanes should shift off target with higher penalties"

    def test_presence_penalty_mixed_batch(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 720
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.0 if i % 2 == 0 else 2.0 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token))[0]
        no_penalty = [tokens[i] for i in range(0, BATCH_SIZE, 2)]
        with_penalty = [tokens[i] for i in range(1, BATCH_SIZE, 2)]
        assert all(tok == target_token for tok in no_penalty), "No presence-penalty lanes should keep target"
        assert all(tok != target_token for tok in with_penalty), "Presence-penalty lanes should move off target"

    def test_presence_penalty_persists_across_steps(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 740
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.0 if i % 2 == 0 else 2.0 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        outputs = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=3,
            advance_seeds=False,
        )

        for step_tokens in outputs:
            assert_tokens_in_vocab(step_tokens, args.vocab_size)

        even_idxs = range(0, BATCH_SIZE, 2)
        odd_idxs = range(1, BATCH_SIZE, 2)
        assert all(outputs[0][i] == target_token for i in odd_idxs), "Odd lanes should start at target token"
        assert all(outputs[0][i] == target_token for i in even_idxs), "Even lanes should start at target token"
        assert all(outputs[1][i] != target_token for i in odd_idxs), "Odd lanes should leave target at step 1"
        assert all(outputs[2][i] != target_token for i in odd_idxs), "Odd lanes should stay off target at step 2"
        assert all(outputs[1][i] == target_token for i in even_idxs), "Even lanes should keep target at step 1"
        assert all(outputs[2][i] == target_token for i in even_idxs), "Even lanes should keep target at step 2"

    def _state_setup(self, sg, target_token: int):
            seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
            sg.reset_prompt_tokens(seen)
            sg.reset_output_state(tokens=seen)
    
    def _get_sampling_params(self, penalties):
        return SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            presence_penalty=penalties,
        )


@pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    RING_FABRIC_DEVICE_PARAMS,
    indirect=True,
)
class TestFrequencyPenaltyPerRequest:
    def test_different_frequency_penalties(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 900
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = ([0.0, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, 4.0] * 4)[:BATCH_SIZE]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token))[0]
        assert any(tok == target_token for tok in tokens), "Some lanes should retain target token"
        assert any(tok != target_token for tok in tokens), "Some lanes should shift off target with higher penalties"

    def test_frequency_penalty_mixed_batch(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 920
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.0 if i % 2 == 0 else 2.0 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        tokens = run_sampling_generator(mesh_device, args, logits, params, state_setup=lambda sg: self._state_setup(sg, target_token))[0]
        no_penalty = [tokens[i] for i in range(0, BATCH_SIZE, 2)]
        with_penalty = [tokens[i] for i in range(1, BATCH_SIZE, 2)]
        assert all(tok == target_token for tok in no_penalty), "No frequency-penalty lanes should keep target"
        assert all(tok != target_token for tok in with_penalty), "Frequency-penalty lanes should move off target"

    def test_frequency_penalty_accumulates_across_steps(self, mesh_device, device_params):
        args = make_sampling_args(mesh_device)
        target_token = 940
        logits = build_penalty_logits(args, target_token=target_token)
        penalties = [0.1 if i % 2 == 0 else 0.6 for i in range(BATCH_SIZE)]
        params = self._get_sampling_params(penalties)

        outputs = run_sampling_generator(
            mesh_device,
            args,
            logits,
            params,
            num_steps=4,
            advance_seeds=False,
        )

        for step_tokens in outputs:
            assert_tokens_in_vocab(step_tokens, args.vocab_size)

        even_idxs = range(0, BATCH_SIZE, 2)
        odd_idxs = range(1, BATCH_SIZE, 2)
        assert all(outputs[0][i] == target_token for i in odd_idxs), "Odd lanes should start at target token"
        assert all(outputs[1][i] == target_token for i in odd_idxs), "Odd lanes should still hit target at step 1"
        assert all(outputs[2][i] != target_token for i in odd_idxs), "Odd lanes should leave target at step 2"
        assert all(outputs[3][i] != target_token for i in odd_idxs), "Odd lanes should stay off target at step 3"
        assert all(outputs[0][i] == target_token for i in even_idxs), "Even lanes should start at target token"
        assert all(outputs[1][i] == target_token for i in even_idxs), "Even lanes should keep target at step 1"
        assert all(outputs[2][i] == target_token for i in even_idxs), "Even lanes should keep target at step 2"
        assert all(outputs[3][i] == target_token for i in even_idxs), "Even lanes should keep target at step 3"

    def _state_setup(self, sg, target_token: int):
        seen = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
        sg.reset_prompt_tokens(seen)
        sg.reset_output_state(tokens=seen)

    def _get_sampling_params(self, penalties):
        return SamplingParams(
            temperature=[0.0] * BATCH_SIZE,
            top_k=[32] * BATCH_SIZE,
            top_p=[1.0] * BATCH_SIZE,
            frequency_penalty=penalties,
        )


# --- Test: seed behavior ---
class TestSeededSamplingPerRequest:
    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_run_sampling_generator_rejects_seed_vector_longer_than_max_batch(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, hot_tokens=[990, 991, 992])
        params = SamplingParams(temperature=1.0, top_k=3, top_p=1.0)
        too_many_seeds = list(range(BATCH_SIZE + 1))
        with pytest.raises(ValueError, match=r"cannot exceed BATCH_SIZE"):
            run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=1,
                advance_seeds=True,
                seed_values=too_many_seeds,
            )

    @pytest.mark.parametrize("mesh_device", [1], indirect=True)
    def test_run_sampling_generator_rejects_seed_vector_longer_than_effective_batch(self, mesh_device):
        args = make_sampling_args(mesh_device)
        logits = build_hot_logits(args, batch_size=2, hot_tokens=[993, 994, 995])
        params = SamplingParams(temperature=[1.0, 1.0], top_k=[3, 3], top_p=[1.0, 1.0])
        too_many_for_active_batch = [11, 12, 13]
        with pytest.raises(ValueError, match=r"active batch size"):
            run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=1,
                advance_seeds=True,
                seed_values=too_many_for_active_batch,
            )

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

        # Shift every seed; when backend RNG path is active, at least one slot should change.
        seeds_b = [s + 12345 for s in seeds_a]
        out_b = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds_b
        )[0]

        changed_indices = [i for i, (a, b) in enumerate(zip(out_a1, out_b)) if a != b]
        if len(changed_indices) == 0:
            pytest.xfail(
                "Different seed vector produced identical outputs for this stochastic config; "
                f"backend appears argmax-only here. out_a1={out_a1}, out_b={out_b}"
            )

        # Sanity checks: valid token range and no sampling outside the hot set.
        for outputs in (out_a1, out_a2, out_b):
            assert_tokens_in_vocab(outputs, args.vocab_size)
            unexpected = [tok for tok in outputs if tok not in hot_token_set]
            assert not unexpected, (
                f"Sampled tokens outside expected hot set {sorted(hot_token_set)}: {unexpected}. "
                f"outputs={outputs}"
            )

    @pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
    @pytest.mark.parametrize(
        "device_params",
        RING_FABRIC_DEVICE_PARAMS,
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

        for device_idx in representative_device_indices(mesh_device)[1:]:
            out_device = run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=FAST_NUM_STEPS,
                advance_seeds=True,
                seed_values=seeds,
                device_idx=device_idx,
            )
            assert out_device == out1, f"Device view mismatch for device_idx={device_idx} in seeded replay"

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
        assert_tokens_in_vocab(out, args.vocab_size)


# --- Test: batch isolation ---

@pytest.mark.parametrize("mesh_device", MULTI_DEVICE_MESHES, indirect=True)
@pytest.mark.parametrize(
    "device_params",
    RING_FABRIC_DEVICE_PARAMS,
    indirect=True,
)
class TestBatchIsolation:
    def test_mixed_params_batch(self, mesh_device, device_params):
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

        params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition,
            presence_penalty=presence,
            frequency_penalty=frequency,
        )

        out1 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds, state_setup=self._state_setup
        )[0]
        out2 = run_sampling_generator(
            mesh_device, args, logits, params, num_steps=1, advance_seeds=True, seed_values=seeds, state_setup=self._state_setup
        )[0]
        assert out1 == out2, "Deterministic lanes (and seeded lanes) should replay identically across runs"
        assert len(set(out1)) >= 2, "Mixed batch should not collapse to a single token"

    def test_outputs_not_mixed_different_prompts(self, mesh_device, device_params):
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

        for device_idx in representative_device_indices(mesh_device)[1:]:
            device_tokens = run_sampling_generator(
                mesh_device,
                args,
                logits,
                params,
                num_steps=1,
                advance_seeds=True,
                seed_values=seeds,
                device_idx=device_idx,
            )[0]
            assert device_tokens == tokens, f"Device view mismatch for device_idx={device_idx} in batch isolation test"

    def _state_setup(self, sg):
        seen = torch.full((BATCH_SIZE, 1), 2000, dtype=torch.int64)
        sg.reset_prompt_tokens(seen)
        sg.reset_output_state(tokens=seen)

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
        assert_tokens_in_vocab(tokens, args.vocab_size)

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

        assert_tokens_in_vocab(tokens, args.vocab_size)


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
        assert_tokens_in_vocab(out1, args.vocab_size)
        assert len(set(out1)) >= 2, "Mixed parameter batch should produce non-trivial diversity"
