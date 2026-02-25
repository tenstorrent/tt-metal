# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end on-device sampling tests for GPT-OSS on Galaxy [4,8] mesh.

Tests TTSampling, TTPenalties, and LogProbsCalculator with GPT-OSS
vocab dimensions (201088, TP=8) to verify:
- Greedy (argmax) sampling matches torch reference
- Stochastic sampling respects top-k/top-p constraints
- Presence/frequency/repetition penalties suppress targeted tokens
- Log probabilities are correctly computed
- Sampled token IDs are always < vocab_size (no padding tokens leak through)
"""

from collections import Counter

import pytest
import torch
from loguru import logger

import ttnn
from models.common.sampling.generator import SamplingGenerator, SamplingParams, format_sampling_params
from models.common.sampling.tt_sampling import TTSampling
from models.demos.gpt_oss.tt.model import compute_per_device_vocab

# --- Reference implementation ---


def reference_sampling(input_tensor, sampling_params, num_devices, padded_vocab_size, max_top_k):
    """Reference sampling that mirrors TTSampling's multi-device top-k gather logic."""
    per_device_offset = input_tensor.shape[-1] // num_devices

    tt_indices_device_offsets = torch.ones([1, 1, 32, max_top_k * num_devices], dtype=torch.int32)
    for device_id in range(num_devices):
        tt_indices_device_offsets[:, :, :, device_id * max_top_k : (device_id + 1) * max_top_k] = (
            device_id * per_device_offset
        )

    # Per-device top-k
    per_device_tensors = torch.split(input_tensor, per_device_offset, dim=-1)
    topk_values_list = []
    topk_indices_list = []
    for i in range(num_devices):
        topk_values, topk_indices = torch.topk(per_device_tensors[i], k=max_top_k, dim=-1)
        topk_values_list.append(topk_values)
        topk_indices_list.append(topk_indices)

    topk_values_tensor = torch.cat(topk_values_list, dim=3)
    topk_indices_tensor = torch.cat(topk_indices_list, dim=3)
    topk_indices_tensor += tt_indices_device_offsets

    # Apply temperature
    temperature = sampling_params["temperature"]
    if temperature != 0.0:
        topk_values_tensor /= temperature

    # Global top-k on gathered
    k_final = sampling_params["top_k"] if sampling_params["temperature"] != 0.0 else 1
    topk_values_gathered, topk_indices_gathered = torch.topk(topk_values_tensor, k=k_final, dim=-1)
    topk_indices_gathered = torch.gather(topk_indices_tensor, dim=-1, index=topk_indices_gathered)
    topk_values_gathered = topk_values_gathered[0, 0, :, :]

    # Sample
    if sampling_params["temperature"] == 0.0:
        sampled_indices = torch.argmax(topk_values_gathered, dim=-1, keepdim=True)
    else:
        # Greedy for reference to match device argmax
        sampled_indices = torch.argmax(topk_values_gathered, dim=-1, keepdim=True)

    sampled_indices = torch.gather(topk_indices_gathered.squeeze(0).squeeze(0), dim=-1, index=sampled_indices)
    return sampled_indices


# --- Constants & helpers ---

VOCAB_SIZE = 201088
MAX_TOP_K = 32
BATCH_SIZE = 32

# GPT-OSS device params: FABRIC_1D_RING + large trace region.
# NOT Llama Galaxy's dispatch_core_axis/worker_l1_size/small trace.
GPT_OSS_DEVICE_PARAMS = {
    "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
    "trace_region_size": 30000000,
}


def make_gpt_oss_sampling_args(mesh_device, sampling_dp=1):
    """Create args matching GPT-OSS model on Galaxy [4,8] mesh.

    Args:
        mesh_device: TTNN mesh device.
        sampling_dp: Number of independent sampling groups (1 for basic tests,
            mesh_device.shape[0] for row-sharded production config).
    """

    class _Args:
        pass

    args = _Args()
    args.vocab_size = VOCAB_SIZE
    num_tp = mesh_device.shape[1]  # TP along cols
    per_device_vocab = compute_per_device_vocab(args.vocab_size, num_tp)
    args.padded_vocab_size = per_device_vocab * num_tp
    args.cluster_shape = tuple(mesh_device.shape)
    args.sampling_all_gather_axis = 1  # gather along cols (TP axis)
    args.num_devices = mesh_device.get_num_devices()
    args.is_galaxy = mesh_device.shape[0] > 1
    args.model_config = {}
    args.sampling_dp = sampling_dp
    args.max_top_k = MAX_TOP_K
    args.sub_core_grids = None
    return args


def make_sharded_logits(torch_input, mesh_device, cluster_shape, dtype=ttnn.bfloat8_b):
    """Shard logits across cols (TP axis) for GPT-OSS [4,8] mesh.

    GPT-OSS shards vocab across mesh columns (axis 1). dims=(None, 3) means
    replicate across rows, shard tensor dim 3 across cols.
    """
    return ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 3),  # replicate rows, shard vocab across cols
            mesh_shape=cluster_shape,
        ),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )


def extract_token(tt_out_tok, device_idx=0):
    """Extract user-0 token ID from a multi-device sampling output tensor."""
    device_tensor = ttnn.get_device_tensors(tt_out_tok)[device_idx]
    torch_tensor = ttnn.to_torch(device_tensor)
    return torch_tensor[0, 0, :, :].reshape(-1, 1)[0].item()


def extract_all_tokens(tt_out_tok, batch_size, device_idx=0):
    """Extract token IDs for all batch users from a multi-device output tensor."""
    device_tensor = ttnn.get_device_tensors(tt_out_tok)[device_idx]
    torch_tensor = ttnn.to_torch(device_tensor)
    return torch_tensor[0, 0, :, :].reshape(-1)[:batch_size].tolist()


def make_hot_logits(args, mesh_device, batch_size, num_hot=8):
    """Create logits with controlled hot tokens spread across TP shards.

    Returns (tt_input, hot_tokens) where tt_input is a sharded device tensor
    and hot_tokens is the list of token IDs that have elevated logit values.
    Hot tokens are placed in different shards so the all-gather path is
    exercised, with varied logit values (10.0, 9.75, 9.5, ...) to produce
    a non-uniform softmax distribution.
    """
    num_tp = mesh_device.shape[1]
    per_device_vocab = args.padded_vocab_size // num_tp

    hot_tokens = []
    for i in range(num_hot):
        shard_idx = i % num_tp
        candidate = shard_idx * per_device_vocab + 100 + i
        if candidate >= VOCAB_SIZE:
            candidate = 200 + i
        hot_tokens.append(candidate)

    torch_input = torch.full((1, 1, batch_size, args.padded_vocab_size), 0.001)
    for idx, tok in enumerate(hot_tokens):
        torch_input[:, :, :, tok] = 10.0 - idx * 0.25
    torch_input[:, :, :, VOCAB_SIZE:] = -float("inf")

    tt_input = make_sharded_logits(torch_input, mesh_device, args.cluster_shape, dtype=ttnn.bfloat16)
    return tt_input, hot_tokens


# --- Test: greedy (argmax) sampling ---


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
@pytest.mark.parametrize(
    "sampling_params",
    [
        {"temperature": 0.0, "top_k": 32, "top_p": 0.00, "seed": 42},  # Greedy
        {"temperature": 0.0, "top_k": 32, "top_p": 0.95, "seed": 42},  # Greedy (top_p ignored)
        {"temperature": 1.0, "top_k": 1, "top_p": 0.00, "seed": 42},  # top-k=1 (always argmax)
    ],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [GPT_OSS_DEVICE_PARAMS], indirect=True)
def test_gpt_oss_greedy_sampling(sampling_params, batch_size, mesh_device, device_params, reset_seeds):
    """Test greedy (argmax) on-device sampling matches torch reference."""
    args = make_gpt_oss_sampling_args(mesh_device, sampling_dp=1)
    max_top_k = args.max_top_k
    num_tp = mesh_device.shape[1]

    # Prepare sampling parameters
    top_k = sampling_params["top_k"]
    if isinstance(top_k, int):
        top_k = [top_k] * batch_size
    top_p = sampling_params["top_p"]
    if isinstance(top_p, float):
        top_p = [top_p] * batch_size
    temperature = sampling_params["temperature"]
    if temperature == 0.0:
        temperature = 1.0
        top_k = [1] * batch_size
        top_p = [0.0] * batch_size
    if isinstance(temperature, float):
        temperature = [temperature] * batch_size
    seed = sampling_params["seed"]

    # Create random logits with GPT-OSS padded vocab dimensions
    torch_input = torch.randn(1, 1, batch_size, args.padded_vocab_size)
    torch_input[:, :, :, VOCAB_SIZE:] = -float("inf")

    # Reference argmax — split across num_tp (cols) to match device TP sharding
    ref = reference_sampling(torch_input, sampling_params, num_tp, args.padded_vocab_size, max_top_k)
    ref_token = ref[0].item()

    # Shard input and run device sampling
    tt_input = make_sharded_logits(torch_input, mesh_device, args.cluster_shape, dtype=ttnn.bfloat8_b)

    tt_sampling = TTSampling(
        args=args,
        mesh_device=mesh_device,
        tt_ccl=None,
        k=torch.tensor(top_k),
        p=torch.tensor(top_p),
        temp=torch.tensor(temperature),
    )

    ttnn.manual_seed(seed, device=mesh_device, sub_core_grids=args.sub_core_grids)
    tt_out_tok, _tt_log_probs = tt_sampling(tt_input)
    device_token = extract_token(tt_out_tok)

    assert 0 <= device_token < VOCAB_SIZE, f"Sampled token {device_token} outside vocab range [0, {VOCAB_SIZE})"
    assert (
        device_token == ref_token
    ), f"Argmax mismatch: ref={ref_token}, device={device_token}, params={sampling_params}"
    logger.info(f"Greedy sampling test passed: token={device_token}")


# --- Test: stochastic sampling (top-k/top-p) ---


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
@pytest.mark.parametrize(
    "sampling_params",
    [
        {"temperature": 1.0, "top_k": 8, "top_p": 0.95, "seed": 42},  # Small top-k
        {"temperature": 1.0, "top_k": 32, "top_p": 0.50, "seed": 42},  # Tight top-p
        {"temperature": 1.0, "top_k": 32, "top_p": 0.95, "seed": 42},  # Standard
        {"temperature": 1.0, "top_k": 32, "top_p": 1.00, "seed": 42},  # No top-p filter
    ],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [GPT_OSS_DEVICE_PARAMS], indirect=True)
def test_gpt_oss_stochastic_sampling(sampling_params, batch_size, mesh_device, device_params, reset_seeds):
    """Test stochastic sampling with controlled logits to verify top-k/top-p behavior.

    Uses logits with a known set of "hot" tokens (high logit values) and low
    baseline for all others. Verifies the device only samples from the hot set
    and produces varied output (not stuck on one token).
    """
    args = make_gpt_oss_sampling_args(mesh_device, sampling_dp=1)
    num_samples = 100

    top_k = sampling_params["top_k"]
    top_p = sampling_params["top_p"]
    temperature = sampling_params["temperature"]
    seed = sampling_params["seed"]

    num_hot = min(top_k, 8)
    tt_input, hot_tokens = make_hot_logits(args, mesh_device, batch_size, num_hot=num_hot)

    # SamplingGenerator manages seeds between iterations (TTSampling re-seeds
    # from seeds_tt_tensor each forward call, so seeds must be updated via
    # SeedManager.get_new_values() to get different random samples).
    sg = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None, enable_internal_trace=False)
    params = format_sampling_params(
        SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p, seed=seed),
        batch_size,
    )
    sg.reset_sampling_params(params)

    # Run device sampling
    sampled_tokens = []
    for _ in range(num_samples):
        sg.seed_manager.get_new_values()
        tokens, _ = sg.sample(tt_input, enable_trace=False)
        token_id = extract_token(tokens)
        sampled_tokens.append(token_id)

    # --- Verify properties ---
    hot_set = set(hot_tokens)

    # 1. All tokens in valid range
    for token_id in sampled_tokens:
        assert 0 <= token_id < VOCAB_SIZE, f"Token {token_id} outside vocab range [0, {VOCAB_SIZE})"

    # 2. All tokens should be from the hot set (they dominate by 10.0 vs 0.001)
    sampled_set = set(sampled_tokens)
    unexpected = sampled_set - hot_set
    assert not unexpected, (
        f"Device sampled tokens not in hot set: {unexpected}. " f"Hot set: {hot_set}, sampled unique: {sampled_set}"
    )

    # 3. Should have variety (not stuck on one token) — at least 2 unique tokens
    unique_count = len(sampled_set)
    assert unique_count >= 2, (
        f"Only {unique_count} unique token(s) in {num_samples} samples — "
        f"sampling may be stuck. Tokens: {Counter(sampled_tokens).most_common(5)}"
    )

    logger.info(
        f"Stochastic sampling test passed: top_k={top_k}, top_p={top_p}, "
        f"{unique_count} unique tokens from {num_samples} samples, "
        f"all in hot set of {len(hot_set)}"
    )


# --- Test: penalties suppress previously-seen tokens ---


@torch.no_grad()
@pytest.mark.parametrize(
    "penalty_params",
    [
        {"presence_penalty": 1000.0, "frequency_penalty": 0.0, "repetition_penalty": 1.0},
        {"presence_penalty": 0.0, "frequency_penalty": 1000.0, "repetition_penalty": 1.0},
        {"presence_penalty": 0.0, "frequency_penalty": 0.0, "repetition_penalty": 1000.0},
    ],
    ids=["presence", "frequency", "repetition"],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [GPT_OSS_DEVICE_PARAMS], indirect=True)
def test_gpt_oss_penalties(penalty_params, mesh_device, device_params, reset_seeds):
    """Test that penalties suppress previously-seen tokens.

    Creates logits where a single target token dominates (logit=10.0 vs 0.1
    for all others). After marking the target as seen in both prompt and
    output, greedy sampling with a large penalty value should select a
    different token.

    Presence penalty subtracts from output tokens, frequency penalty subtracts
    proportional to count, and repetition penalty divides positive logits.
    """
    args = make_gpt_oss_sampling_args(mesh_device, sampling_dp=1)
    target_token = 5000

    # All valid tokens at 0.1, target dominates at 10.0
    torch_input = torch.full((1, 1, BATCH_SIZE, args.padded_vocab_size), 0.1)
    torch_input[:, :, :, target_token] = 10.0
    torch_input[:, :, :, VOCAB_SIZE:] = -float("inf")

    # Confirm target_token is the host-side argmax
    assert torch.argmax(torch_input[0, 0, 0, :]).item() == target_token

    # Shard and create SamplingGenerator with penalties
    tt_input = make_sharded_logits(torch_input, mesh_device, args.cluster_shape, dtype=ttnn.bfloat16)

    sg = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None, enable_internal_trace=False)

    params = format_sampling_params(
        SamplingParams(
            temperature=0.0,
            top_k=32,
            top_p=0.0,
            presence_penalty=penalty_params["presence_penalty"],
            frequency_penalty=penalty_params["frequency_penalty"],
            repetition_penalty=penalty_params["repetition_penalty"],
            seed=42,
        ),
        BATCH_SIZE,
    )
    sg.reset_sampling_params(params)
    sg.seed_manager.get_new_values()

    # Mark target_token as seen in both prompt and output so all penalty
    # types can trigger (presence/frequency use output_mask/output_counts,
    # repetition uses prompt_mask + output_mask).
    target_tokens = torch.full((BATCH_SIZE, 1), target_token, dtype=torch.int64)
    sg.reset_prompt_tokens(target_tokens)
    sg.reset_output_state(tokens=target_tokens)

    tokens, _log_probs = sg.sample(tt_input, enable_trace=False)
    token = extract_token(tokens)

    assert token != target_token, (
        f"Penalty {penalty_params} failed to suppress token {target_token} — " f"got {token}. Penalty not effective."
    )
    assert 0 <= token < VOCAB_SIZE, f"Penalized token {token} outside vocab range"
    logger.info(f"Penalty test passed: {penalty_params} changed {target_token} -> {token}")


# --- Test: log probabilities ---


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [GPT_OSS_DEVICE_PARAMS], indirect=True)
def test_gpt_oss_logprobs(mesh_device, device_params, reset_seeds):
    """Test that log probability calculation produces valid results.

    Verifies that with enable_log_probs=True:
    - Log probs are finite
    - Log probs are <= 0 (log of probability)
    - Log probs are not all identical (actual computation ran)
    """
    args = make_gpt_oss_sampling_args(mesh_device, sampling_dp=1)

    torch_input = torch.randn(1, 1, BATCH_SIZE, args.padded_vocab_size)
    torch_input[:, :, :, VOCAB_SIZE:] = -float("inf")

    tt_input = make_sharded_logits(torch_input, mesh_device, args.cluster_shape, dtype=ttnn.bfloat16)

    sg = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None, enable_internal_trace=False)

    params = format_sampling_params(
        SamplingParams(
            temperature=1.0,
            top_k=32,
            top_p=0.95,
            enable_log_probs=True,
            seed=42,
        ),
        BATCH_SIZE,
    )
    sg.reset_sampling_params(params)
    sg.seed_manager.get_new_values()

    tokens, log_probs = sg.sample(tt_input, enable_trace=False)

    # Extract log probs from first device
    log_probs_device = ttnn.get_device_tensors(log_probs)[0]
    log_probs_torch = ttnn.to_torch(log_probs_device).float()

    logger.info(f"Log probs shape: {log_probs_torch.shape}")
    logger.info(f"Log probs range: [{log_probs_torch.min().item():.4f}, {log_probs_torch.max().item():.4f}]")

    assert torch.isfinite(log_probs_torch).all(), "Log probs contain non-finite values"
    assert (log_probs_torch <= 0).all(), f"Log probs should be <= 0, got max={log_probs_torch.max().item():.4f}"

    unique_vals = log_probs_torch.unique()
    assert len(unique_vals) > 1, "All log probs identical — computation may not have run"

    # Extract token to verify it's valid
    token = extract_token(tokens)
    assert 0 <= token < VOCAB_SIZE, f"Sampled token {token} outside vocab range"

    logger.info("Log probs test passed!")


# --- Test: seed determinism ---


def _make_seeded_generator(args, mesh_device, batch_size, per_user_seeds):
    """Create a SamplingGenerator with specific per-user seeds for stochastic sampling.

    Seeds the per-user Python RNGs but does NOT call get_new_values() — the
    caller is responsible for calling it before each sample() to advance the
    RNG state and copy fresh seeds to the device.
    """
    sg = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=None, enable_internal_trace=False)
    params = format_sampling_params(
        SamplingParams(temperature=1.0, top_k=32, top_p=0.95, seed=0),
        batch_size,
    )
    sg.reset_sampling_params(params)
    sg.seed_manager.reset_seed(per_user_seeds, list(range(len(per_user_seeds))))
    return sg


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [GPT_OSS_DEVICE_PARAMS], indirect=True)
def test_gpt_oss_seed_determinism_batch_replay(mesh_device, device_params, reset_seeds):
    """Test that replaying the same seeds reproduces identical tokens for all users.

    Seeds all 32 users with distinct per-slot seeds, runs a multi-iteration
    decode, then repeats with a fresh generator and the same seeds.  Every
    user slot should produce the same token sequence across both runs.

    Note: ttnn.manual_seed mixes user_id into the RNG state, so two *different*
    user slots with the same seed value will produce different tokens.  This is
    by design — it prevents identical outputs for different users that happen
    to share a seed.
    """
    batch_size = BATCH_SIZE
    args = make_gpt_oss_sampling_args(mesh_device, sampling_dp=1)
    tt_input, _ = make_hot_logits(args, mesh_device, batch_size)

    per_user_seeds = [100 + i for i in range(batch_size)]
    num_iterations = 5

    # --- Run 1 ---
    sg1 = _make_seeded_generator(args, mesh_device, batch_size, per_user_seeds)
    run1_all = []
    for _ in range(num_iterations):
        sg1.seed_manager.get_new_values()
        tokens, _ = sg1.sample(tt_input, enable_trace=False)
        run1_all.append(extract_all_tokens(tokens, batch_size))

    # --- Run 2 (fresh generator, same seeds) ---
    sg2 = _make_seeded_generator(args, mesh_device, batch_size, per_user_seeds)
    run2_all = []
    for _ in range(num_iterations):
        sg2.seed_manager.get_new_values()
        tokens, _ = sg2.sample(tt_input, enable_trace=False)
        run2_all.append(extract_all_tokens(tokens, batch_size))

    # Every user at every iteration should match
    for it in range(num_iterations):
        for u in range(batch_size):
            assert run1_all[it][u] == run2_all[it][u], (
                f"Mismatch at iteration {it}, user {u}: " f"run1={run1_all[it][u]}, run2={run2_all[it][u]}"
            )

    # Verify we actually sampled varied tokens (not all stuck on one)
    flat = [tok for row in run1_all for tok in row]
    unique = set(flat)
    assert len(unique) > 1, f"All {len(flat)} tokens identical — stochastic sampling may be broken"

    logger.info(
        f"Batch replay seed determinism passed: {num_iterations} iterations x {batch_size} users, "
        f"{len(unique)} unique tokens, both runs matched exactly"
    )


@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize("device_params", [GPT_OSS_DEVICE_PARAMS], indirect=True)
def test_gpt_oss_seed_determinism_across_requests(mesh_device, device_params, reset_seeds):
    """Test that the same seed produces the same token sequence across independent requests.

    Runs the full sampling pipeline twice with identical setup (same logits, same
    seed, same parameters). The two runs should produce identical token sequences,
    verifying that seed-based determinism works end-to-end.
    """
    batch_size = BATCH_SIZE
    args = make_gpt_oss_sampling_args(mesh_device, sampling_dp=1)
    tt_input, _ = make_hot_logits(args, mesh_device, batch_size)

    per_user_seeds = [42] * batch_size
    num_iterations = 10

    # --- Run 1 ---
    sg1 = _make_seeded_generator(args, mesh_device, batch_size, per_user_seeds)
    run1_all = []
    for _ in range(num_iterations):
        sg1.seed_manager.get_new_values()
        tokens, _ = sg1.sample(tt_input, enable_trace=False)
        run1_all.append(extract_all_tokens(tokens, batch_size))

    # --- Run 2 (fresh generator, same seeds) ---
    sg2 = _make_seeded_generator(args, mesh_device, batch_size, per_user_seeds)
    run2_all = []
    for _ in range(num_iterations):
        sg2.seed_manager.get_new_values()
        tokens, _ = sg2.sample(tt_input, enable_trace=False)
        run2_all.append(extract_all_tokens(tokens, batch_size))

    # Every user at every iteration should match
    for it in range(num_iterations):
        for u in range(batch_size):
            assert run1_all[it][u] == run2_all[it][u], (
                f"Mismatch at iteration {it}, user {u}: " f"run1={run1_all[it][u]}, run2={run2_all[it][u]}"
            )

    logger.info(
        f"Cross-request seed determinism passed: {num_iterations} iterations x {batch_size} users matched exactly"
    )
