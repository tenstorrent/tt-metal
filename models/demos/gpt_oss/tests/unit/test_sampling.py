# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end on-device sampling test for GPT-OSS on Galaxy [4,8] mesh.

Tests TTSampling with GPT-OSS vocab dimensions (201088, TP=8) to verify:
- Greedy (argmax) sampling matches torch reference
- Stochastic sampling token distribution matches reference within KL threshold
- Sampled token IDs are always < vocab_size (no padding tokens leak through)
"""

from collections import Counter

import numpy as np
import pytest
import torch
from loguru import logger
from scipy.stats import entropy

import ttnn
from models.common.sampling.tt_sampling import TTSampling
from models.demos.gpt_oss.tt.model import compute_per_device_vocab

# --- Reference implementation ---


def sample_top_p(values: torch.Tensor, p: float):
    values = torch.nn.functional.softmax(values, dim=-1)
    assert 0 <= p <= 1
    probs_sort, probs_idx = torch.sort(values, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort = probs_sort / probs_sort.sum(dim=-1, keepdim=True)
    probs_sort = torch.where(torch.isnan(probs_sort), torch.zeros_like(probs_sort), probs_sort)
    probs_sort = torch.where(torch.isinf(probs_sort), torch.zeros_like(probs_sort), probs_sort)
    probs_sort = torch.where(probs_sort.sum(dim=-1, keepdim=True) == 0, torch.ones_like(probs_sort), probs_sort)
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def reference_sampling(input_tensor, sampling_params, num_devices, padded_vocab_size, max_top_k):
    """Reference sampling that mirrors TTSampling's multi-device top-k gather logic."""
    k = sampling_params["top_k"]
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
        sampled_indices = sample_top_p(topk_values_gathered, sampling_params["top_p"])

    sampled_indices = torch.gather(topk_indices_gathered.squeeze(0).squeeze(0), dim=-1, index=sampled_indices)
    return sampled_indices


# --- Statistical helpers ---


def counts_to_vector(*samples, return_prob=True, dtype=float):
    all_tokens = set().union(*[set(s) for s in samples])
    token_index = {tok: i for i, tok in enumerate(sorted(all_tokens))}
    size = len(token_index)

    vectors = []
    for s in samples:
        vec = np.zeros(size, dtype=dtype)
        for tok, cnt in Counter(s).items():
            vec[token_index[tok]] = cnt
        if return_prob:
            total = vec.sum()
            if total:
                vec /= total
        vectors.append(vec)

    return vectors if len(vectors) > 1 else vectors[0], token_index


def kl_divergence(p, q, *, base=None, eps=1e-12):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p /= p.sum()
    q /= q.sum()
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return entropy(p, q, base=base)


# --- TTSampling args for GPT-OSS ---

VOCAB_SIZE = 201088
MAX_TOP_K = 32
BATCH_SIZE = 32


def make_gpt_oss_sampling_args(mesh_device):
    """Create args matching GPT-OSS model on Galaxy [4,8] mesh."""

    class _Args:
        pass

    args = _Args()
    args.vocab_size = VOCAB_SIZE
    num_tp = mesh_device.shape[1]
    per_device_vocab = compute_per_device_vocab(args.vocab_size, num_tp)
    args.padded_vocab_size = per_device_vocab * num_tp
    args.cluster_shape = tuple(mesh_device.shape)
    args.sampling_all_gather_axis = 1
    args.num_devices = mesh_device.get_num_devices()
    args.is_galaxy = mesh_device.shape[0] > 1
    args.model_config = {}
    args.sampling_dp = mesh_device.shape[0]
    args.max_top_k = MAX_TOP_K
    args.sub_core_grids = None
    return args


# --- Tests ---


@torch.no_grad()
@pytest.mark.parametrize("batch_size", [BATCH_SIZE])
@pytest.mark.parametrize(
    "num_samples_with_threshold",
    [
        (10, 25.5),  # Quick smoke test — loose KL threshold
        (1000, 2.0),  # Statistical test — tight KL threshold
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "sampling_params",
    [
        {"temperature": 0.0, "top_k": 32, "top_p": 0.00, "seed": 42},  # Greedy
        {"temperature": 0.0, "top_k": 32, "top_p": 0.95, "seed": 42},  # Greedy (top_p ignored)
        {"temperature": 1.0, "top_k": 32, "top_p": 0.95, "seed": 42},  # Stochastic
        {"temperature": 1.0, "top_k": 32, "top_p": 1.00, "seed": 42},  # Stochastic, no top-p filter
    ],
)
@pytest.mark.parametrize(
    "mesh_device",
    [(4, 8)],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 31744,
            "worker_l1_size": 1344544,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_gpt_oss_sampling(
    dtype, sampling_params, batch_size, num_samples_with_threshold, mesh_device, device_params, reset_seeds
):
    """Test on-device sampling with GPT-OSS vocab dimensions on Galaxy mesh."""
    num_samples, kl_required = num_samples_with_threshold
    args = make_gpt_oss_sampling_args(mesh_device)
    max_top_k = args.max_top_k

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
    # Shape: [1, 1, batch_size, padded_vocab_size] — the full padded width
    torch_input = torch.randn(1, 1, batch_size, args.padded_vocab_size)
    # Zero out padding region so padded tokens can't win argmax
    torch_input[:, :, :, VOCAB_SIZE:] = -float("inf")

    # Reference sampling uses the full padded input (mimics multi-device gather)
    reference_outputs = []
    for i in range(num_samples):
        ref = reference_sampling(torch_input, sampling_params, args.cluster_shape[0], args.padded_vocab_size, max_top_k)
        reference_outputs.append(ref[0].item())

    # Shard input across Galaxy mesh columns (TP axis = dim 1 = cols)
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(3, None),
            mesh_shape=args.cluster_shape,
        ),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Initialize TTSampling
    tt_sampling = TTSampling(
        args=args,
        mesh_device=mesh_device,
        tt_ccl=None,
        k=torch.tensor(top_k),
        p=torch.tensor(top_p),
        temp=torch.tensor(temperature),
    )

    # Run device sampling
    tt_outputs_torch = []
    for i in range(num_samples):
        if i == 0:
            ttnn.manual_seed(seed, device=mesh_device, sub_core_grids=args.sub_core_grids)
            tt_outputs = tt_sampling(tt_input)
        else:
            tt_outputs = tt_sampling(tt_input)
        tt_output = ttnn.get_device_tensors(tt_outputs)[0]
        tt_output_torch = ttnn.to_torch(tt_output)
        token_id = tt_output_torch[0, 0, :, :].reshape(-1, 1)[0].item()
        tt_outputs_torch.append(token_id)

    # Verify all sampled tokens are within vocab bounds
    for token_id in tt_outputs_torch:
        assert 0 <= token_id < VOCAB_SIZE, f"Sampled token {token_id} outside vocab range [0, {VOCAB_SIZE})"

    # Compute KL divergence between reference and device distributions
    logger.info(f"reference_outputs (first 10): {reference_outputs[:10]}")
    logger.info(f"tt_outputs_torch  (first 10): {tt_outputs_torch[:10]}")

    vectors, tok2col = counts_to_vector(reference_outputs, tt_outputs_torch, return_prob=True)
    reference_freqs = vectors[0]
    tt_freqs = vectors[1]

    passing = True
    d_kl = kl_divergence(reference_freqs, tt_freqs, base=2)
    logger.info(f"KL(ref‖device) = {d_kl:.4f} bits (threshold: {kl_required})")

    if d_kl > kl_required:
        logger.warning(f"KL divergence {d_kl:.4f} exceeds threshold {kl_required}!")
        passing = False

    # For argmax, require exact match
    if sampling_params["top_k"] == 1 or sampling_params["top_p"] == 0.0 or sampling_params["temperature"] == 0.0:
        match = reference_outputs == tt_outputs_torch
        if not match:
            logger.warning(f"Argmax mismatch: ref={reference_outputs[:5]} vs device={tt_outputs_torch[:5]}")
            passing = False
        else:
            logger.info("Argmax exact match confirmed")

    assert passing, (
        f"GPT-OSS sampling test failed: KL={d_kl:.4f}/{kl_required}, "
        f"params={sampling_params}, num_samples={num_samples}"
    )
    logger.info("GPT-OSS sampling test passed!")
