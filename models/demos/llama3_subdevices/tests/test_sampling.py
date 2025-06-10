# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.utility_functions import (
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL

from models.demos.llama3_subdevices.tt.sampling import TTSampling

import numpy as np
from scipy.stats import entropy  # KL divergence lives here as scipy.stats.entropy
import numpy as np
from collections import Counter

is_RING_6U = os.environ.get("RING_6U", "0") == "1"


def counts_to_vector(*samples, return_prob=True, dtype=float):
    """
    Convert one or more token-ID sample lists into aligned vectors.

    Parameters
    ----------
    *samples : 1-D iterables of int
        Each positional argument is a separate sample list (e.g. P, Q, …).
    return_prob : bool, default True
        If True → normalise each vector so it sums to 1 (probabilities);
        if False → leave as raw counts.
    dtype : NumPy dtype, default float
        Data type of the returned vectors.

    Returns
    -------
    vectors : list[np.ndarray]
        One NumPy array per input list, all of identical length/order.
    token_index : dict[int, int]
        Mapping from token-ID → column index in the vectors.
    """
    # 1. Collect the union of all token IDs
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
            if total:  # protect against empty lists
                vec /= total
        vectors.append(vec)

    return vectors if len(vectors) > 1 else vectors[0], token_index


def kl_divergence(p, q, *, base=None, eps=1e-12):
    """
    Computes KL(P‖Q) for two 1-D NumPy arrays *p* and *q*.

    Parameters
    ----------
    p, q : array-like
        Discrete probability distributions. Lengths must match.
    base : float or None
        Logarithm base. None ⇒ natural log (nats), 2 ⇒ bits, 10 ⇒ bans.
    eps : float
        Tiny constant to avoid log(0). Added only where a zero appears.

    Returns
    -------
    float
        KL divergence (non-negative, 0 iff distributions identical element-wise).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    if p.shape != q.shape:
        raise ValueError("p and q must be the same length")

    # Renormalise in case inputs don’t already sum to 1
    p /= p.sum()
    q /= q.sum()

    # Protect against log(0) by clipping
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)

    return entropy(p, q, base=base)


def sample_top_p(values: torch.Tensor, p: float):
    # sum_values = torch.sum(values, dim=-1)
    # values = values / sum_values
    values = torch.nn.functional.softmax(values, dim=-1)
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(values, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def reference_sampling(input_tensor, sampling_params, num_devices, padded_vocab_size):
    tt_indices_device_offsets = torch.ones([1, 1, 32, 32 * num_devices], dtype=torch.int32)
    per_device_offset = input_tensor.shape[-1] // num_devices
    for device_id in range(num_devices):
        tt_indices_device_offsets[:, :, :, device_id * 32 : (device_id + 1) * 32] = device_id * per_device_offset

    # Split up in per device tensors
    per_device_tensors = torch.split(input_tensor, per_device_offset, dim=-1)
    topk_values_list = []
    topk_indices_list = []
    for i in range(num_devices):
        topk_values, topk_indices = torch.topk(per_device_tensors[i], k=32, dim=-1)
        topk_values_list.append(topk_values)
        topk_indices_list.append(topk_indices)

    topk_values_tensor = torch.cat(topk_values_list, dim=3)
    topk_indices_tensor = torch.cat(topk_indices_list, dim=3)

    topk_indices_tensor += tt_indices_device_offsets

    # Do topk on gathered
    topk_values_gathered, topk_indices_gathered = torch.topk(topk_values_tensor, k=sampling_params["top_k"], dim=-1)
    topk_indices_gathered = torch.gather(topk_indices_tensor, dim=-1, index=topk_indices_gathered)
    topk_values_gathered = topk_values_gathered[0, 0, :, :]

    # Do sampling
    sampled_indices = sample_top_p(topk_values_gathered, sampling_params["top_p"])
    sampled_indices = torch.gather(topk_indices_gathered.squeeze(0).squeeze(0), dim=-1, index=sampled_indices)

    return sampled_indices


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "dtype",
    (ttnn.bfloat8_b,),
)
@pytest.mark.parametrize(
    "sampling_params",
    (
        {"top_k": 1, "top_p": 0.5, "seed": 42},  # argmax
        # {"top_k": 32, "top_p": 0.00, "seed": 42}, # argmax
        # {"top_k": 32, "top_p": 1.00, "seed": 42}, # multinomial sampling from all tok-k tokens
        # {"top_k": 32, "top_p": 0.95, "seed": 42}, # typical top-k top-p sampling
        # # {"top_k": 128, "top_p": 0.95, "seed": 42}, # large top-k # does not work with CCL persistent buffers!!!
        # {"top_k": 32, "top_p": 0.08, "seed": 42}, # small top-p
    ),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(  # Worker size is selected to give 120kB ringbuffer size
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 25600,
            "worker_l1_size": 1344544,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if is_RING_6U else ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_llama_sampling_inference(dtype, sampling_params, batch_size, mesh_device, use_program_cache, reset_seeds):
    use_tracing = True
    load_cached_outputs = True
    num_samples = 10
    num_compile_steps = 1
    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=32, dummy_weights=True)

    if load_cached_outputs:
        # Cached model outputs
        tt_model_output_cache_path = f"models/demos/llama3_subdevices/tests/ref_outputs/test_llama_model/tt_model_layers_80_output_logits_tok_4.bin"
        tt_input_loaded = ttnn.load_tensor(file_name=tt_model_output_cache_path, device=mesh_device)
        tt_input_loaded = tt_input_loaded.reshape(1, 1, 32, -1)
        torch_input = ttnn.to_torch(
            tt_input_loaded,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(3, 1), mesh_shape=model_args.cluster_shape),
        )
        torch_input = torch_input[:, :1, :, :]  # select first cluster row (others are duplicates)

    else:
        # Random inputs
        # torch_input = torch.randn(1, 1, 32, model_args.padded_vocab_size)
        torch_input = torch.randn(1, 1, 32, 512)
        # torch_input = torch.rand(1, 1, 32, 512)

        # torch_input = torch.zeros(1, 1, 32, 512)
        # torch_input[:, :, :, 1:33] = 1

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(3, None) if model_args.is_galaxy else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    model_args.padded_vocab_size = torch_input.shape[-1]

    # print("torch_input:", torch_input)

    # Reference output
    reference_outputs = []
    for i in range(num_samples):
        reference_output = reference_sampling(
            torch_input, sampling_params, model_args.cluster_shape[0], model_args.padded_vocab_size
        )
        reference_outputs.append(reference_output[0].item())

    # Setup prefetcher and CCL
    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=0,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )
    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id)

    # Setup sampling
    tt_sampling = TTSampling(
        args=model_args,
        mesh_device=mesh_device,
        sampling_params=sampling_params,
        tt_ccl=tt_ccl,
    )

    if use_tracing:
        try:
            logger.info("Compile Llama Sampling")

            for i in range(1):
                tt_outputs = tt_sampling(tt_input)
            logger.info("Done comiling Llama Sampling Trace")

            # ttnn.synchronize_device(mesh_device)

            logger.info("Capture Llama Sampling Trace")

            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)

            tt_outputs = tt_sampling(tt_input)

            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)

            ttnn.synchronize_device(mesh_device)

            logger.info("Capture done")

            # Run trace
            tt_outputs_torch = []
            # Resetting the input
            tt_input_reset = ttnn.from_torch(
                torch_input,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device,
                    dims=(3, None) if model_args.is_galaxy else (None, None),
                    mesh_shape=model_args.cluster_shape,
                ),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
            )
            logger.info("Starting sampling...")
            for i in range(num_samples):
                ttnn.copy_host_to_device_tensor(tt_input_reset, tt_input)
                # iteration_time_start = time()

                # Execute trace
                ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
                tt_out_tok_device0 = ttnn.get_device_tensors(tt_outputs)[0]
                tt_out_tok_cpu = tt_out_tok_device0.cpu(blocking=True, cq_id=0)
                # iteration_time = time() - iteration_time_start

                tt_output_torch = ttnn.to_torch(tt_out_tok_cpu)
                tt_output_torch = tt_output_torch[0, 0, :, :]
                tt_output_torch = tt_output_torch.reshape(-1, 1)
                tt_outputs_torch.append(tt_output_torch[0].item())
            logger.info("Sampling done")
        except Exception as e:
            logger.error(f"Error during tracing: {e}")
            raise e

    else:  # No tracing
        tt_outputs_torch = []
        for i in range(num_samples):
            tt_outputs = tt_sampling(tt_input)
            tt_output = ttnn.get_device_tensors(tt_outputs)[0]
            tt_output_torch = ttnn.to_torch(
                tt_output,
            )
            tt_output_torch = tt_output_torch[0, 0, :, :]
            tt_output_torch = tt_output_torch.reshape(-1, 1)
            tt_outputs_torch.append(tt_output_torch[0].item())

    # Compute KL divergence
    print("reference_outputs:", reference_outputs)
    print("tt_outputs_torch:", tt_outputs_torch)

    argmax = torch.argmax(torch_input[:, :, 0, :], dim=-1)
    print("argmax:", argmax)

    vectors, tok2col = counts_to_vector(reference_outputs, tt_outputs_torch, return_prob=True)
    reference_tok_frequencies = vectors[0]
    tt_tok_frequencies = vectors[1]

    logger.info("column order (token_id → index):", tok2col)
    logger.info("reference_tok_frequencies:", reference_tok_frequencies)
    logger.info("tt_tok_frequencies:", tt_tok_frequencies)

    passing = True

    d_kl = kl_divergence(reference_tok_frequencies, tt_tok_frequencies, base=2)
    print(f"KL(P‖Q) = {d_kl:.4f} bits")
    kl_required = 0.01
    if d_kl > kl_required:
        logger.warning(f"KL(P‖Q) = {d_kl:.4f} bits is too high!")
        passing = False

    if sampling_params["top_k"] == 1 or sampling_params["top_p"] == 0.0:  # argmax can be compared directly
        # PCC
        pcc_required = 1.0
        pcc_passing, pcc_message = comp_allclose(reference_output, tt_output_torch, pcc_required)
        passing = passing and pcc_passing

        logger.info(f"PCC: {pcc_message}")

    if passing:
        logger.info("Llama Sampling Passed!")
    else:
        logger.warning("Llama Sampling Failed!")

    tt_ccl.close()

    assert (
        passing
    ), f"Llama Sampling output does not meet KL / PCC requirement {d_kl:.4f}/{kl_required} KL, {pcc_message}/{pcc_required} PCC."
