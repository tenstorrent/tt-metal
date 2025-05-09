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


def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))

    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def reference_sampling(input_tensor, sampling_params, num_devices, padded_vocab_size):
    tt_indices_device_offsets = torch.ones([1, 1, 32, 32 * num_devices], dtype=torch.int32)
    per_device_vocab_size = padded_vocab_size // num_devices
    for device_id in range(num_devices):
        tt_indices_device_offsets[:, :, :, device_id * 32 : (device_id + 1) * 32] = device_id * per_device_vocab_size

    # Split up in per device tensors
    per_device_tensors = torch.split(input_tensor, per_device_vocab_size, dim=-1)
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
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "device_params",
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_llama_sampling_inference(dtype, batch_size, mesh_device, use_program_cache, reset_seeds):
    sampling_params = {"top_k": 32, "top_p": 0.08, "seed": 42}
    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=32, dummy_weights=True)
    torch_input = torch.randn(1, 1, 32, model_args.padded_vocab_size)

    # Reference output
    reference_output = reference_sampling(
        torch_input, sampling_params, model_args.cluster_shape[0], model_args.padded_vocab_size
    )

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

    logger.info("Run Llama Sampling")
    for i in range(1):
        # Input to TTNN
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
        tt_outputs = tt_sampling(tt_input)
        tt_output = ttnn.get_device_tensors(tt_outputs)[0]
        tt_output_torch = ttnn.to_torch(
            tt_output,
        )
    tt_output_torch = tt_output_torch[0, 0, :, :]
    tt_output_torch = tt_output_torch.reshape(-1, 1)

    pcc_required = 0.99
    passing, pcc_message = comp_allclose(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Llama Sampling Passed!")
    else:
        logger.warning("Llama Sampling Failed!")

    tt_ccl.close()

    assert passing, f"Llama Sampling output does not meet PCC requirement {pcc_required}: {pcc_message}."
