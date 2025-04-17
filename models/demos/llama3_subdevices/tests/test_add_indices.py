# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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

from models.demos.llama3_subdevices.tt.add_indices import TTAddIndices


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("FAKE_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize("device_params", [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL}], indirect=True)
def test_llama_add_indices_inference(mesh_device, use_program_cache, reset_seeds):
    mesh_device.enable_async(True)
    model_args = TtModelArgs(mesh_device, max_batch_size=32, max_seq_len=32, dummy_weights=True)
    torch_input = torch.zeros(1, 1, 32, 256)
    for i in range(256):
        torch_input[:, :, :, i] = i % 32

    # Setup prefetcher and CCL
    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=0,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    # Setup add indices
    tt_add_indices = TTAddIndices(
        args=model_args,
        mesh_device=mesh_device,
    )

    logger.info("Run Llama Sampling")
    for i in range(1):
        # Input to TTNN
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        tt_outputs = tt_add_indices(tt_input)
        tt_output_torch = ttnn.to_torch(
            tt_outputs,
            mesh_composer=ttnn.ConcatMesh2dToTensor(  # TODO: get single device tensor only, it's all replicated
                mesh_device, model_args.cluster_shape, dims=(0, 1)
            ),
        )
    tt_output_torch = tt_output_torch[0, 0, :, :]

    reference_output = torch_input
    for i in range(8):
        reference_output[:, :, :, i * 32 : (i + 1) * 32] = i * 5000  # 16*1024

    breakpoint()

    pcc_required = 0.99
    passing, pcc_message = comp_allclose(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Llama Add Indices Passed!")
    else:
        logger.warning("Llama Add Indices Failed!")
