# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import FeedForward
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL


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
@pytest.mark.parametrize(
    "seq_len",
    (32,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
def test_llama_mlp_inference(seq_len, batch_size, mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, dummy_weights=False, max_seq_len=128)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=3,
        n_layers=1,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id)

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtLlamaMLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(
        dim=model_args.dim,
        hidden_dim=4 * model_args.dim,
        multiple_of=model_args.multiple_of,
        ffn_dim_multiplier=model_args.ffn_dim_multiplier,
    )
    reference_model.load_state_dict(partial_state_dict)

    tt_model = TtLlamaMLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_args.get_model_config(),
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    torch_input = torch.randn(1, 1, seq_len, model_args.dim)
    prev_pcc = None

    logger.info("Run Llama_MLP_PF")
    # Explicitly allocate global CB to avoid memory fragmentation
    prefetcher_setup.create_global_cb()
    for i in range(20):
        ttnn.dram_prefetcher(
            prefetcher_setup.get_input_tensors(),
            num_layers=1,
            global_cb=prefetcher_setup.global_circular_buffer,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 3) if model_args.is_galaxy else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
            dtype=ttnn.bfloat8_b,
            memory_config=(
                model_args.model_config["SHARDED_FF12_RING_MEMCFG"]
                if model_args.is_galaxy
                else model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"]
            )
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
        logger.info("Run Llama_MLP")
        tt_output = tt_model(tt_input, mode)

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        logger.info("llama MLP Done")

        tt_output_torch = tt_output_torch[:, :1, :, : model_args.dim]

        reference_output = reference_model(torch_input[:, :, :1, : model_args.dim])

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        if prev_pcc is not None:
            assert prev_pcc == pcc_message, f"PCC changed from {prev_pcc} to {pcc_message} during inference."
        prev_pcc = pcc_message

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Llama_MLP Passed!")
    else:
        logger.warning("Llama_MLP Failed!")
    tt_ccl.close()
    assert passing, f"Llama_MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
