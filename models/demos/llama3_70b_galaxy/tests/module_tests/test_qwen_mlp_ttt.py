# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.tt_transformers.tt.model_config import ModelArgs
from models.tt_transformers.tests.test_utils import get_ref_model_dype
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL


@torch.no_grad()
@skip_for_grayskull("Requires wormhole_b0 to run")
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
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
def test_qwen_mlp_ttt_inference(seq_len, batch_size, mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    # Load tt_transformers reference model args for reference MLP
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args_ref.n_layers = 1  # For the unit test, just run a single layer

    state_dict_ref = model_args_ref.load_state_dict()

    first_layer_prefix = model_args_ref.get_state_dict_prefix("MLP", 0)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict_ref = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict_ref.items() if (k.startswith(first_layer_prefix))
    }

    # Use tt_transformers reference MLP
    reference_model = model_args_ref.reference_mlp()
    reference_model.load_state_dict(partial_state_dict_ref)

    logger.info(f"tt_transformers Reference Model Loaded")

    # Load Qwen3 model using TtQwenModelArgs
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, dummy_weights=False, max_seq_len=128)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    logger.info(f"Qwen3 Model Loaded")

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=3,
        n_layers=1,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, use_qwen_mlp=True)

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

    # Use Qwen model dimensions for input
    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args_ref.model_name)
    )
    prev_pcc = None

    logger.info("Run Qwen_MLP_TTT")
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

        logger.info("Run Qwen_MLP")
        tt_output = tt_model(tt_input, mode)
        logger.info(f"tt_output shape: {tt_output.shape}")

        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )
        logger.info(f"tt_output_torch shape: {tt_output_torch.shape}")
        logger.info("Qwen MLP Done")

        tt_output_torch = tt_output_torch[:, :1, :, : model_args.dim]

        # Adjust reference input to match the reference model's expected dimensions
        # The reference model expects the standard transformer dimensions, not Qwen's
        ref_input = torch_input[:, :, :, : model_args_ref.dim]
        reference_output = reference_model(ref_input)

        # Pad reference output to match Qwen's output dimensions if needed
        if reference_output.shape[-1] != model_args.dim:
            # If reference model has different output dim, we need to handle this
            # For now, we'll just compare the overlapping dimensions
            min_dim = min(reference_output.shape[-1], model_args.dim)
            reference_output_padded = torch.zeros_like(tt_output_torch)
            reference_output_padded[:, :, :, :min_dim] = reference_output[:, :, :, :min_dim]
            reference_output = reference_output_padded

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        if prev_pcc is not None:
            assert prev_pcc == pcc_message, f"PCC changed from {prev_pcc} to {pcc_message} during inference."
        prev_pcc = pcc_message

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Qwen_MLP_TTT Passed!")
    else:
        logger.warning("Qwen_MLP_TTT Failed!")
    tt_ccl.close()
    assert passing, f"Qwen MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
