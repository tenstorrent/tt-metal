# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.lm_head import LMHead
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
    "seq_len",
    (32,),
)
@pytest.mark.parametrize(
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
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
def test_qwen_lm_head_ttt_inference(seq_len, batch_size, mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b

    # Load tt_transformers reference model args for reference LM head
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len, cache_hf=True)
    model_args_ref.n_layers = 1  # For the unit test, just run a single layer

    state_dict_ref = model_args_ref.load_state_dict()

    state_dict_prefix_ref = model_args_ref.get_state_dict_prefix("", None)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict_ref = {
        "weight": state_dict_ref[f"{state_dict_prefix_ref}output.weight"],
    }

    # Use tt_transformers reference LM head
    model_args_ref.WEIGHTS_DTYPE = dtype
    reference_model = model_args_ref.reference_lm_head()
    reference_model.load_state_dict(partial_state_dict_ref)
    logger.info(f"tt_transformers Reference LM Head Model Loaded")

    # Load Qwen3 model using TtQwenModelArgs
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len, dummy_weights=False)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    logger.info(f"Qwen3 LM Head Model Loaded")

    state_dict_prefix = model_args.get_state_dict_prefix("", None)

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=0,
        n_layers=model_args.n_layers,
    )

    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, use_qwen_mlp=True)

    tt_model = LMHead(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_cache_path=model_args.weight_cache_path(dtype),
        tt_ccl=tt_ccl,
        prefetcher_setup=prefetcher_setup,
    )

    # Create input tensor with appropriate dtype for reference model
    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args_ref.model_name)
    )

    # Run reference model
    reference_output = reference_model(torch_input.to(torch.bfloat16))

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 3) if model_args.is_galaxy else (None, None), mesh_shape=model_args.cluster_shape
        ),
        dtype=ttnn.bfloat8_b,
        memory_config=model_args.model_config["SHARDED_LM_HEAD_INPUT_RING_MEMCFG"],
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Run Qwen_LM_Head_TTT")
    # Pre-allocated output of AllReduce in LM Head to avoid memory cloberring
    tt_ccl.tt_lm_head_buffer_l1 = ttnn.to_memory_config(tt_ccl.tt_lm_head_buffer, tt_ccl.lm_head_buffer_mem_cfg)
    tt_outputs = tt_model(tt_input, prefetcher_setup.worker_sub_device_id, mode="decode")
    tt_outputs = [
        ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, model_args.cluster_shape, dims=(3, 1) if model_args.is_galaxy else (1, 3)
            ),
        )
        for tt_output in tt_outputs
    ]
    tt_output_torch = torch.concat(tt_outputs, dim=-1)
    tt_output_torch = tt_output_torch[:, 0:1, :, : model_args.vocab_size]

    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Qwen_LM_Head_TTT Passed!")
    else:
        logger.warning("Qwen_LM_Head_TTT Failed!")

    tt_ccl.close()

    assert passing, f"Qwen_LM_Head_TTT output does not meet PCC requirement {pcc_required}: {pcc_message}."
