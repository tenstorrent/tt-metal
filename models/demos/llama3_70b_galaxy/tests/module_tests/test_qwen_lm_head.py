# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.tt.lm_head import LMHead
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
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": True,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),
)
def test_qwen_lm_head_inference(
    max_seq_len,
    batch_size,
    mesh_device,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b

    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False)

    state_dict = model_args.load_state_dict()

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=5,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, use_qwen_mlp=True)

    # Initialize TT LM Head
    tt_lm_head = LMHead(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        state_dict_prefix=model_args.get_state_dict_prefix("", None),
        weight_cache_path=model_args.weight_cache_path(dtype),
        tt_ccl=tt_ccl,
        prefetcher_setup=prefetcher_setup,
    )

    # Create input tensor with the same shape and memory config as decoder output
    # The decoder output shape is [batch_size, 1, model_args.dim] for decode mode
    seqlen = 1

    # Create random input tensor matching decoder output shape
    pt_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1

    # Convert to TT tensor with DECODE_RESIDUAL_MEMCFG memory configuration
    tt_input = model_args.prepare_residual_tensor_decode(
        pt_input,
        model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
    )

    logger.info(f"Input tensor shape: {tt_input.shape}")
    logger.info(f"Input tensor memory config: {tt_input.memory_config()}")

    # Explicitly allocate global CB to avoid memory fragmentation
    prefetcher_setup.create_global_cb()

    # Run TT LM Head forward pass
    logger.info("Running TT LM Head forward pass...")
    tt_outputs = tt_lm_head.forward(tt_input, prefetcher_setup.worker_sub_device_id, mode="decode")

    # Convert output to torch for validation
    tt_output_torch = ttnn.to_torch(
        tt_outputs[0],
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 3), mesh_shape=model_args.cluster_shape),
    )

    logger.info(f"Output tensor shape: {tt_output_torch.shape}")
    logger.info(f"Expected vocab size: {model_args.vocab_size}")
    logger.info(f"Padded vocab size: {model_args.padded_vocab_size}")

    # Basic shape validation
    expected_shape = (batch_size, seqlen, model_args.padded_vocab_size)
    actual_shape = tt_output_torch.shape[:3]  # Take first 3 dimensions

    assert actual_shape == expected_shape, f"Output shape mismatch: expected {expected_shape}, got {actual_shape}"

    # Reference computation using PyTorch
    lm_head_weight = state_dict[f"{model_args.get_state_dict_prefix('LMHead', None)}output.weight"]

    # Pad the weight to match padded vocab size
    padded_weight = torch.zeros(model_args.padded_vocab_size, model_args.dim)
    padded_weight[: model_args.vocab_size, :] = lm_head_weight

    ref_output = torch.matmul(pt_input.float(), padded_weight.T.float())

    # Compare only the valid vocabulary portion
    tt_output_valid = tt_output_torch[:, :, : model_args.vocab_size]
    ref_output_valid = ref_output[:, :, : model_args.vocab_size]

    passing, pcc_message = comp_pcc(ref_output_valid, tt_output_valid, pcc=0.99)

    logger.info(comp_allclose(ref_output_valid, tt_output_valid))
    logger.info(f"PCC: {pcc_message}")

    tt_ccl.close()

    if passing:
        logger.info("Qwen LM Head Passed!")
    else:
        logger.warning("Qwen LM Head Failed!")
        assert passing, f"PCC value is lower than 0.99. Check Warnings!"
