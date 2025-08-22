# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.lm_head import LMHead
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import ColumnParallelLinear
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
    (128, 4096),
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
    [{"dispatch_core_axis": ttnn.DispatchCoreAxis.COL, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_qwen_lm_head_inference_prefill(seq_len, batch_size, mesh_device, reset_seeds, ensure_gc):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=seq_len, dummy_weights=False)
    model_args.n_layers = 1
    model_args.use_prefetcher = False
    state_dict = model_args.load_state_dict()

    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict = {
        "weight": state_dict[f"{state_dict_prefix}output.weight"],
    }

    model_args.WEIGHTS_DTYPE = dtype
    reference_model = ColumnParallelLinear(model_args.dim, model_args.vocab_size, bias=False, init_method=lambda x: x)
    reference_model.load_state_dict(partial_state_dict)

    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode="prefill")
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode="prefill", use_qwen_mlp=True)

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

    torch_input = torch.randn(1, 1, seq_len, model_args.dim)

    logger.info("Run Qwen_LM_Head_PF")
    for i in range(3):
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device,
                dims=(None, 3) if model_args.is_galaxy else (None, None),
                mesh_shape=model_args.cluster_shape,
            ),
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        logger.info("Run Qwen_LM_Head")
        # Pre-allocated output of AllReduce in LM Head to avoid memory cloberring
        if mode == "prefill":
            tt_outputs = tt_model.forward_prefill(tt_input, prefetcher_setup.worker_sub_device_id)
        else:
            tt_ccl.tt_lm_head_buffer_l1 = ttnn.to_memory_config(tt_ccl.tt_lm_head_buffer, tt_ccl.lm_head_buffer_mem_cfg)
            tt_outputs = tt_model(tt_input, prefetcher_setup.worker_sub_device_id, mode=mode)

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

        reference_output = reference_model(torch_input)

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Qwen_LM_Head Prefill Passed!")
    else:
        logger.warning("Qwen_LM_Head Prefill Failed!")

    tt_ccl.close()

    assert passing, f"Qwen_LM_Head prefill output does not meet PCC requirement {pcc_required}: {pcc_message}."
