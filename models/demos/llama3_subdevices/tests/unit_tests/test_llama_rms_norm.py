# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.common.rmsnorm import RMSNorm as TtRMSNorm
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.t3000.llama2_70b.reference.llama.llama31_8b.model import RMSNorm as RefRMSNorm
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.utility_functions import skip_for_grayskull
from models.demos.llama3_subdevices.tt.distributed_norm import DistributedNorm
from models.demos.llama3_subdevices.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_subdevices.tt.llama_ccl import TT_CCL

is_RING_6U = os.environ.get("RING_6U", "0") == "1"


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
    "batch_size",
    (1,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (128,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize(
    "mode",
    [
        "decode",
    ],
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if is_RING_6U else ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_llama_rms_norm_inference(
    max_seq_len,
    batch_size,
    mode,
    mesh_device,
    use_program_cache,
    reset_seeds,
):
    dtype = ttnn.bfloat16

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=True)

    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()
    state_dict_prefix = model_args.get_state_dict_prefix("", 0)
    first_layer_prefix = state_dict_prefix + "attention_norm."

    prefetcher_setup = TtLlamaPrefetcherSetup(mesh_device, n_tensors=0, n_layers=1, mode=mode)
    mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id, mode=mode)

    # Create the inner RMSNormxw
    tt_inner_norm = TtRMSNorm(
        device=mesh_device,
        dim=model_args.dim,
        state_dict=state_dict,
        state_dict_prefix=state_dict_prefix,
        weight_key="attention_norm",
        weight_dtype=dtype,
        is_distributed=model_args.is_distributed_norm,
        sharded_program_config=model_args.get_model_config()["SHARDED_NORM_ATTN_PRGM_CFG"],
        sharded_output_config=model_args.get_model_config()["SHARDED_ATTN_INPUT_MEMCFG"],
    )

    # Wrap it in DistributedNorm
    tt_model = DistributedNorm(tt_inner_norm, model_args, TG=model_args.is_galaxy, tt_ccl=tt_ccl)

    # Create reference model (unchanged)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = RefRMSNorm(dim=model_args.dim, eps=model_args.norm_eps)
    reference_model.load_state_dict(partial_state_dict)

    input = torch.rand(1, 1, 32, model_args.dim)
    reference_output = reference_model(input)
    for i in range(3):
        # DistributedNorm inputs are fractured across devices and interleaved in DRAM (for prefill) and L1 (for decode)
        tt_input = ttnn.from_torch(
            input,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=model_args.cluster_shape),
            memory_config=model_args.get_model_config()["DECODE_RESIDUAL_MEMCFG"]
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_input_res = ttnn.from_torch(
            input * 0,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, -1), mesh_shape=model_args.cluster_shape),
            memory_config=model_args.get_model_config()["DECODE_RESIDUAL_MEMCFG"]
            if mode == "decode"
            else ttnn.DRAM_MEMORY_CONFIG,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])
        tt_output, res = tt_model(tt_input, tt_input_res, mode=mode)

        # DistributedNorm outputs are replicated across devices
        tt_output_torch = ttnn.to_torch(
            tt_output,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                mesh_device, dims=(0, 3) if model_args.is_galaxy else (3, 0), mesh_shape=model_args.cluster_shape
            ),
        )[:1, :, :, :]

        passing, pcc_message = comp_pcc(reference_output, tt_output_torch)

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

    tt_ccl.close()

    if passing:
        logger.info("Llama_rms_norm Passed!")
    else:
        logger.warning("Llama_rms_norm Failed!")

    assert passing, f"Llama_rms_norm output does not meet PCC requirement {0.99}."
