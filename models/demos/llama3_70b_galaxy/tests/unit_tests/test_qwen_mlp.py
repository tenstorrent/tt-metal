# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Single parametrized test. Wormhole runs the original (main) prefetcher path; Blackhole Galaxy runs
# the no-prefetcher bring-up path. The architecture is detected once at import so the pytest
# parameters (fabric config, batch/seq) and the in-body setup select the right path automatically.
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.llama_mlp import TtLlamaMLP
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.tt_transformers.tt.model_config import ModelArgs
from models.common.utility_functions import (
    comp_pcc,
    comp_allclose,
)
from models.demos.llama3_70b_galaxy.tt.prefetcher_common import TtLlamaPrefetcherSetup
from models.demos.llama3_70b_galaxy.tt.llama_ccl import TT_CCL
from models.demos.llama3_70b_galaxy.reference.qwen import FeedForward
from models.demos.llama3_70b_galaxy.tests.unit_tests.qwen_test_utils import (
    IS_BLACKHOLE as _IS_BLACKHOLE,
    DECODE_FABRIC_CONFIG as _FABRIC_CONFIG,
)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [
        (8, 4),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "seq_len",
    (1,) if _IS_BLACKHOLE else (32,),
)
@pytest.mark.parametrize(
    "batch_size",
    (32,) if _IS_BLACKHOLE else (1,),
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "fabric_config": _FABRIC_CONFIG,
        }
    ],
    indirect=True,
)
def test_qwen_mlp_inference(seq_len, batch_size, mesh_device, reset_seeds):
    dtype = ttnn.bfloat8_b
    mode = "decode" if seq_len <= 32 else "prefill"

    # Load reference model

    # Note that the Llama3 tests use a reference Llama model, here we call MLP from tt_transformers
    model_args_ref = ModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args_ref.n_layers = 1  # For the unit test, just run a single layer

    state_dict_ref = model_args_ref.load_state_dict()

    first_layer_prefix = model_args_ref.get_state_dict_prefix("MLP", 0)
    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    partial_state_dict_ref = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict_ref.items() if (k.startswith(first_layer_prefix))
    }

    # reference_model = model_args_ref.reference_mlp()
    reference_model = FeedForward(
        dim=5120,
        hidden_dim=25600,
        multiple_of=1,
        ffn_dim_multiplier=None,
    )
    reference_model.load_state_dict(partial_state_dict_ref)

    logger.info(f"Reference Model Loaded")

    # Load Qwen3 model
    model_args = TtQwenModelArgs(mesh_device, max_batch_size=batch_size, dummy_weights=False, max_seq_len=128)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    logger.info(f"Qwen3 Model Loaded")

    model_config = model_args.get_model_config()
    if _IS_BLACKHOLE:
        # Blackhole bring-up runs the unit test without the runtime DRAM prefetcher.
        model_args.use_prefetcher = False
        model_config["USE_PREFETCHER"] = False
        prefetcher_setup = None
        worker_sub_device_id = None
    else:
        prefetcher_setup = TtLlamaPrefetcherSetup(
            mesh_device,
            n_tensors=3,
            n_layers=1,
            is_qwen=True,
        )
        mesh_device.set_sub_device_stall_group(
            [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
        )
        worker_sub_device_id = prefetcher_setup.worker_sub_device_id

    tt_ccl = TT_CCL(mesh_device, model_args, worker_sub_device_id, is_qwen=True)

    tt_model = TtLlamaMLP(
        mesh_device=mesh_device,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=model_args.weight_cache_path(dtype),
        layer_num=0,
        dtype=dtype,
        model_config=model_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    prev_pcc = None
    logger.info("Run Qwen_MLP_PF")
    if not _IS_BLACKHOLE:
        # Wormhole reuses one input across iterations to assert PCC stability of the prefetcher path.
        torch_input = torch.randn(1, 1, seq_len, model_args.dim)
        # Explicitly allocate global CB to avoid memory fragmentation
        prefetcher_setup.create_global_cb()

    for i in range(20):
        if _IS_BLACKHOLE:
            torch_input = (torch.rand(batch_size, seq_len, model_args.dim) * 2) - 1
            tt_input = model_args.prepare_residual_tensor_decode(
                torch_input,
                # Decode MLP input memcfg (decoder residual memcfg is attention-oriented).
                model_args.model_config["SHARDED_MLP_INPUT_MEMCFG"],
            )
        else:
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
                    dims=(None, 3),
                    mesh_shape=model_args.cluster_shape,
                ),  # When both dims are None, the mapper used is `ReplicateTensorToMesh`
                dtype=ttnn.bfloat8_b,
                memory_config=model_args.model_config["SHARDED_FF12_RING_MEMCFG"]
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

        if _IS_BLACKHOLE:
            tt_output_torch = tt_output_torch[:, 0:1, : model_args.max_batch_size, : model_args.dim].view(
                -1, 1, model_args.dim
            )
            reference_output = reference_model(torch_input)
        else:
            tt_output_torch = tt_output_torch[:, :1, :, : model_args.dim]
            ref_input = torch_input[:, :, :, : model_args.dim]
            reference_output = reference_model(ref_input)[:, :, :1, :]

        pcc_required = 0.99
        passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

        if not _IS_BLACKHOLE:
            if prev_pcc is not None:
                assert prev_pcc == pcc_message, f"PCC changed from {prev_pcc} to {pcc_message} during inference."
            prev_pcc = pcc_message

        logger.info(comp_allclose(reference_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")
    if passing:
        logger.info("Qwen_MLP Passed!")
    else:
        logger.warning("Qwen_MLP Failed!")
    tt_ccl.close()
    assert passing, f"Qwen MLP output does not meet PCC requirement {pcc_required}: {pcc_message}."
