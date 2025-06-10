# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import os
import ttnn
from models.demos.llama3_subdevices.tt.model_config import TtModelArgs
from models.demos.llama3_subdevices.tt.llama_decoder import TtTransformerBlock
from models.demos.llama3_subdevices.tt.llama_rope import TtLlamaRotarySetup
from models.utility_functions import skip_for_grayskull
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
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "dispatch_core_axis": ttnn.DispatchCoreAxis.COL,
            "trace_region_size": 165136000,
            "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING if is_RING_6U else ttnn.FabricConfig.FABRIC_1D,
        }
    ],
    indirect=True,
)
def test_llama_decoder_same(
    max_seq_len,
    batch_size,
    mesh_device,
    use_program_cache,
    reset_seeds,
    ensure_gc,
):
    dtype = ttnn.bfloat8_b
    seqlen = 1

    model_args = TtModelArgs(mesh_device, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=True)
    model_args.n_layers = 1

    state_dict = model_args.load_state_dict()

    prefetcher_setup = TtLlamaPrefetcherSetup(
        mesh_device,
        n_tensors=5,
        n_layers=model_args.n_layers,
    )
    mesh_device.set_sub_device_stall_group(
        [prefetcher_setup.prefetcher_sub_device_id, prefetcher_setup.worker_sub_device_id]
    )

    tt_ccl = TT_CCL(mesh_device, model_args, prefetcher_setup.worker_sub_device_id)

    generation_start_pos = 127
    generation_length = 100

    # Setup RoPE transformation matrices
    rope_setup = TtLlamaRotarySetup(
        mesh_device,
        model_args.max_batch_size,
        model_args.head_dim,
        model_args.max_seq_len,
        model_args.rope_theta,
        model_args.use_scaled_rope,
        model_args.rope_scaling_factor,
    )
    transformation_mats = rope_setup.get_both_trans_mats()

    # Prepare page table for paged attention
    page_table_tt = None
    paged_attention_config = None

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        n_layers=model_args.n_layers,
        weight_cache_path=model_args.weight_cache_path(dtype),
        transformation_mats=transformation_mats,
        paged_attention_config=paged_attention_config,
        prefetcher_setup=prefetcher_setup,
        tt_ccl=tt_ccl,
    )

    ##### Set up input tensors #####

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    current_pos_tensor = ttnn.from_torch(
        current_pos,
        device=mesh_device,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device,
            dims=(None, 0) if (model_args.is_galaxy and batch_size > 1) else (None, None),
            mesh_shape=model_args.cluster_shape,
        ),
    )

    # input = torch.randn(1, 32, 4096)
    pt_decode_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1
    tt_decode_input = pt_decode_input.clone()

    decode_input = model_args.prepare_residual_tensor_decode(
        tt_decode_input,
        model_args.model_config["DECODE_RESIDUAL_MEMCFG"],
    )

    # Get cos/sin matrices for the current position of each user
    rot_mats = rope_setup.get_rm_rot_mats(current_pos)
    tt_pf = prefetcher_setup.get_input_tensors()

    # Explicitly allocate global CB to avoid memory fragmentation
    prefetcher_setup.create_global_cb()

    ##### Run the decoder #####
    outs = []
    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")
        ttnn.dram_prefetcher(
            tt_pf,
            num_layers=1,
            global_cb=prefetcher_setup.global_circular_buffer,
        )
        mesh_device.set_sub_device_stall_group([prefetcher_setup.worker_sub_device_id])

        # Run TT model
        res = None
        tt_out, res = tt_model(
            decode_input,
            res,
            current_pos_tensor,
            rot_mats=rot_mats,
            mode="decode",
            page_table=page_table_tt,
        )
        tt_out = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
        )

        outs.append(tt_out)

    ##### Check outputs #####
    for arr in [outs]:
        golden = arr[0]
        all_passing = True
        for i in range(len(arr)):
            logger.info(f"Checking output for iteration {i}")

            passing = torch.all(arr[i] == golden)

            if passing:
                logger.info(f"Output for iteration {i} is equal to golden")
            else:
                logger.warning(f"Output for iteration {i} is NOT equal to golden")
            all_passing = all_passing and passing

    ##### Close CCL #####
    tt_ccl.close()

    assert all_passing, "Not all outputs are equal to the golden output"
