# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.qwen.reference.model import TransformerBlock
from models.demos.qwen.tt.model_config import TtModelArgs
from models.demos.qwen.tt.qwen_common import get_single_rot_mat, precompute_freqs
from models.demos.qwen.tt.qwen_decoder import TtTransformerBlock
from models.utility_functions import comp_allclose, comp_pcc, skip_for_grayskull


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
def test_qwen_decoder_inference(mesh_device, use_program_cache, reset_seeds, ensure_gc):
    if mesh_device.shape != (1, 1):
        pytest.skip("Only N150 is supported")
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(mesh_device)
    model_args.n_layers = 1
    state_dict = model_args.load_state_dict()

    # Ref model needs partial state dict, but our models use full state dict keys as cached weight names
    first_layer_prefix = model_args.get_state_dict_prefix("TtTransformerBlock", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) :]: v for k, v in state_dict.items() if (k.startswith(first_layer_prefix))
    }
    reference_model = TransformerBlock(args=model_args)
    reference_model.load_state_dict(partial_state_dict)

    generation_start_pos = 0
    generation_length = 10
    all_tests_pass = True

    # pre-compute the rotational embedding matrix and send to device
    current_rot_mat, rot_matrix = get_single_rot_mat(
        model_args.head_dim,
        mesh_device,
        model_args.num_devices,
        start_pos=0,
    )

    # Initialize TT model
    tt_model = TtTransformerBlock(
        args=model_args,
        mesh_device=mesh_device,
        dtype=dtype,
        state_dict=state_dict,
        layer_num=0,
        weight_cache_path=model_args.weight_cache_path(dtype),
    )

    seqlen = 1
    batch = model_args.max_batch_size

    cos, sin = precompute_freqs(model_args.head_dim, model_args.max_seq_len * 2)
    freqs_cis = torch.complex(cos, sin)

    for i in range(generation_length):
        logger.info(f"[Decoder] Generating token {i}")

        # input = torch.randn(1, 32, 4096)
        pt_decode_input = (torch.rand(batch, seqlen, model_args.dim) * 2) - 1
        tt_decode_input = pt_decode_input.clone()
        current_pos = generation_start_pos + i
        current_pos_tensor = ttnn.from_torch(
            torch.tensor([current_pos] * batch),
            device=mesh_device,
            dtype=ttnn.int32,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        decode_input = model_args.prepare_inputs_ttnn_decode(
            tt_decode_input,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        # Run TT model
        tt_out = tt_model(decode_input, current_pos_tensor, rot_mat=current_rot_mat)
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=-1))[
                :1, :, :, : model_args.dim
            ]
            .permute(2, 1, 0, 3)
            .squeeze(1)[: model_args.max_batch_size, :, :]
        )  # [seq, batch, dim]

        freqs_cis_i = freqs_cis[current_pos, :].unsqueeze(0)
        positions = torch.tensor([current_pos])
        # Reference model
        ref_output = reference_model(pt_decode_input, freqs_cis_i, positions, mask=None)

        passing, pcc_message = comp_pcc(ref_output, tt_output_torch)

        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(f"PCC: {pcc_message}")

        if passing:
            logger.info("Qwen Decoder Block Passed!")
        else:
            logger.warning("Qwen Decoder Block Failed!")
            all_tests_pass = False

        # Update rotation matrix for next iteration
        current_rot_mat = ttnn.linear(rot_matrix, current_rot_mat)

    if all_tests_pass:
        logger.info(f"All {generation_length} Qwen decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Qwen decode Failed!")
        assert all_tests_pass, f"PCC value is lower than {0.99} for some of the outputs. Check Warnings!"
