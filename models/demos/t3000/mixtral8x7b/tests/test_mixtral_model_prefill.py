# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import numpy as np
from loguru import logger
from sklearn.metrics import top_k_accuracy_score

# Set Mixtral flags for CI, if CI environment is setup
if os.getenv("CI") == "true":
    os.environ["MIXTRAL_CKPT_DIR"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_TOKENIZER_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["MIXTRAL_CACHE_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/"
    os.environ["TT_METAL_ASYNC_DEVICE_QUEUE"] = "1"
    os.environ["WH_ARCH_YAML"] = "wormhole_b0_80_arch_eth_dispatch.yaml"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn_prefill,
    prepare_rotation_mat_ttnn,
    get_rot_transformation_mat,
    set_model_args,
)
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.reference.model import Transformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs
from models.utility_functions import comp_pcc, comp_allclose


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


@pytest.mark.parametrize(
    "seq_len",
    (128, 1024, 2048, 8192),
)
@pytest.mark.parametrize(
    "n_layers",
    (
        1,
        32,
    ),
)
def test_mixtral_model_inference(t3k_device_mesh, use_program_cache, reset_seeds, n_layers, seq_len):
    pcc = 0.96
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    model_args = set_model_args(model_args)

    model_args.n_layers = n_layers
    batch = 1
    state_dict = model_args.load_state_dict()

    tokenizer = Tokenizer(model_args.tokenizer_path)
    prompt_file = "models/demos/t3000/mixtral8x7b/demo/tale-of-two-cities.txt"
    with open(prompt_file, "r") as f:
        prompt = f.read()
    encoded_prompts = tokenizer.encode(prompt)[:seq_len]

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
    rot_mats = prepare_rotation_mat_ttnn(
        model_args.head_dim, model_args.max_seq_len, t3k_device_mesh, mode="prefill", seq_len=seq_len
    )
    head_dim = model_args.dim // model_args.n_heads
    transformation_mat_torch = get_rot_transformation_mat(head_dim)
    transformation_mats = ttnn.as_tensor(
        transformation_mat_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=t3k_device_mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ReplicateTensorToMesh(t3k_device_mesh),
    )
    # Load TTNN model
    tt_model = TtTransformer(
        device_mesh=t3k_device_mesh,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        dtype=dtype,
    )

    # Select the corresponding seq_len of tokens for prefill
    encoded_prompts_tensor = torch.tensor(encoded_prompts)
    pt_decode_input = embd(encoded_prompts_tensor).view(batch, seq_len, -1)
    tt_decode_input = pt_decode_input

    start_pos = 0
    current_pos = start_pos % model_args.sliding_window

    for iter in range(2):
        decode_input, attn_mask, attn_mask_torch = prepare_inputs_ttnn_prefill(
            tt_decode_input,
            tt_model.device_mesh,
        )

        # Run TT model
        tt_out = tt_model(
            decode_input, start_pos, current_pos, attn_mask, rot_mats, transformation_mats, 0, mode="prefill"
        )
        # Work around program cache issue https://github.com/tenstorrent/tt-metal/issues/7159
        del decode_input, attn_mask
        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .view(batch, seq_len, -1)
            .detach()
            .float()
        )

    # Measure PCC
    if seq_len == 8192:
        ref_output = torch.load("models/demos/t3000/mixtral8x7b/reference/ref_output_prefil_32L_8192.pt")
    else:
        reference_model = Transformer(args=model_args)
        reference_model.load_state_dict(state_dict)
        reference_model.eval()
        positions = torch.LongTensor(range(seq_len))
        ref_output = reference_model(pt_decode_input, positions, attn_mask_torch, mode="prefill").detach().float()

    passing, pcc_message = comp_pcc(ref_output.view(batch, seq_len, -1), tt_output_torch.view(batch, seq_len, -1), pcc)
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info(f"Mistral model Passed!")
    else:
        logger.warning("Mistral model Failed!")
        assert passing, f"PCC value is lower than {pcc} for some of the outputs. Check Warnings!"
