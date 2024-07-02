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
    os.environ["REF_OUTPUT_PATH"] = "/mnt/MLPerf/tt_dnn-models/Mistral/Mixtral-8x7B-v0.1/prefill"

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.demos.t3000.mixtral8x7b.tt.mixtral_common import (
    prepare_inputs_ttnn_prefill,
    get_prefill_rot_mat,
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


@pytest.mark.skipif(os.getenv("CI") == "False", reason="CI-only test")
@pytest.mark.parametrize(
    "seq_len",
    (128, 1024, 2048, 4096, 8192, 8192 * 2, 8192 * 4),
)
def test_mixtral_model_inference(t3k_device_mesh, use_program_cache, reset_seeds, seq_len):
    n_layers = 32
    pcc = 0.93
    dtype = ttnn.bfloat8_b

    model_args = TtModelArgs(t3k_device_mesh.get_device(0))
    model_args = set_model_args(model_args, seq_len)

    model_args.n_layers = n_layers
    batch = 1
    state_dict = model_args.load_state_dict()

    tokenizer = Tokenizer(model_args.tokenizer_path)
    prompt_file = os.environ["REF_OUTPUT_PATH"] + "tale-of-two-cities.txt"
    with open(prompt_file, "r") as f:
        prompt = f.read()
    encoded_prompts = tokenizer.encode(prompt)[:seq_len]

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})
    rot_mats = get_prefill_rot_mat(model_args.head_dim, model_args.max_seq_len, t3k_device_mesh, seq_len=seq_len)
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
        start_pos=seq_len,
    )

    # Select the corresponding seq_len of tokens for prefill
    encoded_prompts_tensor = torch.tensor(encoded_prompts)
    pt_decode_input = embd(encoded_prompts_tensor).view(batch, seq_len, -1)
    tt_decode_input = pt_decode_input

    start_pos = 0
    current_pos = start_pos % model_args.sliding_window

    for iter in range(1):
        decode_input, attn_mask, attn_mask_torch = prepare_inputs_ttnn_prefill(
            tt_decode_input,
            tt_model.device_mesh,
        )

        # Run TT model
        tt_out = tt_model(
            decode_input, start_pos, current_pos, attn_mask, rot_mats, transformation_mats, 0, mode="prefill"
        )

        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .view(batch, seq_len, -1)
            .detach()
            .float()
        )

    # Measure PCC
    if seq_len in [8192, 8192 * 2, 8192 * 4]:
        ref_output = torch.load(os.environ["REF_OUTPUT_PATH"] + f"ref_output_prefil_{n_layers}L_{seq_len//1000}k.pt")
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
        logger.info(f"Mistral model prefill Passed!")
    else:
        logger.warning("Mistral model prefill Failed!")

    decode_after_prefill = False
    passing_decode = True

    # only running decode upto 4096 tokens, till we do decode via FlashDecode
    if seq_len < 4096:
        decode_after_prefill = True
    if decode_after_prefill:
        # Run TT model
        from models.demos.t3000.mixtral8x7b.tt.mixtral_common import prepare_inputs_ttnn

        start_pos = seq_len
        current_pos = start_pos
        seqlen = 1
        batch = model_args.max_batch_size

        prompts = ["Once"] * batch
        encoded_prompts_tensor = torch.tensor([tokenizer.encode(prompt) for prompt in prompts])
        decode_input_torch = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)
        decode_input, attn_mask = prepare_inputs_ttnn(
            decode_input_torch,
            model_args.dim,
            start_pos,
            model_args,
            tt_model.device_mesh,
        )
        tt_out = tt_model(decode_input, start_pos, current_pos, attn_mask, mode="decode")

        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_device_mesh, dim=0))[0]
            .view(batch, seqlen, -1)
            .detach()
            .float()
        )
        positions = torch.LongTensor([start_pos])
        ref_output = reference_model(decode_input_torch, positions).detach().float()

        passing_decode, pcc_message_decode = comp_pcc(
            ref_output.view(batch, seqlen, -1), tt_output_torch.view(batch, seqlen, -1), pcc
        )
        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message_decode)

        if passing_decode:
            logger.info(f"Mistral model decode after prefill Passed!")
        else:
            logger.warning("Mistral model decode after prefill Failed!")

    assert passing and passing_decode, f"PCC value is lower for some of the outputs. Check Warnings!"
