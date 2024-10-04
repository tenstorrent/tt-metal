# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
import numpy as np
from loguru import logger
from sklearn.metrics import top_k_accuracy_score

import ttnn
from ttnn import ReplicateTensorToMesh, ConcatMeshToTensor

from models.demos.t3000.mixtral8x7b.tt.mixtral_common import prepare_inputs_ttnn
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
    "batch",
    (
        16,
        32,
    ),
)
def test_mixtral_model_inference(t3k_mesh_device, use_program_cache, reset_seeds, batch):
    t3k_mesh_device.enable_async(True)

    valid_pcc = 0.97
    dtype = ttnn.bfloat8_b
    iterations = 10

    if batch == 32:
        generation_start_pos = 0
        max_seq_len = 16384
    elif batch in [4, 8, 16]:
        generation_start_pos = 0
        max_seq_len = 32768
    else:
        raise ValueError(f"Batch size {batch} not supported")

    model_args = TtModelArgs(t3k_mesh_device.get_device(0), max_seq_len=max_seq_len, max_batch_size=batch)
    state_dict = model_args.load_state_dict()
    tokenizer = Tokenizer(model_args.tokenizer_path)

    prompts = ["Once"] * batch
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    # Load reference model
    reference_model = Transformer(args=model_args)
    reference_model.load_state_dict(state_dict)
    reference_model.eval()

    # Embedding on host
    embd = Emb()
    embd.load_state_dict({"emb.weight": state_dict["tok_embeddings.weight"]})

    # Load TTNN model
    tt_model = TtTransformer(
        mesh_device=t3k_mesh_device,
        state_dict=state_dict,
        args=model_args,
        layers=list(range(model_args.n_layers)),
        start_pos_ids=[generation_start_pos for _ in range(batch)],
        dtype=dtype,
    )

    generation_length = iterations
    all_tests_pass = True

    seqlen = 1  # Generating one token per user at a time

    # Select the first token from the prompts for initial decoding
    encoded_prompts_tensor = torch.tensor(encoded_prompts)  # [:,0]
    pt_decode_input = embd(encoded_prompts_tensor[:, 0]).view(batch, seqlen, -1)

    tt_decode_input = pt_decode_input
    ref_tokens = []
    tt_tokens = []

    for i in range(generation_length):
        logger.info(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i
        start_pos_ids = [start_pos for _ in range(batch)]

        decode_input = prepare_inputs_ttnn(
            tt_decode_input,
            model_args.dim,
            tt_model.mesh_device,
        )

        # Run TT model
        tt_out = tt_model(decode_input, start_pos_ids)

        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
            .squeeze(1)
            .view(32, seqlen, -1)
            .detach()
            .float()
        )[:batch, ...]

        # Measure PCC
        positions = torch.LongTensor([start_pos])
        ref_output = reference_model(pt_decode_input, positions).detach().float()

        passing, pcc_message = comp_pcc(
            ref_output.view(batch, seqlen, -1), tt_output_torch.view(batch, seqlen, -1), valid_pcc
        )
        logger.info(comp_allclose(ref_output, tt_output_torch))
        logger.info(pcc_message)

        reference_top1 = np.argmax(ref_output, axis=-1).squeeze()
        top1_acc = top_k_accuracy_score(
            reference_top1, tt_output_torch.squeeze(), k=1, labels=np.arange(tt_output_torch.shape[-1])
        )
        top5_acc = top_k_accuracy_score(
            reference_top1, tt_output_torch.squeeze(), k=5, labels=np.arange(tt_output_torch.shape[-1])
        )
        logger.info(f"Mean Top-1: {top1_acc}")
        logger.info(f"Mean Top-5: {top5_acc}")

        ref_token_batch = ref_output.squeeze().argmax(axis=-1)
        tt_token_batch = tt_output_torch.squeeze().argmax(axis=-1)
        logger.info(f"ref_output: {tokenizer.decode(ref_token_batch[0].item())}")
        logger.info(f"tt_output: {tokenizer.decode(tt_token_batch[0].item())}")
        pt_decode_input = embd(ref_token_batch).view(batch, seqlen, -1)
        tt_decode_input = pt_decode_input  # teacher forcing for PCC test

        if passing:
            logger.info("Mistral Model Passed!")
        else:
            logger.warning("Mistral Model Failed!")
            all_tests_pass = False

    if all_tests_pass:
        logger.info(f"All {generation_length} Mistral decode iterations Passed!")
    else:
        logger.warning("One or more iterations of Mistral decode Failed!")
        assert (
            all_tests_pass
        ), f"PCC value is lower than the expected {valid_pcc} for some of the outputs. Check Warnings!"
