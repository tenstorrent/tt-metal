# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import os
import torch
import pytest
from loguru import logger
from sklearn.metrics import top_k_accuracy_score

import ttnn
from models.demos.t3000.mixtral8x7b.tt.mixtral_common import prepare_inputs_ttnn, preprocess_inputs, load_inputs
from models.demos.t3000.mixtral8x7b.tt.mixtral_model import TtTransformer
from models.demos.t3000.mixtral8x7b.reference.model import Transformer
from models.demos.t3000.mixtral8x7b.reference.tokenizer import Tokenizer
from models.demos.t3000.mixtral8x7b.tt.model_config import TtModelArgs


class Emb(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(32000, 4096)

    def forward(self, x):
        return self.emb(x)


# Expected times for each test: 64seqlen: 11 min, 128seqlen: 22 min, 256seqlen: 44 min
@pytest.mark.timeout(3600)
@pytest.mark.parametrize(
    "iterations, expected_top1, expected_top5",
    (
        (64, 0.93, 0.99),
        # (128, 0.92, 0.99),
        # (256, 0.92, 0.99),
    ),
    ids=(
        "64seqlen",
        # "128seqlen",
        # "256seqlen"
    ),
)
def test_mixtral_model_inference(
    t3k_mesh_device, use_program_cache, reset_seeds, iterations, expected_top1, expected_top5
):
    for device in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device).enable_async(True)

    dtype = ttnn.bfloat8_b
    seqlen = 1  # Generating one token per user at a time
    batch = 32
    generation_start_pos = 0
    running_top1 = 0
    running_top5 = 0
    inputs_file = "models/demos/t3000/mixtral8x7b/demo/input_data.json"

    model_args = TtModelArgs(t3k_mesh_device.get_device(0))
    state_dict = model_args.load_state_dict()
    tokenizer = Tokenizer(model_args.tokenizer_path)

    # Prepare inputs
    input_prompts = load_inputs(inputs_file, 32)
    input_tokens_tt, max_prompt_len, input_mask, input_tokens_pt, input_mask_pt = preprocess_inputs(
        input_prompts, tokenizer, model_args, dtype, False, t3k_mesh_device
    )

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
        dtype=dtype,
        rotary_on_host=False,
        start_pos_ids=[generation_start_pos] * batch,
    )

    # Select the first token from the prompts for initial decoding
    pt_decode_input = embd(input_tokens_pt[:, 0]).view(batch, seqlen, -1)

    for i in range(iterations):
        logger.info(f"[Decode] Generating token {i}")

        start_pos = generation_start_pos + i

        decode_input = prepare_inputs_ttnn(
            pt_decode_input,
            model_args.dim,
            start_pos,
            model_args,
            tt_model.mesh_device,
        )

        # Run TT model
        tt_out = tt_model(decode_input, [start_pos] * batch)
        # Convert ttnn tensor to torch tensor
        tt_output_torch = (
            ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(t3k_mesh_device, dim=0))[0]
            .squeeze(1)
            .view(32, seqlen, -1)
            .detach()
            .float()
        )[:batch, ...]

        # Run reference model
        positions = torch.LongTensor([start_pos])
        ref_output = reference_model(pt_decode_input, positions).detach().float()

        # Measure top-1 and top-5
        reference_top1 = torch.argmax(ref_output, axis=-1)
        top1_acc = top_k_accuracy_score(
            reference_top1.squeeze(), tt_output_torch.squeeze(), k=1, labels=torch.arange(tt_output_torch.shape[-1])
        )
        top5_acc = top_k_accuracy_score(
            reference_top1.squeeze(), tt_output_torch.squeeze(), k=5, labels=torch.arange(tt_output_torch.shape[-1])
        )
        running_top1 += top1_acc
        running_top5 += top5_acc

        # Prepare next input - teacher forcing (top-1), after initial prompts are done
        if i < max_prompt_len:  # Check if user has finished initial prompt or not
            reference_top1 = torch.where(input_mask_pt[:, i], input_tokens_pt[:, i], reference_top1[:, 0]).unsqueeze(1)
        pt_decode_input = embd(reference_top1).view(batch, seqlen, -1)

    # Validate accuracy metrics against the expected values
    final_top1 = running_top1 / iterations
    final_top5 = running_top5 / iterations
    logger.info(f"Mean Top-1 accuracy: {final_top1:.4f}")
    logger.info(f"Mean Top-5 accuracy: {final_top5:.4f}")

    assert final_top1 >= expected_top1, f"Top-1 accuracy is lower than {expected_top1}"
    assert final_top5 >= expected_top5, f"Top-5 accuracy is lower than {expected_top5}"
