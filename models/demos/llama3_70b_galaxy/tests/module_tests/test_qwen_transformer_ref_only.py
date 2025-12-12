# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from loguru import logger
import ttnn
from models.demos.llama3_70b_galaxy.tt.qwen_model_config import TtQwenModelArgs
from models.demos.llama3_70b_galaxy.reference.qwen import Transformer


@torch.no_grad()
@pytest.mark.parametrize(
    "batch_size",
    (32,),
)
@pytest.mark.parametrize(
    "max_seq_len",
    (256,),  # For decode-only unit test, there's no need to run with large sequence lengths
)
def test_qwen_transformer_inference(
    max_seq_len,
    batch_size,
):
    dtype = ttnn.bfloat8_b

    model_args = TtQwenModelArgs(
        mesh_device=None, max_batch_size=batch_size, max_seq_len=max_seq_len, dummy_weights=False
    )
    model_args.n_layers = 3

    state_dict = model_args.load_state_dict()

    # Setup reference model using full Transformer (ignoring embedding as requested)
    state_dict_prefix = model_args.get_state_dict_prefix("", None)
    reference_state_dict = {
        k[len(state_dict_prefix) :]: v
        for k, v in state_dict.items()
        if (
            any([f"{state_dict_prefix}layers.{i}." in k for i in range(model_args.n_layers)])
            or any(
                [
                    f"{state_dict_prefix}{name}" in k
                    for name in ["norm.weight", "output.weight", "tok_embeddings.weight"]
                ]
            )
        )
    }
    reference_model = Transformer(model_args)
    reference_model.load_state_dict(reference_state_dict)

    generation_start_pos = 0
    generation_length = 2

    seqlen = 1

    # Initial positions
    current_pos = torch.tensor([generation_start_pos for _ in range(batch_size)])
    # Create random input tensor (skipping embedding as requested)
    pt_decode_input = (torch.rand(batch_size, seqlen, model_args.dim) * 2) - 1

    for i in range(generation_length):
        logger.info(f"[Qwen Transformer] Generating token {i}")

        # Reference model - skip embedding, directly use input tensor
        ref_output = reference_model(pt_decode_input, current_pos[0], mode="decode")

        print(ref_output)
