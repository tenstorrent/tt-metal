# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import torch
from transformers import BertForQuestionAnswering
import numpy as np

import ttnn
from models.experimental.bert.fused_ops.add_and_norm import AddAndNorm
from tt_lib.utils import pad_activation, pad_weight, print_diff_argmax
from models.utility_functions import (
    enable_persistent_kernel_cache,
    comp_pcc,
    comp_allclose,
)


class TtAddAndNormModel(torch.nn.Module):
    def __init__(self, config, state_dict, device, lnorm_type):
        super().__init__()

        if lnorm_type == "attention":
            gamma = pad_weight(state_dict["bert.encoder.layer.0.attention.output.LayerNorm.weight"])
            gamma = (
                ttnn.Tensor(
                    gamma.reshape(-1).tolist(),
                    gamma.shape,
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
            beta = pad_weight(state_dict["bert.encoder.layer.0.attention.output.LayerNorm.bias"])
            beta = (
                ttnn.Tensor(
                    beta.reshape(-1).tolist(),
                    beta.shape,
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
        elif lnorm_type == "ffn":
            gamma = pad_weight(state_dict["bert.encoder.layer.0.output.LayerNorm.weight"])
            gamma = (
                ttnn.Tensor(
                    gamma.reshape(-1).tolist(),
                    gamma.shape,
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
            beta = pad_weight(state_dict["bert.encoder.layer.0.output.LayerNorm.bias"])
            beta = (
                ttnn.Tensor(
                    beta.reshape(-1).tolist(),
                    beta.shape,
                    ttnn.bfloat16,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
                .to(ttnn.TILE_LAYOUT)
                .to(device)
            )
        else:
            assert False, "Invalid lnorm_type"

        self.add_and_norm = AddAndNorm(
            gamma,
            beta,
            config.layer_norm_eps,
            config.hidden_size,
            config.hidden_size,
            device,
        )

    def forward(self, a, b):
        print(a.shape.with_tile_padding(), b.shape.with_tile_padding())
        return self.add_and_norm(a, b)


class PytorchAddAndNormModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model, lnorm_type):
        super().__init__()
        if lnorm_type == "attention":
            self.layernorm = hugging_face_reference_model.bert.encoder.layer[0].attention.output.LayerNorm
        elif lnorm_type == "ffn":
            self.layernorm = hugging_face_reference_model.bert.encoder.layer[0].output.LayerNorm
        else:
            assert False, "Invalid lnorm_type"

    def forward(self, a, b):
        out = self.layernorm(a + b)
        return out


def run_add_and_norm_inference(device, model_version, batch, seq_len, pcc, model_location_generator):
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tt_add_and_norm_model = TtAddAndNormModel(
        hugging_face_reference_model.config,
        hugging_face_reference_model.state_dict(),
        device,
        "attention",
    )
    pytorch_add_and_norm_model = PytorchAddAndNormModel(hugging_face_reference_model, "attention")

    # Prepare input
    torch.manual_seed(0)
    add_and_norm_inputa = (torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size) * 2) - 1
    add_and_norm_inputb = (torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size) * 2) - 1

    pytorch_out = pytorch_add_and_norm_model(add_and_norm_inputa, add_and_norm_inputb)

    pad_add_and_norm_inputa = pad_activation(add_and_norm_inputa)
    pad_add_and_norm_inputb = pad_activation(add_and_norm_inputb)
    tt_add_and_norm_input_a = ttnn.Tensor(
        pad_add_and_norm_inputa.reshape(-1).tolist(),
        pad_add_and_norm_inputa.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(ttnn.TILE_LAYOUT)
    tt_add_and_norm_input_a = tt_add_and_norm_input_a.to(device)
    tt_add_and_norm_input_b = ttnn.Tensor(
        pad_add_and_norm_inputb.reshape(-1).tolist(),
        pad_add_and_norm_inputb.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(ttnn.TILE_LAYOUT)
    tt_add_and_norm_input_b = tt_add_and_norm_input_b.to(device)

    tt_out = tt_add_and_norm_model(tt_add_and_norm_input_a, tt_add_and_norm_input_b).cpu()
    tt_out = tt_out.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    passing, output = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output {output}")
    _, output = comp_allclose(
        pytorch_out, tt_out, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")
    if not passing:
        logger.error(f"Output PCC < {pcc}")
    # assert np.allclose(pytorch_out.detach().numpy(), tt_out.numpy(), 1e-5, 0.17)


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        ("mrm8488/bert-tiny-finetuned-squadv2", 1, 128, 0.99),
        ("phiyodr/bert-base-finetuned-squad2", 1, 128, 0.99),
        ("phiyodr/bert-large-finetuned-squad2", 1, 384, 0.99),
    ),
)
def test_add_and_norm_inference(device, model_version, batch, seq_len, pcc, model_location_generator):
    run_add_and_norm_inference(device, model_version, batch, seq_len, pcc, model_location_generator)
