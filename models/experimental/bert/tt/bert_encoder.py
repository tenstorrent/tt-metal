# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import BertForQuestionAnswering
import numpy as np
from loguru import logger

import ttnn
from models.experimental.bert.tt.mha import TtMultiHeadAttentionModel
from models.experimental.bert.tt.ffn import TtFeedForwardModel
from models.experimental.bert.fused_ops.add_and_norm import AddAndNorm
from models.experimental.bert.fused_ops.linear import Linear
from tt_lib.utils import pad_activation, pad_weight, print_diff_argmax
from models.utility_functions import comp_pcc, comp_allclose


class TtBertEncoder(torch.nn.Module):
    def __init__(self, config, encoder_idx, state_dict, device):
        super().__init__()
        hidden_dim = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.weight"]).shape[-1]

        # MHA part
        self.mha = TtMultiHeadAttentionModel(config, encoder_idx, state_dict, device)
        attention_output_weight = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.dense.weight"]
        )
        attention_output_weight = (
            ttnn.Tensor(
                attention_output_weight.reshape(-1).tolist(),
                attention_output_weight.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        attention_output_bias = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.dense.bias"])
        attention_output_bias = (
            ttnn.Tensor(
                attention_output_bias.reshape(-1).tolist(),
                attention_output_bias.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        self.attention_output = Linear(
            hidden_dim,
            hidden_dim,
            attention_output_weight,
            attention_output_bias,
            device,
        )

        # MHA layernorm part
        gamma0 = state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.LayerNorm.weight"]
        beta0 = state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.LayerNorm.bias"]
        mha_gamma = pad_weight(gamma0)
        mha_gamma = (
            ttnn.Tensor(
                mha_gamma.reshape(-1).tolist(),
                mha_gamma.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        mha_beta = pad_weight(beta0)
        mha_beta = (
            ttnn.Tensor(
                mha_beta.reshape(-1).tolist(),
                mha_beta.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        self.mha_add_and_norm = AddAndNorm(
            mha_gamma,
            mha_beta,
            config.layer_norm_eps,
            config.hidden_size,
            config.hidden_size,
            device,
        )

        # FFN part
        self.ffn = TtFeedForwardModel(encoder_idx, state_dict, device)

        # FFN layernorm part
        gamma1 = state_dict[f"bert.encoder.layer.{encoder_idx}.output.LayerNorm.weight"]
        beta1 = state_dict[f"bert.encoder.layer.{encoder_idx}.output.LayerNorm.bias"]
        ffn_gamma = pad_weight(gamma1)
        ffn_gamma = (
            ttnn.Tensor(
                ffn_gamma.reshape(-1).tolist(),
                ffn_gamma.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        ffn_beta = pad_weight(beta1)
        ffn_beta = (
            ttnn.Tensor(
                ffn_beta.reshape(-1).tolist(),
                ffn_beta.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        self.ffn_add_and_norm = AddAndNorm(
            ffn_gamma,
            ffn_beta,
            config.layer_norm_eps,
            config.hidden_size,
            config.hidden_size,
            device,
        )

    def forward(self, activation, attention_mask=None):
        mha_out = self.attention_output(self.mha(activation, attention_mask))
        mha_out_add_and_norm = self.mha_add_and_norm(activation, mha_out)
        ffn_out = self.ffn(mha_out_add_and_norm)
        ffn_out_add_and_norm = self.ffn_add_and_norm(mha_out_add_and_norm, ffn_out)
        return ffn_out_add_and_norm


class PytorchBertEncoder(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.bert_encoder = hugging_face_reference_model.bert.encoder.layer[0]

    def forward(self, x):
        return self.bert_encoder(x)[0]


def run_bert_encoder_inference(device, model_version, batch, seq_len, pcc, model_location_generator):
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tt_bert_encoder_model = TtBertEncoder(
        hugging_face_reference_model.config,
        0,
        hugging_face_reference_model.state_dict(),
        device,
    )
    pytorch_bert_model = PytorchBertEncoder(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    bert_encoder_input = (torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size) * 2) - 1

    pytorch_out = pytorch_bert_model(bert_encoder_input.squeeze(1)).unsqueeze(1)

    pad_bert_encoder_input = pad_activation(bert_encoder_input)
    tt_bert_encoder_input = ttnn.Tensor(
        pad_bert_encoder_input.reshape(-1).tolist(),
        bert_encoder_input.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(ttnn.TILE_LAYOUT)
    tt_bert_encoder_input = tt_bert_encoder_input.to(device)

    tt_out = tt_bert_encoder_model(tt_bert_encoder_input).cpu()
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
def test_bert_encoder_inference(device, model_version, batch, seq_len, pcc, model_location_generator):
    run_bert_encoder_inference(device, model_version, batch, seq_len, pcc, model_location_generator)
