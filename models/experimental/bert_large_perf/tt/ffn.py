# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
from loguru import logger

import torch
from transformers import BertForQuestionAnswering
import ttnn
from tt_lib.utils import pad_activation, pad_weight, print_diff_argmax
from models.utility_functions import comp_pcc, comp_allclose


def feed_forward(ffn_dim, hidden_dim, ff1_weighta, ff1_biasa, ff2_weighta, ff2_biasa, device):
    # Weights pre-transposed on host​. No on-the fly transpose of W.
    ff1_weighta = ttnn.transpose(ff1_weighta, -2, -1)
    ff2_weighta = ttnn.transpose(ff2_weighta, -2, -1)

    # activation = [1, 9, 384, 1024]
    # ff1_weighta = [1, 1, 1024, 4096]
    # output = [1, 9, 384, 4096]
    def op13_MM_bias_gelu(activation, ff1_weighta, ff1_biasa):
        # profiler.start("___op13_MM_bias_gelu")
        output = ttnn.matmul(activation, ff1_weighta)
        output_plus_bias = ttnn.add(output, ff1_biasa)
        output_plus_bias_act = ttnn.gelu(output_plus_bias)
        # profiler.end("___op13_MM_bias_gelu")

        return output_plus_bias_act

    # activation = [1, 9, 384, 4096]
    # ff2_weighta = [1, 1, 4096, 1024]
    # output = [1, 9, 384, 1024]
    def op14_MM_bias(activation, ff2_weighta, ff2_biasa):
        # profiler.start("___op14_MM_bias")
        output = ttnn.matmul(activation, ff2_weighta)
        output_plus_bias = ttnn.add(output, ff2_biasa)
        # profiler.end("___op14_MM_bias")

        return output_plus_bias

    def feed_forward_(activation):
        # profiler.start("__ffn")
        ff1_output_plus_bias_act = op13_MM_bias_gelu(activation, ff1_weighta, ff1_biasa)
        ff2_output_plus_bias = op14_MM_bias(ff1_output_plus_bias_act, ff2_weighta, ff2_biasa)
        # profiler.end("__ffn")

        return ff2_output_plus_bias

    return feed_forward_


class TtFeedForwardModel(torch.nn.Module):
    def __init__(self, encoder_idx, state_dict, device):
        super().__init__()

        # FF1 params
        encoder0_ff1_weight = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.intermediate.dense.weight"])
        encoder0_ff1_bias = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.intermediate.dense.bias"])

        encoder0_ff1_weight_shape = encoder0_ff1_weight.shape
        encoder0_ff1_bias_shape = encoder0_ff1_bias.shape

        encoder0_ff1_weight = (
            ttnn.Tensor(
                encoder0_ff1_weight.reshape(-1).tolist(),
                encoder0_ff1_weight.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        encoder0_ff1_bias = (
            ttnn.Tensor(
                encoder0_ff1_bias.reshape(-1).tolist(),
                encoder0_ff1_bias.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )

        # FF2 params
        encoder0_ff2_weight = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.output.dense.weight"])
        encoder0_ff2_bias = pad_weight(state_dict[f"bert.encoder.layer.{encoder_idx}.output.dense.bias"])

        encoder0_ff2_weight_shape = encoder0_ff2_weight.shape
        encoder0_ff2_bias_shape = encoder0_ff2_bias.shape

        encoder0_ff2_weight = (
            ttnn.Tensor(
                encoder0_ff2_weight.reshape(-1).tolist(),
                encoder0_ff2_weight.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )
        encoder0_ff2_bias = (
            ttnn.Tensor(
                encoder0_ff2_bias.reshape(-1).tolist(),
                encoder0_ff2_bias.shape,
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device)
        )

        self.ffn = feed_forward(
            *encoder0_ff1_weight_shape[-2:],
            encoder0_ff1_weight,
            encoder0_ff1_bias,
            encoder0_ff2_weight,
            encoder0_ff2_bias,
            device,
        )

    def forward(self, activation):
        return self.ffn(activation)


class PytorchFeedForwardModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.ff1 = hugging_face_reference_model.bert.encoder.layer[0].intermediate
        self.ff2 = hugging_face_reference_model.bert.encoder.layer[0].output.dense

    def forward(self, x):
        return self.ff2(self.ff1(x))


def summarize_stats(t, name):
    mean = t.mean()
    std = t.std()
    mag = t.norm()
    max = t.max()
    print(f"STATS FOR {name}")
    print(f"mean {mean}")
    print(f"std {std}")
    print(f"mag {mag}")
    print(f"max {max}")
    print()


def run_ffn_inference(device, model_version, batch, seq_len, pcc, model_location_generator):
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tt_ffn_model = TtFeedForwardModel(0, hugging_face_reference_model.state_dict(), device)
    pytorch_ffn_model = PytorchFeedForwardModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    ffn_input = (torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size) * 2) - 1

    pytorch_out = pytorch_ffn_model(ffn_input)

    pad_ffn_input = pad_activation(ffn_input)
    tilized_ffn_input = ttnn.Tensor(
        pad_ffn_input.reshape(-1).tolist(),
        pad_ffn_input.shape,
        ttnn.bfloat16,
        ttnn.ROW_MAJOR_LAYOUT,
    ).to(ttnn.TILE_LAYOUT)
    tilized_ffn_input = tilized_ffn_input.to(device)

    tt_out = tt_ffn_model(tilized_ffn_input).cpu()
    tt_out = tt_out.to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    passing, output = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output {output}")

    _, output = comp_allclose(
        pytorch_out, tt_out, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")

    if not passing:
        logger.error(f"Output PCC < {pcc}")


@pytest.mark.parametrize(
    "model_version, batch, seq_len, pcc",
    (
        ("mrm8488/bert-tiny-finetuned-squadv2", 1, 128, 0.99),
        ("phiyodr/bert-base-finetuned-squad2", 1, 128, 0.99),
        ("phiyodr/bert-large-finetuned-squad2", 1, 384, 0.99),
    ),
)
def test_ffn_inference(device, model_version, batch, seq_len, pcc, model_location_generator):
    run_ffn_inference(device, model_version, batch, seq_len, pcc, model_location_generator)
