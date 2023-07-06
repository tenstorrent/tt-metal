import pytest
import torch
from transformers import BertForQuestionAnswering
import numpy as np
from loguru import logger
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from tests.python_api_testing.models.conftest import model_location_generator_
import tt_lib as ttl
from python_api_testing.models.metal_BERT_large_15.mha import TtMultiHeadAttentionModel
from python_api_testing.models.metal_BERT_large_15.ffn import TtFeedForwardModel
from tt_lib.utils import pad_activation, pad_weight
from utility_functions import comp_pcc, comp_allclose, profiler


class TtBertEncoder(torch.nn.Module):
    def __init__(self, config, encoder_idx, state_dict, device, mem_config):
        super().__init__()
        self.device = device
        self.mem_config = mem_config

        # MHA part
        self.mha = TtMultiHeadAttentionModel(config, encoder_idx, state_dict, device, mem_config)

        self.attention_output_weight = pad_weight(
            torch.transpose(
                state_dict[
                    f"bert.encoder.layer.{encoder_idx}.attention.output.dense.weight"
                ],
                -2,
                -1,
            )
        )
        self.attention_output_weight = (
            ttl.tensor.Tensor(
                self.attention_output_weight.reshape(-1).tolist(),
                self.attention_output_weight.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )
        self.attention_output_bias = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.output.dense.bias"]
        )
        self.attention_output_bias = (
            ttl.tensor.Tensor(
                self.attention_output_bias.reshape(-1).tolist(),
                self.attention_output_bias.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )

        # Weights pre-transposed on host​. No on-the fly transpose of W.
        # self.attention_output_weight = ttl.tensor.transpose(
        #     self.attention_output_weight
        # )

        # MHA layernorm part
        gamma0 = state_dict[
            f"bert.encoder.layer.{encoder_idx}.attention.output.LayerNorm.weight"
        ]
        beta0 = state_dict[
            f"bert.encoder.layer.{encoder_idx}.attention.output.LayerNorm.bias"
        ]
        mha_gamma = pad_weight(gamma0)
        self.mha_gamma = (
            ttl.tensor.Tensor(
                mha_gamma.reshape(-1).tolist(),
                mha_gamma.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )
        mha_beta = pad_weight(beta0)
        self.mha_beta = (
            ttl.tensor.Tensor(
                mha_beta.reshape(-1).tolist(),
                mha_beta.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )

        # FFN part
        self.ffn = TtFeedForwardModel(encoder_idx, state_dict, device, mem_config)

        # FFN layernorm part
        gamma1 = state_dict[f"bert.encoder.layer.{encoder_idx}.output.LayerNorm.weight"]
        beta1 = state_dict[f"bert.encoder.layer.{encoder_idx}.output.LayerNorm.bias"]
        ffn_gamma = pad_weight(gamma1)
        self.ffn_gamma = (
            ttl.tensor.Tensor(
                ffn_gamma.reshape(-1).tolist(),
                ffn_gamma.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )
        ffn_beta = pad_weight(beta1)
        self.ffn_beta = (
            ttl.tensor.Tensor(
                ffn_beta.reshape(-1).tolist(),
                ffn_beta.shape,
                ttl.tensor.DataType.BFLOAT16,
                ttl.tensor.Layout.ROW_MAJOR,
            )
            .to(ttl.tensor.Layout.TILE)
            .to(device)
        )

        self.layer_norm_eps = config.layer_norm_eps

    def op11_mm_plus_bias(
        self, mha_res, attention_output_weight, attention_output_bias
    ):
        # profiler.start("__op11_mm_plus_bias")
        mha_out = ttl.tensor.bert_large_selfout_matmul(mha_res, attention_output_weight, attention_output_bias, mem_config=self.mem_config)
        # profiler.end("__op11_mm_plus_bias")

        return mha_out

    def op12_add_layernorm(self, activation, mha_out):
        # profiler.start("__op12_add_layernorm")
        mha_out_add_and_norm = ttl.tensor.add_layernorm_gamma_beta(
            activation,
            mha_out,
            self.layer_norm_eps,
            self.mha_gamma,
            self.mha_beta,
            self.mem_config,
        )
        # profiler.end("__op12_add_layernorm")

        return mha_out_add_and_norm

    def op15_add_layernorm(self, mha_out_add_and_norm, ffn_out):
        # profiler.start("__op15_add_layernorm")
        ffn_out_add_and_norm = ttl.tensor.add_layernorm_gamma_beta(
            mha_out_add_and_norm,
            ffn_out,
            self.layer_norm_eps,
            self.ffn_gamma,
            self.ffn_beta,
            self.mem_config,
        )
        # profiler.end("__op15_add_layernorm")

        return ffn_out_add_and_norm

    def forward(self, activation, attention_mask=None):
        assert activation.shape() == [9, 1, 384, 1024]
        # MHA - OP1 - OP10 ------------------------------->
        mha_res = self.mha(activation, attention_mask)
        # Don't deallocate activations here since it is used by more ops
        # activation.deallocate()

        mha_out = self.op11_mm_plus_bias(
            mha_res, self.attention_output_weight, self.attention_output_bias
        )
        mha_res.deallocate()
        mha_out_add_and_norm = self.op12_add_layernorm(activation, mha_out)
        activation.deallocate()
        mha_out.deallocate()

        # FFN - OP13 - OP14 ----------------------------->
        ffn_out = self.ffn(mha_out_add_and_norm)

        ffn_out_add_and_norm = self.op15_add_layernorm(mha_out_add_and_norm, ffn_out)
        mha_out_add_and_norm.deallocate()
        ffn_out.deallocate()
        return ffn_out_add_and_norm


class PytorchBertEncoder(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.bert_encoder = hugging_face_reference_model.bert.encoder.layer[0]

    def forward(self, x, attention_mask=None):
        return self.bert_encoder(x, attention_mask)[0]


def run_bert_encoder_inference(
    model_version, batch, seq_len, on_weka, dram, pcc, model_location_generator
):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    # Initialize the device
    ttl.device.InitializeDevice(device, ttl.device.MemoryAllocator.BASIC if dram else ttl.device.MemoryAllocator.L1_BANKING)
    host = ttl.device.GetHost()
    mem_config = ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM if dram else ttl.tensor.BufferType.L1)

    if on_weka:
        model_name = str(
            model_location_generator(
                "tt_dnn-models/Bert/BertForQuestionAnswering/models/"
            )
            / model_version
        )
    else:
        model_name = model_version

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(
        model_name, torchscript=False
    )
    config = hugging_face_reference_model.config

    tt_bert_encoder_model = TtBertEncoder(
        hugging_face_reference_model.config,
        0,
        hugging_face_reference_model.state_dict(),
        device,
        mem_config
    )
    pytorch_bert_model = PytorchBertEncoder(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    bert_encoder_input = (
        torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size)
        * 2
    ) - 1
    bert_attention_mask = torch.zeros(batch, 1, 1, seq_len)
    extended_bert_attention_mask = torch.zeros(batch, 1, 32, seq_len)

    pytorch_out = pytorch_bert_model(
        bert_encoder_input.squeeze(1), bert_attention_mask
    ).unsqueeze(1)

    pad_bert_encoder_input = pad_activation(bert_encoder_input)
    tt_bert_encoder_input = (
        ttl.tensor.Tensor(
            pad_bert_encoder_input.reshape(-1).tolist(),
            bert_encoder_input.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, mem_config)
    )
    tt_bert_attention_mask = (
        ttl.tensor.Tensor(
            extended_bert_attention_mask.reshape(-1).tolist(),
            extended_bert_attention_mask.shape,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(device, mem_config)
    )

    tt_out = tt_bert_encoder_model(tt_bert_encoder_input, tt_bert_attention_mask).to(
        host
    )
    tt_out = torch.Tensor(tt_out.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        tt_out.shape()
    )

    ttl.device.CloseDevice(device)

    passing, output = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output {output}")

    _, output = comp_allclose(
        pytorch_out, tt_out, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")

    if not passing:
        logger.error(f"Output PCC < {pcc}")
    assert(passing)

@pytest.mark.parametrize(
    "model_version, batch, seq_len, on_weka, dram, pcc",
    (
        ("phiyodr/bert-large-finetuned-squad2", 9, 384, True, True, 0.99),
        ("phiyodr/bert-large-finetuned-squad2", 9, 384, True, False, 0.99),
    ),
    ids=["DRAM", "L1"],
)
def test_bert_encoder_inference(
    model_version, batch, seq_len, on_weka, dram, pcc, model_location_generator
):
    ttl.profiler.set_profiler_flag(False)
    ttl.profiler.set_profiler_location(
        "tt_metal/tools/profiler/logs/BERT_large_1_encoder"
    )

    ttl.profiler.start_profiling("entire_run")
    run_bert_encoder_inference(
        model_version, batch, seq_len, on_weka, dram, pcc, model_location_generator
    )
    ttl.profiler.stop_profiling("entire_run")


if __name__ == "__main__":
    run_bert_encoder_inference(
        "phiyodr/bert-large-finetuned-squad2",
        9,
        384,
        True,
        True,
        0.99,
        model_location_generator_,
    )
