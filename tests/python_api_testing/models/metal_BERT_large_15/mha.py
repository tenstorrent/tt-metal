import pytest
from loguru import logger
import math
from pathlib import Path
import sys

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import torch
from transformers import BertForQuestionAnswering
import numpy as np

from tests.python_api_testing.models.conftest import model_location_generator_
import tt_lib as ttl
from tt_lib.utils import pad_activation, pad_weight, print_diff_argmax
from utility_functions import enable_compile_cache, comp_pcc, comp_allclose, profiler


def torch2tt_tensor(py_tensor: torch.Tensor, tt_device):
    size = list(py_tensor.size())

    while len(size) < 4:
        size.insert(0, 1)

    tt_tensor = (
        ttl.tensor.Tensor(
            py_tensor.reshape(-1).tolist(),
            size,
            ttl.tensor.DataType.BFLOAT16,
            ttl.tensor.Layout.ROW_MAJOR,
        )
        .to(ttl.tensor.Layout.TILE)
        .to(tt_device)
    )

    return tt_tensor


def tt2torch_tensor(tt_tensor):
    host = ttl.device.GetHost()
    tt_output = tt_tensor.to(host)
    if tt_output.layout() != ttl.tensor.Layout.ROW_MAJOR:
        tt_output = tt_output.to(ttl.tensor.Layout.ROW_MAJOR)
    py_output = torch.Tensor(tt_output.data()).reshape(tt_output.shape())
    return py_output


def mha(qw, qb, kw, kb, vw, vb, hidden_dim, num_heads, device, mem_config):
    assert isinstance(num_heads, int) and num_heads > 0

    # Weights pre-transposed on host​. No on-the fly transpose of W​
    qw = torch.transpose(qw, -1, -2)
    kw = torch.transpose(kw, -1, -2)
    vw = torch.transpose(vw, -1, -2)

    qkv_weight = torch.cat((qw, kw, vw), -1)
    qkv_bias = torch.cat((qb, kb, vb), -1)

    qkv_weight = torch2tt_tensor(qkv_weight, device)
    qkv_bias = torch2tt_tensor(qkv_bias, device)

    # Used to scale down the input to the softmax
    freciprocal_of_sqrt_hidden_dim = 1 / math.sqrt(hidden_dim // num_heads)
    reciprocal_of_sqrt_hidden_dim_tensor = ttl.tensor.Tensor(
        [1 / math.sqrt(hidden_dim // num_heads)] + [0 for _ in range(32 * 32 - 1)],
        [1, 1, 32, 32],
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.TILE,
        device,
    )

    def multiply_by_sqrt_hidden_dim(x):
        return ttl.tensor.bcast(
            x,
            reciprocal_of_sqrt_hidden_dim_tensor,
            ttl.tensor.BcastOpMath.MUL,
            ttl.tensor.BcastOpDim.HW,
            mem_config,
        )

    def op1_qkv_fused(activation, qkv_weight, qkv_bias):
        # profiler.start("___op1_qkv_fused")
        qkv = ttl.tensor.bert_large_fused_qkv_matmul(activation, qkv_weight, qkv_bias, mem_config=mem_config)
        # profiler.end("___op1_qkv_fused")

        return qkv

    def op2to6_create_qkv_heads(qkv):
        # profiler.start("___op2to6_create_qkv_heads")
        q_heads, kt_heads, v_heads = ttl.tensor.bert_large_create_qkv_heads(
            qkv, mem_config
        )
        # profiler.end("___op2to6_create_qkv_heads")

        return q_heads, kt_heads, v_heads

    def op7_bmm(Q_heads, K_T_heads):
        # profiler.start("___op7_bmm")
        qkt = ttl.tensor.bert_large_pre_softmax_bmm(Q_heads, K_T_heads, mem_config=ttl.tensor.MemoryConfig(True, ttl.tensor.BufferType.DRAM))
        # profiler.end("___op7_bmm")

        return qkt

    def op8_scale_mask_softmax(qkt, attention_mask):
        # Attention scores computation
        # profiler.start("___op8_scale_mask_softmax")

        # Input and output tensors of this fused op is: [9, 1, 6144, 384] instead of [9, 16, 384, 384]
        # No-op reshapes are handled within pre-softmax (op 7) and post-softmax bmms (op 9)
        if attention_mask is not None:
            attention_scores = ttl.tensor.scale_mask_softmax_in_place(
                freciprocal_of_sqrt_hidden_dim, attention_mask, qkt
            )
        else:
            attention_score_input = multiply_by_sqrt_hidden_dim(qkt)
            attention_scores = ttl.tensor.softmax_in_place(attention_score_input)
        # profiler.end("___op8_scale_mask_softmax")

        return attention_scores

    def op9_bmm(attention_scores, V_heads):
        # profiler.start("___op9_bmm")
        weighted_activation = ttl.tensor.bert_large_post_softmax_bmm(attention_scores, V_heads, mem_config=mem_config)
        # profiler.end("___op9_bmm")

        return weighted_activation

    def op10_unmake_attention_heads(x):
        if num_heads == 1:
            # profiler.start("___op10_unmake_attention_heads")
            # profiler.end("___op10_unmake_attention_heads")
            return x
        else:
            # profiler.start("___op10_unmake_attention_heads")
            retval = ttl.tensor.bert_large_concat_heads(x, mem_config)
            # profiler.end("___op10_unmake_attention_heads")

            return retval

    def mha_(activation, attention_mask):
        # profiler.start("__mha")
        qkv = op1_qkv_fused(activation, qkv_weight, qkv_bias)
        # activation.deallocate()

        Q_heads, K_T_heads, V_heads = op2to6_create_qkv_heads(qkv)
        qkv.deallocate()

        qkt = op7_bmm(Q_heads, K_T_heads)
        Q_heads.deallocate()
        K_T_heads.deallocate()

        attention_scores = op8_scale_mask_softmax(qkt, attention_mask)
        # Should be a no-op deallocate since it was moved?
        # qkt.deallocate()
        weighted_activation = op9_bmm(attention_scores, V_heads)
        attention_scores.deallocate()
        V_heads.deallocate()

        res = op10_unmake_attention_heads(
            weighted_activation
        )  # [N, num heads, seq len, hid size / num heads] -> [N, seq len, hid size]
        # profiler.end("__mha")
        weighted_activation.deallocate()

        return res

    return mha_


class TtMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, config, encoder_idx, state_dict, device, mem_config):
        super().__init__()
        qw = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.weight"]
        )
        qb = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.bias"]
        )
        kw = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.weight"]
        )
        kb = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.bias"]
        )
        vw = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.weight"]
        )
        vb = pad_weight(
            state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.bias"]
        )

        # Hidden dim
        hidden_dim = qw.shape[-1]

        self.mha = mha(
            qw,
            qb,
            kw,
            kb,
            vw,
            vb,
            hidden_dim,
            config.num_attention_heads,
            device,
            mem_config,
        )

    def forward(self, activation, attention_mask=None):
        result = self.mha(activation, attention_mask)
        return result


class PytorchMultiHeadAttentionModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.mha = hugging_face_reference_model.bert.encoder.layer[0].attention.self

        # Disable dropout
        self.mha.eval()

    def forward(self, x, attention_mask):
        result = self.mha(x, attention_mask)[0]
        return result


def run_mha_inference(
    model_version, batch, seq_len, on_weka, dram, pcc, model_location_generator
):
    device = ttl.device.CreateDevice(ttl.device.Arch.GRAYSKULL, 0)
    # Initialize the device
    ttl.device.InitializeDevice(
        device,
        ttl.device.MemoryAllocator.BASIC
        if dram
        else ttl.device.MemoryAllocator.L1_BANKING,
    )
    host = ttl.device.GetHost()
    mem_config = ttl.tensor.MemoryConfig(
        True, ttl.tensor.BufferType.DRAM if dram else ttl.tensor.BufferType.L1
    )

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
    tt_mha_model = TtMultiHeadAttentionModel(
        hugging_face_reference_model.config,
        0,
        hugging_face_reference_model.state_dict(),
        device,
        mem_config,
    )
    pytorch_mha_model = PytorchMultiHeadAttentionModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    mha_input = (
        torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size)
        * 2
    ) - 1
    bert_attention_mask = torch.zeros(batch, 1, 1, seq_len)
    extended_bert_attention_mask = torch.zeros(batch, 1, 32, seq_len)
    pytorch_out = pytorch_mha_model(
        mha_input.squeeze(1), bert_attention_mask
    ).unsqueeze(1)

    pad_mha_input = pad_activation(mha_input)
    tt_mha_input = ttl.tensor.Tensor(
        pad_mha_input.reshape(-1).tolist(),
        pad_mha_input.shape,
        ttl.tensor.DataType.BFLOAT16,
        ttl.tensor.Layout.ROW_MAJOR,
    ).to(ttl.tensor.Layout.TILE)
    tt_mha_input = tt_mha_input.to(device, mem_config)

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

    tt_out = tt_mha_model(tt_mha_input, tt_bert_attention_mask).to(host)
    tt_out1 = torch.Tensor(tt_out.to(ttl.tensor.Layout.ROW_MAJOR).data()).reshape(
        tt_out.shape()
    )

    ttl.device.CloseDevice(device)

    passing, output = comp_pcc(pytorch_out, tt_out1, pcc)
    logger.info(f"Output {output}")
    _, output = comp_allclose(
        pytorch_out, tt_out1, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")
    if not passing:
        logger.error(f"Output PCC < {pcc}")

    assert passing
    # print_diff_argmax(pytorch_out, tt_out1)
    # assert np.allclose(pytorch_out.detach().numpy(), tt_out1, 1e-5, 0.17)


@pytest.mark.parametrize(
    "model_version, batch, seq_len, on_weka, dram, pcc",
    (
        ("phiyodr/bert-large-finetuned-squad2", 9, 384, True, True, 0.99),
        ("phiyodr/bert-large-finetuned-squad2", 9, 384, True, False, 0.99),
    ),
    ids=["DRAM", "L1"],
)
def test_mha_inference(
    model_version, batch, seq_len, on_weka, dram, pcc, model_location_generator
):
    # enable_compile_cache()

    run_mha_inference(
        model_version, batch, seq_len, on_weka, dram, pcc, model_location_generator
    )


if __name__ == "__main__":
    test_mha_inference(
        "phiyodr/bert-large-finetuned-squad2",
        9,
        384,
        True,
        True,
        0.99,
        model_location_generator_,
    )
