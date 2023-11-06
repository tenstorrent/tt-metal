# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from loguru import logger
from transformers import BertForQuestionAnswering

import tt_lib

from tt_lib.utils import pad_activation
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
    profiler,
)
from models.demos.metal_BERT_large_15.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.metal_BERT_large_15.tt.mha import TtMultiHeadAttentionModel


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
    device, model_version, batch, seq_len, pcc, model_config, tt_cache_path, model_location_generator
):
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tt_mha_model = TtMultiHeadAttentionModel(
        hugging_face_reference_model.config,
        0,
        hugging_face_reference_model.state_dict(),
        device,
        model_config,
        tt_cache_path,
    )
    pytorch_mha_model = PytorchMultiHeadAttentionModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    mha_input = (torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size) * 2) - 1
    bert_attention_mask = torch.zeros(batch, 1, 1, seq_len)
    extended_bert_attention_mask = torch.zeros(batch, 1, 32, seq_len)
    pytorch_out = pytorch_mha_model(mha_input.squeeze(1), bert_attention_mask).unsqueeze(1)

    pad_mha_input = pad_activation(mha_input)
    tt_mha_input = tt_lib.tensor.Tensor(
        pad_mha_input.reshape(-1).tolist(),
        pad_mha_input.shape,
        model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"],
        tt_lib.tensor.Layout.ROW_MAJOR,
    ).to(tt_lib.tensor.Layout.TILE)
    tt_mha_input = tt_mha_input.to(device, model_config["OP1_FUSED_QKV_MM_INPUT_MEMCFG"])

    tt_bert_attention_mask = (
        tt_lib.tensor.Tensor(
            extended_bert_attention_mask.reshape(-1).tolist(),
            extended_bert_attention_mask.shape,
            model_config["OP8_SOFTMAX_ATTENTION_MASK_DTYPE"],
            tt_lib.tensor.Layout.ROW_MAJOR,
        )
        .to(tt_lib.tensor.Layout.TILE)
        .to(device, model_config["OP8_SOFTMAX_ATTENTION_MASK_MEMCFG"])
    )

    tt_out = tt_mha_model(tt_mha_input, tt_bert_attention_mask).cpu()
    tt_out1 = tt_out.to(tt_lib.tensor.Layout.ROW_MAJOR).to_torch().reshape(tt_out.shape())

    passing, output = comp_pcc(pytorch_out, tt_out1, pcc)
    logger.info(f"Output {output}")
    _, output = comp_allclose(
        pytorch_out, tt_out1, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")
    if not passing:
        logger.error(f"Output PCC < {pcc}")

    if model_config["DEFAULT_DTYPE"] == tt_lib.tensor.DataType.BFLOAT8_B and not passing:
        pytest.xfail("PCC is garbage for BFLOAT8_B. Numbers are for perf only!")

    assert passing


@pytest.mark.parametrize(
    "batch, model_config_str",
    (
        (9, "BFLOAT8_B-DRAM"),
        (9, "BFLOAT16-DRAM"),
        (9, "BFLOAT8_B-L1"),
        (9, "BFLOAT16-L1"),
        (9, "MIXED_PRECISION_BATCH9"),
        (8, "MIXED_PRECISION_BATCH8"),
    ),
    ids=[
        "batch_9-BFLOAT8_B-DRAM",
        "batch_9-BFLOAT16-DRAM",
        "batch_9-BFLOAT8_B-L1",
        "batch_9-BFLOAT16-L1",
        "batch_9-MIXED_PRECISION_BATCH9",
        "batch_8-MIXED_PRECISION_BATCH8",
    ],
)
@pytest.mark.parametrize(
    "model_version, seq_len, pcc",
    (("phiyodr/bert-large-finetuned-squad2", 384, 0.99),),
    ids=["BERT_LARGE"],
)
def test_mha_inference(
    device,
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    model_location_generator,
    request,
):
    model_config = get_model_config(model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    tt_lib.profiler.set_profiler_location(f"BERT_large_mha_{request.node.callspec.id}")

    run_mha_inference(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
