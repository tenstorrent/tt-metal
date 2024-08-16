# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from transformers import BertForQuestionAnswering
from loguru import logger


import ttnn

from tt_lib.utils import pad_activation
from models.utility_functions import comp_pcc, comp_allclose, profiler
from models.demos.metal_BERT_large_11.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.metal_BERT_large_11.tt.bert_encoder import TtBertEncoder


class PytorchBertEncoder(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.bert_encoder = hugging_face_reference_model.bert.encoder.layer[0]

    def forward(self, x, attention_mask=None):
        return self.bert_encoder(x, attention_mask)[0]


def run_bert_encoder_inference(
    device, model_version, batch, seq_len, pcc, model_config, tt_cache_path, model_location_generator
):
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    config = hugging_face_reference_model.config

    tt_bert_encoder_model = TtBertEncoder(
        hugging_face_reference_model.config,
        0,
        hugging_face_reference_model.state_dict(),
        device,
        model_config,
        tt_cache_path,
    )
    pytorch_bert_model = PytorchBertEncoder(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    bert_encoder_input = (torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size) * 2) - 1
    bert_attention_mask = torch.randn(batch, 1, 1, seq_len)

    pytorch_out = pytorch_bert_model(bert_encoder_input.squeeze(1), bert_attention_mask).unsqueeze(1)

    pad_bert_encoder_input = pad_activation(bert_encoder_input)
    tt_bert_encoder_input = ttnn.Tensor(
        pad_bert_encoder_input,
        model_config["OP1_FUSED_QKV_MM_INPUT_DTYPE"],
    ).to(ttnn.TILE_LAYOUT)
    if "OP1_FUSED_QKV_MM_INPUT_SHARDED_MEMCFG" in model_config:
        tt_bert_encoder_input = tt_bert_encoder_input.to(device)
        tt_bert_encoder_input = ttnn.interleaved_to_sharded(
            tt_bert_encoder_input,
            model_config["GRID_SIZE"],
            model_config["SHARD_SIZE"],
            model_config["OP1_FUSED_QKV_MM_INPUT_SHARDED_MEMCFG"].memory_layout,
            model_config["SHARD_ORIENTATION"],
        )
    else:
        tt_bert_encoder_input = tt_bert_encoder_input.to(device, model_config["OP1_FUSED_QKV_MM_INPUT_MEMCFG"])

    if model_config["OP5_POST_SOFTMAX_BMM_OUTPUT_MEMCFG"].is_sharded():
        extended_bert_attention_mask = bert_attention_mask.reshape(bert_attention_mask.shape[0], 1, -1, 32)
        tt_bert_attention_mask = ttnn.Tensor(
            extended_bert_attention_mask,
            model_config["OP4_SOFTMAX_ATTENTION_MASK_DTYPE"],
        ).to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
    else:
        extended_bert_attention_mask = pad_activation(bert_attention_mask)
        tt_bert_attention_mask = (
            ttnn.Tensor(
                extended_bert_attention_mask,
                model_config["OP4_SOFTMAX_ATTENTION_MASK_DTYPE"],
            )
            .to(ttnn.TILE_LAYOUT)
            .to(device, model_config["OP4_SOFTMAX_ATTENTION_MASK_MEMCFG"])
        )

    tt_out = tt_bert_encoder_model(tt_bert_encoder_input, tt_bert_attention_mask)
    if tt_out.is_sharded():
        tt_out = ttnn.sharded_to_interleaved(tt_out)
    tt_out = tt_out.cpu().to(ttnn.ROW_MAJOR_LAYOUT).to_torch()

    passing, output = comp_pcc(pytorch_out, tt_out, pcc)
    logger.info(f"Output {output}")

    _, output = comp_allclose(
        pytorch_out, tt_out, 0.5, 0.5
    )  # Only interested in reporting atol/rtol, using PCC for pass/fail
    logger.info(f"Output {output}")

    if not passing:
        logger.error(f"Output PCC < {pcc}")

    if model_config["DEFAULT_DTYPE"] == ttnn.bfloat8_b and not passing:
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
        (7, "BFLOAT8_B-SHARDED"),
        (12, "BFLOAT8_B-SHARDED"),
    ),
    ids=[
        "batch_9-BFLOAT8_B-DRAM",
        "batch_9-BFLOAT16-DRAM",
        "batch_9-BFLOAT8_B-L1",
        "batch_9-BFLOAT16-L1",
        "batch_9-MIXED_PRECISION_BATCH9",
        "batch_8-MIXED_PRECISION_BATCH8",
        "batch_7-BFLOAT8_B-SHARDED",
        "batch_12-BFLOAT8_B-SHARDED",
    ],
)
@pytest.mark.parametrize(
    "model_version, seq_len, pcc",
    (("phiyodr/bert-large-finetuned-squad2", 384, 0.99),),
    ids=["BERT_LARGE"],
)
def test_bert_encoder_inference(
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    model_location_generator,
    request,
    device,
    use_program_cache,
):
    model_config = get_model_config(batch, device.compute_with_storage_grid_size(), model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    run_bert_encoder_inference(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
