# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from loguru import logger
from transformers import BertForQuestionAnswering

import ttnn

from tt_lib.utils import pad_activation
from models.utility_functions import comp_pcc, comp_allclose
from models.demos.metal_BERT_large_11.tt.model_config import get_model_config, get_tt_cache_path
from models.demos.metal_BERT_large_11.tt.ffn import TtFeedForwardModel


class PytorchFeedForwardModel(torch.nn.Module):
    def __init__(self, hugging_face_reference_model):
        super().__init__()
        self.ff1 = hugging_face_reference_model.bert.encoder.layer[0].intermediate
        self.ff2 = hugging_face_reference_model.bert.encoder.layer[0].output.dense

    def forward(self, x):
        return self.ff2(self.ff1(x))


def run_ffn_inference(
    device, model_version, batch, seq_len, pcc, model_config, tt_cache_path, model_location_generator
):
    model_name = str(model_location_generator(model_version, model_subdir="Bert"))

    hugging_face_reference_model = BertForQuestionAnswering.from_pretrained(model_name, torchscript=False)
    tt_ffn_model = TtFeedForwardModel(
        0,
        hugging_face_reference_model.state_dict(),
        device,
        model_config,
        tt_cache_path,
    )
    pytorch_ffn_model = PytorchFeedForwardModel(hugging_face_reference_model)

    # Prepare input
    torch.manual_seed(0)
    ffn_input = (torch.rand(batch, 1, seq_len, hugging_face_reference_model.config.hidden_size) * 2) - 1

    pytorch_out = pytorch_ffn_model(ffn_input)

    pad_ffn_input = pad_activation(ffn_input)
    tilized_ffn_input = ttnn.Tensor(
        pad_ffn_input,
        model_config["OP8_LAYERNORM_OUTPUT_DTYPE"],
    ).to(ttnn.TILE_LAYOUT)
    if model_config["OP8_LAYERNORM_OUTPUT_MEMCFG"].is_sharded():
        tilized_ffn_input = tilized_ffn_input.to(device)
        tilized_ffn_input = ttnn.interleaved_to_sharded(
            tilized_ffn_input,
            model_config["GRID_SIZE"],
            model_config["SHARD_SIZE"],
            model_config["OP8_LAYERNORM_OUTPUT_MEMCFG"].memory_layout,
            model_config["SHARD_ORIENTATION"],
        )
    else:
        tilized_ffn_input = tilized_ffn_input.to(device, model_config["OP8_LAYERNORM_OUTPUT_MEMCFG"])

    tt_out = tt_ffn_model(tilized_ffn_input)
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
def test_ffn_inference(
    device,
    model_version,
    batch,
    seq_len,
    pcc,
    model_config_str,
    model_location_generator,
    request,
):
    model_config = get_model_config(batch, device.compute_with_storage_grid_size(), model_config_str)
    tt_cache_path = get_tt_cache_path(model_version)

    run_ffn_inference(
        device,
        model_version,
        batch,
        seq_len,
        pcc,
        model_config,
        tt_cache_path,
        model_location_generator,
    )
