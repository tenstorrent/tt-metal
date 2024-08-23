# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import math
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, skip_for_grayskull, get_devices_for_t3000
from models.demos.t3000.falcon40b.tt.model_config import (
    get_model_config,
)

from torch.nn import functional as F


class PytorchFalconSoftmax(torch.nn.Module):
    def __init__(self, head_dim=64):
        super().__init__()
        self.scale = math.sqrt(head_dim)

    def forward(self, attention_scores, attention_mask):
        attention_mask_float = (attention_mask * 1.0).masked_fill(attention_mask, -100000.0).to(torch.bfloat16)

        attention_scores /= self.scale

        attention_scores = F.softmax(
            attention_scores + attention_mask_float,
            dim=-1,
            dtype=torch.bfloat16,
        )
        return attention_scores


class TtFalconSoftmax:
    def __init__(self, device, model_config, head_dim: int = 64, seqlen: int = 32):
        super().__init__()

        self.model_config = model_config
        self.seqlen = seqlen
        self.scalar = 1 / math.sqrt(head_dim)

    def __call__(
        self, x: ttnn.experimental.tensor.Tensor, attention_mask: ttnn.experimental.tensor.Tensor
    ) -> ttnn.experimental.tensor.Tensor:
        out = ttnn.scale_causal_mask_hw_dims_softmax_in_place(
            x,
            self.scalar,
            attention_mask,
            program_config=self.model_config["SOFTMAX_PROGCFG"],
        )

        return out


def run_test_FalconSoftmax_inference(
    pcc,
    device,
    model_location_generator,
):
    head_dim = 64
    seqlen = 64
    num_attention_heads = 16

    # Prepare input
    torch.manual_seed(0)
    model_input_shape = [1, seqlen]

    model_config = get_model_config("BFLOAT8_B-DRAM", "prefill", model_input_shape, 8)

    input_shape = [1, num_attention_heads, seqlen, seqlen]
    input_torch = (torch.rand(input_shape) * 2) - 1
    input = torch2tt_tensor(input_torch, None, tt_dtype=ttnn.experimental.tensor.DataType.BFLOAT16)
    input = input.to(device, ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM))

    attention_mask_bool = torch.ones(1, 1, seqlen, seqlen, dtype=bool).triu(diagonal=1)

    input = ttnn.interleaved_to_sharded(input, model_config["SOFTMAX_HEIGHT_SHARDED_MEMCFG"])

    attn_mask_bool = torch.ones(1, 1, seqlen, seqlen, dtype=bool)
    attn_mask_bool = attn_mask_bool.triu(diagonal=1)
    attention_mask_memconfig = ttnn.DRAM_MEMORY_CONFIG

    tt_attn_mask = ttnn.as_tensor(
        tensor=attn_mask_bool,
        dtype=model_config["BFLOAT16_DTYPE"],
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=attention_mask_memconfig,
        preprocess=lambda x: (x * -1e5),
    )

    tt_attn_mask = ttnn.tilize(
        tt_attn_mask,
        memory_config=attention_mask_memconfig,
        dtype=model_config["ATTN_MASK_DTYPE"],
    )

    # PyTorch output --------------------------------------------------------------------
    pytorch_falcon_softmax = PytorchFalconSoftmax(head_dim=head_dim)
    torch_out = pytorch_falcon_softmax(input_torch, attention_mask_bool)

    # TT hardware execution -------------------------------------------------------------
    tt_falcon_softmax = TtFalconSoftmax(device, model_config=model_config, head_dim=head_dim, seqlen=seqlen)
    tt_out = tt_falcon_softmax(input, tt_attn_mask)

    tt_out = tt2torch_tensor(tt_out)

    # check outputs ----------------------------------------------------------------------
    does_pass, output_pcc = comp_pcc(torch_out, tt_out, pcc)
    logger.info(f"PCC value: {output_pcc}")

    if does_pass:
        logger.info("Falcon MLP output Passed!")
    else:
        logger.warning("Falcon MLP output Failed!")
        assert does_pass, f"PCC value is lower than {pcc}"


@skip_for_grayskull("Requires eth connected devices to run")
@pytest.mark.parametrize("pcc", [(0.99)])
def test_FalconSoftmax_inference(
    pcc,
    all_devices,
    model_location_generator,
):
    device = all_devices[0]

    run_test_FalconSoftmax_inference(
        pcc,
        device,
        model_location_generator,
    )
