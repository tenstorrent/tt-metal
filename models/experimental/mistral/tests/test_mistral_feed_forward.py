# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import tt_lib
import pytest
from loguru import logger
import json
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.tt.mistral_feed_forward import TtFeedForward
from models.experimental.mistral.reference.model import FeedForward
from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.utility_functions import (
    comp_pcc,
    comp_allclose,
)


@pytest.mark.parametrize(
    "dtype",
    (tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.DataType.BFLOAT8_B),
)
@pytest.mark.parametrize(
    "pcc",
    ((0.99),),
)
def test_mistral_feed_forward_inference(pcc, model_location_generator, device, dtype, reset_seeds):
    mistral_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    state_dict = torch.load(mistral_path / "consolidated.00.pth")
    base_address = f"layers.0.feed_forward."
    with open(mistral_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    state_dict = {k[22:]: v for k, v in state_dict.items() if (k.startswith("layers.0.feed_forward"))}
    model_args.max_batch_size = 1
    model_args.WEIGHTS_DTYPE = dtype
    reference_model = FeedForward(args=model_args)
    reference_model.load_state_dict(state_dict)
    output_mem_config = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    )
    tt_cache_path = "/mnt/MLPerf/tt_dnn-models/tt/Mistral/"

    tt_model = TtFeedForward(
        args=model_args,
        device=device,
        base_address=base_address,
        tt_cache_path=tt_cache_path,
        output_mem_config=output_mem_config,
    )
    input = torch.rand(1, 11, 4096)
    reference_output = reference_model(input)

    tt_input = torch_to_tt_tensor_rm(input, device, put_on_device=False)

    tt_output = tt_model(tt_input)
    tt_output_torch = tt_to_torch_tensor(tt_output).squeeze(0)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(pcc_message)

    if passing:
        logger.info("Mistral_Feed_Forward Passed!")
    else:
        logger.warning("Mistral_feed_Forward Failed!")

    assert passing, f"Mistral_Feed_forward output does not meet PCC requirement {pcc}."
