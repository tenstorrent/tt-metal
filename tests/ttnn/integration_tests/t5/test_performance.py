# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

from loguru import logger
import pytest
import torch
import transformers

from models.experimental.functional_t5.tt import ttnn_functional_t5 as functional_t5
from models.utility_functions import torch_random
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters


MODEL_NAME = "google/flan-t5-small"


@pytest.mark.parametrize("model_name", [MODEL_NAME])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [32])
def test_t5_for_conditional_generation(device, use_program_cache, model_name, batch_size, sequence_size):
    config = transformers.T5Config.from_pretrained(model_name)

    torch_input_ids = torch_random((batch_size, sequence_size), 0, 1, dtype=torch.int64)
    torch_decoder_input_ids = torch_random((batch_size, sequence_size), 0, 1, dtype=torch.int64)

    parameters = preprocess_model_parameters(
        f"ttnn_functional_t5",
        initialize_model=lambda: transformers.T5ForConditionalGeneration.from_pretrained(model_name).eval(),
        custom_preprocessor=functional_t5.custom_preprocessor,
        device=device,
    )

    tt_input_ids = ttnn.from_torch(torch_input_ids)
    tt_input_ids = ttnn.to_device(tt_input_ids, device)
    tt_decoder_input_ids = ttnn.from_torch(torch_decoder_input_ids)
    tt_decoder_input_ids = ttnn.to_device(tt_decoder_input_ids, device)
    for _ in range(2):
        start = time.time()
        tt_output = functional_t5.t5_for_conditional_generation(
            tt_input_ids,
            tt_decoder_input_ids,
            parameters=parameters,
            num_heads=config.num_heads,
        )
        tt_output = ttnn.from_device(tt_output)
        end = time.time()
        duration = end - start
        logger.info(f"Duration: {duration}")
        logger.info(f"Samples per second: {1 / duration * batch_size}")
