# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from transformers import AutoTokenizer
from models.generation_utils import run_generate


def run_demo_t5(t5_model_constructor):
    input_sentance = "translate English to German: The house is wonderful."
    tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=32)

    device = ttnn.open_device(0)

    output_sentance = run_generate(
        input_sentance,
        tokenizer,
        t5_model_constructor,
        device,
        run_tt_model=True,
        log=False,
    )

    logger.info(f"Input sentance: '{input_sentance}'")
    logger.info(f"Tt output: '{output_sentance}'")

    ttnn.close_device(device)
