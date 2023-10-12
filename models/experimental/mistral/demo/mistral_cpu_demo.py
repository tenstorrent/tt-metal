# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.mistral.reference.tokenizer import Tokenizer, generate
from models.experimental.mistral.reference.model import Transformer
from pathlib import Path
import pytest
from loguru import logger


@pytest.mark.parametrize(
    "max_tokens,max_batch_size",
    ((35, 1),),
)
def test_demo_mistral_whole_model(model_location_generator, max_tokens, max_batch_size):
    context = [
        "My name is Julien and I like to make music. I've been writing for years and I've now decided to write more and ",
    ] * max_batch_size

    model_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(
        Path(model_path), n_layers=32, max_batch_size=max_batch_size, is_whole_model=True
    )

    res, _logprobs = generate(
        context,
        transformer,
        tokenizer,
        max_tokens=max_tokens,
    )

    logger.info(f"Mistral Model input context: {context}")
    logger.info(f"Mistral CPU Demo Output: {res}")


@pytest.mark.parametrize(
    "max_tokens,max_batch_size",
    ((35, 1),),
)
def test_demo_mistral_single_layer(model_location_generator, max_tokens, max_batch_size):
    sample_context = [
        "This is a sample text for single layer execution ",
    ] * max_batch_size

    model_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(
        Path(model_path), n_layers=1, max_batch_size=max_batch_size, is_whole_model=False
    )

    res, _ = generate(
        sample_context,
        transformer,
        tokenizer,
        max_tokens=max_tokens,
    )

    logger.info(f"Mistral Model input context: {sample_context}")
    logger.info(f"Mistral CPU Demo Output: {res}")
