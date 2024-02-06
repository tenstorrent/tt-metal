# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib
import pytest
import json
from pathlib import Path
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.reference.model import Transformer
from models.experimental.mistral.reference.tokenizer import Tokenizer
from models.experimental.functional_mistral.tt.ttnn_functional_transformer import mistral_transformer
from models.experimental.functional_mistral.tt.mistral_utility import custom_preprocessor
from models.utility_functions import skip_for_wormhole_b0

from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_wormhole_b0()
@pytest.mark.parametrize(
    "batch_size",
    (
        (1),
        (8),
    ),
)
def test_mistral_transformer_inference(batch_size, model_location_generator, device, reset_seeds):
    prompts = [
        "'A man is sitting on a roof ",
    ] * batch_size
    model_path = model_location_generator("mistral-7B-v0.1", model_subdir="Mistral")
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), n_layers=1, max_batch_size=batch_size, is_whole_model=False)
    with open(model_path / "params.json", "r") as f:
        model_args = TtModelArgs(**json.loads(f.read()))

    model_args.max_batch_size = batch_size
    model_args.n_layers = 1

    reference_model = Transformer(args=model_args)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )
    output_mem_config = tt_lib.tensor.MemoryConfig(
        tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
    )
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)
    input_tokens = torch.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=torch.long, device="cpu")
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = torch.tensor(encoded, dtype=torch.long)
    positions = torch.arange(0, min_prompt_len)

    reference_ouput = reference_model(input_tokens[:, :min_prompt_len], positions)
    del reference_model
    ttnn_input = ttnn.from_torch(input_tokens, dtype=ttnn.uint32, device=device)

    output = mistral_transformer(
        model_args,
        ttnn_input,
        positions,
        parameters=parameters,
        device=device,
        mem_config=output_mem_config,
    )
    output = ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)
    output = ttnn.to_torch(ttnn.from_device(output))

    assert_with_pcc(reference_ouput, output.to(reference_ouput.dtype), 0.98)
