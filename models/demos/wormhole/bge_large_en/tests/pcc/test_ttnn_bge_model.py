# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.bge_large_en.common import load_torch_model
from models.demos.bge_large_en.ttnn.ttnn_bge_model import TtnnBGEModel
from models.demos.sentence_bert.reference.sentence_bert import BertModel, custom_extended_mask
from models.demos.wormhole.bge_large_en.ttnn.common import BGE_L1_SMALL_SIZE, custom_preprocessor, preprocess_inputs
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize(
    "inputs",
    [["BAAI/bge-large-en-v1.5", [8, 512], [8, 1, 1, 512]]],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": BGE_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_bge_model(device, inputs, model_location_generator):
    """Test BGE-large-en-v1.5 model with PCC validation against PyTorch reference."""
    config = transformers.BertConfig.from_pretrained(inputs[0])

    # Generate random inputs
    random_seed = 42
    torch.manual_seed(random_seed)
    input_ids = torch.randint(low=0, high=config.vocab_size - 1, size=inputs[1], dtype=torch.int64)
    attention_mask = torch.ones(inputs[1][0], inputs[1][1])
    extended_mask = custom_extended_mask(attention_mask, dtype=torch.bfloat16)
    token_type_ids = torch.zeros(inputs[1], dtype=torch.int64)
    position_ids = torch.arange(0, inputs[1][1], dtype=torch.int64).unsqueeze(dim=0)

    # Load PyTorch reference model
    reference_module = BertModel(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix="", model_location_generator=model_location_generator
    )
    reference_out = reference_module(
        input_ids,
        extended_attention_mask=extended_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    # Preprocess model parameters for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    # Create TTNN model
    ttnn_module = TtnnBGEModel(parameters=parameters, config=config)

    # Preprocess inputs for TTNN
    (
        ttnn_input_ids,
        ttnn_token_type_ids,
        ttnn_position_ids,
        ttnn_extended_attention_mask,
        ttnn_attention_mask,
    ) = preprocess_inputs(input_ids, token_type_ids, position_ids, extended_mask, attention_mask, device)

    # Run TTNN inference
    ttnn_out = ttnn_module(
        ttnn_input_ids,
        ttnn_extended_attention_mask,
        ttnn_attention_mask,
        ttnn_token_type_ids,
        ttnn_position_ids,
        device=device,
    )

    # Convert to PyTorch for comparison
    ttnn_out = ttnn.to_torch(ttnn_out[0])

    # Validate with PCC (BGE-large may have slightly lower PCC due to larger model)
    assert_with_pcc(reference_out.post_processed_output, ttnn_out, 0.98)
