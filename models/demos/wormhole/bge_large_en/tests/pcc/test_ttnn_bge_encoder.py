# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.demos.bge_large_en.common import load_torch_model
from models.demos.bge_large_en.ttnn.ttnn_bge_encoder import TtnnBGEEncoder
from models.demos.sentence_bert.reference.sentence_bert import BertEncoder
from models.demos.wormhole.bge_large_en.ttnn.common import BGE_L1_SMALL_SIZE, custom_preprocessor
from tests.ttnn.utils_for_testing import assert_with_pcc, comp_pcc


@pytest.mark.parametrize(
    "inputs",
    [
        ["BAAI/bge-large-en-v1.5", [8, 512, 1024], [8, 1, 1, 512]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": BGE_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_bge_encoder(device, inputs, model_location_generator):
    """Test BGE encoder (all 24 layers) PCC."""
    target_prefix = f"encoder."
    config = transformers.BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    attention_mask = torch.randn(inputs[2], dtype=torch.bfloat16)
    reference_module = BertEncoder(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix=target_prefix, model_location_generator=model_location_generator
    )
    reference_out = reference_module(
        hidden_states,
        attention_mask,
    )
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_module = TtnnBGEEncoder(parameters=parameters, config=config)
    ttnn_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(dim=1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_attention_mask = ttnn.from_torch(attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    sharded_input = ttnn.to_memory_config(
        ttnn_hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            ttnn_hidden_states.shape,
            core_grid=ttnn.CoreGrid(y=8, x=8),  # BGE uses (8, 8) grid
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    ttnn_out = ttnn_module(
        sharded_input,
        ttnn_attention_mask,
        device=device,
    )
    ttnn_out = ttnn.to_torch(ttnn_out).squeeze(dim=1)
    assert_with_pcc(reference_out.last_hidden_state, ttnn_out, 0.98)


@pytest.mark.parametrize(
    "inputs",
    [
        ["BAAI/bge-large-en-v1.5", [8, 512, 1024], [8, 1, 1, 512]],
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": BGE_L1_SMALL_SIZE}], indirect=True)
def test_ttnn_bge_encoder_layer_by_layer(device, inputs, model_location_generator):
    """Debug: Test PCC after each encoder layer to find where divergence occurs."""
    target_prefix = f"encoder."
    config = transformers.BertConfig.from_pretrained(inputs[0])
    hidden_states = torch.randn(inputs[1], dtype=torch.bfloat16)
    attention_mask = torch.randn(inputs[2], dtype=torch.bfloat16)

    # Load PyTorch reference model
    reference_module = BertEncoder(config).to(torch.bfloat16)
    reference_module = load_torch_model(
        reference_module, target_prefix=target_prefix, model_location_generator=model_location_generator
    )

    # Preprocess for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_module,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    ttnn_module = TtnnBGEEncoder(parameters=parameters, config=config)

    # Prepare TTNN inputs
    ttnn_hidden_states = ttnn.from_torch(
        hidden_states.unsqueeze(dim=1), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    ttnn_attention_mask = ttnn.from_torch(attention_mask, layout=ttnn.TILE_LAYOUT, device=device)
    sharded_input = ttnn.to_memory_config(
        ttnn_hidden_states,
        memory_config=ttnn.create_sharded_memory_config(
            ttnn_hidden_states.shape,
            core_grid=ttnn.CoreGrid(y=8, x=8),
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    # Process attention mask
    if ttnn_attention_mask.is_sharded():
        attention_mask_interleaved = ttnn.sharded_to_interleaved(ttnn_attention_mask, ttnn.L1_MEMORY_CONFIG)
        attention_mask_interleaved = ttnn.to_layout(attention_mask_interleaved, ttnn.TILE_LAYOUT)
        ttnn.deallocate(ttnn_attention_mask)
    else:
        attention_mask_interleaved = ttnn_attention_mask

    # Reference: Run through layers and capture outputs
    ref_hidden = hidden_states.clone()
    ref_outputs = []
    for i, layer in enumerate(reference_module.layer):
        layer_outputs = layer(ref_hidden, attention_mask)
        ref_hidden = layer_outputs[0]
        ref_outputs.append(ref_hidden.clone())

    # TTNN: Run through layers and capture outputs
    ttnn_hidden = sharded_input

    print(f"\n{'Layer':<6} {'PCC':<12} {'Status':<10} {'Memory Config'}")
    print("-" * 60)

    for i in range(len(ttnn_module.layers)):
        # Run TTNN layer
        layer_outputs = ttnn_module.layers[i](ttnn_hidden, attention_mask_interleaved, device=device)
        ttnn_hidden = layer_outputs

        # Convert to torch for comparison
        ttnn_out_torch = ttnn.to_torch(ttnn_hidden).squeeze(dim=1)
        ref_out_torch = ref_outputs[i]

        # Calculate PCC
        pcc_passed, pcc_message = comp_pcc(ref_out_torch, ttnn_out_torch, 0.0)
        # Extract PCC value from message (could be string or float)
        if isinstance(pcc_message, (int, float)):
            pcc_value = float(pcc_message)
        elif isinstance(pcc_message, str) and "PCC:" in pcc_message:
            pcc_value = float(pcc_message.split("PCC: ")[1].split(",")[0])
        else:
            # Try to extract from string format
            import re

            match = re.search(r"PCC:\s*([\d.]+)", str(pcc_message))
            pcc_value = float(match.group(1)) if match else 0.0

        # Get memory config info
        mem_config = "sharded" if ttnn_hidden.is_sharded() else "interleaved"
        if ttnn_hidden.is_sharded():
            mem_config += f" (grid: {ttnn_hidden.memory_config().shard_spec.grid})"

        status = "✓ PASS" if pcc_value >= 0.98 else "✗ FAIL"
        print(f"{i:<6} {pcc_value:<12.6f} {status:<10} {mem_config}")

        # Warn if PCC drops significantly but continue to see full picture
        if pcc_value < 0.90 and i < len(ttnn_module.layers) - 1:
            print(f"  ⚠️  PCC dropped below 0.90 at layer {i}, continuing...")

    # Final comparison
    final_ttnn = ttnn.to_torch(ttnn_hidden).squeeze(dim=1)
    final_ref = ref_outputs[-1]
    pcc_passed, pcc_message = comp_pcc(final_ref, final_ttnn, 0.0)
    # Extract PCC value
    if isinstance(pcc_message, (int, float)):
        final_pcc = float(pcc_message)
    elif isinstance(pcc_message, str) and "PCC:" in pcc_message:
        final_pcc = float(pcc_message.split("PCC: ")[1].split(",")[0])
    else:
        import re

        match = re.search(r"PCC:\s*([\d.]+)", str(pcc_message))
        final_pcc = float(match.group(1)) if match else 0.0
    print(f"\n{'Final':<6} {final_pcc:<12.6f} {'✓ PASS' if final_pcc >= 0.98 else '✗ FAIL'}")
