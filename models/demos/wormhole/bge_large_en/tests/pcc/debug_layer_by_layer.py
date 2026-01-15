# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Debug script to check PCC after each encoder layer to find where divergence occurs.
"""

import torch
import transformers
from ttnn.model_preprocessing import preprocess_model_parameters

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.bge_large_en.common import load_torch_model
from models.demos.bge_large_en.ttnn.ttnn_bge_encoder import TtnnBGEEncoder
from models.demos.sentence_bert.reference.sentence_bert import BertEncoder
from models.demos.wormhole.bge_large_en.ttnn.common import custom_preprocessor


def test_layer_by_layer(device):
    """Test PCC after each layer to find where divergence occurs."""
    model_name = "BAAI/bge-large-en-v1.5"
    config = transformers.BertConfig.from_pretrained(model_name)

    # Generate random inputs
    hidden_states = torch.randn([8, 384, 1024], dtype=torch.bfloat16)
    attention_mask = torch.randn([8, 1, 1, 384], dtype=torch.bfloat16)

    # Load PyTorch reference model
    reference_encoder = BertEncoder(config).to(torch.bfloat16)
    reference_encoder = load_torch_model(reference_encoder, target_prefix="encoder.", model_location_generator=None)

    # Preprocess for TTNN
    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_encoder,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )

    ttnn_encoder = TtnnBGEEncoder(parameters=parameters, config=config)

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
    for i, layer in enumerate(reference_encoder.layer):
        layer_outputs = layer(ref_hidden, attention_mask)
        ref_hidden = layer_outputs[0]
        ref_outputs.append(ref_hidden.clone())

    # TTNN: Run through layers and capture outputs
    ttnn_hidden = sharded_input
    ttnn_outputs = []

    print(f"\n{'Layer':<6} {'PCC':<10} {'Status'}")
    print("-" * 30)

    for i in range(len(ttnn_encoder.layers)):
        # Run TTNN layer
        layer_outputs = ttnn_encoder.layers[i](ttnn_hidden, attention_mask_interleaved, device=device)
        ttnn_hidden = layer_outputs

        # Convert to torch for comparison
        ttnn_out_torch = ttnn.to_torch(ttnn_hidden).squeeze(dim=1)
        ref_out_torch = ref_outputs[i]

        # Calculate PCC
        pcc_passed, pcc_message = comp_pcc(ref_out_torch, ttnn_out_torch, 0.0)
        pcc_value = float(pcc_message.split("PCC: ")[1].split(",")[0]) if "PCC:" in pcc_message else 0.0

        status = "✓" if pcc_value >= 0.98 else "✗"
        print(f"{i:<6} {pcc_value:<10.6f} {status}")

        ttnn_outputs.append(ttnn_hidden)

        # Stop if PCC drops significantly
        if pcc_value < 0.90:
            print(f"\n⚠️  Significant PCC drop detected at layer {i}!")
            break

    # Final comparison
    final_ttnn = ttnn.to_torch(ttnn_hidden).squeeze(dim=1)
    final_ref = ref_outputs[-1]
    pcc_passed, pcc_message = comp_pcc(final_ref, final_ttnn, 0.0)
    final_pcc = float(pcc_message.split("PCC: ")[1].split(",")[0]) if "PCC:" in pcc_message else 0.0
    print(f"\n{'Final':<6} {final_pcc:<10.6f} {'✓' if final_pcc >= 0.98 else '✗'}")


if __name__ == "__main__":
    pass

    # This would need to be run with pytest fixture, but for now just print usage
    print("This script needs to be run with pytest device fixture")
    print(
        "Run: pytest models/demos/wormhole/bge_large_en/tests/pcc/test_ttnn_bge_encoder.py -k test_layer_by_layer -v -s"
    )
