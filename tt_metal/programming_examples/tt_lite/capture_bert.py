# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Capture BERT-large QA trace and export to .ttb file.

Uses the optimized sharded BERT model from models/demos/bert.

Usage:
    python capture_bert.py [--output bert.ttb] [--batch-size 8]

Requirements:
    - Tenstorrent device available
    - transformers library installed (for BERT weights)
    - TT_METAL_HOME and PYTHONPATH set to tt-metal root
"""

import argparse
import sys
import torch

import ttnn
from trace_binary import export_trace


def main():
    parser = argparse.ArgumentParser(description="Capture BERT trace to .ttb")
    parser.add_argument("--output", default="bert.ttb", help="Output .ttb file path")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--sequence-size", type=int, default=384, help="Sequence length")
    parser.add_argument(
        "--model-name",
        default="phiyodr/bert-large-finetuned-squad2",
        help="HuggingFace model name",
    )
    args = parser.parse_args()

    import transformers
    from transformers import BertForQuestionAnswering, BertTokenizer
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.demos.bert.tt import ttnn_optimized_sharded_bert as bert_module
    from models.experimental.functional_common.attention_mask_functions import get_extended_attention_mask

    batch_size = args.batch_size
    sequence_size = args.sequence_size

    print(f"Opening device (l1_small_size=24576, trace_region_size=8388608)...")
    device = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        l1_small_size=24576,
        trace_region_size=8 * 1024 * 1024,
    )

    print(f"Loading BERT model: {args.model_name}")
    hugging_face_model = BertForQuestionAnswering.from_pretrained(args.model_name, torchscript=False)
    hugging_face_model.eval()
    config = hugging_face_model.config
    config = bert_module.update_model_config(config, batch_size)

    print("Preprocessing model parameters...")
    parameters = preprocess_model_parameters(
        model_name=f"ttnn_{args.model_name}_optimized_sharded",
        initialize_model=lambda: transformers.BertForQuestionAnswering.from_pretrained(
            args.model_name, torchscript=False
        ).eval(),
        custom_preprocessor=bert_module.custom_preprocessor,
        device=device,
    )

    # Create dummy inputs
    print("Creating input tensors...")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, sequence_size))
    token_type_ids = torch.zeros_like(input_ids)
    position_ids = torch.arange(sequence_size).unsqueeze(0).expand(batch_size, -1)
    attention_mask_torch = torch.ones(batch_size, sequence_size, dtype=torch.long)

    # Preprocess inputs to ttnn tensors
    tt_input_ids = ttnn.from_torch(input_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_token_type_ids = ttnn.from_torch(
        token_type_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    tt_position_ids = ttnn.from_torch(
        position_ids, dtype=ttnn.uint32, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    extended_attention_mask = get_extended_attention_mask(attention_mask_torch, input_ids.shape, torch.bfloat16)
    extended_attention_mask = extended_attention_mask.expand((batch_size, -1, -1, -1))
    extended_attention_mask = torch.clamp(extended_attention_mask, min=-100000)
    tt_attention_mask = ttnn.from_torch(
        extended_attention_mask,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Compile run
    print("Running compile pass...")
    compile_output = bert_module.bert_for_question_answering(
        config,
        tt_input_ids,
        tt_token_type_ids,
        tt_position_ids,
        tt_attention_mask,
        parameters=parameters,
    )
    print("Compile pass complete.")

    # Deallocate compile output
    ttnn.deallocate(compile_output, force=True)

    # Capture trace
    print("Capturing trace...")
    trace_id = ttnn.begin_trace_capture(device)
    output_tensor = bert_module.bert_for_question_answering(
        config,
        tt_input_ids,
        tt_token_type_ids,
        tt_position_ids,
        tt_attention_mask,
        parameters=parameters,
    )
    ttnn.end_trace_capture(device, trace_id)
    print("Trace captured.")

    # Verify trace replay
    print("Verifying trace replay...")
    ttnn.execute_trace(device, trace_id, blocking=True)
    result = ttnn.to_torch(output_tensor)
    print(f"Output shape: {result.shape}")

    # Export to .ttb
    io_tensors = {
        "input_ids": tt_input_ids,
        "token_type_ids": tt_token_type_ids,
        "position_ids": tt_position_ids,
        "attention_mask": tt_attention_mask,
        "qa_output": output_tensor,
    }
    export_trace(device, trace_id, args.output, io_tensors)

    ttnn.release_trace(device, trace_id)
    ttnn.close_mesh_device(device)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
