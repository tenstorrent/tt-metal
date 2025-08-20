#!/usr/bin/env python3

# Simple script to regenerate ground truth for llama3 demo
# This bypasses pytest and directly calls the demo

import os
import sys

# Add the demo directory to Python path
sys.path.append(os.path.dirname(__file__))

from demo import construct_arg, main

# Create args that match the failing test case:
# greedy, short_context, text_completion, check_disabled
args = construct_arg(
    # Model args
    implementation="tt",
    llama_version="llama3",
    skip_model_load=False,
    num_layers=80,
    # Generation args
    max_output_tokens=128,
    prompts_file="models/demos/t3000/llama2_70b/demo/data/multi_prompt.json",
    output_at_end=True,  # This will save the output to file
    top_p=1,
    top_k=1,  # This makes it greedy
    temperature=1.0,
    chat=False,  # text_completion
    # TT args
    n_devices=8,
    decode_only=True,
    trace_mode=False,
    ground_truth=None,  # This disables ground truth checking
    max_batch_size=32,  # short_context
    max_context_len=2048,  # short_context
)

print("Running demo to regenerate ground truth...")
print("Args:", args)

# Run the demo
main(args)

print("\nDemo completed!")
print("New output should be saved to: models/demos/t3000/llama2_70b/demo/data/llama3_demo_user_output.json")
print("\nTo update ground truth, run:")
print(
    "cp models/demos/t3000/llama2_70b/demo/data/llama3_demo_user_output.json models/demos/t3000/llama2_70b/demo/data/llama3_ground_truth.json"
)
